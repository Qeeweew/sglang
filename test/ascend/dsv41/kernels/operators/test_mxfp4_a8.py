"""CPU packing and opt-in 910B eager/Graph checks against checkpoint E2M1."""

import os

import numpy as np
import pytest
import torch
from dsv41.kernels.references.mxfp4_grouped import a8_embedding, fixture, weight
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import mxfp4_a8


def tensors(*arrays):
    return [torch.from_numpy(value) for value in arrays]


@pytest.mark.parametrize("underflow", [False, True])
def test_packing(underflow):
    for depth in (256, 288, 2304, 5120):
        codes, scales = fixture(1, 64, depth)
        if underflow:
            codes[0] = np.arange(codes[0].size).astype(np.uint8).reshape(64, -1)
            scales[0] = np.resize([117, 118, 119, 120, 121, 122, 123, 125], depth // 32)
        packed, factors, channel = mxfp4_a8.pack_expert(*tensors(codes[0], scales[0]))
        assert (
            packed.nbytes + factors.nbytes + channel.nbytes
            <= codes.nbytes + scales.nbytes
        )

        def unpack(data):
            raw = data.numpy()
            nibbles = np.stack((raw & 15, raw >> 4), axis=-1).reshape(-1)
            return ((nibbles.astype(np.int16) + 8) % 16) - 8

        panel = max(v for v in range(32, 2049, 32) if depth % v == 0)
        quantum = max(v for v in range(1024, 16385, 1024) if (panel * 32) % v == 0)
        words = packed.numpy().view(np.uint16).reshape(-1, quantum // 4)
        symbols = np.stack(
            [
                ((words >> mag) & 7) | (((words >> sign) & 1) << 3)
                for mag, sign in ((0, 6), (3, 7), (8, 14), (11, 15))
            ],
            axis=1,
        )
        symbols = (
            symbols.reshape(1, -1, 2, 32, 32).swapaxes(-1, -2).reshape(1, -1, 64, 32)
        )
        rank = np.where(symbols & 8, 1, -1) * (symbols & 7)
        factor = unpack(factors).reshape(1, -1, 64, 1)
        quantized = (
            np.sign(rank) * np.array([0, 1, 2, 3, 4, 6, 8, 12])[abs(rank)] * factor
        )
        restored = (
            quantized.transpose(0, 2, 1, 3).reshape(64, depth)
            * channel.numpy()[:, None]
        )
        original = weight(codes[0], scales[0], depth).astype(np.float64)
        expected = a8_embedding(original, scales[0]) if underflow else original
        np.testing.assert_array_equal(restored, expected)
        assert np.all(np.abs(restored) <= np.abs(original))
        assert np.all(np.abs(restored - original) < channel.numpy()[:, None])


def test_outliers():
    codes = torch.zeros((64, 128), dtype=torch.uint8)
    scales = torch.full((64, 8), 125, dtype=torch.uint8)
    scales[:, 0] = 120
    mxfp4_a8.pack_expert(codes, scales)
    codes[3, 0] = 0xD5
    with pytest.raises(ValueError, match="non-exact"):
        mxfp4_a8.pack_expert(codes, scales, allow_rounding=False)
    packed, _, channel = mxfp4_a8.pack_expert(codes, scales)
    assert packed.view(torch.int16)[3] & 0x47 == 1
    assert packed.view(torch.int16)[35] & 0x47 == 0x41
    assert channel[3] == 2**-6
    scales[0, 0] = 255
    with pytest.raises(ValueError, match="NaN"):
        mxfp4_a8.pack_expert(codes, scales, allow_rounding=True)


@pytest.mark.skipif(os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B")
def test_integer_group_factors():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    width, depth = 64, 256
    codes, exps = fixture(1, width, depth)
    exps.fill(128)
    packed, _, _ = mxfp4_a8.pack_expert(*tensors(codes[0], exps[0]))
    factor = (np.arange(width)[:, None] + np.arange(depth // 32)) % 8 + 1
    neg = (-factor.T).astype(np.uint8)
    metadata = ((neg[:, ::2] & 15) | (neg[:, 1::2] << 4)).ravel()
    channel = np.linspace(0.011, 0.039, width, dtype=np.float32)
    weights = tuple(
        value[None, :].npu() for value in (packed, *tensors(metadata, channel))
    )
    activation = torch.eye(depth, dtype=torch.int8).npu()
    route = torch.tensor([1, depth, 0, 1, 0, 0, depth, 0], dtype=torch.int32).npu()
    token = torch.ones(depth, dtype=torch.float32, device="npu")
    out = torch.empty((depth, width), dtype=torch.float16, device="npu")
    expected = (
        (weight(codes[0], exps[0], depth) * factor.repeat(32, axis=1)).astype(
            np.float32
        )
        * channel[:, None]
    ).T.astype(np.float16)
    invoke = mxfp4_a8.bind(activation, weights, token, route, out=out)
    invoke()
    np.testing.assert_array_equal(out.cpu().numpy(), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        invoke()
    activation.neg_()
    out.fill_(float("nan"))
    graph.replay()
    np.testing.assert_array_equal(out.cpu().numpy(), -expected)


@pytest.mark.skipif(os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B")
def test_projection_requantizes_on_replay():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    width, depth = 64, 256
    raw, exps = fixture(1, width, depth)
    weights = tuple(
        item[None].npu() for item in mxfp4_a8.pack_expert(*tensors(raw[0], exps[0]))
    )
    x = torch.linspace(-2, 3, 2 * depth).reshape(2, depth).half().npu()
    route = torch.tensor([1, 2, 0, 1, 0, 0, 2, 0], dtype=torch.int32).npu()
    out = x.new_empty((2, width))
    projection = mxfp4_a8.A8GroupedGemm(depth, width, experts=1, device=x.device)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        invoke = projection.bind(x, *weights, route, out=out)
        invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        invoke()
    torch.npu.current_stream().wait_stream(stream)
    x.copy_(x.flip([1]) * 0.25)
    graph.replay()
    quantized, scale = mxfp4_a8.quantize_tokens(x)
    ref = weight(raw[0], exps[0], depth).astype(np.float64)
    expected = (quantized.cpu().numpy() @ ref.T).astype(
        np.float32
    ) * scale.cpu().numpy()[:, None]
    np.testing.assert_array_equal(out.cpu().numpy(), expected.astype(np.float16))


@pytest.mark.skipif(os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B")
@pytest.mark.parametrize(
    "depth,width", [(256, 64), (288, 576), (832, 640), (2304, 5120), (5120, 4608)]
)
def test_routed_graph(depth, width):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    experts, capacity = 2, 24
    codes, exps = fixture(experts, width, depth)
    rounded = depth == 288
    if rounded:
        exps[:, :, 0] = 120
    weights = [
        mxfp4_a8.pack_expert(*tensors(raw, scale), allow_rounding=rounded)
        for raw, scale in zip(codes, exps)
    ]
    weights = tuple(torch.stack(items).npu() for items in zip(*weights))
    reference = [
        a8_embedding(weight(raw, scale, depth), scale)
        for raw, scale in zip(codes, exps)
    ]
    rng = np.random.default_rng(31)
    x = torch.from_numpy(rng.normal(size=(capacity, depth)).astype(np.float16)).npu()
    x[0].zero_()
    quantized, token = mxfp4_a8.quantize_tokens(x)
    out = x.new_full((capacity, width), -19)
    route = torch.zeros(4 + 4 * experts, dtype=torch.int32, device=x.device)
    if depth == 256:
        with pytest.raises(ValueError, match="contiguous"):
            mxfp4_a8.bind(quantized[::2], weights, token[::2], route, out=out[::2])
        source = x[::2]
        codes_nd, scales_nd = mxfp4_a8.quantize_tokens(source)
        restored = codes_nd.cpu().float() * scales_nd.cpu()[:, None]
        assert torch.all(
            (restored - source.cpu().float()).abs()
            <= scales_nd.cpu()[:, None] * 0.501 + 1e-6
        )
    invoke = mxfp4_a8.bind(quantized, weights, token, route, out=out)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        invoke()

    for replay, spans in [
        (False, [(0, 0, 4), (1, 6, 17)]),
        (True, [(1, 2, 1)]),
        (True, []),
    ]:
        if replay:
            x.neg_()
            new_q, new_s = mxfp4_a8.quantize_tokens(x.bfloat16())
            quantized.copy_(new_q)
            token.copy_(new_s)
        records = [len(spans), sum(count for _, _, count in spans), 0, 1]
        for expert, first, count in spans:
            records += [expert, first, count, 0]
        route[: len(records)].copy_(torch.tensor(records, dtype=torch.int32))
        out.fill_(-19)
        if replay:
            graph.replay()
        else:
            with torch.npu.stream(stream):
                stream.wait_stream(torch.npu.default_stream())
                invoke()
            torch.npu.current_stream().wait_stream(stream)
        result = out.cpu().numpy()
        expected = np.full_like(result, -19)
        for expert, first, count in spans:
            block = slice(first, first + count)
            dot = (
                quantized.cpu().numpy()[block].astype(np.float64)
                @ reference[expert].astype(np.float64).T
            )
            expected[block] = dot.astype(np.float32) * token.cpu().numpy()[block, None]
        np.testing.assert_array_equal(result, expected)
