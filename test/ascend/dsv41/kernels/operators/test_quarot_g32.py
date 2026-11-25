import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import int4_g32_a8
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.h32_quant import h32_quant


pytestmark = pytest.mark.skipif(
    os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B"
)


@pytest.mark.parametrize(
    "width,interleaved,rows",
    [(256, False, 4096), (2304, True, 65), (5120, False, 65)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_h32(width, interleaved, rows, dtype):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    gen = torch.Generator().manual_seed(32)
    source = torch.randint(-127, 128, (rows, width), generator=gen).to(dtype)
    source[0].zero_()
    matrix = torch.ones(1, 1)
    for _ in range(5):
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1))
        )
    actual = source.npu()
    codes, scale = h32_quant(actual, interleaved)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        replay_codes, replay_scale = h32_quant(actual, interleaved)
    for replay in (False, True):
        if replay:
            source = source.flip([1])
            actual.copy_(source)
            graph.replay()
            codes, scale = replay_codes, replay_scale
        natural = (
            source.reshape(-1, 32, 8).transpose(1, 2).reshape_as(source)
            if interleaved
            else source
        )
        rotated = (natural.float().reshape(-1, 32) @ matrix).reshape_as(source)
        expected = rotated.abs().amax(1) / 127
        expected[0] = 1
        torch.testing.assert_close(scale.cpu(), expected, rtol=2e-6, atol=0)
        assert (codes.cpu().int() - rotated / expected[:, None]).abs().max() <= 0.50005


def g32_reference(dot, channel, token, dtype):
    return torch.from_numpy(dot * channel * token[:, None]).to(dtype)


@pytest.mark.parametrize("depth,width", [(256, 64), (2304, 5120), (5120, 4608)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_g32_cube(depth, width, dtype):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    rng = np.random.default_rng(43)
    signed = rng.integers(-8, 8, size=(width, depth), dtype=np.int8)
    mult = rng.integers(1, 17, size=(width, depth // 32), dtype=np.int8)

    def pack(values):
        return ((values[:, ::2] & 15) | (values[:, 1::2] << 4)).view(np.uint8)

    channel = np.linspace(1 / 8192, 1 / 4096, width, dtype=np.float32)
    meta = np.concatenate(
        (pack(mult - 9), channel.view(np.uint8).reshape(width, 4)), axis=1
    )
    weights = tuple(
        value[None].npu()
        for value in int4_g32_a8.pack_g32(
            torch.from_numpy(pack(signed)), torch.from_numpy(meta)
        )
    )
    product = (signed.reshape(width, -1, 32) * mult[:, :, None]).reshape(width, -1)
    capacity = 1051 if width >= 4608 else 259
    host = rng.integers(-127, 128, (capacity, depth), dtype=np.int8)
    activation = torch.from_numpy(host).npu()
    token = torch.linspace(0.125, 1.0, len(host) + 1).npu()[1:]
    weights = tuple(
        torch.cat((tensor, tensor * 0.5 if index == 2 else tensor))
        for index, tensor in enumerate(weights)
    )

    def metadata(first, second):
        return torch.tensor(
            [2, first + second, 0, 1, 0, 0, first, 0, 1, first, second, 0],
            dtype=torch.int32,
        )

    route = metadata(3, 0).npu()
    out = torch.empty((len(host), width), dtype=dtype, device=activation.device)
    invoke = int4_g32_a8.bind(activation, weights, token, route, out=out)
    invoke()
    expected = g32_reference(
        (host[:3].astype(np.int32) @ product.astype(np.int32).T).astype(np.float32),
        channel,
        token[:3].cpu().numpy(),
        dtype,
    )
    torch.testing.assert_close(out[:3].cpu(), expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        invoke()
    activation.neg_()
    graph.replay()
    torch.testing.assert_close(out[:3].cpu(), -expected, rtol=0, atol=0)
    selected = np.unique(np.r_[np.arange(0, width, 63), width - 1])
    dot = (-host.astype(np.int32) @ product[selected].astype(np.int32).T).astype(
        np.float32
    )
    routes = [(17, 129), (127, 128), (240, 1), (3, 0)]
    if capacity > 1024:
        routes.extend([(1025, 17), (1, 2), (513, 529), (0, 1024)])
    for first, second in routes:
        route.copy_(metadata(first, second))
        out.fill_(-19)
        graph.replay()
        live = first + second
        scale = (
            channel[selected][None, :]
            * np.where(np.arange(live) < first, 1, 0.5).astype(np.float32)[:, None]
        )
        reference = g32_reference(dot[:live], scale, token[:live].cpu().numpy(), dtype)
        actual = out.cpu()
        torch.testing.assert_close(actual[:live, selected], reference, rtol=0, atol=0)
        assert (actual[live:] == -19).all()
    route.zero_()
    out.fill_(-19)
    graph.replay()
    assert (out == -19).all().item()


def test_g32_fp16_full_scale_and_overflow_across_tile_sizes():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    depth, width = 2304, 5120
    channel = torch.tensor([1.0, 1 + 3 * 2**-12, 2**-24, 69000.0]).repeat(width // 4)
    codes = torch.zeros(width, depth // 2, dtype=torch.uint8)
    codes[:, 0] = 1
    factors = torch.full((width, depth // 64), 0x88, dtype=torch.uint8)
    metadata = torch.cat((factors, channel.view(torch.uint8).reshape(width, 4)), dim=1)
    weights = tuple(
        value[None].npu() for value in int4_g32_a8.pack_g32(codes, metadata)
    )
    activation = torch.zeros(130, depth, dtype=torch.int8)
    activation[:, 0] = torch.tensor([1, -1], dtype=torch.int8).repeat(65)
    activation = activation.npu()
    token = torch.ones(130, device=activation.device)
    route = torch.tensor([1, 130, 0, 1, 0, 0, 130, 0], dtype=torch.int32).npu()
    out = torch.empty(130, width, device=activation.device, dtype=torch.float16)
    invoke = int4_g32_a8.bind(activation, weights, token, route, out=out)
    invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        invoke()
    for live in (1, 5, 6, 128, 130):
        route[1:2].fill_(live)
        route[6:7].fill_(live)
        graph.replay()
        sign = activation[:live, :1].cpu().float()
        torch.testing.assert_close(
            out[:live, :4].cpu(), (sign * channel[:4]).half(), rtol=0, atol=0
        )


def test_joint_route_gather_combine():
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    width, rows, topk = 5120, 67, 3
    library = load_library()
    owner = Router(device="npu:0")
    ids = torch.tensor([[0, 1, 14], [1, 3, 0], [12, 9, 15]] * rows, dtype=torch.int32)[
        :rows
    ].npu()
    weights = torch.tensor([[0.25, 0.375, 0.5]]).expand(rows, topk).contiguous().npu()
    plan = owner.bucket(ids, weights, 0, 4)
    codes = torch.randint(-127, 128, (rows, width), dtype=torch.int8).npu()
    scales = torch.rand(rows).npu()
    sorted_codes = codes.new_full((ids.numel(), width), -128)
    sorted_scales = scales.new_full((ids.numel(),), -1)
    inverse = ids.new_full((ids.numel(),), -1)
    values = torch.randn(ids.numel(), width).bfloat16().npu()
    result = values.new_empty((rows, width))

    def run():
        library.route_gather_a8(
            codes,
            plan.order,
            plan.metadata,
            sorted_codes,
            inverse,
            topk,
            scales,
            sorted_scales,
        )
        library.route_combine(values, inverse, plan.ids, weights, result)

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        run()
    for shift in (0, 11, 1):
        updated_ids = (ids + shift) % 16
        owner.bucket(updated_ids, weights, 0, 4, out=plan)
        graph.replay()
        live = plan.metadata[1].item()
        order = plan.order[:live].cpu().long()
        assert torch.equal(sorted_codes[:live].cpu(), codes.cpu()[order // topk])
        assert torch.equal(sorted_scales[:live].cpu(), scales.cpu()[order // topk])
        assert torch.equal(inverse.cpu()[order], torch.arange(live))
        contributions = torch.zeros(ids.numel(), width)
        contributions[order] = (
            values[:live].float().cpu() * weights.flatten().cpu()[order, None]
        )
        torch.testing.assert_close(
            result.cpu(),
            contributions.view(rows, topk, width).sum(1).bfloat16(),
            rtol=0,
            atol=0,
        )


def test_shared_w8a8_output_dtype():
    from sglang.srt.hardware_backend.npu.dsv41.linear import SharedW8A8Method
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(190)
    source = torch.randint(-32, 32, (17, 256), generator=generator, dtype=torch.int8)
    weights = torch.randint(-32, 32, (256, 512), generator=generator, dtype=torch.int8)
    channel = torch.arange(512, dtype=torch.float32).add_(512).div_(2**17)
    token = torch.arange(19, dtype=torch.float32).add_(16).div_(2**8)
    layer = SimpleNamespace(
        weight=npu_format_cast(weights.npu()), weight_scale_inv=channel.npu()
    )
    inp, per_token = source.npu(), token.npu()[1:-1]
    reference = (source.int() @ weights.int()).float() * channel * token[1:-1, None]

    def run():
        op = SharedW8A8Method.from_quantized
        return op(layer, inp, per_token, output_dtype=torch.float16), op(
            layer, inp, per_token
        )

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        fp16, default = run()
    inp.neg_()
    graph.replay()
    torch.testing.assert_close(fp16.cpu(), -reference.half(), rtol=0, atol=0)
    torch.testing.assert_close(default.cpu(), -reference.bfloat16(), rtol=0, atol=0)
    assert not torch.equal(fp16.cpu(), default.cpu().half())


def test_routed_g32_bf16_activations():
    """FP16 intermediates change SwiGLU rounding and overflow finite down outputs."""
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router
    from sglang.srt.hardware_backend.npu.dsv41.moe import ResidentMXFP4Method

    torch.npu.set_device(0)
    width = 256

    def weights(channel):
        n = channel.numel()
        codes = torch.full((n, width // 2), 0x11, dtype=torch.uint8)
        factors = torch.full((n, width // 64), 0x88, dtype=torch.uint8)
        metadata = torch.cat((factors, channel.view(torch.uint8).view(n, 4)), dim=1)
        return tuple(t[None].npu() for t in int4_g32_a8.pack_g32(codes, metadata))

    gate_scale = 1.003 / width
    down_scale = torch.tensor([1.0, 1024.0]).repeat(width // 2)
    gate = weights(torch.full((2 * width,), gate_scale))
    down = weights(down_scale)
    method = ResidentMXFP4Method(g32=True)
    method.h = method.i = width
    method.e = method.ep = 1
    method.limit = 10.0
    method.local_experts = method.bind_local_experts(gate, down)
    hidden = torch.ones(1, width, dtype=torch.bfloat16, device="npu")
    ids = torch.zeros((1, 1), dtype=torch.int32, device="npu")
    plan = Router().bucket(ids, torch.ones_like(ids, dtype=torch.float32), 0, 1)

    def run():
        return method.routed(hidden, plan)

    def check(output, sign):
        z = torch.tensor(sign * width * gate_scale).bfloat16().float()
        mid = (torch.nn.functional.silu(z) * z).bfloat16().float()
        expected = (width * mid * down_scale).bfloat16().unsqueeze(0)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        check(run(), 1)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            output = run()
        hidden.neg_()
        graph.replay()
        check(output, -1)


@pytest.mark.parametrize("capacity", [7, 6, 150])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefix_swiglu_and_h32_quant(capacity, dtype):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
        PackedClippedSwiGLU,
    )

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    width = 2304
    dtype_name = "f16" if dtype == torch.float16 else "bf16"
    split = PackedClippedSwiGLU(dimension=width, dtype=dtype_name)
    prefix = PackedClippedSwiGLU(dimension=width, dtype=dtype_name, prefix=True)
    routed = torch.randn(capacity, 2 * width).mul_(8).to(dtype).npu()
    out = routed.new_full((capacity, width), -123)
    count = torch.tensor([capacity], dtype=torch.int32).npu()

    def run():
        prefix(routed, out=out, rows=count)
        return h32_quant(out, True, count=count)

    def check(live, codes, scales):
        if live:
            expected = split(routed[:live])
            torch.testing.assert_close(out[:live], expected, atol=0, rtol=0)
            ref_codes, ref_scales = h32_quant(expected, True)
            assert torch.equal(codes[:live], ref_codes)
            assert torch.equal(scales[:live], ref_scales)
        assert (out[live:] == -123).all().item()

    check(capacity, *run())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        codes, scales = run()
    for live in (0, capacity // 2, capacity):
        out.fill_(-123)
        count.fill_(live)
        routed.neg_()
        graph.replay()
        check(live, codes, scales)


def test_all_signed_products():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    codes = torch.arange(-8, 8, dtype=torch.int8).repeat(16).to(torch.uint8) & 15
    factors = torch.arange(1, 17, dtype=torch.int8).repeat_interleave(16)
    stored = (factors - 9).to(torch.uint8) & 15
    weights = (
        (codes | (codes << 4))[:, None].expand(256, 128),
        torch.cat(
            (
                (stored | (stored << 4))[:, None].expand(256, 4),
                torch.ones(256, 1).view(torch.uint8),
            ),
            1,
        ),
    )
    packed = [value[None].npu() for value in int4_g32_a8.pack_g32(*weights)]
    x = torch.zeros(1, 256, dtype=torch.int8)
    x[0, 0] = 1
    output = torch.empty((1, 256), dtype=torch.float16, device="npu")
    route = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0], dtype=torch.int32).npu()
    invoke = int4_g32_a8.bind(x.npu(), packed, torch.ones(1).npu(), route, out=output)
    invoke()
    expected = torch.arange(-8, 8).repeat(16) * factors
    torch.testing.assert_close(output.cpu().flatten(), expected.half(), atol=0, rtol=0)
