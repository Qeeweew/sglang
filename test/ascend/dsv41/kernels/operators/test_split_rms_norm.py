"""Packed, row-strided dual RMSNorm on a single 910B."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv41.attention import projections

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module")
def native():
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or torch.npu.device_count() != 1:
        pytest.skip("select one 910B with ASCEND_RT_VISIBLE_DEVICES")
    return load_library().split_rms_norm


def check_outputs(outputs, packed, weights, epsilons):
    parts = packed.split([w.numel() for w in weights], dim=-1)
    for result, part, weight, eps in zip(outputs, parts, weights, epsilons):
        assert result.is_contiguous() and result.dtype == torch.bfloat16
        assert result.shape == part.shape
        source = part.cpu().double()
        oracle = (
            source
            * (source.square().mean(-1, keepdim=True) + eps).rsqrt()
            * weight.cpu().double()
        ).bfloat16()
        torch.testing.assert_close(
            result.cpu(), oracle, rtol=1 / 128, atol=0, equal_nan=True
        )
        if part.numel():
            vendor = torch_npu.npu_rms_norm(part, weight, eps)[0]
            torch.testing.assert_close(
                result.cpu(), vendor.cpu(), rtol=1 / 128, atol=0, equal_nan=True
            )


@pytest.mark.parametrize("rows", [0, 1, 11, 12, 21, 47, 129])
@pytest.mark.parametrize("widths,padding", [((1280, 512), 0), ((320, 64), 13)])
def test_split_norm(native, rows, widths, padding):
    torch.manual_seed(rows + 101)
    width = sum(widths)
    storage = torch.randn(rows, width + padding, dtype=torch.bfloat16)
    source = storage[:, padding:]
    if rows:
        source[0].zero_()
    if rows > 2:
        source[1].mul_(1e-9)
        source[2, widths[0]] = float("nan")
        source[-1, 0] = float("inf")
    packed = storage.npu()[:, padding:]
    weights = tuple(torch.randn(n, dtype=torch.bfloat16).npu() for n in widths)
    eps = (1e-20, 1e-5)
    check_outputs(native(packed, *weights, *eps), packed, weights, eps)


def test_changed_graph_inputs(native):
    widths, eps = (1280, 512), (1e-6, 1e-4)
    packed = torch.randn(12, sum(widths), device="npu", dtype=torch.bfloat16)
    weights = tuple(torch.randn_like(packed[0, :width]) for width in widths)
    native(packed, *weights, *eps)
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        outputs = native(packed, *weights, *eps)
    for scale in (1.0, 1e-7):
        packed.copy_(torch.randn_like(packed) * scale)
        for weight in weights:
            weight.copy_(torch.randn_like(weight))
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        check_outputs(outputs, packed, weights, eps)


def test_precompile_for_first_graph(native):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    widths, rows = (192, 128), 12
    workers = torch.npu.get_device_properties(0).vector_core_num
    load_library().precompile_elementwise(
        "split_rms_norm", rows, widths[0], workers, second_width=widths[1]
    )
    packed = torch.randn(rows, sum(widths), device="npu", dtype=torch.bfloat16)
    weights = tuple(torch.randn_like(packed[0, :width]) for width in widths)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        outputs = native(packed, *weights, 1e-20, 1e-20)
    graph.replay()
    check_outputs(outputs, packed, weights, (1e-20, 1e-20))


@pytest.mark.parametrize(
    "path", ["fused", "old_library", "separate", "unsupported_width", "large"]
)
def test_projection_dispatch(native, monkeypatch, path):
    calls = []

    class Norm:
        def __init__(self, width, eps):
            self.weight = torch.randn(width, device="npu", dtype=torch.bfloat16)
            self.variance_epsilon = eps

        def __call__(self, value):
            calls.append("vendor")
            return torch_npu.npu_rms_norm(value, self.weight, self.variance_epsilon)[0]

    def fused(*args):
        calls.append("native")
        return native(*args)

    widths = (80 if path == "unsupported_width" else 128, 64)
    eps = (1e-6, 1e-3)
    rows = 2049 if path == "large" else 12
    packed = torch.randn(rows, sum(widths), device="npu", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        fuse_wqa_wkv=path != "separate",
        q_lora_rank=widths[0],
        q_norm=Norm(widths[0], eps[0]),
        kv_norm=Norm(widths[1], eps[1]),
        wqkv_a=lambda value: (value, None),
        wq_a=lambda value: (value[:, : widths[0]], None),
        wkv=lambda value: (value[:, widths[0] :], None),
    )
    monkeypatch.setattr(
        projections,
        "_split_rms_operator",
        lambda: None if path == "old_library" else fused,
    )
    outputs = projections._normalized_qkv(layer, packed)
    assert calls == (["native"] if path == "fused" else ["vendor", "vendor"])
    check_outputs(outputs, packed, (layer.q_norm.weight, layer.kv_norm.weight), eps)
