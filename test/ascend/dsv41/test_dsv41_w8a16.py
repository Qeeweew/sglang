"""Checkpoint repack and actual model-method eager/Graph integration."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("SGLANG_DSV41_TEST_NPU") != "1",
    reason="requires explicit physical NPU binding and native library",
)
@pytest.mark.parametrize(
    "rows,batches,n,k,merged",
    [
        (1, 1, 256, 512, False),
        (5, 1, 256, 512, True),
        (6, 1, 256, 512, True),
        (17, 1, 256, 512, False),
        (48, 1, 256, 512, False),
        (129, 1, 256, 512, False),
        (2049, 1, 256, 512, False),
        (6, 8, 1024, 4096, False),
        (48, 8, 1024, 4096, False),
        (6, 1, 4608, 5120, True),
        (129, 1, 4608, 5120, True),
        (6, 1, 32768, 1280, False),
    ],
)
def test_loaded_method_and_changed_input_graph(rows, batches, n, k, merged):
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") is not None
    assert os.environ.get("ASCEND_DEVICE_ID") == "0"
    import torch_npu
    from dsv41.kernels.references.w8a16 import quantize_g256, reference
    from sglang.srt.hardware_backend.npu.dsv41.linear import (
        W8A16LinearMethod,
        project_wo_a,
    )

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(161614)
    total_n = batches * n
    source = (torch.randn(total_n, k) * 0.02).bfloat16()
    expected_q, expected_scale = quantize_g256(source)
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty(total_n, k, dtype=torch.int8, device="npu"), requires_grad=False
        ),
    )

    def loader(param, value, shard=None):
        if shard is None:
            param.data.copy_(value)
        else:
            begin = shard * (total_n // 2)
            param.data[begin : begin + total_n // 2].copy_(value)

    layer.weight.weight_loader = loader
    layer.weight.input_dim, layer.weight.output_dim = 1, 0
    original_parameter = layer.weight
    method = W8A16LinearMethod()
    method.setup(layer, (total_n // 2, total_n // 2) if merged else (total_n,))
    method.batches = batches
    layer.quant_method = method
    if merged:
        layer.weight.weight_loader(layer.weight, source[total_n // 2 :].npu(), 1)
        with pytest.raises(RuntimeError, match="incomplete"):
            method.process_weights_after_loading(layer)
        layer.weight.weight_loader(layer.weight, source[: total_n // 2].npu(), 0)
    else:
        layer.weight.weight_loader(layer.weight, source.npu())
    assert method.pending is None
    torch.testing.assert_close(layer.weight.cpu(), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(layer.w8a16_scale.cpu(), expected_scale, rtol=0, atol=0)
    method.process_weights_after_loading(layer)
    assert layer.weight is original_parameter and layer.weight.shape == source.shape
    assert (layer.weight.input_dim, layer.weight.output_dim) == (1, 0)
    assert torch_npu.get_npu_format(layer.weight) in (0, 2)
    restored = method.weight.cpu().permute(0, 2, 1).reshape(total_n, k)
    restored_scale = layer.w8a16_scale.cpu().permute(1, 0, 2).reshape(k // 256, total_n)
    torch.testing.assert_close(restored, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(restored_scale, expected_scale, rtol=0, atol=0)
    pointers = [layer.weight.data_ptr(), layer.w8a16_scale.data_ptr()]
    method.process_weights_after_loading(layer)  # No second repack.
    assert pointers == [layer.weight.data_ptr(), layer.w8a16_scale.data_ptr()]

    xc = (
        torch.randn((rows, batches, k) if batches > 1 else (rows, k)) * 0.125
    ).bfloat16()
    xc.reshape(-1, k)[0, 0] = (
        64  # Keep small neighboring activations; do not requantize them.
    )
    x = xc.npu()
    bias = torch.randn(n).bfloat16() * 0.01 if batches == 1 else None
    bias_npu = bias.npu() if bias is not None else None
    stream = torch.npu.Stream()
    out = torch.empty(
        (rows, batches, n) if batches > 1 else (rows, n),
        dtype=torch.bfloat16,
        device="npu",
    )
    torch.npu.synchronize()

    def invoke():
        if batches > 1:
            method._launch(layer, x, out)
        else:
            method.apply_into(layer, x, out, bias_npu)

    def check():
        expected = reference(xc, expected_q, expected_scale)
        if bias is not None:
            expected = (expected.float() + bias.float()).bfloat16()
        torch.testing.assert_close(
            out.cpu().float(), expected.float(), rtol=0.01, atol=0.004
        )
        assert pointers == [layer.weight.data_ptr(), layer.w8a16_scale.data_ptr()]

    # Loader quantization has completed; every forward must bypass it.
    with patch(
        "sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16.quantize_bf16_weight",
        side_effect=AssertionError("activation quantization is forbidden"),
    ):
        with torch.npu.stream(stream):
            invoke()
            eager = (
                project_wo_a(x, layer.weight.view(batches, n, k))
                if batches > 1
                else method.apply(layer, x, bias_npu)
            )
        stream.synchronize()
        check()
        torch.testing.assert_close(eager.cpu(), out.cpu(), rtol=0, atol=0)
        output_pointer = out.data_ptr()
        workspaces = [entry[2].data_ptr() for entry in method.plans.values()]
        graph = torch.npu.NPUGraph()
        with patch.object(
            method.op, "bind", side_effect=AssertionError("Graph/hot path cannot bind")
        ):
            with torch.npu.graph(graph, stream=stream):
                invoke()
            for _ in range(3):
                xc.neg_()
                with torch.npu.stream(stream):
                    x.copy_(xc)
                    graph.replay()
                stream.synchronize()
                check()
                assert out.data_ptr() == output_pointer
                assert workspaces == [
                    entry[2].data_ptr() for entry in method.plans.values()
                ]
        del graph
    if batches == 1:
        addend = torch.full_like(out, 0.25)
        actual = method.apply_with_addend(layer, x, addend, bias_npu)
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual.cpu(), (out.cpu().float() + 0.25).bfloat16(), rtol=0, atol=0
        )
    target = os.environ.get("SGLANG_DSV41_W8A16_CHECK_OUTPUT")
    if target:
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"m{rows}-b{batches}-n{n}-k{k}.json").write_text(
            json.dumps(
                dict(
                    rows=rows,
                    batches=batches,
                    n=n,
                    k=k,
                    merged_shards=merged,
                    raw_scales_and_repack="passed",
                    changed_input_graph_replays=3,
                    numerical_reference="independent CPU weight quantization and FP64 W8A16 dot",
                    stable_weights_workspace_and_output=True,
                )
            )
            + "\n"
        )
