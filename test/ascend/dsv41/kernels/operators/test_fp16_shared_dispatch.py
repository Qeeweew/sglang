import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B"
)


def test_fp16_down_rounds_accumulator_directly_to_bf16():
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
        W8A16,
        repack_weight,
    )

    torch.npu.set_device(0)
    width = 256
    codes = torch.ones((128, width), dtype=torch.int8)
    scales = torch.ones((1, 128), dtype=torch.float16)
    weight, scales = (tensor.npu() for tensor in repack_weight(codes, scales))
    host = torch.zeros(2, width, dtype=torch.float16)
    host[0, :3] = torch.tensor([1.0, 2**-8, 2**-12])
    host[1, :2] = 65504
    expected = host.float().sum(1, keepdim=True).expand(2, 128).bfloat16()
    assert not torch.equal(
        expected,
        host.float().sum(1, keepdim=True).half().bfloat16().expand_as(expected),
    )
    source = host.npu()
    output = source.new_empty((2, 128), dtype=torch.bfloat16)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        call = W8A16().bind(source, weight, scales, out=output)
        call()
        torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            call()
        source.neg_()
        graph.replay()
        torch.testing.assert_close(output.cpu(), -expected, atol=0, rtol=0)
