import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B"
)


@pytest.mark.parametrize(
    "tokens,capacity,first,owned",
    [(3, 0, 1, 2), (8, 23, 8, 0), (8, 23, 3, 3), (257, 512, 200, 31)],
)
def test_shared_dispatch_graph(tokens, capacity, first, owned):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    torch.npu.set_device(0)
    library = load_library()
    width, topk = 5120, 3
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        hidden = torch.randn(tokens, width).bfloat16().npu()
        slots = torch.randperm(
            tokens * topk, generator=torch.Generator().manual_seed(32)
        ).int()
        order = slots.npu()
        meta = torch.zeros(8, dtype=torch.int32, device="npu")
        inverse = torch.empty_like(order)
        shared = hidden.new_empty((owned, width), dtype=torch.float16)
        routed = shared.new_empty((capacity, width))

        def run():
            library.route_gather(
                hidden, order, meta, routed, inverse, topk, shared, first
            )

        run()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            run()
        for live in (capacity, 0, capacity // 2):
            hidden.neg_()
            slots = slots.flip(0)
            order.copy_(slots)
            meta[1:2].fill_(live)
            routed.fill_(-123)
            shared.fill_(-123)
            inverse.fill_(-1)
            graph.replay()
            expected = hidden.cpu().half()
            assert torch.equal(shared.cpu(), expected[first : first + owned])
            assert torch.equal(routed[:live].cpu(), expected[slots[:live] // topk])
            assert (routed[live:].cpu() == -123).all()
            reference = torch.full_like(slots, -1)
            reference[slots[:live]] = torch.arange(live, dtype=torch.int32)
            assert torch.equal(inverse.cpu(), reference)


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
