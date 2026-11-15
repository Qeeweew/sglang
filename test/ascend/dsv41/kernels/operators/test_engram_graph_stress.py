"""Twenty-dispatch Graphs exercise tail consumer events across replay."""

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram import (
    EngramTable,
    HostBuffer,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram_ops import EngramOps
from dsv41.kernels.references.engram import check, reference
from dsv41.kernels.references.engram_hash import oracle_gate


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(414181)
    count = 0
    for width in (32, 288, 768):
        for host in (False, True):
            codes = torch.randint(0, 127, (19, width), dtype=torch.uint8)
            scales = torch.randint(123, 131, (19, width // 32), dtype=torch.uint8)
            owners = (
                [HostBuffer(t.shape, device=0) for t in (codes, scales)] if host else []
            )
            device_tables = (
                [o.tensor for o in owners] if host else [codes.npu(), scales.npu()]
            )
            for target, source in zip(device_tables, (codes, scales)):
                target.copy_(source)
            table = EngramTable(*owners) if host else EngramTable(*device_tables)
            for rows in (6, 41):
                ids = torch.zeros(rows, device="npu", dtype=torch.int64)
                out = torch.empty((rows, width), device="npu", dtype=torch.bfloat16)
                call = table.bind(ids, out=out)
                call()
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    for _ in range(20):
                        call()
                for generation in range(6):
                    ids_cpu = torch.randint(-1, 20, (rows,), dtype=torch.int64)
                    if generation == 3:
                        # At least one actor loads a valid row, then sees
                        # only invalid rows and performs no later raw DMA.
                        ids_cpu.fill_(-1)
                        ids_cpu[: rows // 2] = 0
                    elif generation == 4:
                        # The reverse order first reaches raw storage only
                        # after one or more zero-output tiles.
                        ids_cpu.fill_(0)
                        ids_cpu[: rows // 2] = -1
                    elif generation == 5:
                        ids_cpu.fill_(-1)
                    codes.random_(0, 127)
                    scales.random_(123, 131)
                    for target, source in zip(device_tables, (codes, scales)):
                        target.copy_(source)
                    ids.copy_(ids_cpu)
                    out.fill_(float("nan"))
                    torch.npu.synchronize()
                    for _ in range(20):
                        graph.replay()
                    stream.synchronize()
                    check(out, reference(codes, scales, ids_cpu))
                    count += 1
                del graph
            for owner in owners:
                owner.close()
    op = EngramOps()
    for width in (256, 5120):
        for rows in (1, 6, 41):
            cpu = [
                torch.randn(rows, 4, width).bfloat16(),
                torch.randn(rows, 5 * width).bfloat16(),
                torch.randn(4, width).bfloat16(),
                torch.randn(4, width).bfloat16(),
                torch.ones(rows, dtype=torch.int32),
            ]
            tensors = [x.npu() for x in cpu]
            out = torch.empty_like(tensors[0])
            call = op.bind_gate(*tensors, out=out)
            call()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                for _ in range(20):
                    call()
            for generation in range(3):
                for source in cpu[:4]:
                    source.normal_()
                cpu[4].random_(0, 2)
                for target, source in zip(tensors, cpu):
                    target.copy_(source)
                call()
                torch.npu.synchronize()
                eager = out.clone()
                torch.testing.assert_close(
                    eager.cpu(), oracle_gate(*cpu), rtol=0.008, atol=0.001
                )
                out.fill_(float("nan"))
                torch.npu.synchronize()
                for _ in range(20):
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(out, eager, rtol=0, atol=0)
                count += 1
            del graph
    print(
        f"PASS {count} changed-input Graph stress cases;20 dispatches x20 replays each",
        flush=True,
    )


if __name__ == "__main__":
    run()
