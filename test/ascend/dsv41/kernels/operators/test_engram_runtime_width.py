"""Encoding groups and decode capacity are independent of table row width."""

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram import (
    EngramTable,
    HostBuffer,
)


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(20409)
    count = 0
    for width in (32, 96, 256, 288, 512, 768):
        codes = torch.randint(0, 127, (19, width), dtype=torch.uint8, generator=rng)
        scales = torch.randint(
            120, 130, (19, width // 32), dtype=torch.uint8, generator=rng
        )
        decoded = codes.view(torch.float8_e4m3fn).float() * torch.pow(
            2.0, scales.float().sub(127).repeat_interleave(32, dim=1)
        )
        for host in (False, True):
            owners = []
            if host:
                owners = [HostBuffer(t.shape, device=0) for t in (codes, scales)]
                for owner, source in zip(owners, (codes, scales)):
                    owner.tensor.copy_(source)
                table = EngramTable(*owners)
            else:
                table = EngramTable(codes.npu(), scales.npu())
            ids_cpu = torch.tensor([0, 18, -1, 19, 7, 7], dtype=torch.int64)
            ids = ids_cpu.npu()
            out = torch.full(
                (6, width), float("nan"), device="npu", dtype=torch.bfloat16
            )
            call = table.bind(ids, out=out)
            call()
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.graph(graph, stream=stream):
                call()
            stream.synchronize()
            pointers = (ids.data_ptr(), out.data_ptr())
            for generation in range(2):
                ids_cpu = ids_cpu.roll(generation + 1)
                ids.copy_(ids_cpu)
                out.fill_(float("nan"))
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                reference = torch.zeros((6, width), dtype=torch.bfloat16)
                live = (ids_cpu >= 0) & (ids_cpu < 19)
                reference[live] = decoded[ids_cpu[live]].bfloat16()
                torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
                assert pointers == (ids.data_ptr(), out.data_ptr())
                count += 1
            for owner in owners:
                owner.close()
            print("Engram runtime width", width, "registered_host", host, flush=True)
    print(f"PASS {count} Engram runtime-width changed-input Graph cases", flush=True)


if __name__ == "__main__":
    run()
