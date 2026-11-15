"""Run on one reserved 910B: ASCEND_RT_VISIBLE_DEVICES=N python -m ..."""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram import (
    EngramTable,
    HostBuffer,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.engram import check, reference
from dsv41.kernels.support.engram_cpu import cpu_function


def run(library=None, allocation="mmap"):
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(4162)
    count = 0
    with (
        HostBuffer((256, 256), allocation=allocation) as codes,
        HostBuffer((256, 8), allocation=allocation) as scales,
    ):
        codes.tensor.copy_(torch.arange(256, dtype=torch.uint8).repeat(256, 1))
        scales.tensor.copy_(torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, 8))
        host = EngramTable(codes, scales, library=library)
        hbm = EngramTable(codes.tensor.npu(), scales.tensor.npu(), library=library)
        ids = torch.arange(256, dtype=torch.int64)
        oracle = reference(codes.tensor, scales.tensor, ids)
        cpu = cpu_function(ROOT / "build/tests/engram_cpu")
        cpu_output = torch.empty_like(oracle)
        cpu(
            codes.tensor.data_ptr(),
            scales.tensor.data_ptr(),
            ids.data_ptr(),
            cpu_output.data_ptr(),
            ids.numel(),
            host.vocab,
        )
        check(cpu_output, oracle)
        count += 1
        for table in (host, hbm):
            check(table.gather(ids.npu()), oracle)
            count += 1
        for rows in (0, 1, 2, 6, 24, 39, 40, 41, 144, 256, 3072, 24576):
            ids = torch.randint(-2, 258, (rows,), generator=generator)
            if rows >= 6:
                ids[:6] = torch.tensor([-1, 256, 2**40, -(2**40), 0, 255])
            oracle = reference(codes.tensor, scales.tensor, ids)
            device_ids = ids.npu()
            # Output view catches writes outside the requested logical rows.
            storage = torch.full(
                (rows + 2, 256), 71.0, dtype=torch.bfloat16, device="npu"
            )
            for table in (host, hbm):
                result = table.gather(device_ids, storage[1:-1])
                assert result.data_ptr() == storage[1:-1].data_ptr()
                check(result, oracle)
                assert torch.equal(
                    storage[[0, -1]].cpu(),
                    torch.full((2, 256), 71.0, dtype=torch.bfloat16),
                )
                count += 1
        # A nondefault stream plus explicitly closed host allocation exercises
        # allocator tracking and close()'s cross-stream lifetime synchronization.
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            ids = torch.randint(0, 256, (6, 24), generator=generator)
            result = host.gather(ids.npu())
        stream.synchronize()
        check(result, reference(codes.tensor, scales.tensor, ids))
        count += 1
        # Captured fixed addresses, updating IDs and compressed bytes between
        # replays; table/mappings stay live throughout the graph's lifetime.
        device_ids = torch.zeros((6, 24), dtype=torch.int64, device="npu")
        out = torch.empty((6, 24, 256), dtype=torch.bfloat16, device="npu")
        for _ in range(3):
            host.gather(device_ids, out)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            host.gather(device_ids, out)
        for iteration in range(3):
            ids = torch.randint(-1, 257, (6, 24), generator=generator)
            ids.reshape(-1)[0] = iteration  # Ensure the mutated row is read.
            codes.tensor[iteration].fill_(iteration + 60)
            scales.tensor[iteration].fill_(127)
            device_ids.copy_(ids)
            graph.replay()
            check(out, reference(codes.tensor, scales.tensor, ids))
            count += 1
        del graph
    # A read submitted on another stream must finish before close unmaps it.
    codes = HostBuffer((8, 256))
    scales = HostBuffer((8, 8))
    codes.tensor.fill_(56)
    scales.tensor.fill_(127)
    table = EngramTable(codes, scales, library=library)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        out = table.gather(torch.arange(8, dtype=torch.int64, device="npu"))
    codes.close()
    scales.close()
    assert torch.equal(out.cpu(), torch.ones((8, 256), dtype=torch.bfloat16))
    count += 1
    print(
        f"PASS: {count} Engram cases (full 65,536 encoding pairs, HBM/host, tails, streams, graph, lifetime)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--allocation", choices=["mmap", "acl"], default="mmap")
    args = parser.parse_args()
    run(args.library, args.allocation)
