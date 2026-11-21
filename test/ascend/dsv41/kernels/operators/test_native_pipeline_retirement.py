"""Preserve distinct results across consecutive private-arena kernel calls.

Single-call checks cannot detect a tail consumer still touching UB when the
next kernel starts. Keep twenty distinct generations resident, submit without
host synchronization, and validate every result after eager and Graph batches.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.fp8_weight import FP8Weight
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.moe_routes import MoERoutes
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import RoutingPlan
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.row_copy import (
    gather_rows,
    scatter_rows,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.health import health_snapshot
from dsv41.kernels.references.fp8_weight import reference as fp8_reference


def guarded(shape, dtype):
    elements = shape[0] * shape[1]
    storage = torch.full((elements + 64,), -77, dtype=dtype, device="npu")
    return storage[32:-32].view(shape), storage


def make_generation(routes, fp8, rows, width, generation):
    generator = torch.Generator().manual_seed(71209 + generation)
    source = torch.randn(rows, width, generator=generator).bfloat16()
    device_source = source.npu()
    mapping = torch.tensor([rows - 1, 0, 0, -1, rows, 1], dtype=torch.int32)
    device_mapping = mapping.npu()
    gathered, gather_guard = guarded((mapping.numel(), width), torch.bfloat16)
    scattered, scatter_guard = guarded((rows, width), torch.bfloat16)
    expected_gather = torch.zeros_like(gathered, device="cpu")
    expected_scatter = torch.full_like(source, -77)
    updates = (source[: mapping.numel()].float() * (generation + 1)).bfloat16()
    device_updates = updates.npu()
    for i, row in enumerate(mapping.tolist()):
        if 0 <= row < rows:
            expected_gather[i] = source[row]
            expected_scatter[row] = updates[i]

    topk = 3
    slots = rows * topk
    weights = torch.rand(rows, topk, generator=generator)
    routing = RoutingPlan(
        torch.zeros(rows, topk, dtype=torch.int32, device="npu"),
        weights.npu(),
        torch.tensor([slots], dtype=torch.int32, device="npu"),
        torch.tensor([0, slots], dtype=torch.int32, device="npu"),
        torch.arange(slots, dtype=torch.int32, device="npu"),
        torch.tensor([1, slots, 0, 1, 0, 0, slots, 0], dtype=torch.int32, device="npu"),
    )
    values, inverse = routes.gather(device_source, routing)
    combined, combine_guard = guarded((rows, width), torch.bfloat16)
    expected_values = source.repeat_interleave(topk, 0).half()
    expected_combined = torch.zeros(rows, width)
    for k in range(topk):
        expected_combined += expected_values[k::topk].float() * weights[:, k, None]

    codes = torch.randint(0, 256, (rows, width), dtype=torch.uint8, generator=generator)
    scales = torch.randint(
        120,
        131,
        ((rows + 31) // 32, (width + 31) // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    device_codes, device_scales = codes.npu(), scales.npu()
    decoded, decode_guard = guarded((rows, width), torch.bfloat16)
    expected_decoded = fp8_reference(codes, scales)

    def invoke():
        gather_rows(device_source, device_mapping, out=gathered)
        scatter_rows(device_updates, device_mapping, scattered)
        routes.gather(device_source, routing, out=values, inverse=inverse)
        routes.combine(values, inverse, routing, out=combined)
        fp8.convert(device_codes, device_scales, out=decoded)

    def check():
        for actual, expected in (
            (gathered, expected_gather),
            (scattered, expected_scatter),
            (values, expected_values),
            (combined, expected_combined.bfloat16()),
            (decoded, expected_decoded),
        ):
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=0, atol=0, equal_nan=True
            )
        for storage in (gather_guard, scatter_guard, combine_guard, decode_guard):
            assert bool((storage[:32].cpu() == -77).all())
            assert bool((storage[-32:].cpu() == -77).all())

    return invoke, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    health = health_snapshot(args.physical, args.output)
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    os.environ["DSV41_NATIVE_DIR"] = str(args.library_dir.resolve())
    routes = MoERoutes()
    fp8 = FP8Weight(args.library_dir / "libdsv41_kernels.so")
    records = []
    for rows, width in ((6, 257), (33, 513), (6, 5120)):
        generations = [make_generation(routes, fp8, rows, width, i) for i in range(20)]

        def invoke_batch():
            for invoke, _ in generations:
                invoke()

        invoke_batch()
        torch.npu.synchronize()
        for _, check in generations:
            check()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke_batch()
        for _ in range(20):
            graph.replay()
        torch.npu.synchronize()
        for _, check in generations:
            check()
        records.append(dict(rows=rows, width=width, generations=20, replays=20))
        del graph
    (args.output / "result.json").write_text(
        json.dumps(
            dict(
                status="passed",
                health=health,
                library=str(args.library_dir),
                records=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(status="passed", groups=len(records))), flush=True)


if __name__ == "__main__":
    main()
