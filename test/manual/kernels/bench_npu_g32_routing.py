"""Paired G32 / MXFP4 target GEMM latency on synthetic EP8 decode routes."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import median
from time import monotonic, time_ns

import numpy as np
import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import int4_g32_a8
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.quant_grouped_gemm import (
    NativeGroupedGemm,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


EXPERTS, TOPK, EP, VERIFY = 48, 6, 8, 5
PROJECTIONS = {"gate_up": (5120, 4608), "down": (2304, 5120)}
INT8_PEAK_TOPS_910B3 = 20 * 1.8e9 * 16 * 16 * 32 * 2 / 1e12


def distributions(tokens):
    slots = tokens * TOPK
    average = max(1, slots // EP)
    cases = {"empty": ()}
    for active in (1, 2, 4, 8, 12, 15, 16, 20, 24, 32, 40, 48):
        if active <= average:
            quotient, extra = divmod(average, active)
            cases[f"balanced_{active}"] = tuple(
                quotient + (index < extra) for index in range(active)
            )
        if active in (1, 2, 4, 8, 16, 32, 48):
            for rows in (
                1,
                5,
                15,
                16,
                17,
                31,
                32,
                33,
                63,
                64,
                65,
                96,
                127,
                128,
                129,
                160,
                191,
                239,
                255,
                256,
                257,
                tokens,
            ):
                if rows <= tokens and active * rows <= slots:
                    cases[f"grid_{active}_{rows}"] = (rows,) * active
    if tokens >= 240:
        cases.update(
            skew8=(17,) * 8 + (13,) * 8,
            skew12=(17,) * 12 + (9,) * 4,
            skew14=(17,) * 14 + (1,) * 2,
            one_hot=(225,) + (1,) * 15,
            skew48=(17,) * 8 + (3,) * 24 + (2,) * 16,
        )
    return cases


def metadata(counts):
    result = np.zeros((4, (4 * (EXPERTS + 1) + 7) // 8 * 8), dtype=np.int32)
    for record, shift in zip(result, (0, 13, 26, 39), strict=True):
        record[:4] = (len(counts), sum(counts), 0, 1)
        offset = 0
        for slot, (expert, count) in enumerate(
            sorted(
                ((index + shift) % EXPERTS, count) for index, count in enumerate(counts)
            ),
            1,
        ):
            record[4 * slot : 4 * slot + 4] = (expert, offset, count, 0)
            offset += count
    return torch.from_numpy(result).npu()


def capture(calls, repeats):
    for call in calls:
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        for index in range(repeats):
            calls[index % len(calls)]()
    return graph


def pipeline_groups():
    yield 5, {"1x1": (1,), "4x1": (1,) * 4}
    yield (
        320,
        {
            "48x1": (1,) * 48,
            "48x5": (5,) * 48,
            "16x15": (15,) * 16,
            "16x17": (17,) * 16,
            "1x96": (96,),
            "1x120": (120,),
            "1x128": (128,),
            "skew240": (17,) * 14 + (1,) * 2,
        },
    )
    for experts, rows in ((4, 1024), (48, 128), (48, 256), (48, 512)):
        yield (
            max(rows, (experts * rows * EP + TOPK - 1) // TOPK),
            {f"{experts}x{rows}": (rows,) * experts},
        )


def measure(args, projection, tokens, weights, rng, cases):
    depth, width = PROJECTIONS[projection]
    codes, factors, channel, mx_scales = weights
    budget = tokens * TOPK
    capacity = max(tokens, max(map(sum, cases.values()))) if args.pipeline else budget
    activation = torch.from_numpy(
        rng.integers(-32, 32, (capacity, depth), dtype=np.int8)
    ).npu()
    half = activation.half() / 32
    units = torch.full((capacity,), 1 / 32, device="npu", dtype=torch.float32)
    fixed = {name: metadata(counts) for name, counts in cases.items()}
    routing = next(
        (fixed[name] for name, counts in cases.items() if counts),
        next(iter(fixed.values())),
    ).clone()
    mx = NativeGroupedGemm(depth, width, experts=EXPERTS)
    scratch = codes.new_empty(load_library().int4_g32_a8_scratch(depth, width, 0))
    output = torch.empty(
        (capacity, width), device="npu", dtype=getattr(torch, args.g32_dtype)
    )
    mx_output = torch.empty_like(output, dtype=torch.float16)
    grouped = [
        int4_g32_a8.bind(
            activation,
            (codes, factors, channel),
            units,
            record,
            out=output,
            workspace=scratch,
        )
        for record in routing.unbind()
    ]
    baseline = [
        mx.bind(
            half,
            codes.view(torch.int32),
            mx_scales,
            record,
            out=mx_output,
            max_expert_rows=tokens,
            mean_expert_rows=budget / (EXPERTS * EP),
            tiling_rows=budget,
        )
        for record in routing.unbind()
    ]
    graphs = {
        arm: capture(calls, args.repeats)
        for arm, calls in [("g32", grouped), ("mx", baseline)]
        if arm in args.arms
    }
    samples = {name: {arm: [] for arm in graphs} for name in cases}
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    for trial in range(args.rounds):
        order = list(cases)
        random.Random(trial + 837).shuffle(order)
        for index, name in enumerate(order):
            routing.copy_(fixed[name])
            for arm in reversed(graphs) if (index + trial) % 2 else graphs:
                graph = graphs[arm]
                graph.replay()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples[name][arm].append(start.elapsed_time(end) * 1000 / args.repeats)
    for name, counts in cases.items():
        times = samples[name]
        if args.sustain_seconds:
            routing.copy_(fixed[name])
            graph = next(iter(graphs.values()))
            graph.replay()
            torch.npu.synchronize()
            start_ns = time_ns()
            deadline = monotonic() + args.sustain_seconds
            while monotonic() < deadline:
                for _ in range(16):
                    graph.replay()
                torch.npu.synchronize()
            print(
                json.dumps(
                    dict(window=[start_ns, time_ns()], projection=projection, case=name)
                ),
                flush=True,
            )
        latency = median(times["g32"]) if "g32" in times else None
        payload = len(counts) * (depth * width // 2 + depth * width // 64 + 4 * width)
        operations = 2 * sum(counts) * depth * width
        yield dict(
            batch="" if args.pipeline else tokens // VERIFY,
            tokens=tokens,
            projection=projection,
            case=name,
            active=len(counts),
            rows=sum(counts),
            counts=json.dumps(counts),
            mx_bm=baseline[0].block_m,
            g32_us=latency,
            g32_dtype=args.g32_dtype,
            g32_effective_weight_GBs=payload / (latency * 1e3) if latency else None,
            g32_useful_mfu_pct=100 * operations / (latency * 1e6 * INT8_PEAK_TOPS_910B3)
            if latency
            else None,
            mx_us=median(times["mx"]) if "mx" in times else None,
            samples=json.dumps(times),
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument(
        "--projection", nargs="+", choices=PROJECTIONS, default=list(PROJECTIONS)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--check", action="store_true", help="exit nonzero for any slower median"
    )
    parser.add_argument(
        "--pipeline", action="store_true", help="sparse decode and dense expert tiles"
    )
    parser.add_argument("--cases", nargs="+", help="only measure these route names")
    parser.add_argument(
        "--arms", nargs="+", choices=("g32", "mx"), default=["g32", "mx"]
    )
    parser.add_argument(
        "--g32-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--sustain-seconds",
        type=float,
        default=0,
        help="single-arm replay window for HBM sampling",
    )
    args = parser.parse_args()
    if args.check and set(args.arms) != {"g32", "mx"}:
        parser.error("--check needs both arms")
    if args.sustain_seconds < 0 or (args.sustain_seconds and len(args.arms) != 1):
        parser.error("--sustain-seconds needs one arm and a nonnegative duration")
    if min(args.batch) < 1 or args.rounds < 1 or args.repeats < 4 or args.repeats % 4:
        parser.error(
            "positive batch/rounds and repeats a positive multiple of four required"
        )
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    writer = None
    failures = 0
    with args.output.open("w", newline="") as dest:
        for projection in args.projection:
            depth, width = PROJECTIONS[projection]

            def payload(elements):
                return torch.from_numpy(
                    rng.integers(0, 256, (EXPERTS, elements), dtype=np.uint8)
                ).npu()

            codes = payload(width * depth // 2)
            weights = (
                codes,
                payload(width * depth // 64),
                torch.full(
                    (EXPERTS, width), 1 / 128, device="npu", dtype=torch.float32
                ),
                codes.new_full((EXPERTS, width * depth // 32), 48),
            )
            groups = (
                pipeline_groups()
                if args.pipeline
                else (
                    (batch * VERIFY, distributions(batch * VERIFY))
                    for batch in args.batch
                )
            )
            for tokens, cases in groups:
                cases = {
                    name: counts
                    for name, counts in cases.items()
                    if not args.cases or name in args.cases
                }
                if not cases:
                    continue
                rows = list(measure(args, projection, tokens, weights, rng, cases))
                if writer is None:
                    writer = csv.DictWriter(dest, fieldnames=rows[0].keys())
                    writer.writeheader()
                writer.writerows(rows)
                dest.flush()
                slower = [
                    row["case"]
                    for row in rows
                    if None not in (row["g32_us"], row["mx_us"])
                    and row["g32_us"] > row["mx_us"]
                ]
                failures += len(slower)
                print(
                    f"{projection} tokens={tokens}: {len(rows)} cases, slower: {slower}",
                    flush=True,
                )
    if writer is None:
        parser.error("no matching cases")
    if args.check and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
