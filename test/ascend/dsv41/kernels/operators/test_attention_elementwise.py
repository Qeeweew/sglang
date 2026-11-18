"""Independent FP32 CPU oracles, eager calls and updated-input NPU Graph replay.

ASCEND_RT_VISIBLE_DEVICES=4 ASCEND_DEVICE_ID=0 python \
  -m dsv41.kernels.operators.test_attention_elementwise --physical 4 --build
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.builders.native import build_native
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
    SparseAttention,
    WindowAttention,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
    ClippedSwiGLU,
    PackedClippedSwiGLU,
    RMSNorm,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import (
    ROOT,
    native_library_path,
)
from dsv41.kernels.references.attention import (
    attention_data,
    attention_oracle,
)


def compare(actual, expected, label, *, atol=0.005, rtol=0.02):
    actual = actual.cpu().float()
    assert torch.isfinite(actual).all(), label
    torch.testing.assert_close(
        actual, expected.float(), atol=atol, rtol=rtol, msg=label
    )
    return {
        "case": label,
        "max_abs": (actual - expected.float()).abs().max().item()
        if actual.numel()
        else 0.0,
    }


def elementwise(directory, records):
    norm = RMSNorm()
    swiglu = ClippedSwiGLU()
    for mode, op, d in (("rms", norm, 5120), ("swiglu", swiglu, 2304)):
        rows = (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257)
        if mode == "swiglu":
            rows += (2048, 32768)
        for m in rows:
            x = torch.randn(m, d).bfloat16()
            if mode == "rms":
                if m:
                    x[0].zero_()
                if m > 1:
                    x[1].fill_(1e-30)
                if m > 2:
                    x[2].fill_(1e-10)
                extra = torch.randn(d).bfloat16()
                expected = (
                    x.float()
                    * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
                    * extra.float()
                ).bfloat16()
            else:
                extra = (torch.randn(m, d) * 20).bfloat16()
                if m:
                    x[0, :7] = torch.tensor(
                        [-1e30, -1000, -100, -10, 0, 10, 1000]
                    ).bfloat16()
                gate = x.float().clamp(max=10)
                expected = (
                    torch.nn.functional.silu(gate) * extra.float().clamp(-10, 10)
                ).bfloat16()
            a, b = x.npu(), extra.npu()
            output = torch.empty_like(a)
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                for _ in range(2):
                    op(a, b, out=output)
            stream.synchronize()
            records.append(
                compare(output, expected, f"{mode}/M{m}/eager", atol=0.0001, rtol=0.008)
            )
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                op(a, b, out=output)
            for replay in range(2):
                x = torch.randn(m, d).bfloat16()
                extra = torch.randn_like(extra)
                if mode == "rms":
                    expected = (
                        x.float()
                        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
                        * extra.float()
                    ).bfloat16()
                else:
                    gate = x.float().clamp(max=10)
                    expected = (
                        torch.nn.functional.silu(gate) * extra.float().clamp(-10, 10)
                    ).bfloat16()
                a.copy_(x)
                b.copy_(extra)
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                records.append(
                    compare(
                        output,
                        expected,
                        f"{mode}/M{m}/graph{replay}",
                        atol=0.0001,
                        rtol=0.008,
                    )
                )

    packed_swiglu(records)
    task_queue_order(directory, records)


def packed_swiglu(records):
    op = PackedClippedSwiGLU(dimension=2304)
    for m in (0, 1, 6, 33, 2048, 32768):
        x = torch.empty((m, 4608), device="npu", dtype=torch.bfloat16)
        storage = torch.full((m * 2304 + 64,), 123, device="npu", dtype=torch.bfloat16)
        out = storage[32:-32].view(m, 2304)
        call = op.bind(x, out=out)
        stream = torch.npu.Stream()
        graph = None
        for replay in range(3):
            source = (torch.randn(m, 4608) * 15).bfloat16()
            if m:
                source[0, :16] = torch.tensor(
                    [
                        -1e30,
                        -1000,
                        -100,
                        -85,
                        -80,
                        -30,
                        -10,
                        -1,
                        0,
                        1,
                        10,
                        1000,
                        -90,
                        -89,
                        -88,
                        -87,
                    ]
                ).bfloat16()
                source[0, 2307:2309] = 10
            gate, up = source.float().chunk(2, -1)
            expected = (
                torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            ).bfloat16()
            with torch.npu.stream(stream):
                x.copy_(source)
                if graph is None:
                    call()
                else:
                    graph.replay()
            stream.synchronize()
            records.append(
                compare(
                    out,
                    expected,
                    f"swiglu_packed/M{m}/{'eager' if replay == 0 else 'graph' + str(replay)}",
                    atol=0.0001,
                    rtol=0.008,
                )
            )
            if m:
                # Tiny BF16 outputs must pass a relative check; the ordinary
                # absolute tolerance would hide an accidental lower clamp.
                torch.testing.assert_close(
                    out[0, 3:5].cpu(), expected[0, 3:5], rtol=0.008, atol=0
                )
            assert bool((storage[:32] == 123).all())
            assert bool((storage[-32:] == 123).all())
            if graph is None:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    call()


def task_queue_order(directory, records):
    """Torch producers must be submitted before a cached native consumer.

    Synchronize only after each producer/consumer chain. An intermediate CPU
    readback would drain the Torch-NPU host task queue and hide stale reads.
    """
    swiglu = ClippedSwiGLU()
    gate = torch.ones((128, 2304), device="npu", dtype=torch.bfloat16)
    up, output = torch.zeros_like(gate), torch.empty_like(gate)
    expected = (torch.nn.functional.silu(torch.ones((128, 2304))) * 8).bfloat16()
    for stream in (torch.npu.current_stream(), torch.npu.Stream()):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            invoke = swiglu.bind(gate, up, out=output)
            invoke()
            stream.synchronize()
            for mode in ("bound", "ordinary"):
                for _ in range(80):
                    up.zero_()
                    for _ in range(32):
                        up.add_(0.25)
                    if mode == "bound":
                        invoke()
                    else:
                        swiglu(gate, up, out=output)
                    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
                records.append(
                    {
                        "case": f"swiglu/task-queue/{mode}/stream{int(stream.npu_stream)}",
                        "trials": 80,
                        "max_abs": 0.0,
                    }
                )


def attention(directory, records, variant=False, large_tokens=0, max_sequence=0):
    b, t, max_seq = (4, 16, 1024) if variant else (1, large_tokens or 6, 4096)
    max_seq = max_sequence or max_seq
    qlens_cases = (
        [[t]]
        if not variant
        else [[0, 2, 0, 14], [1, 2, 6, 7], [3, 4, 5, 4], [8, 0, 8, 0], [6, 6, 4, 0]]
    )
    for mode in ("swa", "c1", "c2"):
        cls = WindowAttention if mode == "swa" else SparseAttention
        op = cls(
            **({"ratio": int(mode[1])} if mode != "swa" else {}),
        )
        for qlens in qlens_cases:
            for case in (
                "empty",
                "short",
                "mixed",
                "narrow",
                "duplicate",
                "padding",
                "full",
                "sinks",
            ):
                cpu_args = attention_data(mode, b, t, max_seq, case, qlens)
                expected = attention_oracle(mode, cpu_args)
                device_args = [x.npu() for x in cpu_args]
                output = torch.empty_like(device_args[0])
                op(*device_args, out=output)
                records.append(
                    compare(output, expected, f"{mode}/{qlens}/{case}/eager")
                )
        # Stable allocation capture, then change Q, caches, indices, Cu and lengths.
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for _ in range(2):
                op(*device_args, out=output)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            op(*device_args, out=output)
        for replay, case in enumerate(("short", "padding", "empty", "sinks")):
            cpu_args = attention_data(
                mode, b, t, max_seq, case, qlens_cases[replay % len(qlens_cases)]
            )
            for dest, source in zip(device_args, cpu_args):
                dest.copy_(source)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            records.append(
                compare(
                    output,
                    attention_oracle(mode, cpu_args),
                    f"{mode}/var{variant}/graph{replay}",
                )
            )
        # Empty wrapper invocation performs no launch and has no batch cap.
        empty = list(device_args)
        empty[0] = empty[0][:0]
        assert op(*empty).shape[0] == 0
        for table_index in (2,) if mode == "swa" else (4, 5):
            invalid = list(device_args)
            invalid[table_index] = invalid[table_index][:, :0]
            try:
                op(*invalid)
            except ValueError as error:
                assert "page-table capacit" in str(error)
            else:
                raise AssertionError("zero page-table capacity accepted")
        # The same binary accepts short, non-DMA-aligned page tables and high
        # logical positions. This is local attention/index qualification, not
        # evidence that an end-to-end 1M prompt fits or generates correctly.
        for columns in (1, 8, 257, 513, 1025, 8192):

            def capacity_inputs(odd=0):
                values = attention_data(
                    mode,
                    b,
                    t,
                    columns * 128,
                    "full",
                    qlens_cases[0],
                    physical_blocks=(17, 23),
                )
                length_index = 4 if mode == "swa" else 7
                values[length_index].sub_(odd)
                if mode != "swa":
                    r = 1 if mode == "c1" else 2
                    values[8].copy_(values[7] // r)
                    values[9].copy_(values[7] % r)
                    for batch in range(b):
                        available = int(values[8][batch])
                        for token in range(
                            int(values[6][batch]), int(values[6][batch + 1])
                        ):
                            # Include the final page and causal exclusions at
                            # the end of the six-token verification window.
                            count = min(512, available)
                            values[3][token, 0, :count] = torch.arange(
                                available - count, available
                            )
                return values

            cpu = capacity_inputs()
            dev = [x.npu() for x in cpu]
            result = torch.empty_like(dev[0])
            op(*dev, out=result)
            records.append(
                compare(
                    result, attention_oracle(mode, cpu), f"{mode}/table{columns}/eager"
                )
            )
            with torch.npu.stream(stream):
                op(*dev, out=result)
            stream.synchronize()
            capacity_graph = torch.npu.NPUGraph()
            with torch.npu.graph(capacity_graph, stream=stream):
                op(*dev, out=result)
            for odd in (1, 2):
                cpu = capacity_inputs(odd)
                for dst, src in zip(dev, cpu):
                    dst.copy_(src)
                torch.npu.synchronize()
                capacity_graph.replay()
                torch.npu.synchronize()
                records.append(
                    compare(
                        result,
                        attention_oracle(mode, cpu),
                        f"{mode}/table{columns}/graph-tail{odd}",
                    )
                )
        # Physical pool capacities do not specialize the artifact. Move all
        # logical pages upward, including the last physical page, and use
        # independently sized original/compressed pools with the same binary.
        for extra in (1, 7):

            def expanded_pools():
                values = attention_data(mode, b, t, max_seq, "full", qlens_cases[0])
                values[1] = torch.cat(
                    (
                        values[1].new_full((extra, *values[1].shape[1:]), float("nan")),
                        values[1],
                    )
                )
                table_index = 2 if mode == "swa" else 4
                values[table_index] += extra
                if mode != "swa":
                    cmp_extra = extra + 3
                    values[2] = torch.cat(
                        (
                            values[2].new_full(
                                (cmp_extra, *values[2].shape[1:]), float("nan")
                            ),
                            values[2],
                        )
                    )
                    values[5] += cmp_extra
                return values

            pool_cpu = expanded_pools()
            pool_device = [x.npu() for x in pool_cpu]
            pool_output = torch.empty_like(pool_device[0])
            op(*pool_device, out=pool_output)
            records.append(
                compare(
                    pool_output,
                    attention_oracle(mode, pool_cpu),
                    f"{mode}/dynamic-pool+{extra}/eager",
                )
            )
            with torch.npu.stream(stream):
                op(*pool_device, out=pool_output)
            stream.synchronize()
            pool_graph = torch.npu.NPUGraph()
            with torch.npu.graph(pool_graph, stream=stream):
                op(*pool_device, out=pool_output)
            for replay in range(2):
                pool_cpu = expanded_pools()
                for dst, src in zip(pool_device, pool_cpu):
                    dst.copy_(src)
                torch.npu.synchronize()
                pool_graph.replay()
                torch.npu.synchronize()
                records.append(
                    compare(
                        pool_output,
                        attention_oracle(mode, pool_cpu),
                        f"{mode}/dynamic-pool+{extra}/graph{replay}",
                    )
                )


def dynamic_attention(records):
    """One binary across exact B/T, with independent oracles and output guards."""
    stream = torch.npu.Stream()
    shapes = [(b, b, [1] * b) for b in range(1, 9)]
    shapes += [
        (3, 15, [0, 6, 9]),
        (6, 193, [1, 0, 31, 7, 128, 26]),
        (8, 257, [0, 1, 0, 7, 64, 128, 55, 2]),
        (1, 193, [193]),
        (1, 257, [257]),
    ]
    for mode in ("swa", "c1", "c2"):
        policies = (False,) if mode == "swa" else (False, True)
        for reuse in policies:
            op = (
                WindowAttention()
                if mode == "swa"
                else SparseAttention(ratio=int(mode[1]), window_reuse=reuse)
            )
            artifact = native_library_path()
            for batch, tokens, counts in shapes:
                if reuse and batch != 1:
                    continue
                cpu = attention_data(mode, batch, tokens, 512, "mixed", counts)
                values = [value.npu() for value in cpu]
                guard = torch.full(
                    (tokens * 64 * 512 + 512,), 123, dtype=torch.bfloat16, device="npu"
                )
                output = guard[256:-256].view(tokens, 64, 512)
                label = f"{mode}/reuse{reuse}/B{batch}T{tokens}"
                with torch.npu.stream(stream):
                    output.fill_(float("nan"))
                    op(*values, out=output)
                stream.synchronize()
                records.append(
                    compare(output, attention_oracle(mode, cpu), label + "/eager")
                )
                with torch.npu.graph(graph := torch.npu.NPUGraph(), stream=stream):
                    op(*values, out=output)
                fresh = attention_data(mode, batch, tokens, 512, "narrow", counts)
                with torch.npu.stream(stream):
                    for destination, source in zip(values, fresh):
                        destination.copy_(source)
                    output.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                records.append(
                    compare(output, attention_oracle(mode, fresh), label + "/graph")
                )
                assert bool((guard[:256] == 123).all().cpu()), label
                assert bool((guard[-256:] == 123).all().cpu()), label
                assert native_library_path() == artifact
                print(
                    "PASS",
                    label,
                    "eager, changed Graph, canaries, shared binary",
                    flush=True,
                )
                del graph


def prefill_attention(directory, records, *, tokens, capacity):
    """Run the full deployment bucket; sample independent CPU oracle rows.

    First/last queries, page crossings and each request boundary are checked.
    All device output rows are written and checked for finite values, and the
    trailing canary catches writes beyond the 8192-query allocation.
    """
    for batch in sorted({1, capacity}):
        counts = [tokens] if batch == 1 else [1] * (batch - 1) + [tokens - batch + 1]
        cu = [0] + list(torch.tensor(counts).cumsum(0).tolist())
        samples = sorted(
            {
                0,
                1,
                127,
                128,
                255,
                511,
                tokens // 2,
                tokens - 2,
                tokens - 1,
                *(v for v in cu[:-1]),
                *(v - 1 for v in cu[1:]),
            }
        )
        samples = [t for t in samples if 0 <= t < tokens]
        for mode in ("swa", "c1", "c2"):
            cls = WindowAttention if mode == "swa" else SparseAttention
            op = cls(
                **({"ratio": int(mode[1])} if mode != "swa" else {}),
            )
            storage = torch.full(
                (tokens * 64 * 512 + 256,), 123, dtype=torch.bfloat16, device="npu"
            )
            output = storage[:-256].view(tokens, 64, 512)
            stream = torch.npu.Stream()
            graph = None
            device_args = None
            for case in ("short", "full"):
                cpu_args = attention_data(
                    mode,
                    batch,
                    tokens,
                    tokens * 2,
                    case,
                    counts,
                    physical_blocks=(137, 139),
                )
                if device_args is None:
                    device_args = [value.npu() for value in cpu_args]
                else:
                    for destination, source in zip(device_args, cpu_args):
                        destination.copy_(source)
                expected = attention_oracle(mode, cpu_args, token_indices=samples)
                if graph is None:
                    with torch.npu.stream(stream):
                        op(*device_args, out=output)
                    stream.synchronize()
                    records.append(
                        compare(
                            output[samples], expected, f"{mode}/B{batch}T{tokens}/eager"
                        )
                    )
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        op(*device_args, out=output)
                output.fill_(float("nan"))
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                assert bool(torch.isfinite(output).all().cpu()), (mode, batch, case)
                assert bool((storage[-256:].cpu() == 123).all()), (
                    "prefill output canary"
                )
                records.append(
                    compare(
                        output[samples],
                        expected,
                        f"{mode}/B{batch}T{tokens}/graph-{case}",
                    )
                )
            print(
                f"PASS {mode} B{batch}T{tokens}: eager, updated-input Graph, boundaries and canary",
                flush=True,
            )
            del graph, device_args, storage, output, cpu_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical", type=int, default=4)
    p.add_argument("--build", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "build/native")
    p.add_argument(
        "--report-dir", type=Path, default=ROOT / "runs/native-attention-elementwise"
    )
    p.add_argument("--only", choices=("all", "elementwise", "attention"), default="all")
    p.add_argument("--variant", action="store_true")
    p.add_argument("--dynamic-shapes", action="store_true")
    p.add_argument("--large-tokens", type=int, default=0)
    p.add_argument("--max-seq", type=int, default=0)
    p.add_argument(
        "--production-prefill",
        action="store_true",
        help="validate a deployment bucket with a sampled CPU oracle",
    )
    p.add_argument(
        "--batch", type=int, default=8, help="request capacity for --production-prefill"
    )
    args = p.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    if args.production_prefill and (
        args.only != "attention" or args.large_tokens < args.batch
    ):
        p.error(
            "--production-prefill requires --only attention and --large-tokens >= --batch"
        )
    if args.build:
        components = []
        if args.only in {"all", "attention"}:
            components.append("attention")
        if args.only in {"all", "elementwise"}:
            components.append("elementwise")
        build_native(args.output, components=components)
        os.environ["DSV41_NATIVE_DIR"] = str(args.output)
    health = subprocess.check_output(["npu-smi", "info"], text=True)
    assert health.count("OK") >= 8, health
    assert "No running processes found in NPU " + str(args.physical) in health, health
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(2394)
    records = []
    if args.only in {"all", "elementwise"}:
        elementwise(args.output, records)
    if args.only in {"all", "attention"}:
        if args.dynamic_shapes:
            dynamic_attention(records)
        elif args.production_prefill:
            prefill_attention(
                args.output,
                records,
                tokens=args.large_tokens,
                capacity=args.batch,
            )
        else:
            attention(
                args.output,
                records,
                args.variant,
                args.large_tokens,
                args.max_seq,
            )
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    report = out / (
        args.only
        + ("-var" if args.variant else "")
        + (f"-t{args.large_tokens}" if args.large_tokens else "")
        + (f"-s{args.max_seq}" if args.max_seq else "")
        + ".json"
    )
    report.write_text(
        json.dumps(
            {
                "physical_device": args.physical,
                "status": "passed",
                "cases": records,
                "health": health,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS {len(records)} comparisons; {report}", flush=True)


if __name__ == "__main__":
    main()
