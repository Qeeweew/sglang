"""Exact native schedule/survival, lex ties, canaries and changing Graph input."""

# ruff: noqa: E402 - check exclusive device availability before Torch bootstrap
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path

_HEALTH = (
    subprocess.check_output(["npu-smi", "info"], text=True)
    if __name__ == "__main__"
    else None
)

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.dspark_schedule import (
    DSparkSchedule,
)
from dsv41.kernels.references.dspark import (
    ROOT,
    SOURCE,
    load_official,
    ranking_oracle,
    test_data,
)


def guarded(shape, dtype, value):
    count = 1
    for size in shape:
        count *= size
    storage = torch.full((count + 64,), value, dtype=dtype, device="npu")
    return storage[32 : 32 + count].view(shape), storage


def check(op, scope, confidence, budget, cfg, scratch, output, guards, label, records):
    survival = scope["compute_sort_survival"](confidence).cpu()
    torch.testing.assert_close(
        output.cpu(), ranking_oracle(survival, budget, cfg), rtol=0, atol=0, msg=label
    )
    if budget > 0:
        expected = torch.full(scratch.shape, -float("inf"))
        columns = scratch.shape[0]
        expected[:, : len(confidence)] = survival[:, :columns].T.masked_fill(
            ~(survival[:, :columns].T >= cfg.survival_eps), -float("inf")
        )
        assert torch.equal(
            scratch.cpu().view(torch.int32), expected.view(torch.int32)
        ), label + "/FP32 survival bitwise"
    for guard, value in guards:
        assert bool((guard[:32].cpu() == value).all()) and bool(
            (guard[-32:].cpu() == value).all()
        ), label + "/canary"
    records.append(
        {
            "case": label,
            "requests": len(confidence),
            "gamma": confidence.shape[1],
            "budget": budget,
            "min": cfg.min_verify_len,
            "max": cfg.resolved_max_verify_len(),
            "eps": str(cfg.survival_eps),
            "status": "passed",
            "survival_bitwise": budget > 0,
            "canaries": True,
        }
    )


def cases(op, scope, records):
    Config = scope["DSparkScheduleConfig"]
    shapes = [(b, g) for b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33) for g in (1, 6, 17)]
    shapes += [(257, 6), (1024, 6), (2049, 1), (2, 65), (2, 257)]
    for b, g in shapes:
        patterns = ("random", "ones", "special", "zeros", "nextafter")
        for pattern in patterns:
            cpu = test_data(b, g, pattern)
            confidence = cpu.npu()
            configs = [
                (1, 0, 1e-6),
                (0, 1, 0.0),
                (0, g, 0.25),
                (min(3, g + 1), 0, 1.0),
                (1, 0, float("inf")),
                (1, 0, float("nan")),
            ]
            if b > 33:
                configs = [(1, 0, 1e-6), (0, min(3, g), 0.0)]
            for minimum, maximum, eps in configs:
                cfg = Config(
                    gamma=g,
                    min_verify_len=minimum,
                    max_verify_len=maximum,
                    survival_eps=eps,
                )
                cfg.validate()
                scratch, sg = guarded(
                    op.workspace_shape(b, g, maximum), torch.float32, 123.0
                )
                output, og = guarded((b,), torch.int32, -987654)
                for budget in sorted(set((-2, 0, 1, b + 1, 3 * b, b * g + 1))):
                    op(
                        confidence,
                        budget,
                        min_verify_len=minimum,
                        max_verify_len=maximum,
                        survival_eps=eps,
                        survival=scratch,
                        out=output,
                    )
                    check(
                        op,
                        scope,
                        confidence,
                        budget,
                        cfg,
                        scratch,
                        output,
                        ((sg, 123.0), (og, -987654)),
                        f"native/{b}x{g}/{pattern}",
                        records,
                    )
        print(f"PASS native numeric {b}x{g}", flush=True)
    # Values below normal FP32 and near normal underflow, with eps=0 so any
    # unexpected flush changes exact ranks instead of being hidden by masking.
    cpu = torch.tensor(
        [
            [1e-38, 0.5, 1.0, 1.0, 1.0, 1.0],
            [2e-38, 0.25, 1.0, 1.0, 1.0, 1.0],
            [1e-45, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    confidence = cpu.npu()
    cfg = Config(gamma=6, survival_eps=0.0)
    scratch, sg = guarded(op.workspace_shape(4, 6), torch.float32, 123.0)
    output, og = guarded((4,), torch.int32, -987654)
    for budget in (1, 5, 12, 23):
        op(confidence, budget, survival_eps=0.0, survival=scratch, out=output)
        check(
            op,
            scope,
            confidence,
            budget,
            cfg,
            scratch,
            output,
            ((sg, 123.0), (og, -987654)),
            "native/subnormal",
            records,
        )


def graphs(op, scope, records, shapes=None):
    Config = scope["DSparkScheduleConfig"]
    for b, g in shapes or ((1, 6), (6, 6), (33, 6), (257, 6), (2, 65)):
        for budget in (0, b + 1, 3 * b):
            cfg = Config(gamma=g)
            confidence = torch.rand(b, g, device="npu")
            scratch, sg = guarded(op.workspace_shape(b, g), torch.float32, 123.0)
            output, og = guarded((b,), torch.int32, -987654)
            tensors = confidence, scratch, output
            pointers = [t.data_ptr() for t in tensors]
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            managed_output = torch.empty_like(output)
            with torch.npu.stream(stream):
                managed = op.bind(confidence, budget, out=managed_output)
                for _ in range(2):
                    op(confidence, budget, survival=scratch, out=output)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                managed()
                op(confidence, budget, survival=scratch, out=output)
            for pattern in ("ones", "special", "zeros", "nextafter", "random"):
                confidence.copy_(test_data(b, g, pattern))
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                check(
                    op,
                    scope,
                    confidence,
                    budget,
                    cfg,
                    scratch,
                    output,
                    ((sg, 123.0), (og, -987654)),
                    f"native/graph/{b}x{g}/{pattern}",
                    records,
                )
                torch.testing.assert_close(
                    managed_output.cpu(), output.cpu(), rtol=0, atol=0
                )
                assert pointers == [t.data_ptr() for t in tensors]
            del graph
        print(f"PASS native Graph {b}x{g}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical", type=int, default=2)
    p.add_argument("--phase", choices=("all", "numeric", "graph"), default="all")
    p.add_argument(
        "--graph-shapes",
        nargs="+",
        metavar="B:G",
        help="extra request/gamma specializations for Graph qualification",
    )
    p.add_argument(
        "--library",
        type=Path,
        default=ROOT / "build/native/libdsv41_kernels.so",
    )
    p.add_argument(
        "--report", type=Path, default=ROOT / "runs/integration/dspark_native.json"
    )
    a = p.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(a.physical)
    health = _HEALTH or subprocess.check_output(["npu-smi", "info"], text=True)
    assert (
        health.count("OK") >= 8
        and f"No running processes found in NPU {a.physical}" in health
    ), health
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.set_num_threads(4)
    torch.manual_seed(772619)
    scope = load_official()
    op = DSparkSchedule(library=a.library)
    records = []
    report = {
        "physical_device": a.physical,
        "phase": a.phase,
        "cases": records,
        "official_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(
            (ROOT / "native/dspark_schedule/dspark_schedule.asc").read_bytes()
        ).hexdigest(),
        "library": str(a.library.resolve()),
        "library_sha256": hashlib.sha256(a.library.read_bytes()).hexdigest(),
    }
    try:
        if a.phase in ("all", "numeric"):
            cases(op, scope, records)
        if a.phase in ("all", "graph"):
            shapes = (
                None
                if a.graph_shapes is None
                else [tuple(map(int, x.split(":"))) for x in a.graph_shapes]
            )
            if shapes is not None and any(len(s) != 2 or min(s) < 1 for s in shapes):
                raise ValueError("graph shapes require positive B:G pairs")
            graphs(op, scope, records, shapes)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(records)} native records; {a.report}", flush=True)


if __name__ == "__main__":
    main()
