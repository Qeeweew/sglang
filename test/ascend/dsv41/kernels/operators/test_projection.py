"""BF16/FP32 dense, token-major wo_a and shared MLP: CPU oracle + NPU Graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.builders.native import build_native
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.projection import (
    DenseProjection,
    WoAProjection,
    repack_linear,
    repack_wo_a,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.projection import (
    compare,
    output_canary,
    shared_reference_checks,
)

CASES = {
    "dense-small": ("dense_projection", {"K": 256, "COLUMNS": 128, "OUT_DTYPE": "f32"}),
    "dense-c2": ("dense_projection", {"K": 5120, "COLUMNS": 512, "OUT_DTYPE": "f32"}),
    "dense-up": ("dense_projection", {"K": 5120, "COLUMNS": 2304}),
    "dense-down": ("dense_projection", {"K": 2304, "COLUMNS": 5120}),
    "wo-a-small": (
        "wo_a_projection",
        {"GROUP_COUNT": 8, "K": 256, "RANK": 128, "BN": 128},
    ),
    "wo-a-real": ("wo_a_projection", {"GROUP_COUNT": 8, "K": 4096, "RANK": 1024}),
}


def dense(base, records):
    for case in ("dense-small", "dense-c2", "dense-up", "dense-down"):
        _, params = CASES[case]
        k, n = params["K"], params["COLUMNS"]
        op = DenseProjection(k=k, columns=n, out_dtype=params.get("OUT_DTYPE", "bf16"))
        fp32 = params.get("OUT_DTYPE") == "f32"
        weight = (torch.randn(n, k) * 0.1).bfloat16()
        packed = repack_linear(weight).npu()
        for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257):
            x = (torch.randn(m, k) * 0.3).bfloat16()
            inputs = x.npu()
            output, canary = output_canary((m, n), op.dtype)
            reference = x.float() @ weight.float().T
            if not fp32:
                reference = reference.bfloat16()
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                for _ in range(2):
                    op(inputs, packed, out=output)
            stream.synchronize()
            compare(output, reference, f"{case}/M{m}/eager", records, fp32)
            assert (canary[:32].cpu() == 7).all() and (canary[-32:].cpu() == 7).all()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                op(inputs, packed, out=output)
            for replay in range(2):
                x = (torch.randn(m, k) * 0.3).bfloat16()
                weight = (torch.randn(n, k) * 0.1).bfloat16()
                inputs.copy_(x)
                packed.copy_(repack_linear(weight))
                reference = x.float() @ weight.float().T
                if not fp32:
                    reference = reference.bfloat16()
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                compare(output, reference, f"{case}/M{m}/graph{replay}", records, fp32)
                assert (canary[:32].cpu() == 7).all() and (
                    canary[-32:].cpu() == 7
                ).all()
        print("PASS", case, flush=True)


def wo_a(base, records):
    for case in ("wo-a-small", "wo-a-real"):
        _, params = CASES[case]
        g, k, n = params["GROUP_COUNT"], params["K"], params["RANK"]
        op = WoAProjection(k=k, groups=g, rank=n)
        weight = (torch.randn(g, n, k) * 0.1).bfloat16()
        viewed = repack_wo_a(weight)
        assert (
            viewed.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        )
        packed = viewed.npu()
        for m in (
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257)
            if case.endswith("small")
            else (0, 1, 6, 33, 257)
        ):
            x = (torch.randn(m, g, k) * 0.3).bfloat16()
            inputs = x.npu()
            output, canary = output_canary((m, g, n), torch.bfloat16)
            reference = torch.einsum(
                "mgk,gnk->mgn", x.float(), weight.float()
            ).bfloat16()
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                for _ in range(2):
                    op(inputs, packed, out=output)
            stream.synchronize()
            compare(output, reference, f"{case}/M{m}/eager", records)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                op(inputs, packed, out=output)
            for replay in range(2):
                x = (torch.randn(m, g, k) * 0.3).bfloat16()
                weight = (torch.randn(g, n, k) * 0.1).bfloat16()
                inputs.copy_(x)
                packed.copy_(repack_wo_a(weight))
                reference = torch.einsum(
                    "mgk,gnk->mgn", x.float(), weight.float()
                ).bfloat16()
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                compare(output, reference, f"{case}/M{m}/graph{replay}", records)
                assert (canary[:32].cpu() == 7).all() and (
                    canary[-32:].cpu() == 7
                ).all()
        if case == "wo-a-real":
            for m in (8192, 32768):
                inputs = torch.randn((m, g, k), dtype=torch.bfloat16, device="npu")
                output, canary = output_canary((m, g, n), torch.bfloat16)
                samples = torch.linspace(0, m - 1, 129, device="npu").long()
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                before = torch.npu.memory_allocated()
                op(inputs, packed, out=output)
                torch.npu.synchronize()
                extra = torch.npu.max_memory_allocated() - before
                assert extra < 1024 * 1024, ("wo_a allocated a layout copy", extra)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    op(inputs, packed, out=output)
                for replay in range(2):
                    inputs.mul_(0.5)
                    packed.mul_(0.5)
                    torch.npu.synchronize()
                    graph.replay()
                    torch.npu.synchronize()
                    x_sample = inputs[samples].cpu().float()
                    w_sample = packed.cpu().float()
                    reference = torch.einsum(
                        "mgk,gnk->mgn", x_sample, w_sample
                    ).bfloat16()
                    compare(
                        output[samples],
                        reference,
                        f"{case}/M{m}/graph{replay}",
                        records,
                    )
                    assert torch.isfinite(output).all()
                    assert (canary[:32].cpu() == 7).all() and (
                        canary[-32:].cpu() == 7
                    ).all()
                records.append(
                    {"case": f"{case}/M{m}/allocation", "extra_allocated_bytes": extra}
                )
                del graph, inputs, output, canary
        print("PASS", case, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, default=4)
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--only", choices=("all", "dense", "wo-a", "shared"), default="all"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "runs/projection-agent/validation.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "build/tests/projection")
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    if args.build:
        build_native(args.output, components=["projection", "elementwise"])
    os.environ["DSV41_NATIVE_DIR"] = str(args.output)
    health = subprocess.check_output(["npu-smi", "info"], text=True)
    assert health.count("OK") >= 8
    assert f"No running processes found in NPU {args.physical}" in health
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(4325)
    torch.npu.config.allow_internal_format = False
    records = []
    if args.only in ("all", "dense"):
        dense(args.output, records)
    if args.only in ("all", "wo-a"):
        wo_a(args.output, records)
    if args.only in ("all", "shared"):
        for seed in (4325, 7193, 15401):
            torch.manual_seed(seed)
            first = len(records)
            for record in records[first:]:
                record["seed"] = seed
    from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path

    path = native_library_path(directory=args.output)
    report = {
        "status": "passed",
        "physical": args.physical,
        "cases": records,
        "library": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "health": health,
    }
    target = args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(records)} comparisons; {target}", flush=True)


if __name__ == "__main__":
    main()
