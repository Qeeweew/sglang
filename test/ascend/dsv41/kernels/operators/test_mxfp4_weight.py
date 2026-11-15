"""NPU compressed weight repack: independent CPU pack, real shards and timings."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main(args):
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(args.physical)
    args.output.mkdir(parents=True, exist_ok=False)
    info = subprocess.run(
        ["npu-smi", "info"], check=True, text=True, capture_output=True
    ).stdout
    (args.output / "health-before.txt").write_text(info)
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    from dsv41.kernels.references.quant_packing import pack_experts
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import MXFP4WeightPacker
    from dsv41.kernels.references.mxfp4_grouped import checkpoint

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(41)
    records = []

    def check(codes, scales, k, n, pk, label):
        op = MXFP4WeightPacker(k, n, panel_k=pk, compact_scales=args.compact_scales)
        p = torch.empty(n * k // 8, dtype=torch.int32, device="npu:0")
        s = torch.empty(
            n * k // 32,
            dtype=torch.uint8 if args.compact_scales else torch.float16,
            device="npu:0",
        )
        source, exponents = (
            torch.from_numpy(np.array(codes)),
            torch.from_numpy(np.array(scales)),
        )
        expected_p, expected_s = pack_experts(
            [(codes, scales)],
            k=k,
            n=n,
            count=1,
            compact_scales=args.compact_scales,
        )
        times = []
        for repeat in range(4):
            start = time.perf_counter()
            op.load_cpu(source, exponents, packed=p, s2=s)
            if repeat:
                times.append((time.perf_counter() - start) * 1e3)
            assert np.array_equal(p.cpu().numpy(), expected_p[0]), label
            assert np.array_equal(s.cpu().numpy(), expected_s[0]), label
        # The physical worker plan is fixed by this geometry. Change both
        # checkpoint bytes and exponents in place under the same captured grid.
        bound = op.bind(op.codes, op.scales, packed=p, s2=s)
        stream = torch.npu.Stream()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            bound()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            bound()
        pointers = [t.data_ptr() for t in (op.codes, op.scales, op.tables, p, s)]
        for generation in (1, 2):
            changed_codes = np.bitwise_xor(codes, np.uint8((0x89 * generation) & 255))
            changed_scales = (
                104 + (scales.astype(np.int16) - 104 + 7 * generation) % 40
            ).astype(np.uint8)
            expected_p, expected_s = pack_experts(
                [(changed_codes, changed_scales)],
                k=k,
                n=n,
                count=1,
                compact_scales=args.compact_scales,
            )
            with torch.npu.stream(stream):
                op.codes.copy_(torch.from_numpy(changed_codes))
                op.scales.copy_(torch.from_numpy(changed_scales))
                graph.replay()
            stream.synchronize()
            assert np.array_equal(p.cpu().numpy(), expected_p[0]), label
            assert np.array_equal(s.cpu().numpy(), expected_s[0]), label
            assert pointers == [
                t.data_ptr() for t in (op.codes, op.scales, op.tables, p, s)
            ]
        record = dict(
            changed_graph_generations=2,
            label=label,
            k=k,
            n=n,
            load_ms=statistics.median(times),
        )
        records.append(record)
        print(record, flush=True)

    for k, n, pk in (
        (32, 64, 32),
        (64, 64, 64),
        (96, 64, 96),
        (160, 64, 160),
        (5120, 576, 256),
        (288, 5120, 288),
    ):
        codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        scales = (np.arange(n * k // 32).reshape(n, k // 32) % 40 + 104).astype(
            np.uint8
        )
        check(codes, scales, k, n, pk, "synthetic-all-nibbles-and-exponents")
    if args.checkpoint:
        for expert in (0, 383):
            c13, s13 = checkpoint(args.checkpoint, [expert], "gate_up")
            c2, s2 = checkpoint(args.checkpoint, [expert], "down")
            for rank in (0, 7):
                lo, hi = rank * 288, (rank + 1) * 288
                c = np.concatenate((c13[0][lo:hi], c13[0][2304 + lo : 2304 + hi]))
                s = np.concatenate((s13[0][lo:hi], s13[0][2304 + lo : 2304 + hi]))
                check(c, s, 5120, 576, 256, f"expert{expert}-rank{rank}-up")
                check(
                    c2[0][:, lo // 2 : hi // 2],
                    s2[0][:, lo // 32 : hi // 32],
                    288,
                    5120,
                    288,
                    f"expert{expert}-rank{rank}-down",
                )
    (args.output / "result.json").write_text(
        json.dumps(dict(status="passed", cases=records), indent=2) + "\n"
    )
    print("PASS: NPU pack is byte-identical to CPU converter", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--compact-scales", action="store_true")
    main(p.parse_args())
