"""Independent engram boundaries oracle and canary qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram_ops import (
    EngramOps,
    HashParameters,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT


def hash_reference(ids, positions, cu, slots, history, active, arrays):
    """Python integer formula, including all-pad rows beyond the live Cu prefix."""
    result = torch.empty(2, len(ids), 24, dtype=torch.int64)

    def row(t, context):
        for layer in range(2):
            for n in (2, 3, 4):
                value = 0
                for lag in range(n):
                    value ^= context[lag] * int(arrays["multipliers"][layer, lag])
                for h in range(8):
                    column = (n - 2) * 8 + h
                    result[layer, t, column] = value % int(
                        arrays["primes"][layer, n - 2, h]
                    ) + int(arrays["offsets"][layer, column])

    for t in range(len(ids)):
        row(t, [int(arrays["pad_id"])] * 4)
    for b, slot in enumerate(slots.tolist()):
        for t in range(int(cu[b]), int(cu[b + 1])):
            enabled = 0 <= slot < len(history) and bool(active[t])
            context, blocked = [], not enabled
            for lag in range(4):
                if enabled:
                    local = t - int(cu[b])
                    token = (
                        int(ids[t - lag])
                        if lag <= local
                        else int(history[slot, 3 + local - lag])
                    )
                    blocked |= (
                        int(positions[t]) < lag
                        or token < 0
                        or token >= len(arrays["token_map"])
                    )
                    blocked |= token == arrays["image_id"] or token >= 1_000_000
                else:
                    token = 0
                context.append(
                    int(arrays["pad_id"])
                    if blocked
                    else int(arrays["token_map"][token])
                )
            row(t, context)
    return result


def history_reference(ids, cu, slots, keep, history):
    result = history.clone()
    for b, slot in enumerate(slots.tolist()):
        begin, end, n = int(cu[b]), int(cu[b + 1]), int(keep[b])
        if not 0 <= slot < len(history) or not 0 < n <= end - begin:
            continue
        context = history[slot, :3].tolist() + ids[begin : begin + n].tolist()
        result[slot, :3] = torch.tensor(context[-3:], dtype=torch.int32)
        result[slot, 3:] = 0
    return result


def check_engram_boundaries(library, records):
    arrays = HashParameters.model_arrays()
    params = HashParameters(**arrays)
    op = EngramOps(library=library)
    batch, slots_count = 83, 88
    counts = [i % 9 for i in range(batch)]
    cu = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32
    )
    total = int(cu[-1]) + 3  # deliberate graph-padding tail outside the live Cu
    slots = torch.randperm(batch).int()
    slots[2] = -1
    slots[5] = slots_count + 1
    history = torch.zeros(slots_count, 3, dtype=torch.int32)
    history[:, :3] = torch.randint(
        0, len(arrays["token_map"]), (slots_count, 3), dtype=torch.int32
    )
    ids = torch.randint(0, len(arrays["token_map"]), (total,), dtype=torch.int32)
    ids[1] = -3
    ids[7] = arrays["image_id"]
    ids[13] = 1_000_003
    positions = torch.cat(
        [torch.arange(b % 5, b % 5 + n) for b, n in enumerate(counts)]
        + [torch.arange(3)]
    ).long()
    active = torch.ones(total, dtype=torch.int32)
    active[-3:] = 0
    active[::13] = 0
    cpu = (ids, positions, cu, slots, history, active)
    device = [t.npu() for t in cpu]
    out_base = torch.full(
        (2 * total * 24 + 64,), -987654321, dtype=torch.int64, device="npu"
    )
    out = out_base[32:-32].view(2, total, 24)
    op.hash(*device, params, out=out)
    assert torch.equal(out.cpu(), hash_reference(*cpu, arrays))
    assert torch.equal(device[4].cpu(), history)
    assert (out_base[:32].cpu() == -987654321).all() and (
        out_base[-32:].cpu() == -987654321
    ).all()
    keep = torch.tensor(
        [min(n, b % 7) for b, n in enumerate(counts)], dtype=torch.int32
    )
    keep[1] = -1
    keep[3] = counts[3] + 1
    op.commit(device[0], device[2], device[3], keep.npu(), device[4])
    assert torch.equal(
        device[4].cpu(), history_reference(ids, cu, slots, keep, history)
    )
    records.append(
        {
            "case": "engram/model-parameters/B83/padding-invalid-slots/guarded-hash-commit"
        }
    )

    # Independent original-basis equation: normalize first, then affine q/k,
    # rather than copying the native weighted-dot factorization.
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 33, 257):
        x = torch.randn(m, 4, 5120).bfloat16()
        kv = torch.randn(m, 5 * 5120).bfloat16()
        qw, kw = [torch.randn(4, 5120).bfloat16() for _ in range(2)]
        active = (torch.arange(m) % 3 != 1).int()
        source = [t.npu() for t in (x, kv, qw, kw, active)]
        allocation = torch.full(
            (m * 4 * 5120 + 64,), 17.0, dtype=torch.bfloat16, device="npu"
        )
        output = allocation[32:-32].view(m, 4, 5120)
        op.gate(*source, out=output)
        h, key = x.double(), kv[:, :20480].reshape(m, 4, 5120).double()
        q = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-20) * qw.double()
        k = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-20) * kw.double()
        dot = (q * k).sum(-1) / (5120**0.5)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate[active == 0] = 0
        expected = (h + gate[:, :, None] * kv[:, 20480:].double()[:, None]).bfloat16()
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), atol=0.001, rtol=0.008
        )
        assert (allocation[:32].cpu() == 17).all() and (
            allocation[-32:].cpu() == 17
        ).all()
        records.append(
            {
                "case": f"engram/original-basis-FP64/M{m}/prefix-suffix-canaries",
                "max_abs": (output.cpu().float() - expected.float()).abs().max().item(),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, default=6)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/native/libdsv41_kernels.so"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "runs/tests/engram_boundaries.json"
    )
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    records = []
    check_engram_boundaries(args.library, records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"status": "passed", "physical": args.physical, "cases": records}, indent=2
        )
        + "\n"
    )
    print(f"PASS {len(records)} checks", flush=True)


if __name__ == "__main__":
    main()
