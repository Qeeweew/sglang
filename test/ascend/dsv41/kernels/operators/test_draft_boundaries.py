"""Independent draft boundaries oracle and canary qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import DraftAttention
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT


def check_draft_accuracy(records, large_window=False):
    # Separate source-selected rows and an independent CPU FP64 softmax/PV.
    cases = [
        (1, 6, [6], 256),
        (4, 16, [0, 1, 8, 7], 256),
        (1, 33, [33], 256),
        (43, 257, [6] * 42 + [5], 256),
    ]
    if large_window:
        cases.append((1, 257, [257], 512))
    for batch, total, qlens, capacity in cases:
        q = torch.randn(total, 64, 512).bfloat16()
        kv = (torch.randn(batch * 8, 128, 1, 512) * 0.2).bfloat16()
        table = torch.randperm(batch * 8).int().reshape(batch, 8)
        cu = torch.tensor(
            [0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32
        )
        lengths = torch.tensor(
            [129 + b % 129 + n for b, n in enumerate(qlens)], dtype=torch.int32
        )
        sinks = torch.randn(64)
        args = [t.npu() for t in (q, kv, table, cu, lengths, sinks)]
        guard_out = torch.full(
            (total * 64 * 512 + 64,), 13.0, dtype=torch.bfloat16, device="npu"
        )
        output = guard_out[32:-32].view(total, 64, 512)
        op = DraftAttention(capacity=capacity)

        def launch(args=args, op=op, output=output):
            op(*args, out=output)

        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for _ in range(2):
                launch()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            launch()
        for replay in range(3):
            q = torch.randn_like(q)
            kv = torch.randn_like(kv) * 0.2
            # Prefixes cross page boundaries. Every query sees the final
            # draft row, including the earliest query in a request.
            first_request = next(i for i, n in enumerate(qlens) if n)
            t0 = int(cu[first_request])
            q[t0].fill_(1)
            final_pos = int(lengths[first_request]) - 1
            kv[int(table[first_request, final_pos // 128]), final_pos % 128, 0].fill_(2)
            args[0].copy_(q)
            args[1].copy_(kv)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            expected = torch.zeros_like(q)
            for b, n in enumerate(qlens):
                if not n:
                    continue
                first = max(0, int(lengths[b]) - n - 128)
                rows = torch.stack(
                    [
                        kv[int(table[b, p // 128]), p % 128, 0]
                        for p in range(first, int(lengths[b]))
                    ]
                ).double()
                for t in range(int(cu[b]), int(cu[b + 1])):
                    logits = q[t].double() @ rows.T / (512**0.5)
                    probability = torch.cat(
                        (logits, sinks.double()[:, None]), 1
                    ).softmax(1)[:, :-1]
                    expected[t] = (probability @ rows).bfloat16()
            torch.testing.assert_close(
                output.cpu().float(), expected.float(), atol=0.005, rtol=0.02
            )
            assert (output[t0].cpu().float() > 1.9).all(), (
                "future draft row was causally hidden"
            )
            for tensor, n, value in ((guard_out, 32, 13),):
                assert (tensor[:n].cpu() == value).all() and (
                    tensor[-n:].cpu() == value
                ).all()
            records.append(
                {
                    "case": f"draft/B{batch}T{total}S{capacity}/FP64/noncausal/guarded/graph{replay}",
                    "max_abs": (output.cpu().float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                }
            )
        del graph


def check_draft_guards(records):
    """Metadata rejection must not read invalid pages or publish nonzero output."""
    kv = torch.randn(4, 128, 1, 512, device="npu", dtype=torch.bfloat16)
    table = torch.arange(4, dtype=torch.int32, device="npu").view(1, 4)
    cu = torch.tensor([0, 6], dtype=torch.int32, device="npu")
    lengths = torch.tensor([134], dtype=torch.int32, device="npu")
    q = torch.randn(6, 64, 512, dtype=torch.bfloat16, device="npu")
    sinks = torch.randn(64, device="npu")
    output = torch.empty_like(q)
    op = DraftAttention()
    # Page IDs are an input contract, matching C0/C1/C2. These guards
    # cover inconsistent request lengths without introducing a host sync.
    for name, count, length in (
        ("capacity", 129, 257),
        ("length-below-query", 6, 5),
        ("length-above-table", 6, 513),
        ("negative-query-count", -1, 6),
    ):
        table.copy_(torch.arange(4, dtype=torch.int32, device="npu").view(1, 4))
        cu.copy_(torch.tensor([0, count], dtype=torch.int32))
        lengths.fill_(length)
        op(q, kv, table, cu, lengths, sinks, out=output)
        assert torch.count_nonzero(output.cpu()).item() == 0, name
        records.append({"case": "draft/invalid-metadata/" + name})

    # One live query, five graph-padding rows outside Cu: only first is used.
    table.copy_(torch.arange(4, dtype=torch.int32, device="npu").view(1, 4))
    cu.copy_(torch.tensor([0, 1], dtype=torch.int32))
    lengths.fill_(1)
    op(q, kv, table, cu, lengths, sinks, out=output)
    assert torch.count_nonzero(output[1:].cpu()).item() == 0
    assert torch.count_nonzero(output[:1].cpu()).item() > 0
    records.append({"case": "draft/Cu-tail-padding-zero"})

    # Disjoint external tensors are a launch-time contract, not a promise
    # that writes may overwrite Q after Cube has consumed it.
    try:
        op(q, kv, table, cu, lengths, sinks, out=q)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("overlapping Q/output reached the kernel")
    records.append({"case": "draft/alias-rejected"})

    empty = torch.empty(0, 64, 512, dtype=torch.bfloat16, device="npu")
    assert op(empty, kv, table, cu, lengths, sinks, out=empty) is empty
    invalid = torch.empty(0, 63, 512, dtype=torch.bfloat16, device="npu")
    try:
        op(invalid, kv, table, cu, lengths, sinks)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid empty query layout accepted")
    records.append({"case": "draft/empty-no-launch-and-layout-guard"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, default=6)
    parser.add_argument("--large-window", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "runs/tests/draft_boundaries.json"
    )
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    records = []
    check_draft_accuracy(records, args.large_window)
    check_draft_guards(records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            dict(status="passed", physical=args.physical, cases=records), indent=2
        )
        + "\n"
    )
    print(f"PASS {len(records)} checks", flush=True)


if __name__ == "__main__":
    main()
