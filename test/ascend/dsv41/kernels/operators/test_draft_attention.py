"""DSpark full-block visibility from exact paged inputs and Graph updates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.builders.native import build_native
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import DraftAttention
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.draft import oracle_attention, oracle_window


def run_case(
    batch,
    total,
    counts,
    prefix,
    records,
    capture=False,
    *,
    capacity=256,
    heads=64,
):
    columns = max(8, (max(p + n for p, n in zip(prefix, counts)) + 127) // 128)
    blocks = batch * columns
    op = DraftAttention(capacity=capacity, heads=heads)
    q = torch.randn(total, heads, 512).bfloat16()
    kv = (torch.randn(blocks, 128, 1, 512) * 0.2).bfloat16()
    table = torch.randperm(blocks).int().reshape(batch, columns)
    cu = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32
    )
    lengths = torch.tensor([p + n for p, n in zip(prefix, counts)], dtype=torch.int32)
    sinks = torch.randn(heads)
    if counts[0] > 1:
        # Earliest query must attend the last *future* draft row. A causal
        # substitution cannot pass this deliberately separated numerical case.
        q[0].fill_(1)
        last = int(lengths[0]) - 1
        kv[int(table[0, last // 128]), last % 128, 0].fill_(2)
    device = [x.npu() for x in (q, kv, table, cu, lengths, sinks)]
    output = torch.empty_like(device[0])

    def chain():
        return op(*device, out=output)

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(2):
            chain()
    stream.synchronize()
    selected = None if total < 1024 else sorted({0, 1, total // 2, total - 1})

    def checked_output():
        return output.cpu() if selected is None else output[selected].cpu()

    expected_window, expected_visible = oracle_window(kv, table, cu, lengths, capacity)
    expected = oracle_attention(
        q, expected_window, cu, expected_visible, sinks, token_indices=selected
    )
    torch.testing.assert_close(
        checked_output().float(), expected.float(), atol=0.005, rtol=0.02
    )
    if counts[0] > 1 and int(expected_visible[0, 0]) > 0:
        assert bool((output[0].cpu().float() > 1.9).all()), (
            "draft future block was causally masked"
        )
    records.append(
        {
            "kind": "eager",
            "heads": heads,
            "counts": counts,
            "prefix": prefix,
            "max_abs": (checked_output().float() - expected.float()).abs().max().item(),
        }
    )
    if not capture:
        return
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        chain()
    for generation in range(4):
        q = torch.randn_like(q)
        kv = torch.randn_like(kv) * 0.2
        if generation % 2:
            counts = list(reversed(counts))
        cu = torch.tensor(
            [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32
        )
        lengths = torch.tensor([n + generation * 63 for n in counts], dtype=torch.int32)
        table = torch.randperm(blocks).int().reshape(batch, columns)
        sinks = torch.randn(heads) * (100 if generation == 2 else 1)
        for dst, src in zip(device, (q, kv, table, cu, lengths, sinks)):
            dst.copy_(src)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        w, v = oracle_window(kv, table, cu, lengths, capacity)
        expected = oracle_attention(q, w, cu, v, sinks, token_indices=selected)
        torch.testing.assert_close(
            checked_output().float(), expected.float(), atol=0.005, rtol=0.02
        )
        records.append(
            {
                "kind": "graph",
                "heads": heads,
                "generation": generation,
                "max_abs": (checked_output().float() - expected.float())
                .abs()
                .max()
                .item(),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, default=6)
    parser.add_argument("--heads", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--build-output", type=Path, default=ROOT / "build/tests/draft-attention"
    )
    parser.add_argument("--large-window", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "runs/draft-attention/validation.json"
    )
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    if args.build:
        build_native(args.build_output, components=["attention"])
        os.environ["DSV41_NATIVE_DIR"] = str(args.build_output)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(12309)
    records = []
    cases = [
        ([6], [prefix], prefix == 255, 256) for prefix in (0, 1, 127, 128, 255, 511)
    ]
    cases += [
        ([0, 3, 6, 7], [0, 128, 255, 511], True, 256),
        ([33], [127], False, 256),
        ([6] * 42 + [5], [127] * 43, False, 256),
    ]
    if args.large_window:
        cases.append(([257], [127], True, 512))
        cases += [
            ([tokens], [127], True, capacity)
            for capacity in (2048, 4096)
            for tokens in (6, 1800)
        ]
    for heads in args.heads:
        for counts, prefix, capture, capacity in cases:
            run_case(
                len(counts),
                sum(counts),
                counts,
                prefix,
                records,
                capture,
                capacity=capacity,
                heads=heads,
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            dict(
                status="passed",
                physical_device=args.physical,
                paged_kv=True,
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"PASS {len(records)} direct paged draft attention cases", flush=True)


if __name__ == "__main__":
    main()
