"""Independent CPU metadata oracle, sparse writes and changing NPU Graph inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.page_metadata import (
    DraftPageMetadata,
)
from dsv41.kernels.references.page_metadata import oracle
from dsv41.kernels.support.page_metadata import fixture, input_values


def run_case(op, batch, block, context, records, graph=True):
    tokens, page_ids, _, requests, prefix, full_slots, outputs, guards = fixture(
        batch, block, context
    )
    mapping = torch.zeros(4096 * 128, dtype=torch.int64, device="npu")

    def launch():
        op(tokens, mapping, requests, prefix, full_slots, **outputs, block=block)

    expected = {key: value.cpu() for key, value in outputs.items()}
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        launch()
    stream.synchronize()
    captured = None
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured, stream=stream):
            launch()
    # Restore all outputs: the first warm-up read all dummy requests.
    for tensor in outputs.values():
        tensor.fill_(-787878)
    for generation in range(5):
        values = input_values(batch, block, context, page_ids, generation)
        for destination, value in zip((requests, prefix, full_slots, mapping), values):
            destination.copy_(value)
        expected = oracle(expected, page_ids, *values, block, context)
        if captured is None:
            launch()
        else:
            captured.replay()
        for name, value in outputs.items():
            torch.testing.assert_close(
                value.cpu(),
                expected[name],
                rtol=0,
                atol=0,
                msg=f"B{batch}/block{block}/C{context}/{generation}/{name}",
            )
        for guard in guards:
            assert bool((guard[:32].cpu() == -787878).all())
            assert bool((guard[-32:].cpu() == -787878).all())
    records.append(
        {
            "batch": batch,
            "block": block,
            "context": context,
            "mode": "graph" if graph else "eager",
            "replays": 5,
            "status": "passed",
            "inactive_table_unchanged": True,
        }
    )
    print(
        f"PASS metadata B{batch} block{block} context{context} graph={graph}",
        flush=True,
    )


def check_attention(op, records):
    """Run the actual consumer with canaries in every untranslated table entry."""

    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
        DraftAttention,
    )

    batch, block, context = 2, 5, 1048832
    tokens, page_ids, _, requests, prefix, full_slots, outputs, _ = fixture(
        batch, block, context
    )
    values = input_values(batch, block, context, page_ids, 0)
    mapping = values[3].npu()
    requests.copy_(torch.tensor([1, 2], dtype=torch.int64))
    kv = (
        torch.randn(1024, 128, 1, 512, dtype=torch.bfloat16, device="npu") * 0.2
    ).contiguous()
    q = torch.randn(batch * block, 64, 512, dtype=torch.bfloat16, device="npu")
    sink = torch.randn(64, device="npu")
    sparse_output, dense_output = torch.empty_like(q), torch.empty_like(q)
    dense_table = (values[3][page_ids[1:].long() * 128] // 128).int().npu()
    attention = DraftAttention(capacity=256, heads=64)
    for positions in ((0, 129), (16383, 16384), (context - block, context - 129)):
        prefix.copy_(torch.tensor(positions, dtype=torch.int64))
        locations = torch.empty((batch, block), dtype=torch.int64)
        for b in range(batch):
            for k in range(block):
                pos = positions[b] + k
                locations[b, k] = int(page_ids[b + 1, pos // 128]) * 128 + pos % 128
        full_slots.copy_(locations.flatten())
        op(tokens, mapping, requests, prefix, full_slots, **outputs, block=block)
        attention(
            q,
            kv,
            outputs["table"],
            outputs["cu"],
            outputs["lengths"],
            sink,
            out=sparse_output,
        )
        attention(
            q,
            kv,
            dense_table,
            outputs["cu"],
            outputs["lengths"],
            sink,
            out=dense_output,
        )
        torch.testing.assert_close(
            sparse_output.cpu(), dense_output.cpu(), rtol=0, atol=0
        )
        records.append(
            {
                "kind": "actual_draft_attention",
                "prefix": positions,
                "status": "passed",
                "bitwise_equal": True,
            }
        )
        print(f"PASS actual draft attention prefix={positions}", flush=True)


def check_output_alignment(op, records):
    """Reject a contiguous table whose group8 boundary is shifted within 32B."""
    tokens, _, _, requests, prefix, full_slots, outputs, _ = fixture(9, 5, 256)
    mapping = torch.zeros(4096 * 128, dtype=torch.int64, device="npu")
    shape = outputs["table"].shape
    storage = torch.full(
        (outputs["table"].numel() + 8,), -787878, dtype=torch.int32, device="npu"
    )
    outputs["table"] = storage[1 : 1 + outputs["table"].numel()].view(shape)
    assert outputs["table"].is_contiguous() and outputs["table"].data_ptr() % 32 == 4
    try:
        op(tokens, mapping, requests, prefix, full_slots, **outputs, block=5)
    except (ValueError, RuntimeError) as error:
        assert "32B-aligned" in str(error), str(error)
    else:
        raise AssertionError("misaligned output table reached the metadata kernel")
    for output in outputs.values():
        assert bool((output.cpu() == -787878).all())
    assert bool((storage.cpu() == -787878).all())
    records.append(
        {
            "kind": "interface_misaligned_table_rejected",
            "status": "passed",
            "outputs_unchanged": True,
        }
    )
    print(
        "PASS interface rejects non-32B-aligned output table before launch", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--attention-only", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    op = DraftPageMetadata(library=args.library)
    records = []
    for batch, block, context in (
        ()
        if args.attention_only
        else (
            (0, 5, 256),
            (1, 1, 256),
            (1, 5, 1048832),
            (8, 5, 1048832),
            (8, 6, 16512),
            (9, 5, 16512),
            (33, 5, 1048832),
            (65, 5, 16512),
            (8, 129, 16512),
            (1, 257, 1048832),
        )
    ):
        run_case(op, batch, block, context, records, graph=False)
        run_case(op, batch, block, context, records, graph=True)
    if args.attention_only:
        check_attention(op, records)
    else:
        check_output_alignment(op, records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"physical_device": args.physical, "cases": records}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
