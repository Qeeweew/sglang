"""FP64 checks for coordinate caches, per-AIC state and direct arena reuse.

Every launch computes all query rows. Sparse cases check the reported FP64
subset; direct cases check every row after a full-head predecessor launch.
Whole-output finite checks, output canaries and address checks cover replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
    SparseAttention,
    WindowAttention,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from dsv41.kernels.references.attention import attention_data


def reference(cpu, ratio, selected):
    """Select logical rows independently and perform FP64 softmax and PV."""
    (
        q,
        ori,
        cmp,
        ids,
        table,
        cmp_table,
        cu,
        lengths,
        available,
        residual,
        window,
        sinks,
    ) = cpu
    result = []
    for token in selected:
        batch = next(
            b for b in range(len(lengths)) if int(cu[b]) <= token < int(cu[b + 1])
        )
        count = int(cu[batch + 1] - cu[batch])
        offset = token - int(cu[batch])
        right = max(0, int(lengths[batch]) - count + offset + 1)
        visible = min(128, right)
        requested = int(window[token, 0])
        if requested > 0:
            visible = min(visible, requested)
        rows = [
            ori[int(table[batch, position // 128]), position % 128, 0]
            for position in range(right - visible, right)
        ]
        cutoff = (
            max(
                0,
                int(available[batch]) * ratio
                + int(residual[batch])
                - count
                + offset
                + 1,
            )
            // ratio
        )
        limit = min(int(available[batch]), cutoff)
        page_rows = 128 // ratio
        for index in ids[token, 0, : min(512, limit)].tolist():
            if 0 <= index < limit and index // page_rows < cmp_table.shape[1]:
                rows.append(
                    cmp[
                        int(cmp_table[batch, index // page_rows]),
                        index % page_rows,
                        0,
                    ]
                )
        if not rows:
            result.append(torch.zeros_like(q[token]))
            continue
        values = torch.stack(rows).double()
        logits = q[token].double() @ values.T * (512**-0.5)
        probabilities = torch.cat((logits, sinks.double()[:, None]), 1).softmax(1)
        result.append((probabilities[:, :-1] @ values).bfloat16())
    return torch.stack(result)


def fixture(mode, heads, total, counts, columns, case, generation, cores, alternate):
    cpu = attention_data(
        mode,
        len(counts),
        total,
        16384,
        case,
        counts,
        heads=heads,
        physical_blocks=(4, 4),
    )
    cpu[5] = torch.randint(4, (len(counts), columns), dtype=torch.int32)
    page_rows = 128 // int(mode[1])
    cpu[3][:, 0, :5] = torch.tensor(
        [-2147483648, 2147483647, 0, (columns - 1) * page_rows, columns * page_rows],
        dtype=torch.int32,
    )
    if alternate:
        requested = (128, 17, 0, 1, 127, 129)
        cpu[10][:, 0] = torch.tensor(
            [
                requested[(token // cores + generation) % len(requested)]
                for token in range(total)
            ],
            dtype=torch.int32,
        )
    else:
        cpu[10][::3, 0] = 17
    return cpu


def run_case(heads, mode, columns, counts, selected, cases, cores, alternate, records):
    total = sum(counts)
    cpu = fixture(mode, heads, total, counts, columns, "full", 0, cores, alternate)
    device = [value.npu() for value in cpu]
    storage = torch.full(
        (total + 2, heads, 512), 13.0, device="npu", dtype=torch.bfloat16
    )
    output = storage[1:-1]
    op = SparseAttention(heads=heads, ratio=int(mode[1]), window_reuse=alternate)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        invoke = op.bind(*device, out=output)
        invoke()
        invoke()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        invoke()
    addresses = [value.data_ptr() for value in (*device, output)]
    workspaces = {value.data_ptr() for value in op.kernel._workspaces.values()}
    for generation, case in enumerate(cases):
        changed_counts = list(reversed(counts)) if generation % 2 else counts
        cpu = fixture(
            mode,
            heads,
            total,
            changed_counts,
            columns,
            case,
            generation,
            cores,
            alternate,
        )
        with torch.npu.stream(stream):
            for destination, source in zip(device, cpu):
                destination.copy_(source)
            output.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        expected = reference(cpu, int(mode[1]), selected)
        actual = output[selected].cpu()
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
        assert bool(torch.isfinite(output).all())
        assert bool((storage[0] == 13).all()) and bool((storage[-1] == 13).all())
        assert addresses == [value.data_ptr() for value in (*device, output)]
        assert workspaces == {
            value.data_ptr() for value in op.kernel._workspaces.values()
        }
        records.append(
            dict(
                heads=heads,
                mode=mode,
                batch=len(counts),
                tokens=total,
                compressed_table_columns=columns,
                generation=generation,
                case=case,
                per_aic_full_narrow_alternation=alternate,
                checked_token_indices=selected,
                max_abs=(actual.float() - expected.float()).abs().max().item(),
            )
        )
    print(
        "PASS",
        heads,
        mode,
        columns,
        "window" if alternate else "coordinates",
        flush=True,
    )


def direct_interleaving(records):
    """Overwrite the actors' full head/depth arenas between padded replays."""
    stream = torch.npu.Stream()
    poison_cpu = attention_data(
        "swa", 1, 6, 256, "full", [6], heads=64, physical_blocks=(2, 2)
    )
    poison_cpu[0].fill_(37)
    poison_cpu[1].fill_(-29)
    with torch.npu.stream(stream):
        poison = [x.npu() for x in poison_cpu]
        # Three KV tiles exercise streamed-PV MTE2 credit consumers before
        # the resident-only padded-head successor reuses the same actors.
        poison_op = WindowAttention(heads=64, window_rows=384, cores=6)
        poison_out = torch.empty_like(poison[0])
        poison_invoke = poison_op.bind(*poison, out=poison_out)
        poison_invoke()
    stream.synchronize()
    for width in (16, 128, 512):
        for tokens in (1, 6):
            cpu = attention_data(
                "swa",
                1,
                tokens,
                256,
                "full",
                [tokens],
                heads=16,
                physical_blocks=(2, 2),
            )
            cpu[0] = torch.randn(tokens, 16, width).bfloat16()
            cpu[1] = (torch.randn(2, 128, 1, width) * 0.1).bfloat16()
            op = WindowAttention(heads=16, cores=6)
            with torch.npu.stream(stream):
                device = [x.npu() for x in cpu]
                guard = torch.full(
                    (tokens + 2, 16, width), 19, device="npu", dtype=torch.bfloat16
                )
                out = guard[1:-1]
                invoke = op.bind(*device, out=out)
                invoke()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                for _ in range(10):
                    poison_invoke()
                    invoke()
            pointers = [
                x.data_ptr() for x in (*device, guard, *op.kernel._workspaces.values())
            ]
            for generation in range(4):
                current = [x.clone() for x in cpu]
                current[0] = (current[0].float() + generation * 0.125).bfloat16()
                current[1] = (current[1].float() - generation * 0.03125).bfloat16()
                expected = []
                for token in range(tokens):
                    right = 256 - tokens + token + 1
                    values = torch.stack(
                        [
                            current[1][int(current[2][0, p // 128]), p % 128, 0]
                            for p in range(right - 128, right)
                        ]
                    ).double()
                    score = current[0][token].double() @ values.T * width**-0.5
                    probability = torch.cat(
                        (score, current[5].double()[:, None]), 1
                    ).softmax(1)
                    expected.append((probability[:, :-1] @ values).bfloat16())
                with torch.npu.stream(stream):
                    for dst, src in zip(device, current):
                        dst.copy_(src)
                    out.fill_(float("nan"))
                    poison_invoke()
                    graph.replay()
                stream.synchronize()
                assert bool(torch.isfinite(out).all())
                torch.testing.assert_close(
                    out.cpu(), torch.stack(expected), atol=0.005, rtol=0.02
                )
                assert bool((guard[0] == 19).all()) and bool((guard[-1] == 19).all())
                assert pointers == [
                    x.data_ptr()
                    for x in (*device, guard, *op.kernel._workspaces.values())
                ]
                records.append(
                    dict(
                        kind="direct_interleaved_onchip_state",
                        width=width,
                        tokens=tokens,
                        generation=generation,
                        graph=True,
                        predecessor_heads=64,
                        predecessor_width=512,
                        predecessor_capacity=384,
                        graph_launches=20,
                        reference="FP64 every output row",
                    )
                )
            print("PASS direct interleaving", width, tokens, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical):
        raise ValueError("bind exactly the requested physical NPU")
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(3479)
    properties = torch.npu.get_device_properties(0)
    cores = getattr(properties, "cube_core_num", properties.vector_core_num // 2)
    records = []
    total = cores * 4 + 1
    selected = sorted(
        {0, total - 1}
        | {
            token
            for wave in range(4)
            for token in (wave * cores + 1, (wave + 1) * cores - 1)
        }
        | {wave * cores for wave in range(4)}
    )
    for heads in (16, 32, 64):
        for mode in ("c1", "c2"):
            run_case(
                heads,
                mode,
                128,
                [total],
                selected,
                ("full", "short", "full", "empty", "full"),
                cores,
                True,
                records,
            )
    for mode in ("c1", "c2"):
        for columns in (1, 8, 257, 513, 1024, 1025):
            run_case(
                32,
                mode,
                columns,
                [0, 19, 0, 22],
                [0, 1, 19, 20, 21, 40],
                ("full", "short", "duplicate", "sinks", "empty"),
                cores,
                False,
                records,
            )
    direct_interleaving(records)
    library = native_library_path()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            dict(
                status="passed",
                physical=args.physical,
                aic_cores=cores,
                library=str(library),
                library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                reference="FP64 selected query rows; BF16 final rounding",
                whole_output="finite, NaN write coverage, canaries and stable addresses",
                physical_blocks=[4, 4],
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print("PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()
