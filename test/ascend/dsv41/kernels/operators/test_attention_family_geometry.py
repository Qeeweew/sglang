"""Runtime geometry, independent references and changing-address-stable Graphs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def rotate(x, cos, sin, inverse=False):
    result = x.float().clone()
    rotary = 2 * cos.shape[-1]
    tail = result[..., -rotary:].reshape(*x.shape[:-1], rotary // 2, 2).clone()
    c, s = cos, -sin if inverse else sin
    while c.ndim < tail.ndim - 1:
        c, s = c.unsqueeze(1), s.unsqueeze(1)
    result[..., -rotary::2] = tail[..., 0] * c - tail[..., 1] * s
    result[..., -rotary + 1 :: 2] = tail[..., 1] * c + tail[..., 0] * s
    return result.bfloat16()


def position_cases(records):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.position_quant import (
        PositionQuant,
    )

    for width, rotary, heads, tokens in (
        (80, 16, 3, 6),
        (144, 32, 7, 33),
        (768, 48, 48, 6),
        (2048, 64, 5, 257),
        (2048, 16, 3, 6),
    ):
        op = PositionQuant(rotary_dim=rotary)
        x = torch.randn(tokens, heads, width).bfloat16()
        cos, sin = torch.randn(tokens, rotary // 2), torch.randn(tokens, rotary // 2)
        device = [t.npu() for t in (x, cos, sin)]
        output = torch.empty_like(device[0])
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            op.rope(*device, out=output)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            op.rope(*device, out=output)
        pointers = [t.data_ptr() for t in (*device, output)]
        for generation in range(3):
            cpu = [x + generation, cos * (generation + 1), sin / (generation + 1)]
            with torch.npu.stream(stream):
                for dst, src in zip(device, cpu):
                    dst.copy_(src)
                output.fill_(float("nan"))
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output.cpu(), rotate(*cpu), rtol=0, atol=0)
            assert pointers == [t.data_ptr() for t in (*device, output)]
            records.append(
                dict(
                    component="position_quant",
                    width=width,
                    rotary=rotary,
                    heads=heads,
                    tokens=tokens,
                    generation=generation,
                    graph=True,
                )
            )
        codes, scales = op.quantize_query(device[0])
        maximum = device[0].cpu().float().abs().amax(-1)
        reference_scale = (maximum / 127).half().float().clamp_min(2**-24)
        reference_code = (
            (device[0].cpu().float() / reference_scale[..., None])
            .round()
            .clamp(-127, 127)
        )
        torch.testing.assert_close(scales.cpu(), reference_scale.half(), rtol=0, atol=0)
        assert (codes.cpu().int() - reference_code.int()).abs().max() <= 1
        print("PASS dynamic position", width, rotary, heads, tokens, flush=True)


def compressor_cases(records):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.compressor import (
        CompressorState,
        ProjectedCompressor,
    )
    from dsv41.kernels.references.compressor import (
        oracle_commit,
        oracle_prepare,
    )

    for width in (16, 144, 768, 4096):
        for ratio in (1, 2):
            counts = [0, 3, 3]
            cu = torch.tensor([0, 0, 3, 6], dtype=torch.int32)
            request_ids = torch.tensor([0, 1, 2], dtype=torch.int32)
            kv = torch.randn(6, width)
            if ratio == 1:
                kv = kv.bfloat16()
            scores = torch.randn_like(kv, dtype=torch.float32) if ratio == 2 else None
            gamma = torch.randn(width)
            pending = torch.randn(3, 2, width)
            lengths = torch.tensor([0, 3, 4], dtype=torch.int64)
            state = CompressorState.allocate(3, width=width)
            state.pending.copy_(pending)
            state.committed_lengths.copy_(lengths)
            op = ProjectedCompressor(ratio, gamma.npu(), state)
            inputs = [
                t.npu() if t is not None else None
                for t in (kv, scores, cu, request_ids)
            ]
            accepted = torch.tensor([0, 1, 3], dtype=torch.int32).npu()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                provisional = op.prepare(*inputs)
                commit = op.commit(provisional, accepted)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                p = op.prepare(
                    *inputs,
                    out=provisional.latent,
                    pooled=provisional.pooled,
                    positions=provisional.positions,
                    rope_positions=provisional.rope_positions,
                    starts=provisional.starts,
                )
                op.commit(
                    p,
                    accepted,
                    positions=commit[0],
                    rope_positions=commit[1],
                    status=commit[2],
                )
            for generation in range(3):
                current_lengths = lengths + generation
                expected = oracle_prepare(
                    ratio, kv, scores, gamma, pending, current_lengths, cu, request_ids
                )
                expected_commit = oracle_commit(
                    ratio,
                    kv,
                    scores,
                    pending,
                    current_lengths,
                    cu,
                    request_ids,
                    expected[4],
                    accepted.cpu(),
                )
                with torch.npu.stream(stream):
                    state.pending.copy_(pending)
                    state.committed_lengths.copy_(current_lengths)
                    provisional.latent.fill_(float("nan"))
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    provisional.latent.cpu(), expected[0], rtol=0.01, atol=0.015625
                )
                torch.testing.assert_close(
                    provisional.pooled.cpu(), expected[1], rtol=2e-6, atol=2e-6
                )
                for actual, reference in zip(
                    (state.pending, state.committed_lengths, *commit), expected_commit
                ):
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
                records.append(
                    dict(
                        component="compressor",
                        width=width,
                        ratio=ratio,
                        generation=generation,
                        graph=True,
                        tokens=sum(counts),
                    )
                )
            print("PASS dynamic compressor", width, ratio, flush=True)


def page_cases(records):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.page_metadata import (
        DraftPageMetadata,
    )

    op = DraftPageMetadata()
    for page_rows, window_rows in ((64, 32), (256, 192), (48, 80)):
        batch, block, context = 3, 6, page_rows * 12
        token_table = torch.arange((batch + 1) * context, dtype=torch.int32).reshape(
            batch + 1, context
        )
        mapping = torch.arange(token_table.numel(), dtype=torch.int64) + 2 * page_rows
        requests = torch.tensor([1, 0, 3], dtype=torch.int64)
        prefix = torch.tensor([page_rows + 3, 0, context - block], dtype=torch.int64)
        full_slots = torch.cat(
            [
                token_table[r, p : p + block].long() if r else torch.full((block,), -1)
                for r, p in zip(requests.tolist(), prefix.tolist())
            ]
        )
        inputs = [t.npu() for t in (token_table, mapping, requests, prefix, full_slots)]
        columns = (context + page_rows - 1) // page_rows
        shapes = [(batch + 1,), (batch,), (batch, columns), (batch * block,), (1,)]
        types = [torch.int32, torch.int32, torch.int32, torch.int64, torch.int32]
        outputs = [
            torch.full(s, -3434, dtype=d, device="npu") for s, d in zip(shapes, types)
        ]
        names = ("cu", "lengths", "table", "slots", "routed_rows")

        def invoke():
            op(
                *inputs,
                **dict(zip(names, outputs)),
                block=block,
                page_rows=page_rows,
                window_rows=window_rows,
            )

        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            invoke()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            invoke()
        for generation in range(3):
            active_requests = requests.roll(generation)
            expected_table = torch.full((batch, columns), -3434, dtype=torch.int32)
            expected_slots = torch.full((batch * block,), -1, dtype=torch.int64)
            expected_lengths = torch.zeros(batch, dtype=torch.int32)
            for b in range(batch):
                request, position = int(active_requests[b]), int(prefix[b])
                if request <= 0:
                    continue
                expected_lengths[b] = position + block
                first, end = (
                    max(0, position - window_rows) // page_rows,
                    (position + block + page_rows - 1) // page_rows,
                )
                for page in range(first, end):
                    expected_table[b, page] = (
                        mapping[token_table[request, page * page_rows].long()]
                        // page_rows
                    )
                for t in range(block):
                    value = int(full_slots[b * block + t])
                    expected_slots[b * block + t] = mapping[value] if value >= 0 else -1
            with torch.npu.stream(stream):
                inputs[2].copy_(active_requests)
                for output in outputs:
                    output.fill_(-3434)
                graph.replay()
            stream.synchronize()
            expected = (
                torch.arange(batch + 1, dtype=torch.int32) * block,
                expected_lengths,
                expected_table,
                expected_slots,
                torch.tensor(
                    [int((active_requests > 0).sum()) * block], dtype=torch.int32
                ),
            )
            for actual, reference in zip(outputs, expected):
                torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            records.append(
                dict(
                    component="page_metadata",
                    page_rows=page_rows,
                    window_rows=window_rows,
                    generation=generation,
                    graph=True,
                )
            )
        print("PASS dynamic metadata", page_rows, window_rows, flush=True)


def attention_cases(records):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
        DraftAttention,
        SparseAttention,
        WindowAttention,
    )

    configurations = (
        ("swa", 16, 16, 64, 32, 0, 1, 6, 4, False),
        ("swa", 48, 192, 48, 192, 0, 3, 81, 4, False),
        ("swa", 32, 384, 256, 257, 0, 0, 6, 4, False),
        # A single resident tile with a partial final feature panel, and
        # changing request/window metadata on replay.
        ("swa", 48, 384, 64, 128, 0, 12, 3, 4, False),
        ("swa", 16, 512, 128, 128, 0, 4, 1, 1, False),
        ("c1", 48, 192, 64, 64, 128, 3, 41, 4, False),
        ("c2", 16, 80, 48, 96, 384, 1, 6, 4, False),
        ("c2", 64, 512, 256, 128, 256, 0, 81, 4, False),
        ("c1", 48, 192, 64, 64, 384, 3, 81, 1, True),
        ("draft", 48, 144, 64, 32, 0, 3, 6, 4, False),
        ("draft", 16, 16, 48, 80, 0, 1, 81, 4, False),
        ("draft", 32, 128, 256, 2048, 0, 3, 6, 4, False),
    )

    def reference(cpu, mode, window_rows, ratio):
        (
            q,
            kv,
            compressed,
            ids,
            table,
            compressed_table,
            cu,
            lengths,
            available,
            residual,
            window,
            sinks,
        ) = cpu
        page_rows, width = kv.shape[1], q.shape[-1]
        result = torch.zeros_like(q)
        for b in range(len(lengths)):
            count = int(cu[b + 1] - cu[b])
            for t in range(int(cu[b]), int(cu[b + 1])):
                if mode == "draft":
                    first, right = (
                        max(0, int(lengths[b]) - count - window_rows),
                        int(lengths[b]),
                    )
                else:
                    right = max(0, int(lengths[b]) - count + t - int(cu[b]) + 1)
                    visible = min(window_rows, right)
                    if int(window[t, 0]) > 0:
                        visible = min(visible, int(window[t, 0]))
                    first = right - visible
                values = [
                    kv[int(table[b, p // page_rows]), p % page_rows, 0]
                    for p in range(first, right)
                ]
                if ids is not None:
                    cutoff = (
                        max(
                            0,
                            int(available[b]) * ratio
                            + int(residual[b])
                            - count
                            + t
                            - int(cu[b])
                            + 1,
                        )
                        // ratio
                    )
                    limit = min(int(available[b]), cutoff)
                    for index in ids[t, 0, : min(ids.shape[-1], limit)].tolist():
                        if (
                            0 <= index < limit
                            and index // (page_rows // ratio)
                            < compressed_table.shape[1]
                        ):
                            values.append(
                                compressed[
                                    int(
                                        compressed_table[
                                            b, index // (page_rows // ratio)
                                        ]
                                    ),
                                    index % (page_rows // ratio),
                                    0,
                                ]
                            )
                if values:
                    value = torch.stack(values).double()
                    score = q[t].double() @ value.T * width**-0.5
                    probability = torch.cat(
                        (score, sinks.double()[:, None]), 1
                    ).softmax(1)[:, :-1]
                    result[t] = (probability @ value).bfloat16()
        return result

    for (
        mode,
        heads,
        width,
        page_rows,
        window_rows,
        selected,
        cores,
        tokens,
        batch,
        ring,
    ) in configurations:
        ratio = 2 if mode == "c2" else 1
        columns, physical_pages = 17, 19
        if mode == "swa":
            op = WindowAttention(heads=heads, window_rows=window_rows, cores=cores)
        elif mode == "draft":
            op = DraftAttention(
                heads=heads,
                capacity=max(256, (window_rows + tokens + 127) // 128 * 128),
                window_rows=window_rows,
                cores=cores,
            )
        else:
            op = SparseAttention(
                heads=heads,
                ratio=ratio,
                window_reuse=ring,
                window_rows=window_rows,
                cores=cores,
            )

        def inputs(generation):
            counts = (
                [tokens] if batch == 1 else [0, tokens // 2, 0, tokens - tokens // 2]
            )
            if generation % 2:
                counts = list(reversed(counts))
            cu = torch.tensor(
                [0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32
            )
            lengths = torch.tensor(
                [page_rows * columns - generation * (b + 1) for b in range(batch)],
                dtype=torch.int32,
            )
            q = (torch.randn(tokens, heads, width) * 0.3).bfloat16()
            kv = (torch.randn(physical_pages, page_rows, 1, width) * 0.3).bfloat16()
            table = torch.randint(physical_pages, (batch, columns), dtype=torch.int32)
            window = torch.zeros(tokens, 1, dtype=torch.int32)
            if generation == 1:
                window[:, 0] = torch.arange(tokens) % window_rows + 1
            sinks = torch.randn(heads)
            if generation == 2:
                sinks[::2], sinks[1::2] = 100, -100
            compressed, ids, compressed_table, available, residual = [None] * 5
            if selected:
                compressed = (
                    torch.randn(physical_pages, page_rows // ratio, 1, width) * 0.3
                ).bfloat16()
                compressed_table = torch.randint(
                    physical_pages, (batch, columns), dtype=torch.int32
                )
                available, residual = lengths // ratio, lengths % ratio
                ids = torch.randint(
                    -3,
                    page_rows * columns // ratio,
                    (tokens, 1, selected),
                    dtype=torch.int32,
                )
                ids[:, 0, :3] = torch.tensor(
                    [-2147483648, 2147483647, -1], dtype=torch.int32
                )
                if generation == 1:
                    ids[:, 0, 3::2] = 0
            return [
                q,
                kv,
                compressed,
                ids,
                table,
                compressed_table,
                cu,
                lengths,
                available,
                residual,
                None if mode == "draft" else window,
                sinks,
            ]

        cpu = inputs(0)
        device = [x.npu() if x is not None else None for x in cpu]
        guard = torch.full(
            (tokens + 2, heads, width), 17.0, dtype=torch.bfloat16, device="npu"
        )
        output = guard[1:-1]
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            if mode == "swa":
                invoke = op.bind(
                    *(device[i] for i in (0, 1, 4, 6, 7, 11, 10)), out=output
                )
            elif mode == "draft":
                invoke = op.bind(*(device[i] for i in (0, 1, 4, 6, 7, 11)), out=output)
            else:
                invoke = op.bind(*device, out=output)
            invoke()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            invoke()
        pointers = [
            x.data_ptr()
            for x in (*device, output, *op.kernel._workspaces.values())
            if x is not None
        ]
        for generation in range(3):
            cpu = inputs(generation)
            expected = reference(cpu, mode, window_rows, ratio)
            with torch.npu.stream(stream):
                for dst, src in zip(device, cpu):
                    if dst is not None:
                        dst.copy_(src)
                output.fill_(float("nan"))
                graph.replay()
            stream.synchronize()
            assert bool(torch.isfinite(output).all())
            torch.testing.assert_close(output.cpu(), expected, atol=0.005, rtol=0.02)
            assert bool((guard[0] == 17).all()) and bool((guard[-1] == 17).all())
            assert pointers == [
                x.data_ptr()
                for x in (*device, output, *op.kernel._workspaces.values())
                if x is not None
            ]
            records.append(
                dict(
                    component="attention",
                    mode=mode,
                    heads=heads,
                    width=width,
                    page_rows=page_rows,
                    window_rows=window_rows,
                    selected=selected,
                    cores=cores,
                    tokens=tokens,
                    generation=generation,
                    graph=True,
                    reference="FP64 all query rows",
                )
            )
        print(
            "PASS dynamic attention",
            mode,
            heads,
            width,
            page_rows,
            window_rows,
            selected,
            cores,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        default=["position_quant", "compressor", "page_metadata", "attention"],
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(79152)
    records = []
    functions = {
        "attention": attention_cases,
        "position_quant": position_cases,
        "compressor": compressor_cases,
        "page_metadata": page_cases,
    }
    for component in args.components:
        functions[component](records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            dict(status="passed", physical=args.physical, cases=records), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
