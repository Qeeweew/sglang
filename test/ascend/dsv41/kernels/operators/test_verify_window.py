"""Compare direct target-verify SWA to an independently materialized paged cache."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.attention.request_window import (
    BF16RequestWindow,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import (
    SparseAttention,
    WindowAttention,
)


VERIFY_ROWS = 6
PAGE_ROWS = 128
WIDTH = 512
LAYERS = 2


def _case(batch, ratio, heads):
    dev = torch.device("npu", torch.npu.current_device())
    rows = batch * VERIFY_ROWS
    window = BF16RequestWindow(
        num_slots=batch + 2, layers=LAYERS, page_size=PAGE_ROWS, capacity=134
    )
    columns = 24

    def empty(*shape, dtype=torch.int32):
        return torch.zeros(shape, dtype=dtype, device=dev)

    meta = SimpleNamespace(
        batch=batch,
        rows=rows,
        cu=torch.arange(batch + 1, dtype=torch.int32, device=dev) * VERIFY_ROWS,
        starts=empty(batch, dtype=torch.int64),
        lengths=empty(batch),
        verify_requests=empty(batch, dtype=torch.int64),
        verify_commit_lens=empty(batch),
        table=empty(batch, columns),
        window=empty(rows, 1),
        cmp=empty(batch),
        residual=empty(batch),
    )
    meta.window[1::VERIFY_ROWS] = 3
    inputs = empty(LAYERS, rows, WIDTH, dtype=torch.bfloat16)
    q = empty(rows, heads, WIDTH, dtype=torch.bfloat16)
    sinks = torch.linspace(-2, 4, heads, device=dev)
    indices = torch.arange(512, dtype=torch.int32, device=dev).repeat(rows, 1, 1)
    compressed_table = torch.arange(columns, dtype=torch.int32, device=dev).repeat(
        batch, 1
    )
    compressed = (
        torch.randn(columns, PAGE_ROWS // max(1, ratio), 1, WIDTH).bfloat16().to(dev)
    )
    operator = (
        WindowAttention(heads=heads)
        if not ratio
        else SparseAttention(heads=heads, ratio=ratio, window_reuse=batch == 1)
    )

    # History + current rows occupy at most three aligned pages per request.
    # Page zero contains empty data. Live pages are rebuilt from absolute positions.
    pages = 3
    paged = empty(batch * pages + 1, PAGE_ROWS, 1, WIDTH, dtype=torch.bfloat16)
    row_offsets = torch.arange(pages * PAGE_ROWS, device=dev)
    page_columns = torch.arange(columns, device=dev)
    query_bases = (torch.arange(batch, device=dev) * VERIFY_ROWS)[:, None]
    physical_bases = (1 + torch.arange(batch, device=dev) * pages)[:, None]

    def build_paged_reference():
        prefix = meta.starts[:, None]
        first = (prefix - PAGE_ROWS).clamp_min(0) // PAGE_ROWS
        pos = first * PAGE_ROWS + row_offsets
        ring_rows = (
            meta.verify_requests[:, None] * window.capacity + pos % window.capacity
        )
        pending_rows = query_bases + (pos - prefix).clamp(0, VERIFY_ROWS - 1)
        history = window.ring[0].index_select(0, ring_rows.flatten())
        pending = inputs[0].index_select(0, pending_rows.flatten())
        values = torch.where((pos < prefix).view(-1, 1), history, pending)
        live = (pos < prefix + VERIFY_ROWS) & (meta.verify_requests[:, None] > 0)
        paged[1:].view(-1, WIDTH).copy_(torch.where(live.view(-1, 1), values, 0))
        pages_live = (
            (page_columns >= first)
            & (page_columns * PAGE_ROWS < meta.lengths[:, None])
            & (meta.verify_requests[:, None] > 0)
        )
        meta.table.copy_(
            torch.where(pages_live, physical_bases + page_columns - first, 0)
        )

    def change(generation):
        origins = (0, 2, 255, 256, 257, 385, 511, 1531)
        starts = [origins[(i + generation) % len(origins)] for i in range(batch)]
        requests = [i + 1 for i in range(batch)]
        if batch > 1:
            dummy = (batch - 1 - generation) % batch
            requests[dummy] = starts[dummy] = 0
        lengths = [start + VERIFY_ROWS for start in starts]
        divisor = max(ratio, 1)
        for member, values in (
            ("starts", starts),
            ("lengths", lengths),
            ("verify_requests", requests),
            (
                "cmp",
                [
                    length // divisor if req else 0
                    for length, req in zip(lengths, requests)
                ],
            ),
            ("residual", [length % divisor for length in lengths]),
        ):
            tensor = getattr(meta, member)
            tensor.copy_(torch.tensor(values, dtype=tensor.dtype))
        inputs.normal_()
        q.normal_()
        return starts, requests

    def bind(cache, verify_input=None):
        out = torch.empty_like(q)
        if ratio:
            args = (
                q,
                cache,
                compressed,
                indices,
                meta.table,
                compressed_table,
                meta.cu,
                meta.lengths,
                meta.cmp,
                meta.residual,
                meta.window,
                sinks,
            )
        else:
            args = (q, cache, meta.table, meta.cu, meta.lengths, sinks, meta.window)
        return operator.bind(*args, out=out, verify_ring=verify_input), out

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        starts, requests = change(0)
        window.ring.normal_()
        before = window.ring.cpu()
        window.init_verify(meta, VERIFY_ROWS)
        direct, direct_out = bind(*window.verify_inputs(meta, 0))
        paged_call, paged_out = bind(paged)

        def invoke():
            meta.swa_pending.copy_(inputs)
            direct()
            build_paged_reference()
            paged_call()

        invoke()
        window.commit_verify(meta)
    stream.synchronize()

    def check(starts, requests, generation, before, commit):
        actual, expected = direct_out.cpu(), paged_out.cpu()
        valid = torch.tensor(requests).ne(0).repeat_interleave(VERIFY_ROWS)
        torch.testing.assert_close(actual[valid], expected[valid], atol=0, rtol=0)
        assert torch.count_nonzero(actual[~valid]).item() == 0
        torch.testing.assert_close(window.ring.cpu(), before, atol=0, rtol=0)
        accepts = [
            ((0, 2, 6)[(i + generation) % 3] if request else 0)
            for i, request in enumerate(requests)
        ]
        expected_ring = before.clone()
        source = inputs.cpu()
        for row, (start, request, accept) in enumerate(zip(starts, requests, accepts)):
            for j in range(accept):
                slot = request * window.capacity + (start + j) % window.capacity
                expected_ring[:, slot].copy_(source[:, row * VERIFY_ROWS + j])
        with torch.npu.stream(stream):
            meta.verify_commit_lens.copy_(torch.tensor(accepts, dtype=torch.int32))
            commit()
        stream.synchronize()
        torch.testing.assert_close(window.ring.cpu(), expected_ring, atol=0, rtol=0)

    check(starts, requests, 0, before, lambda: window.commit_verify(meta))
    with torch.npu.stream(stream):
        meta.verify_commit_lens.zero_()
    verify_graph, commit_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(verify_graph, stream=stream):
        invoke()
    with torch.npu.graph(commit_graph, stream=stream):
        window.commit_verify(meta)
    for generation in (1, 2, 3):
        before = window.ring.cpu()
        with torch.npu.stream(stream):
            starts, requests = change(generation)
            verify_graph.replay()
        stream.synchronize()
        check(starts, requests, generation, before, commit_graph.replay)


if __name__ == "__main__":
    torch.manual_seed(42)
    for batch in (1, 5):
        for ratio in (0, 1, 2):
            _case(batch, ratio, 32)
    print("PASS direct verify ring bitwise output, acceptance and Graph replay")
