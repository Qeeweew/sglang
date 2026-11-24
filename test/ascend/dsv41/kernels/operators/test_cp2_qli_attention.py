"""Interleaved query positions preserve QLI ranking and causal attention."""

import os
from itertools import accumulate

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B"
)


def _bind(counts, lengths, pages, qli_inputs, q, window, kv, ratio, *, positions=None):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
        SparseAttention,
        WindowAttention,
    )
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.qli import QLIPlan

    rows, heads, _ = q.shape
    cu_cpu = [0, *accumulate(counts)]
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=q.device)
    device_pages = pages.to(q.device)
    device_lengths, compressed_lengths, residual = (
        torch.empty_like(lengths, device=q.device) for _ in range(3)
    )
    if positions is not None:
        positions = positions.contiguous().to(q.device)

    def update(new_lengths, new_positions=None):
        device_lengths.copy_(new_lengths)
        compressed_lengths.copy_(new_lengths // ratio)
        residual.copy_(new_lengths % ratio)
        if new_positions is not None:
            positions.copy_(new_positions)

    update(lengths)
    plans = {}
    outputs = {}
    # An explicitly masked dummy request may have a row and no keys. Bind
    # live device metadata after initializing the plan's host geometry.
    initial_lengths = torch.maximum(lengths, torch.tensor(counts, dtype=lengths.dtype))
    for mode in ("source", "consumer"):
        plan = QLIPlan(
            cu_cpu,
            initial_lengths // ratio,
            pages.clamp_min(0),
            ratio=ratio,
            residual=initial_lengths % ratio,
            page_size=128 // ratio,
            candidate_mode=mode,
            device=q.device,
        )
        plan.bind_metadata(
            cu,
            compressed_lengths,
            device_pages,
            residual=residual,
            page_count=qli_inputs[1].size(0),
        )
        candidates = plans["source"].candidate_out if mode == "consumer" else None
        indices = torch.empty((rows, 1, 512), dtype=torch.int32, device=q.device)
        plan.bind(
            *qli_inputs, candidates=candidates, positions=positions, out=indices[:, 0]
        )
        if positions is None:
            staged, _ = plan.run(*qli_inputs, candidates=candidates)
            indices.fill_(12345)
            plan.run(*qli_inputs, candidates=candidates, out=indices[:, 0])
            assert torch.equal(indices[:, 0], staged), mode
        plans[mode], outputs[mode] = plan, indices

    swa_kv, compressed_kv, sinks, selected = kv
    swa_out, sparse_out = torch.empty_like(q), torch.empty_like(q)
    swa = WindowAttention(heads=heads).bind(
        q,
        swa_kv,
        device_pages,
        cu,
        device_lengths,
        sinks,
        window,
        out=swa_out,
        positions=positions,
    )
    # Explicit positions disable reuse even if requested. Use index IDs that
    # can include future tokens to exercise attention's own causal cutoff.
    sparse = SparseAttention(
        ratio=ratio, heads=heads, window_reuse=positions is not None
    ).bind(
        q,
        swa_kv,
        compressed_kv,
        selected,
        device_pages,
        device_pages,
        cu,
        device_lengths,
        compressed_lengths,
        residual,
        window,
        sinks,
        out=sparse_out,
        positions=positions,
    )

    def run():
        for mode, plan in plans.items():
            candidates = plans["source"].candidate_out if mode == "consumer" else None
            plan.run(
                *qli_inputs,
                candidates=candidates,
                positions=positions,
                out=outputs[mode][:, 0],
            )
        swa()
        sparse()

    return (
        run,
        update,
        (*outputs.values(), plans["source"].candidate_out, swa_out, sparse_out),
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dummy_request", [False, True])
def test_interleaved_causality_and_graph(ratio, dummy_request):
    import torch_npu  # noqa: F401

    device = torch.device("npu", torch.npu.current_device())
    torch.manual_seed(4102)
    lengths = torch.tensor([257, 512, 1039], dtype=torch.int32)
    counts = [129, 256, 302]
    cu = [0, *accumulate(counts)]
    rows = cu[-1]
    columns = 9
    physical_pages = len(counts) * columns
    pages = torch.randperm(physical_pages, dtype=torch.int32).view(-1, columns)
    qli_inputs = (
        torch.randint(-128, 128, (rows + 1, 32, 128), dtype=torch.int8).to(device),
        torch.randint(
            -128, 128, (physical_pages, 128 // ratio, 128), dtype=torch.int8
        ).to(device),
        (torch.rand(rows + 1, 32) * 0.5 + 0.1).half().to(device),
        (torch.rand(rows + 1, 32) * 0.02 + 0.001).half().to(device),
        (torch.rand(physical_pages, 128 // ratio) * 0.02 + 0.001).half().to(device),
    )
    width = 512 if dummy_request else 128
    q = (torch.randn(rows + 1, 64, width) * 0.2).bfloat16().to(device)
    window = torch.randint(1, 129, (rows + 1, 1), dtype=torch.int32).to(device)
    swa = (torch.randn(physical_pages, 128, 1, width) * 0.2).bfloat16().to(device)
    compressed = (
        (torch.randn(physical_pages, 128 // ratio, 1, width) * 0.2)
        .bfloat16()
        .to(device)
    )
    sinks = torch.randn(64).to(device)
    selected = torch.arange(511, -1, -1, dtype=torch.int32).view(1, 1, -1)
    selected = selected.repeat(rows + 1, 1, 1).to(device)

    def positions_for(seq_lens):
        return torch.cat(
            [
                *(
                    torch.arange(int(n) - c, int(n), dtype=torch.int32)
                    for n, c in zip(seq_lens, counts)
                ),
                torch.tensor([-1], dtype=torch.int32),
            ]
        )

    def bind(part, local_counts, local_lengths, local_pages, positions=None):
        query, keys, weights, qscale, kscale = qli_inputs
        return _bind(
            local_counts,
            local_lengths,
            local_pages,
            (
                query[part].contiguous(),
                keys,
                weights[part].contiguous(),
                qscale[part].contiguous(),
                kscale,
            ),
            q[part].contiguous(),
            window[part].contiguous(),
            (swa, compressed, sinks, selected[part].contiguous()),
            ratio,
            positions=positions,
        )

    ref, update_ref, expected = bind(slice(0, rows), counts, lengths, pages)
    lane_cases = []
    for lane in range(2):
        part = slice(lane, rows + 1, 2)
        local_counts = [
            len(range(start + (lane - start) % 2, end, 2))
            for start, end in zip(cu, cu[1:])
        ]
        local_lengths, local_pages = lengths, pages
        if lane == 1:
            if dummy_request:
                local_counts.append(1)
                local_lengths = torch.cat((lengths, lengths.new_zeros(1)))
                local_pages = torch.cat((pages, pages.new_full((1, columns), -1)))
            else:
                local_counts[-1] += 1
        run, update, actual = bind(
            part, local_counts, local_lengths, local_pages, positions_for(lengths)[part]
        )
        lane_cases.append((run, update, actual, local_lengths))

    def check():
        for lane, (_, _, actual, _) in enumerate(lane_cases):
            live_rows = len(range(lane, rows, 2))
            for i, (result, reference) in enumerate(zip(actual, expected)):
                result, reference = result.cpu(), reference.cpu()
                assert torch.equal(result[:live_rows], reference[lane::2]), (lane, i)
                pad = -1 if i < 3 else 0
                assert (result[live_rows:] == pad).all(), (lane, i)

    ref()
    for run, _, _, _ in lane_cases:
        run()
    torch.npu.synchronize()
    check()

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for run, _, _, _ in lane_cases:
            run()
    for shift in (7, 0):
        with torch.npu.stream(stream):
            update_ref(lengths - shift)
            ref()
            for lane, (_, update, _, local_lengths) in enumerate(lane_cases):
                update(
                    (local_lengths - shift).clamp_min(0),
                    positions_for(lengths - shift)[lane::2],
                )
            graph.replay()
        stream.synchronize()
        check()
