"""QLI paged-input fixture with independent expected indices."""

from __future__ import annotations

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.qli import QLIPlan
from dsv41.kernels.references.qli import compare, oracle


def make_case(
    row_counts,
    lengths,
    *,
    ratio=1,
    residual=None,
    mode="off",
    seed=1,
    library_dir=None,
    padded=False,
    page_size=128,
    signed=False,
    reference_rows=None,
    heads=32,
    depth=128,
    topk=512,
    cores=None,
    candidate_blocks=23,
    plan_factory=None,
):
    torch.manual_seed(seed)
    batch, total = len(row_counts), sum(row_counts)
    residual = [0] * batch if residual is None else residual
    cu = [0]
    for count in row_counts:
        cu.append(cu[-1] + count)
    per_request = max(
        1, max(((x + page_size - 1) // page_size for x in lengths), default=0)
    )
    blocks = max(1, batch * per_request)
    pages = torch.randperm(blocks).reshape(batch, per_request).int()
    q = torch.randint(-128, 128, (total, heads, depth), dtype=torch.int8)
    k = torch.randint(-128, 128, (blocks, page_size, depth), dtype=torch.int8)
    w = (torch.rand((total, heads)) * 0.5 + (-0.25 if signed else 0.1)).half()
    qs = (torch.rand((total, heads)) * 0.02 + 0.001).half()
    ks = (torch.rand((blocks, page_size)) * 0.02 + 0.001).half()
    plan = (QLIPlan if plan_factory is None else plan_factory)(
        cu,
        lengths,
        pages,
        ratio=ratio,
        residual=residual,
        page_size=page_size,
        candidate_mode=mode,
        heads=heads,
        depth=depth,
        topk=topk,
        cores=cores,
        library_dir=library_dir,
    )
    device_inputs = [x.npu() for x in (q, k, w, qs, ks)]
    if padded:
        kb = torch.empty(
            (blocks, page_size + 16, depth), dtype=torch.int8, device="npu"
        )
        ksb = torch.empty((blocks, page_size + 16), dtype=torch.float16, device="npu")
        kb[:, :page_size].copy_(device_inputs[1])
        ksb[:, :page_size].copy_(device_inputs[4])
        device_inputs[1], device_inputs[4] = kb[:, :page_size], ksb[:, :page_size]
    candidate_input = None
    if mode == "consumer":
        candidate_input = torch.full((total, 2048), -1, dtype=torch.int32)
        # Nontrivial sparse, unsorted candidate sets; <512 reachable K exercises
        # upstream fill with reachable noncandidate positions.
        for row in range(total):
            chosen = torch.randperm(max(1, (max(lengths, default=0) + 7) // 8))[
                :candidate_blocks
            ]
            candidate_input[row, : chosen.numel()] = chosen.int()
        candidate_device = candidate_input.npu()
    else:
        candidate_device = None
    score, valid, candidate_expected = oracle(
        q,
        k,
        w,
        qs,
        ks,
        pages,
        cu,
        lengths,
        residual,
        ratio,
        mode,
        candidate_input,
        row_indices=reference_rows,
    )
    output_storage = torch.full(
        (total * topk + 128,), -1234567, dtype=torch.int32, device="npu"
    )
    output = output_storage[: total * topk].view(total, topk)
    ws_storage = torch.full(
        (plan.workspace_bytes + 512,), 173, dtype=torch.uint8, device="npu"
    )
    plan.workspace = ws_storage[: plan.workspace_bytes]
    result, candidate_result = plan.run(
        *device_inputs, candidates=candidate_device, out=output
    )
    torch.npu.synchronize()
    compare(result if reference_rows is None else result[reference_rows], score, valid)
    assert bool((output_storage[total * topk :].cpu() == -1234567).all()), (
        "output canary"
    )
    assert bool((ws_storage[plan.workspace_bytes :].cpu() == 173).all()), (
        "workspace canary"
    )
    if mode == "source":
        actual = candidate_result.cpu()
        for expected_row, valid_n in enumerate(valid):
            row = (
                expected_row if reference_rows is None else reference_rows[expected_row]
            )
            n = min(2048, (valid_n + 7) // 8)
            assert actual[row, :n].unique().numel() == n
            assert bool((actual[row, n:] == -1).all())
            # All blocks for short contexts; for long contexts compare selected
            # candidate-score boundary independently below via expected set.
            assert set(actual[row, :n].tolist()) == set(
                candidate_expected[expected_row, :n].tolist()
            ), ("candidate set", row)
            if n:
                assert int(actual[row, 0]) == (valid_n - 1) // 8, "latest block pin"
    return plan, device_inputs, candidate_device
