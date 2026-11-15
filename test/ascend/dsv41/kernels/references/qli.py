"""Independent numerical references and fixtures for qli."""

from __future__ import annotations

import torch


def oracle(
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
    mode="off",
    candidates=None,
    *,
    row_indices=None,
):
    """Separate integer-dot/FP16-round/FP64-head-sum reference; no vendor code."""
    scores, valid_counts = [], []
    candidate_rows = []
    selected = None if row_indices is None else set(row_indices)
    for b, length in enumerate(lengths):
        # Every supported INT8 dot has at most 128 terms and magnitude < 2**22. FP64 GEMM
        # represents all products and partial sums exactly, while avoiding
        # the scalar INT64 matmul bottleneck for the 1M-position reference.
        logical_k = k[pages[b].long()].reshape(-1, q.shape[-1])[:length].double()
        logical_scale = ks[pages[b].long()].reshape(-1)[:length].double()
        for row in range(cu[b], cu[b + 1]):
            if selected is not None and row not in selected:
                continue
            n_q = cu[b + 1] - cu[b]
            valid = max(
                0,
                min(
                    length,
                    (length * ratio + residual[b] - n_q + row - cu[b] + 1) // ratio,
                ),
            )
            dot = q[row].double() @ logical_k.T
            relu = (dot.double() / 1024).clamp_min(0).half().double()
            head_weight = (w[row] * qs[row]).half().double()
            score = (
                (head_weight[:, None] * relu).sum(0).float().double() * logical_scale
            ).float()
            score[valid:] = -torch.inf
            candidate = torch.full((2048,), -1, dtype=torch.int32)
            if valid:
                padded = torch.nn.functional.pad(
                    score[:valid], (0, -valid % 8), value=-torch.inf
                )
                block_scores = padded.reshape(-1, 8).amax(1)
                block_scores[-1] = torch.inf
                order = block_scores.argsort(descending=True, stable=True)
                candidate[: min(2048, order.numel())] = order[:2048].int()
            candidate_rows.append(candidate)
            if mode == "consumer":
                allowed = candidates[row].long()
                allowed = allowed[(allowed >= 0) & (allowed < (length + 7) // 8)]
                blocks = torch.zeros((length + 7) // 8, dtype=torch.bool)
                blocks[allowed] = True
                score[:valid].masked_fill_(~blocks[torch.arange(valid) // 8], -1e30)
            scores.append(score)
            valid_counts.append(valid)
    return (
        scores,
        valid_counts,
        (
            torch.stack(candidate_rows)
            if candidate_rows
            else torch.empty((0, 2048), dtype=torch.int32)
        ),
    )


def compare(actual, score_rows, valid_counts):
    actual = actual.cpu()
    for row, (score, valid) in enumerate(zip(score_rows, valid_counts)):
        n = min(actual.shape[1], valid)
        selected = actual[row, :n].long()
        assert bool(((selected >= 0) & (selected < valid)).all()), (
            row,
            valid,
            selected,
        )
        assert selected.unique().numel() == n, ("duplicate output", row)
        assert bool((actual[row, n:] == -1).all()), (
            "padding",
            row,
            valid,
            actual[row, n:],
        )
        if n:
            expected = score[:valid].sort(descending=True, stable=True).values[:n]
            got = score[selected]
            # FP32 reduction order can exchange close ranks. Compare every
            # selected score/rank, rather than permit missing an arbitrary %.
            torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-7)
