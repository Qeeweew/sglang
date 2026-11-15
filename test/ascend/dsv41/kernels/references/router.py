"""Independent numerical references and fixtures for router."""

from __future__ import annotations

import torch


def reference(logits, bias, factor=1.5, live=None):
    # Independent high-precision stages rounded back to FP32. Torch CPU's
    # float32 log1p adds one subnormal unit for exp(-100), so it is not used
    # as the mathematical oracle at this boundary.
    exponential = torch.exp(logits.double()).float()
    softplus = torch.log1p(exponential.double()).float()
    softplus = torch.where(logits > 20, logits, softplus)
    score = torch.sqrt(softplus.double()).float()
    rank = score if bias is None else score + bias
    rank = torch.where(rank.isnan(), torch.inf, rank)
    k = 6 if logits.shape[1] == 384 else 3
    ids = rank.argsort(dim=-1, descending=True, stable=True)[:, :k]
    weights = score.gather(1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * factor
    if live is not None:
        ids[live:] = -1
        weights[live:] = 0
    return ids.int(), weights


def compare(ids, weights, expected):
    assert torch.equal(ids.cpu(), expected[0]), (
        "router indices",
        ids.cpu(),
        expected[0],
    )
    torch.testing.assert_close(
        weights.cpu(), expected[1], rtol=2e-6, atol=2e-7, equal_nan=True
    )
