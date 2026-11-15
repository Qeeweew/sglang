"""Independent numerical references and fixtures for dspark_weights."""

from __future__ import annotations

import torch

M, V, R, H = 6, 129280, 256, 5120


def weights():
    gen = torch.Generator().manual_seed(71341)
    return {
        "embedding": (torch.randn(V, R, generator=gen) * 0.1).bfloat16(),
        "w2": torch.randn(V, R, generator=gen) * 0.02,
        "confidence_weight": torch.randn(1, H + R, generator=gen) * 0.01,
        "confidence_bias": torch.tensor([0.17]),
        "sts": torch.tensor(1.7),
    }
