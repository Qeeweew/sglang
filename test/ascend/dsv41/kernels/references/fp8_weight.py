"""Independent numerical references and fixtures for fp8_weight."""

from __future__ import annotations

import torch


def reference(codes, scales):
    rows, cols = codes.shape
    values = codes.view(torch.float8_e4m3fn).float()
    exponent = scales.int()
    scale = torch.ldexp(
        torch.ones_like(exponent, dtype=torch.float64), exponent - 127
    ).float()
    scale[exponent == 255] = float("nan")
    return (
        values * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :cols]
    ).bfloat16()
