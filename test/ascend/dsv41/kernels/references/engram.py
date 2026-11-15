"""Independent numerical references and fixtures for engram."""

from __future__ import annotations

import torch


def reference(codes, scales, ids):
    valid = (ids >= 0) & (ids < codes.shape[0])
    selected = ids.clamp(0, codes.shape[0] - 1)
    # Independent native CPU FP8 decode and FP64 ldexp (exact powers of two),
    # then FP32 multiplication before the specified BF16 output rounding.
    x = codes[selected].contiguous().view(torch.float8_e4m3fn).float()
    exponent = scales[selected].to(torch.int32)
    scale = torch.ldexp(
        torch.ones_like(exponent, dtype=torch.float64), exponent - 127
    ).float()
    scale[exponent == 255] = float("nan")
    out = (x * scale.repeat_interleave(32, dim=-1)).bfloat16()
    out[~valid] = 0
    return out


def check(actual, expected):
    actual = actual.cpu()
    equal = actual.view(torch.int16) == expected.view(torch.int16)
    equal |= actual.isnan() & expected.isnan()
    if not equal.all():
        location = (~equal).nonzero()[0].tolist()
        raise AssertionError(
            f"BF16 mismatch at {location}; total={(~equal).sum().item()}"
        )
