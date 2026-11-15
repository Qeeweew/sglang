"""Independent numerical references and fixtures for position_quant."""

from __future__ import annotations

import torch


def rope_reference(x, c, s, inverse=False):
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    c = c.reshape(shape)
    s = s.reshape(shape) * (-1 if inverse else 1)
    real = x[..., -64::2].float()
    imag = x[..., -63::2].float()
    out = x.clone()
    out[..., -64::2] = (real * c - imag * s).bfloat16()
    out[..., -63::2] = (imag * c + real * s).bfloat16()
    return out


def q_reference(x):
    scale = (x.float().abs().amax(-1) / 127).half().float().clamp_min(2**-24)
    codes = torch.round(x.float() / scale[..., None]).clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


def k_reference(x):
    maximum = x.float().abs().amax(-1)
    # Ascend vector Div is reciprocal then multiplication. A scalar correctly
    # rounded127/amax can land on the other side of a half-integer boundary.
    multiplier = torch.where(maximum == 0, 0, maximum.reciprocal() * 127)
    normalized = x.float() * multiplier[..., None]
    tiny = (maximum != 0) & (maximum < 2**-120)
    if bool(tiny.any()):
        normalized[tiny] = (x.float()[tiny] / maximum[tiny, None]) * 127
    codes = torch.round(normalized).clamp(-128, 127).to(torch.int8)
    return codes, maximum * (torch.tensor(1 / 127, dtype=torch.float32))


def checked_k(actual, expected, x):
    # The native vector divider has architecture-dependent last-bit error.
    # Require exact native equality separately, and permit the independent
    # mathematical oracle to differ only by one at a half-integer decision.
    actual = actual.cpu()
    mask = actual != expected
    if bool(mask.any()):
        maximum = x.float().abs().amax(-1).double()
        ideal = torch.where(
            maximum[..., None] == 0, 0, x.double() / maximum[..., None] * 127
        )
        half_distance = (ideal - ideal.floor() - 0.5).abs()
        assert bool(((actual.int() - expected.int()).abs()[mask] == 1).all())
        assert bool((half_distance[mask] < 2e-5).all()), (
            "nonboundary K mismatch",
            ideal[mask],
        )


def scatter_reference(cache, values, slots):
    result = cache.clone().view(-1, cache.shape[-1])
    for row, slot in enumerate(slots.tolist()):
        if 0 <= slot < result.shape[0]:
            result[slot] = values[row]
    return result.reshape(cache.shape)
