"""Independent numerical references and fixtures for mhc."""

from __future__ import annotations

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import Triplet


def stats_reference(residual, fn, scale, base, dtype=torch.float64):
    flat = residual.to(dtype).flatten(1)
    mixes = (flat @ fn.to(dtype).t()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-20
    )
    scale, base = scale.to(dtype), base.to(dtype)
    branches = residual.shape[1]
    pre = torch.sigmoid(mixes[:, :branches] * scale[0] + base[:branches]) + 1e-6
    post = 2 * torch.sigmoid(
        mixes[:, branches : 2 * branches] * scale[1] + base[branches : 2 * branches]
    )
    logits = (mixes[:, 2 * branches :] * scale[2] + base[2 * branches :]).reshape(
        -1, branches, branches
    )
    comb = torch.softmax(logits, -1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return Triplet(pre, post, comb)


def combine_reference(residual, pre, weight, dtype=torch.float64):
    mixed = (
        residual[:, 0]
        if pre is None
        else torch.einsum("ti,tih->th", pre.to(dtype), residual.to(dtype)).bfloat16()
    )
    mixed = mixed.to(dtype)
    return (
        mixed
        * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-20)
        * weight.to(dtype)
    ).bfloat16()


def post_reference(x, residual, post, comb, dtype=torch.float64):
    mixed = torch.einsum("tji,tjh->tih", comb.to(dtype), residual.to(dtype))
    return (mixed + post.to(dtype)[..., None] * x.to(dtype)[:, None]).bfloat16()


def check_combine(actual, residual, previous, weight):
    """Keep the FP64/FP32 einsum bounds and the declared serial FP32 order.

    A collapse BF16 midpoint can amplify one legitimate FP32 summation-order
    difference after normalization. The integration suite already uses these
    three independent oracles; keep the same element and aggregate tolerances.
    """
    r = residual.float()
    mixed = r[:, 0] if previous is None else r[:, 0] * previous[:, 0, None]
    if previous is not None:
        for stream in range(1, residual.shape[1]):
            mixed = mixed + r[:, stream] * previous[:, stream, None]
    rounded = mixed.bfloat16().float()
    serial = (
        rounded
        * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-20)
        * weight.float()
    ).bfloat16()
    result = check_bf16(
        actual, combine_reference(residual, previous, weight), fp32=serial
    )
    einsum32 = combine_reference(residual, previous, weight, torch.float32)
    error32 = (
        actual.cpu().double() - einsum32.double()
    ).norm() / einsum32.double().norm().clamp_min(1e-30)
    assert error32 < 5e-4, ("FP32 einsum collapse aggregate", error32.item())
    return result


def check_coeff(actual, reference):
    for result, expected in zip(actual, reference):
        torch.testing.assert_close(
            result.cpu().double(), expected.double(), rtol=3e-5, atol=3e-6
        )


def check_bf16(actual, reference, scale=1.0, fp32=None):
    actual, reference = actual.cpu().float(), reference.float()
    error = (actual - reference).abs()
    allowed = reference.abs() * 0.008 + 2e-6 * scale
    close = error <= allowed
    if fp32 is not None:
        # Different legitimate FP32 reduction orders can land on either side
        # of an intermediate BF16 midpoint. Keep the SAME element tolerance
        # for both independent oracles and require a small aggregate error to
        # EACH oracle; do not demand their rounded intermediates be identical.
        close |= (
            actual - fp32.float()
        ).abs() <= fp32.float().abs() * 0.008 + 2e-6 * scale
        relative32 = torch.linalg.vector_norm(
            (actual - fp32.float()).double()
        ) / torch.linalg.vector_norm(fp32.double()).clamp_min(1e-30)
        assert relative32 < 5e-4, relative32.item()
    if not torch.all(close):
        where = (~close).nonzero()[0].tolist()
        raise AssertionError(f"BF16 error at {where}, max={error.max().item()}")
    denominator = torch.linalg.vector_norm(reference.double()).clamp_min(1e-30)
    relative = (
        torch.linalg.vector_norm((actual - reference).double()) / denominator
    ).item()
    assert relative < 5e-4, relative
    return relative


def data(m, generator, amplitude=1.0):
    return (
        torch.randn((m, 4, 5120), generator=generator).bfloat16() * amplitude,
        torch.randn((24, 20480), generator=generator) * 0.007,
        torch.randn((3,), generator=generator),
        torch.randn((24,), generator=generator),
        torch.rand((m, 4), generator=generator),
        torch.randn((5120,), generator=generator),
        torch.randn((m, 5120), generator=generator).bfloat16(),
    )
