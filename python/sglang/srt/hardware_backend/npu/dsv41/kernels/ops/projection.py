"""DeepJIT Ascend C BF16 projections and the shared expert MLP."""

from __future__ import annotations

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.config import V41
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


def repack_linear(weight):
    """Startup-only [N,K] BF16 checkpoint weight -> contiguous [1,K,N]."""
    if weight.ndim != 2 or str(weight.dtype) != "torch.bfloat16":
        raise ValueError("repack_linear expects [N,K] BF16 weight")
    return weight.T.contiguous().unsqueeze(0)


def repack_wo_a(weight, groups=8):
    """View checkpoint [G*R,K] as [G,R,K]; preserve storage without repack."""
    if str(weight.dtype) != "torch.bfloat16" or weight.ndim not in (2, 3):
        raise ValueError("repack_wo_a expects BF16 [G*R,K] or [G,R,K] weight")
    if not weight.is_contiguous():
        raise ValueError("wo_a checkpoint weight must be contiguous")
    if weight.ndim == 2:
        if groups <= 0 or weight.shape[0] % groups:
            raise ValueError("flattened wo_a rows must divide into complete groups")
        weight = weight.reshape(groups, weight.shape[0] // groups, weight.shape[1])
    return weight


class DenseProjection:
    def __init__(
        self,
        *,
        k=V41.hidden,
        columns=V41.head_dim,
        out_dtype="bf16",
    ):
        if (
            k <= 0
            or columns <= 0
            or k % 16
            or columns % 16
            or out_dtype not in {"bf16", "f32"}
        ):
            raise ValueError(
                "positive K/N multiples of 16 and BF16/FP32 output required"
            )
        self.kernel = load_library().projection
        self.k = k
        self.columns = columns
        self.dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float32

    def bind(self, x, weight, *, out):
        self._check(x, weight, out)
        self.prepare(x, weight, out=out)
        return lambda: self.kernel(x, weight, out, False)

    def prepare(self, x, weight, *, out):
        load_library().prepare_projection(x, weight, out, False)
        return self

    def _check(self, x, weight, out):
        if tuple(x.shape) != (x.shape[0], self.k) or tuple(weight.shape) != (
            1,
            self.k,
            self.columns,
        ):
            raise ValueError(
                "dense projection shape differs from the prepared geometry"
            )
        if out.dtype != self.dtype:
            raise ValueError("dense projection output dtype differs")

    def __call__(self, x, weight, *, out=None):
        if out is None:
            out = x.new_empty((x.shape[0], self.columns), dtype=self.dtype)
        self._check(x, weight, out)
        self.kernel(x, weight, out, False)
        return out


class WoAProjection:
    def __init__(
        self,
        *,
        k=V41.wo_k,
        groups=V41.wo_groups,
        rank=V41.wo_rank,
    ):
        if min(k, groups, rank) <= 0 or k % 16 or rank % 16:
            raise ValueError("positive groups and K/N multiples of 16 required")
        self.kernel = load_library().projection
        self.k = k
        self.groups, self.rank = groups, rank

    def bind(self, x, weight, *, out):
        self._check(x, weight)
        self.prepare(x, weight, out=out)
        return lambda: self.kernel(x, weight, out, True)

    def prepare(self, x, weight, *, out):
        load_library().prepare_projection(x, weight, out, True)
        return self

    def _check(self, x, weight):
        if tuple(x.shape) != (x.shape[0], self.groups, self.k) or tuple(
            weight.shape
        ) != (self.groups, self.rank, self.k):
            raise ValueError("wo_a projection shape differs from the prepared geometry")

    def __call__(self, x, weight, *, out=None):
        if out is None:
            out = x.new_empty((x.shape[0], self.groups, self.rank))
        self._check(x, weight)
        self.kernel(x, weight, out, True)
        return out
