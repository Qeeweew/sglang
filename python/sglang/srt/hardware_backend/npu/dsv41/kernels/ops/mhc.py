"""DeepJIT Ascend C mHC specialized to actual geometry with stable stream rings.

``previous_pre`` belongs to the preceding connection. Projection/statistics
remain IEEE FP32; state handoffs explicitly round to BF16.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import NamedTuple

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.config import V41
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    check_tensor,
    load_library,
    require_disjoint,
)

# arch2201 DMA block/Vector repeat widths, matching native/params.h.
_FP32_BYTES = torch.finfo(torch.float32).bits // 8
_VECTOR_ELEMENTS = 256 // _FP32_BYTES
_BLOCK_ELEMENTS = 32 // _FP32_BYTES
_WIDTH_ALIGNMENT = 2 * _VECTOR_ELEMENTS
_WIDTH_CAPACITY = 2 * _VECTOR_ELEMENTS**2


class Triplet(NamedTuple):
    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class Workspace(NamedTuple):
    projection: torch.Tensor
    triplet: Triplet
    square_sum: torch.Tensor


class PreResult(NamedTuple):
    input: torch.Tensor
    triplet: Triplet


def _residual(value):
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bfloat16
        or value.ndim != 3
        or value.device.type != "npu"
        or not value.is_contiguous()
    ):
        raise ValueError(
            "residual must be contiguous BF16 [rows,branches,width] on an NPU"
        )
    m, b, h = value.shape
    if (
        not 1 <= b <= _BLOCK_ELEMENTS
        or h <= 0
        or h % _WIDTH_ALIGNMENT
        or h > _WIDTH_CAPACITY
    ):
        raise ValueError(
            "mHC branches must fit one FP32 block; width must be128-aligned and <=8192"
        )
    return m, value.device


@lru_cache(maxsize=128)
def _geometry(m, b, h, device):
    return tuple(load_library().mhc_geometry(m, b, h, device))


class MHC:
    SLOTS = 2

    def __init__(self, *, rms_eps=1e-20):
        self.rms_eps = float(rms_eps)
        if not math.isfinite(self.rms_eps) or self.rms_eps <= 0:
            raise ValueError("mHC RMS epsilon must be positive and finite")
        self.native = load_library()
        self.project_kernel = self._project_mixed
        self.compact_project_kernel = self._project_mixed
        self.vector_project_kernel = self._project_vector
        self.combine_kernel = self._combine_call
        self.post_kernel = self._post_call
        self.post_combine_kernel = self._post_combine_call
        self.collapse_kernel = self._collapse_call
        self.mean_kernel = self._mean_call
        self.stats_kernel = self._stats_call
        self.batched_stats_kernel = self._stats_call
        self.stats_post_kernel = self._stats_post_call
        self._rings = {}
        self._prepared = set()

    def _call(self, operation, *args):
        geometry = tuple(
            (value.device, value.dtype, tuple(value.shape), tuple(value.stride()))
            if isinstance(value, torch.Tensor)
            else value
            for value in args
        )
        # epsilon is numerical policy and mean column a per-call selector;
        # neither changes the device code or its physical output stride.
        if operation == "mhc_stats":
            geometry = geometry[:-1]
        elif operation == "mhc_stats_post":
            geometry = geometry[:7] + geometry[8:]
        elif operation == "mhc_mean":
            geometry = geometry[:-1]
        key = (operation, geometry)
        if key not in self._prepared:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(
                    "warm the complete mHC call with actual tensors before capture"
                )
            getattr(self.native, "prepare_" + operation)(*args)
            self._prepared.add(key)
        getattr(self.native, operation)(*args)

    @staticmethod
    def geometry(residual):
        m, device = _residual(residual)
        return _geometry(m, residual.shape[1], residual.shape[2], device.index)

    @staticmethod
    def project_geometry(m, branches=V41.mhc_streams, width=V41.hidden, device=None):
        """Compatibility view: row tile, padded rows, runtime K partitions."""
        index = (
            torch.npu.current_device() if device is None else torch.device(device).index
        )
        return _geometry(m, branches, width, index)[:3]

    @staticmethod
    def k_splits(m):
        return MHC.project_geometry(m)[2]

    @staticmethod
    def workspace(residual):
        m, device = _residual(residual)
        b = residual.shape[1]
        _, rows, splits, _, stride, *_ = MHC.geometry(residual)
        return Workspace(
            torch.empty((splits, rows, stride), dtype=torch.float32, device=device),
            Triplet(
                torch.empty((m, b), dtype=torch.float32, device=device),
                torch.empty((m, b), dtype=torch.float32, device=device),
                torch.empty((m, b, b), dtype=torch.float32, device=device),
            ),
            torch.empty((splits, rows), dtype=torch.float32, device=device),
        )

    def _project_mixed(self, *, Residual, Fn, Projection, SquareSum, Inputs, Products):
        bm = self.geometry(Residual)[0]
        self._call(
            "mhc_project",
            Residual,
            Fn,
            Projection,
            SquareSum,
            Inputs,
            Products,
            bm,
            False,
        )

    def _project_vector(self, *, Residual, Fn, Projection, SquareSum):
        if Residual.ndim == 2:
            branches = math.isqrt(Fn.shape[0] + 1) - 1
            Residual = Residual.view(Residual.shape[0], branches, -1)
        bm = self.geometry(Residual)[0]
        self._call(
            "mhc_project", Residual, Fn, Projection, SquareSum, None, None, bm, True
        )

    def _stats_call(
        self,
        *,
        Projection,
        SquareSum,
        Scale,
        Base,
        Pre,
        Post,
        Comb,
        HiddenWidth=V41.hidden,
    ):
        self._call(
            "mhc_stats",
            Projection,
            SquareSum,
            Scale,
            Base,
            Pre,
            Post,
            Comb,
            HiddenWidth,
            self.rms_eps,
        )

    def _combine_call(self, *, Residual, PreviousPre, NormWeight, Out, Initial):
        self._call("mhc_combine", Residual, PreviousPre, NormWeight, Out, bool(Initial))

    def _collapse_call(self, *, Residual, PreviousPre, Out):
        self._call("mhc_combine", Residual, PreviousPre, None, Out, False)

    def _post_call(self, *, X, Residual, Post, Comb, Out):
        self._call("mhc_post", X, Residual, Post, Comb, Out)

    def _mean_call(self, *, Residual, Packed, Column):
        self._call("mhc_mean", Residual, Packed, Column)

    def _stats_post_call(
        self, *, Projection, SquareSum, Scale, Base, X, Residual, Pre, Post, Comb, Out
    ):
        if X.shape[0] <= _BLOCK_ELEMENTS:
            self._call(
                "mhc_stats_post",
                Projection,
                SquareSum,
                Scale,
                Base,
                Pre,
                Post,
                Comb,
                self.rms_eps,
                X,
                Residual,
                Out,
            )
        else:
            self._stats_call(
                Projection=Projection,
                SquareSum=SquareSum,
                Scale=Scale,
                Base=Base,
                Pre=Pre,
                Post=Post,
                Comb=Comb,
                HiddenWidth=Residual.shape[2],
            )
            self._call("mhc_post", X, Residual, Post, Comb, Out)

    def _post_combine_call(
        self, *, X, Residual, Pre, Post, Comb, NormWeight, Updated, Out
    ):
        b = Residual.shape[1]
        self._call(
            "mhc_post_combine",
            X,
            Residual,
            Pre,
            Post,
            Comb.view(-1, b, b),
            NormWeight,
            Updated,
            Out,
        )

    def _stats_workspace(self, residual, fn, scale, base, workspace):
        m, device = _residual(residual)
        b, h = residual.shape[1:]
        check_tensor(fn, torch.float32, (b * (b + 2), b * h), device, "fn")
        check_tensor(scale, torch.float32, (3,), device, "scale")
        check_tensor(base, torch.float32, (b * (b + 2),), device, "base")
        workspace = self.workspace(residual) if workspace is None else workspace
        projection, triplet, squares = workspace
        _, rows, splits, _, stride, *_ = self.geometry(residual)
        check_tensor(
            projection, torch.float32, (splits, rows, stride), device, "projection"
        )
        check_tensor(squares, torch.float32, (splits, rows), device, "square_sum")
        for value, shape, name in (
            (triplet.pre, (m, b), "pre"),
            (triplet.post, (m, b), "post"),
            (triplet.comb, (m, b, b), "comb"),
        ):
            check_tensor(value, torch.float32, shape, device, name)
        require_disjoint((projection, squares, *triplet), (residual, fn, scale, base))
        return workspace

    def _ring(self, residual):
        bm, _, _, panel, stride, groups, *_ = self.geometry(residual)
        stream = torch.npu.current_stream(residual.device)
        key = (residual.device, int(stream.npu_stream), bm, panel, stride, groups)
        if key not in self._rings:
            shape = (groups, self.SLOTS, bm)
            self._rings[key] = (
                torch.empty(
                    (*shape, panel), dtype=torch.float32, device=residual.device
                ),
                torch.empty(
                    (*shape, stride), dtype=torch.float32, device=residual.device
                ),
            )
        return self._rings[key]

    def prepare(self, residual):
        """Allocate this stream's rings; warm actual calls before Graph capture."""
        _residual(residual)
        with torch.npu.device(residual.device):
            if residual.shape[0] > _BLOCK_ELEMENTS:
                self._ring(residual)

    def _project(self, residual, fn, projection, squares):
        if not residual.shape[0]:
            return
        if residual.shape[0] <= _BLOCK_ELEMENTS:
            self.vector_project_kernel(
                Residual=residual, Fn=fn, Projection=projection, SquareSum=squares
            )
        else:
            inputs, products = self._ring(residual)
            self.project_kernel(
                Residual=residual,
                Fn=fn,
                Projection=projection,
                SquareSum=squares,
                Inputs=inputs,
                Products=products,
            )

    def stats(self, residual, fn, scale, base, *, workspace=None):
        projection, triplet, squares = self._stats_workspace(
            residual, fn, scale, base, workspace
        )
        if residual.shape[0]:
            self._project(residual, fn, projection, squares)
            self.stats_kernel(
                Projection=projection,
                SquareSum=squares,
                Scale=scale,
                Base=base,
                Pre=triplet.pre,
                Post=triplet.post,
                Comb=triplet.comb,
                HiddenWidth=residual.shape[2],
            )
        return triplet

    def stats_post(self, x, residual, fn, scale, base, *, workspace=None, out=None):
        projection, triplet, squares = self._stats_workspace(
            residual, fn, scale, base, workspace
        )
        m, device = _residual(residual)
        b, h = residual.shape[1:]
        check_tensor(x, torch.bfloat16, (m, h), device, "x")
        out = torch.empty_like(residual) if out is None else out
        check_tensor(out, torch.bfloat16, (m, b, h), device, "out")
        require_disjoint(
            (projection, squares, *triplet, out), (x, residual, fn, scale, base)
        )
        if m:
            self._project(residual, fn, projection, squares)
            self.stats_post_kernel(
                Projection=projection,
                SquareSum=squares,
                Scale=scale,
                Base=base,
                X=x,
                Residual=residual,
                Pre=triplet.pre,
                Post=triplet.post,
                Comb=triplet.comb,
                Out=out,
            )
        return out, triplet

    def combine(self, residual, previous_pre, norm_weight, *, out=None):
        m, device = _residual(residual)
        b, h = residual.shape[1:]
        check_tensor(norm_weight, torch.float32, (h,), device, "norm_weight")
        initial = previous_pre is None
        if initial:
            previous_pre = torch.empty((m, b), dtype=torch.float32, device=device)
        else:
            check_tensor(previous_pre, torch.float32, (m, b), device, "previous_pre")
        out = (
            torch.empty((m, h), dtype=torch.bfloat16, device=device)
            if out is None
            else out
        )
        check_tensor(out, torch.bfloat16, (m, h), device, "out")
        require_disjoint((out,), (residual, previous_pre, norm_weight))
        if m:
            self.combine_kernel(
                Residual=residual,
                PreviousPre=previous_pre,
                NormWeight=norm_weight,
                Out=out,
                Initial=initial,
            )
        return out

    def pre(
        self,
        residual,
        fn,
        scale,
        base,
        norm_weight,
        previous_pre=None,
        *,
        workspace=None,
        out=None,
    ):
        return PreResult(
            self.combine(residual, previous_pre, norm_weight, out=out),
            self.stats(residual, fn, scale, base, workspace=workspace),
        )

    def post(self, x, residual, post, comb, *, out=None):
        m, device = _residual(residual)
        b, h = residual.shape[1:]
        for value, dtype, shape, name in (
            (x, torch.bfloat16, (m, h), "x"),
            (post, torch.float32, (m, b), "post"),
            (comb, torch.float32, (m, b, b), "comb"),
        ):
            check_tensor(value, dtype, shape, device, name)
        out = torch.empty_like(residual) if out is None else out
        check_tensor(out, torch.bfloat16, (m, b, h), device, "out")
        require_disjoint((out,), (x, residual, post, comb))
        if m:
            self.post_kernel(X=x, Residual=residual, Post=post, Comb=comb, Out=out)
        return out

    def head(self, residual, last_pre, norm_weight, *, out=None):
        if last_pre is None:
            raise ValueError("head requires the last connection's pre")
        return self.combine(residual, last_pre, norm_weight, out=out)

    def post_combine(
        self, x, residual, pre, post, comb, norm_weight, *, updated=None, out=None
    ):
        m, device = _residual(residual)
        b, h = residual.shape[1:]
        for value, dtype, shape, name in (
            (x, torch.bfloat16, (m, h), "x"),
            (pre, torch.float32, (m, b), "pre"),
            (post, torch.float32, (m, b), "post"),
            (comb, torch.float32, (m, b, b), "comb"),
            (norm_weight, torch.float32, (h,), "norm_weight"),
        ):
            check_tensor(value, dtype, shape, device, name)
        updated = torch.empty_like(residual) if updated is None else updated
        out = x.new_empty((m, h)) if out is None else out
        check_tensor(updated, torch.bfloat16, (m, b, h), device, "updated")
        check_tensor(out, torch.bfloat16, (m, h), device, "out")
        require_disjoint((updated, out), (x, residual, pre, post, comb, norm_weight))
        if m:
            self.post_combine_kernel(
                X=x,
                Residual=residual,
                Pre=pre,
                Post=post,
                Comb=comb,
                NormWeight=norm_weight,
                Updated=updated,
                Out=out,
            )
        return updated, out

    def mean_packed(self, residual, packed, column):
        m, device = _residual(residual)
        h = residual.shape[2]
        if (
            not isinstance(column, int)
            or packed.ndim != 2
            or packed.shape[1] % h
            or not 0 <= column < packed.shape[1] // h
        ):
            raise ValueError(
                "capture column must index a complete hidden-width packed segment"
            )
        check_tensor(packed, torch.bfloat16, (m, packed.shape[1]), device, "packed")
        require_disjoint((packed,), (residual,))
        if m:
            self.mean_kernel(Residual=residual, Packed=packed, Column=column)
        return packed[:, column * h : (column + 1) * h]

    def collapse(self, residual, last_pre, *, out=None):
        if residual.ndim == 2 and last_pre.ndim == 2 and last_pre.shape[1] > 0:
            branches = last_pre.shape[1]
            if residual.shape[1] % branches == 0:
                residual = residual.view(
                    residual.shape[0], branches, residual.shape[1] // branches
                )
        m, device = _residual(residual)
        b, h = residual.shape[1:]
        check_tensor(last_pre, torch.float32, (m, b), device, "last_pre")
        out = (
            torch.empty((m, h), dtype=torch.bfloat16, device=device)
            if out is None
            else out
        )
        check_tensor(out, torch.bfloat16, (m, h), device, "out")
        require_disjoint((out,), (residual, last_pre))
        if m:
            self.collapse_kernel(Residual=residual, PreviousPre=last_pre, Out=out)
        return out
