"""Global sqrtsoftplus TopK and one stable global/EP bucketing operation."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
    require_disjoint,
)


class RoutingPlan(NamedTuple):
    ids: torch.Tensor
    weights: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    order: torch.Tensor
    metadata: torch.Tensor


class RoutingWorkspace(NamedTuple):
    ids: torch.Tensor
    histogram: torch.Tensor


class Router(NativeOperator):
    def __init__(self, *, device="npu:0", library_dir=None):
        super().__init__(device)
        self._library = load_library(native_library_path(directory=library_dir))

    @staticmethod
    def _topk(logits, topk):
        result = topk or (
            6 if logits.shape[1] == 384 else 3 if logits.shape[1] == 128 else 0
        )
        if result <= 0:
            raise ValueError("specify topk for this expert count")
        return result

    def workspace(self, logits, *, topk=None, experts=None):
        local = logits.shape[1] if experts is None else experts
        k = self._topk(logits, topk)
        ids = logits.new_empty((logits.shape[0], k), dtype=torch.int32)
        return RoutingWorkspace(ids, self.bucket_workspace(ids, local))

    def bucket_workspace(self, ids, experts):
        return ids.new_empty((min(self.cores, experts), (experts + 15) // 16 * 16))

    @staticmethod
    def bucket_output(ids, weights, experts):
        return RoutingPlan(
            torch.empty_like(ids),
            weights,
            ids.new_empty(experts),
            ids.new_empty(experts + 1, dtype=torch.int32),
            ids.new_empty(ids.numel(), dtype=torch.int32),
            ids.new_empty(4 * (experts + 1), dtype=torch.int32),
        )

    def bucket(
        self, ids, weights, first, experts, *, out=None, workspace=None, _prepare=False
    ):
        """Plan directly from global IDs, retaining global normalized weights."""
        if out is None:
            out = self.bucket_output(ids, weights, experts)
        if (
            out.weights.data_ptr() != weights.data_ptr()
            or out.counts.numel() != experts
        ):
            raise ValueError(
                "bucket must retain global weights and local expert geometry"
            )
        if workspace is None:
            workspace = self.bucket_workspace(ids, experts)
        require_disjoint(
            (out.ids, out.counts, out.offsets, out.order, out.metadata, workspace),
            (ids, weights),
        )
        operation = (
            self._library.prepare_route_bucket
            if _prepare
            else self._library.route_bucket
        )
        operation(
            ids,
            out.ids,
            out.counts,
            out.offsets,
            out.order,
            out.metadata,
            workspace,
            first,
        )
        return out

    def bind_bucket(self, ids, weights, first, experts, *, out=None, workspace=None):
        if out is None:
            out = self.bucket_output(ids, weights, experts)
        if workspace is None:
            workspace = self.bucket_workspace(ids, experts)
        self.bucket(
            ids, weights, first, experts, out=out, workspace=workspace, _prepare=True
        )
        return partial(
            self.bucket, ids, weights, first, experts, out=out, workspace=workspace
        )

    def bind(self, logits, correction_bias=None, *, plan=False, **kwargs):
        """Prepare stable outputs/workspace before capture; no hot allocation."""
        k = self._topk(logits, kwargs.get("topk"))
        local = kwargs.get("experts")
        if local is None:
            local = logits.shape[1]
        kwargs.setdefault(
            "ids_out", logits.new_empty((logits.shape[0], k), dtype=torch.int32)
        )
        kwargs.setdefault("weights_out", logits.new_empty((logits.shape[0], k)))
        if plan:
            kwargs.setdefault(
                "workspace", self.workspace(logits, topk=k, experts=local)
            )
            kwargs.setdefault("counts_out", logits.new_empty(local, dtype=torch.int32))
            kwargs.setdefault(
                "offsets_out", logits.new_empty(local + 1, dtype=torch.int32)
            )
            kwargs.setdefault(
                "order_out", logits.new_empty(logits.shape[0] * k, dtype=torch.int32)
            )
            kwargs.setdefault(
                "metadata_out", logits.new_empty(4 * (local + 1), dtype=torch.int32)
            )
        self(logits, correction_bias, plan=plan, _prepare=True, **kwargs)
        return partial(self, logits, correction_bias, plan=plan, **kwargs)

    def __call__(
        self,
        logits,
        correction_bias=None,
        *,
        factor=1.5,
        valid_rows=None,
        dp_counts=None,
        dp_stride=None,
        alternate_bias=None,
        image_mask=None,
        ids_out=None,
        weights_out=None,
        plan=False,
        workspace=None,
        counts_out=None,
        offsets_out=None,
        order_out=None,
        metadata_out=None,
        topk=None,
        first=0,
        experts=None,
        _prepare=False,
    ):
        k = self._topk(logits, topk)
        local = logits.shape[1] if experts is None else experts
        if first < 0 or local <= 0 or first + local > logits.shape[1]:
            raise ValueError("invalid expert ownership range")
        if plan and workspace is None:
            workspace = self.workspace(logits, topk=k, experts=local)
        topk_ids = workspace.ids if plan else ids_out
        operation = self._library.prepare_router if _prepare else self._library.router
        ids, weights = operation(
            logits,
            correction_bias,
            factor,
            valid_rows,
            dp_counts,
            dp_stride or 0,
            alternate_bias,
            image_mask,
            topk_ids,
            weights_out,
            k,
        )
        if not plan:
            return ids, weights

        def allocate(value, shape, dtype):
            return ids.new_empty(shape, dtype=dtype) if value is None else value

        out = RoutingPlan(
            allocate(ids_out, ids.shape, torch.int32),
            weights,
            allocate(counts_out, (local,), torch.int32),
            allocate(offsets_out, (local + 1,), torch.int32),
            allocate(order_out, (ids.numel(),), torch.int32),
            allocate(metadata_out, (4 * (local + 1),), torch.int32),
        )
        require_disjoint(
            (*out, workspace.ids, workspace.histogram),
            (
                logits,
                correction_bias,
                valid_rows,
                dp_counts,
                alternate_bias,
                image_mask,
            ),
        )
        return self.bucket(
            ids,
            weights,
            first,
            local,
            out=out,
            workspace=workspace.histogram,
            _prepare=_prepare,
        )

    def route(self, logits, correction_bias=None, **kwargs):
        """TopK + one bucketing kernel, returning the same global/EP plan ABI."""
        return self(logits, correction_bias, plan=True, **kwargs)
