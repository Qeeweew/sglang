"""BF16 model boundary around FP16 expert-major activations, without FP32 GM."""

from __future__ import annotations

import torch
from sglang.srt.hardware_backend.npu.dsv41 import parallel as _dsv41_dp_buffers

from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
)


class MoERoutes(NativeOperator):
    def __init__(self, *, device="npu:0"):
        super().__init__(device)
        self.library = load_library()

    def bind_gather(self, x, routing, *, out, inverse):
        from functools import partial

        self.library.prepare_route_gather(
            x, routing.order, routing.metadata, out, inverse, routing.ids.shape[1]
        )
        return partial(self.gather, x, routing, out=out, inverse=inverse)

    def bind_combine(self, values, inverse, routing, *, out):
        from functools import partial

        self.library.prepare_route_combine(
            values, inverse, routing.ids, routing.weights, out
        )
        return partial(self.combine, values, inverse, routing, out=out)

    def gather(self, x, routing, *, out=None, inverse=None):
        """Gather valid expert rows; out capacity must cover metadata[1].

        An eager EP caller may use its exact local count. Inverse always covers
        all original route slots, including high slot IDs in a compact plan.
        """
        rows, topk = routing.ids.shape
        out = (
            x.new_empty((rows * topk, x.shape[1]), dtype=torch.float16)
            if out is None
            else out
        )
        inverse = routing.order.new_empty(rows * topk) if inverse is None else inverse
        self.library.route_gather(
            x, routing.order, routing.metadata, out, inverse, topk
        )
        return out, inverse

    @_dsv41_dp_buffers.adapt_combine
    def combine(self, values, inverse, routing, *, out=None):
        """Reduce local routes; every nonnegative id needs a valid inverse row."""
        out = (
            values.new_empty(
                (routing.ids.shape[0], values.shape[1]), dtype=torch.bfloat16
            )
            if out is None
            else out
        )
        self.library.route_combine(values, inverse, routing.ids, routing.weights, out)
        return out
