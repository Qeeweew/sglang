"""Target shared/routed experts: share pointwise work without changing GEMM operands."""

from typing import NamedTuple

import torch

from .kernels.ops.mxfp4_a8 import A8GroupedGemm, h32_quant
from .linear import SharedW8A8Method, W8A16LinearMethod
from .moe import ResidentMXFP4Method


class G32Projections(NamedTuple):
    gate_up: A8GroupedGemm
    down: A8GroupedGemm
    gate_up_weight: tuple[torch.Tensor, ...]
    down_weight: tuple[torch.Tensor, ...]


def can_join(mlp):
    method = getattr(getattr(mlp, "experts", None), "quant_method", None)
    if (
        not isinstance(method, ResidentMXFP4Method)
        or method.is_draft
        or (method.a8 and not method.g32)
        or not getattr(mlp, "_shared_expert_tp1", False)
    ):
        return False
    shared = getattr(mlp, "shared_experts", None)
    if shared is None:
        return False
    if (
        shared.swiglu_limit != method.limit
        or shared.gate_up_proj.bias is not None
        or shared.down_proj.bias is not None
    ):
        raise ValueError(
            "joint target experts require matching clipping and bias-free projections"
        )
    gate = shared.gate_up_proj.quant_method
    down = shared.down_proj.quant_method
    if method.g32:
        matches = (
            isinstance(gate, SharedW8A8Method)
            and isinstance(down, SharedW8A8Method)
            and shared.gate_up_proj.weight_scale_inv.numel() == 2 * method.i
        )
    else:
        matches = (
            isinstance(gate, W8A16LinearMethod)
            and isinstance(down, W8A16LinearMethod)
            and gate.parts == (method.i, method.i)
            and down.k == method.i
        )
    if not matches:
        raise ValueError(
            "shared projection format or width does not match the joint target experts"
        )
    return True


def fp16_shared_gate(shared, x):
    gate = shared.gate_up_proj
    return gate.quant_method.apply(gate, x, output_dtype=torch.float16)


def _g32_gate_up(method, shared, hidden, routing, owned, compact):
    _, (storage, inverse, gate_up, _, values) = method.routed_storage(
        hidden, routing, compact, torch.float16
    )
    library = method.routes.library
    rows = gate_up.shape[0]
    metadata = routing.metadata
    quantized, scales = h32_quant(hidden, False)
    shared_input = quantized[owned]
    shared_rows = shared_input.shape[0]
    shared_gate = (
        SharedW8A8Method.from_quantized(
            shared.gate_up_proj, shared_input, scales[owned], output_dtype=gate_up.dtype
        )
        if shared_rows
        else gate_up[:0]
    )
    projections = method.local_experts
    if rows:
        sorted_codes = (
            storage.view(torch.int8).flatten()[: rows * method.h].view(rows, method.h)
        )
        sorted_scales = scales.new_empty(rows)
        library.route_gather_a8(
            quantized,
            routing.order,
            metadata,
            sorted_codes,
            inverse,
            routing.ids.shape[1],
            scales,
            sorted_scales,
        )
        projections.gate_up.from_quantized(
            sorted_codes,
            sorted_scales,
            projections.gate_up_weight,
            metadata,
            out=gate_up,
        )
    total = rows + shared_rows
    needed = total * method.i
    mid = (
        storage.flatten()[:needed].view(total, method.i)
        if needed <= storage.numel()
        else gate_up.new_empty((total, method.i))
    )
    return inverse, values, shared_gate, gate_up, mid


def run_joint_experts(method, shared, hidden, routing, owned, compact):
    first, stop, step = owned.indices(hidden.shape[0])
    if step != 1:
        raise ValueError("shared owner rows must be contiguous")
    shared_rows = max(stop - first, 0)
    if method.g32:
        inverse, values, shared_gate, gate_up, mid = _g32_gate_up(
            method, shared, hidden, routing, owned, compact
        )
        shared_mid, routed_mid = mid[:shared_rows], mid[shared_rows:]
    else:
        shared_input = hidden.new_empty((shared_rows, method.h), dtype=torch.float16)
        inverse, gate_up, routed_mid, values, projections = method.prepare_routed(
            hidden, routing, compact, shared_input=shared_input, first=first
        )
        shared_gate = (
            fp16_shared_gate(shared, shared_input) if shared_rows else gate_up[:0]
        )
        shared_mid = shared_gate.new_empty((shared_rows, method.i))
        if projections is not None:
            projections[0]()

    rows = gate_up.shape[0]
    count = routing.metadata[1:2]
    method.routes.library.joint_swiglu(
        shared_gate, gate_up, count, shared_mid, routed_mid, method.limit
    )
    shared_out = None
    if method.g32 and (rows or shared_rows):
        codes, scale = h32_quant(mid, True, count=count, shared_rows=shared_rows)
        if rows:
            projections = method.local_experts
            projections.down.from_quantized(
                codes[shared_rows:],
                scale[shared_rows:],
                projections.down_weight,
                routing.metadata,
                out=values,
            )
        if shared_rows:
            shared_out = SharedW8A8Method.from_quantized(
                shared.down_proj, codes[:shared_rows], scale[:shared_rows]
            )
    elif not method.g32:
        if projections is not None:
            projections[1]()
        if shared_rows:
            shared_out, _ = shared.down_proj(shared_mid)
    return method.finish_routed(values, inverse, routing), shared_out
