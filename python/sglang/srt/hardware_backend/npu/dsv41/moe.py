"""Resident MXFP4 experts for V4.1 on Ascend 910B.

The patched decoder owns DP dispatch/combine and TP completion. This module
owns bounded checkpoint repacking, routing, local projections and token combine.
"""

from __future__ import annotations

import logging
from functools import cache, wraps
from typing import NamedTuple

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

from .policy import enabled

logger = logging.getLogger(__name__)

NO_SYNC_PREFILL_TOKENS = 8192


@cache
def live_dp_counts(device, ranks):
    """Stable real-row counts, separate from the padded collective sizes."""
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("prepare DP row counts before Graph capture")
    return torch.ones(ranks, dtype=torch.int32, device=device)


@cache
def _expert_projections(experts, hidden, width, device, g32):
    device = torch.device("npu", device)
    shapes = ((hidden, 2 * width), (width, hidden))
    if g32:
        from .kernels.ops.int4_g32_a8 import G32GroupedGemm

        return tuple(G32GroupedGemm(k, n, device=device) for k, n in shapes)

    from .kernels.ops import NativeGroupedGemm

    return tuple(
        NativeGroupedGemm(k, n, experts=experts, device=device) for k, n in shapes
    )


class G32Projections(NamedTuple):
    gate_up: object
    down: object
    gate_up_weight: tuple[torch.Tensor, ...]
    down_weight: tuple[torch.Tensor, ...]


@cache
def activation(width, *, limit=10.0, dtype="f16", prefix=True):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
        PackedClippedSwiGLU,
    )

    return PackedClippedSwiGLU(
        dimension=width, limit=limit, dtype=dtype, prefix=prefix
    )


class ExpertScratch:
    """Serial layers share intermediate activations on one execution stream.

    Graphs keep their old generation when an eager prefill grows the arena.
    Flat views reuse the dead input for activation and the dead gate/up region
    for down output. Every individual kernel still has disjoint input/output.
    A fourfold drop in the route-slot budget releases the eager arena. Short
    windows keep the current forward's budget so the next layer can reuse it.
    """

    def __init__(self, device, hidden, width, dtype=torch.float16):
        self.device, self.hidden, self.width = device, hidden, width
        self.dtype = dtype
        self.capacity, self.buffers, self.retired = 0, None, []
        self.bindings, self.graph_bindings, self.retired_bindings = {}, {}, []
        self.captured = False

    def get(self, rows, *, reserve=0):
        if rows < 0:
            raise ValueError("expert row count must be non-negative")
        capacity = max(rows, reserve)
        capturing = torch.npu.is_current_stream_capturing()
        shrink = (
            not capturing and not self.captured and 0 < capacity * 4 <= self.capacity
        )
        if self.buffers is None or capacity > self.capacity or shrink:
            if capturing:
                raise RuntimeError("warm resident expert scratch before Graph capture")
            # Large prefill arenas occupy several GiB. Retire only Graph-owned
            # storage and release eager storage before allocating its replacement;
            # keeping both generations (and doubling) exceeds the 910B budget.
            if self.buffers is not None and self.captured:
                self.retired.append(self.buffers)
                self.retired_bindings.append(self.graph_bindings)
            self.bindings, self.graph_bindings = {}, {}
            self.buffers = None
            self.capacity = 0
            self.captured = False
            h, i = capacity * self.hidden, capacity * self.width
            start = max(h, i)
            storage = torch.empty(
                (start + 2 * i,), dtype=self.dtype, device=self.device
            )
            sorted_x = storage[:h].view(capacity, self.hidden)
            gate_up = storage[start : start + 2 * i].view(capacity, 2 * self.width)
            activated = storage[:i].view(capacity, self.width)
            values = storage[i : i + h].view(capacity, self.hidden)
            inverse = torch.empty((capacity,), dtype=torch.int32, device=self.device)
            self.capacity = capacity
            self.buffers = (sorted_x, inverse, gate_up, activated, values)
        self.captured |= capturing
        return tuple(value[:rows] for value in self.buffers)


@cache
def expert_scratch(device, stream, hidden, width, dtype=torch.float16):
    return ExpertScratch(device, hidden, width, dtype)


@cache
def weight_packer(k, n, device):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import MXFP4WeightPacker

    return MXFP4WeightPacker(k, n, device=device)


class ResidentMXFP4Method(FusedMoEMethodBase):
    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, *, g32=False, is_draft=False):
        self.g32 = g32
        self.is_draft = is_draft

    def _weights(self, name):
        fields = ("packed", "scales", "channel") if self.g32 else ("packed", "scales")
        return tuple(getattr(self.layer, f"{name}_{field}") for field in fields)

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **attrs,
    ):
        if layer.num_fused_shared_experts or attrs.get("with_bias", False):
            raise ValueError(
                "resident V4.1 MXFP4 requires separate shared experts and no bias"
            )
        self.e, self.h, self.i = (
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )
        self.global_i = attrs["moe_intermediate_size"]
        if self.g32 and (self.i % 256 or self.h % 256):
            raise ValueError(
                "G32 TP partitions must preserve H32 interleave panels of 256"
            )
        self.rank, self.tp = layer.moe_tp_rank, layer.moe_tp_size
        self.ep = layer.moe_ep_size
        self.first_expert = layer._expert_storage_rank * self.e
        if layer._num_global_routed != self.e * self.ep:
            raise ValueError("resident EP requires contiguous equal expert shards")
        if self.global_i != self.tp * self.i or self.i % 32 or self.h % 64:
            raise ValueError(
                "resident expert TP must preserve group32 scales and fractal columns"
            )
        self.device = torch.device("npu", torch.npu.current_device())
        self.loaded = [[False] * 6 for _ in range(self.e)]
        self.pending, self.pending_expert = {}, None
        # Only compressed final payloads persist. The loader consumes one raw
        # expert at a time and never retains an unpacked weight tensor.
        for name, depth, width in (
            ("gate_up", self.h, 2 * self.i),
            ("down", self.i, self.h),
        ):
            if self.g32 and not (256 <= depth <= 8192 and width % 64 == 0):
                raise ValueError("unsupported 910B INT4 G32 expert geometry")
            elements = depth * width
            fields = (
                (
                    ("packed", elements // 2, torch.uint8),
                    ("scales", elements // 64, torch.uint8),
                    ("channel", width, torch.float32),
                )
                if self.g32
                else (
                    ("packed", elements // 8, torch.int32),
                    ("scales", elements // 32, torch.uint8),
                )
            )
            for field, size, dtype in fields:
                layer.register_buffer(
                    f"{name}_{field}",
                    torch.empty((self.e, size), dtype=dtype, device=self.device),
                    persistent=False,
                )
        for name in (
            "w13_weight",
            "w13_weight_scale_inv",
            "w2_weight",
            "w2_weight_scale_inv",
        ):
            param = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.uint8, device="cpu"), requires_grad=False
            )
            param.weight_loader = self.weight_loader
            layer.register_parameter(name, param)
        self.layer = layer

    def create_moe_runner(self, layer, moe_runner_config):
        if (
            not moe_runner_config.is_gated
            or moe_runner_config.activation != "silu"
            or moe_runner_config.apply_router_weight_on_input
        ):
            raise ValueError("resident MXFP4 requires output-weighted gated SiLU")
        self.limit = float(moe_runner_config.swiglu_limit)
        self.runner = None

    def weight_loader(
        self, param, loaded_weight, weight_name, shard_id, expert_id, **kwargs
    ):
        # These parameters install this method directly as param.weight_loader;
        # DeepseekV4 passes global checkpoint IDs, bypassing FusedMoE's loader.
        if not self.first_expert <= expert_id < self.first_expert + self.e:
            return False
        expert_id -= self.first_expert
        if (
            shard_id not in ("w1", "w3", "w2")
            or not 0 <= expert_id < self.e
            or loaded_weight.element_size() != 1
        ):
            raise ValueError("invalid resident MXFP4 checkpoint projection")
        scale = "scale" in weight_name
        projection = {"w1": 0, "w3": 1, "w2": 2}[shard_id]
        slot = 2 * projection + int(scale)
        if (
            self.pending_expert not in (None, expert_id)
            or self.loaded[expert_id][slot]
            or slot in self.pending
        ):
            raise ValueError(
                "resident MXFP4 requires complete, unique expert-major checkpoint input"
            )
        raw = loaded_weight.view(torch.uint8)
        if raw.device.type != "cpu":
            raise ValueError("load raw MXFP4 bytes directly from the CPU checkpoint")
        expected = (
            (self.global_i, self.h // (32 if scale else 2))
            if projection < 2
            else (self.h, self.global_i // (32 if scale else 2))
        )
        if scale and self.g32:
            expected = (expected[0], expected[1] // 2 + 4)
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"MXFP4 {shard_id} source shape {tuple(raw.shape)} != {expected}"
            )
        lo, hi = self.rank * self.i, (self.rank + 1) * self.i
        if projection < 2:
            shard = raw[lo:hi]
        elif self.tp == 1:
            shard = raw
        elif scale and self.g32:
            shard = torch.cat((raw[:, lo // 64 : hi // 64], raw[:, -4:]), dim=1)
        else:
            divisor = 32 if scale else 2
            shard = raw[:, lo // divisor : hi // divisor]
        self.pending_expert = expert_id
        self.pending[slot] = shard.contiguous()
        if len(self.pending) == 6:
            c = torch.cat((self.pending[0], self.pending[2]))
            s = torch.cat((self.pending[1], self.pending[3]))
            for pair, k, n, target in (
                (
                    (c, s),
                    self.h,
                    2 * self.i,
                    self._weights("gate_up"),
                ),
                (
                    (self.pending[4], self.pending[5]),
                    self.i,
                    self.h,
                    self._weights("down"),
                ),
            ):
                if self.g32:
                    from .kernels.ops.int4_g32_a8 import pack_g32

                    for buffer, value in zip(target, pack_g32(*pair), strict=True):
                        buffer[expert_id].copy_(value)
                else:
                    weight_packer(k, n, self.device).load_cpu(
                        *pair, packed=target[0][expert_id], s2=target[1][expert_id]
                    )
            self.loaded[expert_id] = [True] * 6
            self.pending.clear()
            self.pending_expert = None
        return True

    def bind_local_experts(self, gate_up_weight, down_weight):
        up, down = _expert_projections(
            self.e, self.h, self.i, torch.npu.current_device(), self.g32
        )
        if self.g32:
            return G32Projections(up, down, gate_up_weight, down_weight)
        from .kernels.ops import MXFP4LocalExperts

        return MXFP4LocalExperts(
            up,
            down,
            gate_up_weight,
            down_weight,
            limit=self.limit,
            activation_op=activation(self.i, limit=self.limit),
        )

    def process_weights_after_loading(self, layer):
        if self.pending or not all(all(parts) for parts in self.loaded):
            raise RuntimeError("incomplete resident expert weights")
        weights = self._weights("gate_up"), self._weights("down")
        self.local_experts = self.bind_local_experts(*weights)
        logger.debug(
            "Resident Ascend C MXFP4 layer %s: rank %d/%d, %d experts, H%d/J%d, %.3f GiB compressed NPU weights (%s scales)",
            layer.layer_id,
            self.rank,
            self.tp,
            self.e,
            self.h,
            self.i,
            sum(buffer.nbytes for projection in weights for buffer in projection)
            / 2**30,
            "per-token A8, signed K32 factors"
            if self.g32
            else "uint8 preprocessed FP16",
        )

    def apply(self, layer, dispatch_output):
        raise RuntimeError("resident MXFP4 uses the V4.1 native routing entry")

    def routed_storage(self, x, routing, dtype, *, window_budget=0):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.moe_routes import (
            MoERoutes,
        )

        if not hasattr(self, "routes"):
            self.routes = MoERoutes(device=x.device)
        slots = routing.ids.numel()
        arena = expert_scratch(
            x.device,
            torch.npu.current_stream(x.device).npu_stream,
            self.h,
            self.i,
            dtype,
        )
        # Every slot may belong to this EP rank. Kernels read the live local
        # count on device, while the arena reserves the worst case.
        return arena, arena.get(slots, reserve=window_budget * routing.ids.shape[1])

    def prepare_routed(self, x, routing, *, window_budget=0):
        arena, (sorted_x, inverse, gate_up, activated, values) = self.routed_storage(
            x, routing, torch.float16, window_budget=window_budget
        )
        rows = sorted_x.shape[0]
        if rows:
            self.routes.gather(x, routing, out=sorted_x, inverse=inverse)
        if not rows:
            return inverse, gate_up, activated, values, None
        bound = x.shape[0]
        mean = rows / (self.e * self.ep)
        key = (routing.metadata.data_ptr(), rows, bound)
        previous_key, invoke = arena.bindings.get(self.local_experts, (None, None))
        if previous_key != key:
            invoke = self.local_experts.bind(
                sorted_x,
                routing.metadata,
                gate_up=gate_up,
                activation=activated,
                out=values,
                max_expert_rows=bound,
                mean_expert_rows=mean,
                tiling_rows=rows,
            )
            # Keep only this layer's latest binding. Captured bindings live
            # with the arena generation.
            arena.bindings[self.local_experts] = (key, invoke)
        if torch.npu.is_current_stream_capturing():
            arena.graph_bindings[self.local_experts, key] = invoke
        return inverse, gate_up, activated, values, invoke

    def _routed_g32(self, x, routing, *, out=None, window_budget=0):
        from .kernels.ops.h32_quant import h32_quant

        _, (storage, inverse, gate_up, mid, values) = self.routed_storage(
            x, routing, torch.bfloat16, window_budget=window_budget
        )
        rows = gate_up.shape[0]
        metadata = routing.metadata
        if rows:
            quantized, scales = h32_quant(x, False)
            sorted_codes = (
                storage.view(torch.int8).flatten()[: rows * self.h].view(rows, self.h)
            )
            sorted_scales = scales.new_empty(rows)
            self.routes.library.route_gather_a8(
                quantized,
                routing.order,
                metadata,
                sorted_codes,
                inverse,
                routing.ids.shape[1],
                scales,
                sorted_scales,
            )
            projections = self.local_experts
            projections.gate_up.from_quantized(
                sorted_codes,
                sorted_scales,
                projections.gate_up_weight,
                metadata,
                out=gate_up,
            )
            count = metadata[1:2]
            activation(self.i, limit=self.limit, dtype="bf16", prefix=True)(
                gate_up, out=mid, rows=count
            )
            codes, scale = h32_quant(mid, True, count=count)
            projections.down.from_quantized(
                codes,
                scale,
                projections.down_weight,
                metadata,
                out=values,
            )
        return self.finish_routed(values, inverse, routing, out=out)

    def routed(self, x, routing, *, out=None, window_budget=0):
        if x.shape[0] == 0:
            return torch.empty_like(x)
        if self.g32:
            return self._routed_g32(x, routing, out=out, window_budget=window_budget)
        inverse, _, _, values, invoke = self.prepare_routed(
            x, routing, window_budget=window_budget
        )
        if invoke is not None:
            invoke()
        return self.finish_routed(values, inverse, routing, out=out)

    def finish_routed(self, values, inverse, routing, *, out=None):
        from sglang.srt.runtime_context import get_forward

        if get_forward().npu_defer_route_combine:
            from .peer_collectives import RoutedOutput

            return RoutedOutput(values, inverse, routing.ids, routing.weights)
        return self.routes.combine(values, inverse, routing, out=out)


def _run_expert_windows(hidden, run, *, counts, stride, valid, image_mask):
    from sglang.srt.runtime_context import get_forward

    from .parallel import moe_routed_output

    rows = hidden.shape[0]
    if rows <= NO_SYNC_PREFILL_TOKENS:
        return run(hidden, counts, stride, valid, image_mask)
    if get_forward().npu_defer_route_combine:
        raise ValueError("windowed MoE requires immediate route combine")

    budget = NO_SYNC_PREFILL_TOKENS
    split_chunks = counts is not None and stride > budget
    if split_chunks:
        chunk_rows = stride
    else:
        chunk_rows = rows
        if counts is not None:
            budget = (budget // stride) * stride

    routed = moe_routed_output(hidden)
    for chunk in range(0, rows, chunk_rows):
        chunk_end = min(chunk + chunk_rows, rows)
        for start in range(chunk, chunk_end, budget):
            stop = min(start + budget, chunk_end)
            win_counts, win_stride, win_valid = None, None, None
            if split_chunks:
                rank = chunk // stride
                win_valid = (
                    (counts[rank : rank + 1] - (start - chunk))
                    .clamp(0, stop - start)
                    .to(torch.int32)
                )
            elif counts is not None:
                win_counts, win_stride = (
                    counts[start // stride : stop // stride],
                    stride,
                )
            elif valid is not None:
                win_valid = (valid - start).clamp(0, stop - start)

            run(
                hidden[start:stop],
                win_counts,
                win_stride,
                win_valid,
                image_mask[start:stop] if image_mask is not None else None,
                out=routed[start:stop],
                window_budget=budget,
            )
    return routed


def adapt_forward(original):
    if not enabled():
        return original

    @wraps(original)
    def forward(
        self,
        hidden_states,
        forward_batch=None,
        gemm_output_zero_allocator=None,
        input_ids=None,
        input_ids_global=None,
        skip_shared_experts=False,
        return_moe_output=False,
        leave_output_sum=False,
    ):
        from sglang.srt.layers.moe.utils import post_experts_all_reduce
        from sglang.srt.models.deepseek_v2 import MoEOutput

        method = self.experts.quant_method
        if not isinstance(method, ResidentMXFP4Method):
            return original(
                self,
                hidden_states,
                forward_batch,
                gemm_output_zero_allocator,
                input_ids,
                input_ids_global,
                skip_shared_experts,
                return_moe_output,
                leave_output_sum=leave_output_sum,
            )
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import Router
        from sglang.srt.runtime_context import get_parallel

        from .parallel import moe_routing_metadata
        from .peer_collectives import RoutedOutput

        if self._enable_a2a_moe or self.is_hash:
            raise ValueError(
                "resident V4.1 TP/EP uses all-gather and partial sums; A2A/hash routing is unsupported"
            )
        if not hasattr(self, "_dsv41_resident_router"):
            self._dsv41_resident_router = Router(device=hidden_states.device)
        layout = getattr(forward_batch, "_dsv41_group_moe_rows", None)
        alternate = getattr(self.gate, "e_score_correction_bias_vl", None)
        image_mask = None
        if alternate is not None and input_ids_global is not None:
            use_layout_mask = (
                getattr(layout, "global_input_ids", None) is input_ids_global
            )
            if use_layout_mask:
                image_mask = layout.image_mask
            if image_mask is None:
                image_mask = (
                    (input_ids_global.reshape(-1) == self.config.image_token_id)
                    .int()
                    .contiguous()
                )
                if use_layout_mask:
                    layout.image_mask = image_mask
        parallel = get_parallel()
        valid = (
            forward_batch.moe_num_token_non_padded()
            if forward_batch is not None and parallel.attn_dp_size == 1
            else None
        )
        counts, stride = None, None
        if layout is not None:
            counts, stride = layout.counts, layout.stride
        elif (
            forward_batch is not None
            and parallel.attn_dp_size > 1
            and forward_batch.dp_padding_mode.is_max_len()
        ):
            from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper

            counts = live_dp_counts(hidden_states.device, parallel.attn_dp_size)
            stride = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
        rows = hidden_states.shape[0]
        if counts is not None and (not stride or stride * counts.numel() != rows):
            raise ValueError(
                f"resident TP MoE DP chunks: stride={stride}, ranks={counts.numel()}, rows={rows}"
            )
        metadata = moe_routing_metadata(hidden_states, method.e)

        def run_window(x, counts, stride, valid, mask, *, out=None, window_budget=0):
            routing = self._dsv41_resident_router.route(
                self.gate(x, gemm_output_zero_allocator).float(),
                self.gate.e_score_correction_bias,
                factor=self.routed_scaling_factor,
                first=method.first_expert,
                experts=method.e,
                valid_rows=valid,
                dp_counts=counts,
                dp_stride=stride,
                alternate_bias=alternate if mask is not None else None,
                image_mask=mask,
                metadata_out=metadata,
            )
            return method.routed(x, routing, out=out, window_budget=window_budget)

        routed = _run_expert_windows(
            hidden_states,
            run_window,
            counts=counts,
            stride=stride,
            valid=valid.reshape(1) if valid is not None else None,
            image_mask=image_mask,
        )
        if isinstance(routed, RoutedOutput):
            if not skip_shared_experts or return_moe_output or leave_output_sum:
                raise ValueError(
                    "deferred native route combine requires the SP return boundary"
                )
            return routed
        shared = None
        if (
            not skip_shared_experts
            and hasattr(self, "shared_experts")
            and hidden_states.shape[0]
        ):
            shared = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator
            )
        if return_moe_output or leave_output_sum:
            return MoEOutput(
                routed,
                shared,
                self.experts,
                self.routed_scaling_factor,
                self._shared_expert_tp1,
            )
        if shared is not None and (not self._shared_expert_tp1):
            routed = routed + shared
        if self.reduce_results:
            routed = post_experts_all_reduce(routed)
        if (
            shared is not None
            and self._shared_expert_tp1
            and self._adds_replicated_output()
        ):
            routed = routed + shared
        return routed

    return forward
