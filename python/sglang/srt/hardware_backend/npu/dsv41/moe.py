"""Resident MXFP4 experts for V4.1 on Ascend 910B.

The patched decoder owns DP dispatch/combine and TP completion. This module
owns bounded checkpoint repacking, routing, local projections and token combine.
"""

from __future__ import annotations

import logging
from functools import cache, wraps

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

from .policy import enabled

logger = logging.getLogger(__name__)

NO_SYNC_PREFILL_TOKENS = 4096


@cache
def live_dp_counts(device, ranks):
    """Stable real-row counts, separate from the padded collective sizes."""
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("prepare DP row counts before Graph capture")
    return torch.ones(ranks, dtype=torch.int32, device=device)


@cache
def _expert_projections(experts, hidden, width, device):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import NativeGroupedGemm

    result = []
    for k, n in ((hidden, 2 * width), (width, hidden)):
        result.append(
            NativeGroupedGemm(
                k,
                n,
                experts=experts,
                device=torch.device("npu", device),
            )
        )
    return tuple(result)


@cache
def activation(width):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
        PackedClippedSwiGLU,
    )

    return PackedClippedSwiGLU(dimension=width, dtype="f16", prefix=True)


class ExpertScratch:
    """Serial layers share intermediate activations on one execution stream.

    Graphs keep their old generation when an eager prefill grows the arena.
    Flat views reuse the dead input for activation and the dead gate/up region
    for down output. Every individual kernel still has disjoint input/output.
    Inverse indices retain the original route-slot capacity even when expert
    activations use only this rank's compact rows. A fourfold global route-slot
    budget decrease releases an eager arena; local routing skew within that
    budget keeps the high-water capacity and does not invalidate bindings.
    """

    def __init__(self, device, hidden, width):
        self.device, self.hidden, self.width = device, hidden, width
        self.capacity, self.slot_capacity, self.buffers, self.retired = 0, 0, None, []
        self.bindings, self.graph_bindings, self.retired_bindings = {}, {}, []
        self.captured = False

    def get(self, rows, *, slots=None):
        slots = rows if slots is None else slots
        if not 0 <= rows <= slots:
            raise ValueError("expert rows must fit the original route slot capacity")
        capturing = torch.npu.is_current_stream_capturing()
        shrink = (
            not capturing and not self.captured and 0 < slots * 4 <= self.slot_capacity
        )
        if (
            self.buffers is None
            or rows > self.capacity
            or slots > self.slot_capacity
            or shrink
        ):
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
            self.capacity = self.slot_capacity = 0
            self.captured = False
            capacity = rows
            h, i = capacity * self.hidden, capacity * self.width
            start = max(h, i)
            storage = torch.empty(
                (start + 2 * i,), dtype=torch.float16, device=self.device
            )
            sorted_x = storage[:h].view(capacity, self.hidden)
            gate_up = storage[start : start + 2 * i].view(capacity, 2 * self.width)
            activated = storage[:i].view(capacity, self.width)
            values = storage[i : i + h].view(capacity, self.hidden)
            inverse = torch.empty((slots,), dtype=torch.int32, device=self.device)
            buffers = (sorted_x, inverse, gate_up, activated, values)
            self.capacity, self.slot_capacity, self.buffers = capacity, slots, buffers
        self.captured |= capturing
        return tuple(
            value[: slots if index == 1 else rows]
            for index, value in enumerate(self.buffers)
        )


@cache
def expert_scratch(device, stream, hidden, width):
    return ExpertScratch(device, hidden, width)


def compact_prefill(batch):
    """Include idle/decode EP peers participating in a global eager prefill."""
    return (
        batch is not None
        and not torch.npu.is_current_stream_capturing()
        and (
            batch.is_extend_in_batch
            or (
                batch.forward_mode.is_extend_without_speculative()
                and not batch.forward_mode.is_cuda_graph()
            )
        )
    )


@cache
def weight_packer(k, n, device):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import MXFP4WeightPacker

    return MXFP4WeightPacker(k, n, device=device)


class ResidentMXFP4Method(FusedMoEMethodBase):
    fuse_routed_scaling_factor_in_topk = True

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
        for name, shape, dtype in (
            ("gate_up_packed", (self.e, self.h * 2 * self.i // 8), torch.int32),
            ("gate_up_scales", (self.e, self.h * 2 * self.i // 32), torch.uint8),
            ("down_packed", (self.e, self.h * self.i // 8), torch.int32),
            ("down_scales", (self.e, self.h * self.i // 32), torch.uint8),
        ):
            layer.register_buffer(
                name,
                torch.empty(shape, dtype=dtype, device=self.device),
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
        _expert_projections(self.e, self.h, self.i, torch.npu.current_device())

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
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"MXFP4 {shard_id} source shape {tuple(raw.shape)} != {expected}"
            )
        lo, hi = self.rank * self.i, (self.rank + 1) * self.i
        shard = (
            raw[lo:hi]
            if projection < 2
            else raw[:, lo // (32 if scale else 2) : hi // (32 if scale else 2)]
        )
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
                    (self.layer.gate_up_packed, self.layer.gate_up_scales),
                ),
                (
                    (self.pending[4], self.pending[5]),
                    self.i,
                    self.h,
                    (self.layer.down_packed, self.layer.down_scales),
                ),
            ):
                converter = weight_packer(
                    k,
                    n,
                    self.device,
                )
                converter.load_cpu(
                    *pair, packed=target[0][expert_id], s2=target[1][expert_id]
                )
            self.loaded[expert_id] = [True] * 6
            self.pending.clear()
            self.pending_expert = None
        return True

    def process_weights_after_loading(self, layer):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import MXFP4LocalExperts

        if self.pending or not all(all(parts) for parts in self.loaded):
            raise RuntimeError("incomplete resident MXFP4 expert weights")
        up, down = _expert_projections(
            self.e, self.h, self.i, torch.npu.current_device()
        )
        self.local_experts = MXFP4LocalExperts(
            up,
            down,
            (layer.gate_up_packed, layer.gate_up_scales),
            (layer.down_packed, layer.down_scales),
            limit=self.limit,
            activation_op=activation(self.i),
        )
        logger.debug(
            "Resident Ascend C MXFP4 layer %s: rank %d/%d, %d experts, H%d/J%d, %.3f GiB compressed NPU weights (%s scales)",
            layer.layer_id,
            self.rank,
            self.tp,
            self.e,
            self.h,
            self.i,
            sum(
                getattr(layer, n).numel() * getattr(layer, n).element_size()
                for n in (
                    "gate_up_packed",
                    "gate_up_scales",
                    "down_packed",
                    "down_scales",
                )
            )
            / 2**30,
            "uint8 preprocessed FP16",
        )

    def apply(self, layer, dispatch_output):
        raise RuntimeError("resident MXFP4 uses the V4.1 native routing entry")

    def routed(self, x, routing, *, compact=False):
        if x.shape[0] == 0:
            return torch.empty_like(x)
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.moe_routes import (
            MoERoutes,
        )

        if not hasattr(self, "routes"):
            self.routes = MoERoutes(device=x.device)
        slots = routing.ids.numel()
        # Only eager prefill opts in: Graph warmup/capture keeps conservative,
        # stable capacities and never reads a device scalar back to the CPU.
        capturing = torch.npu.is_current_stream_capturing()
        compact = compact and self.ep > 1 and not capturing
        arena = expert_scratch(
            x.device, torch.npu.current_stream(x.device).npu_stream, self.h, self.i
        )
        # kB is a proven upper bound even for a completely skewed EP rank.
        # Reuse an already sufficient arena, or spend a bounded amount for a
        # small global batch. Neither case needs the device's exact local L.
        fits_bound = x.shape[0] <= NO_SYNC_PREFILL_TOKENS or arena.capacity >= slots
        rows = int(routing.metadata[1].item()) if compact and not fits_bound else slots
        if not 0 <= rows <= slots:
            raise ValueError("invalid compact local expert row count")
        sorted_x, inverse, gate_up, activated, values = arena.get(rows, slots=slots)
        if not rows:
            return self.routes.combine(values, inverse, routing)
        bound = min(x.shape[0], rows)
        mean = slots / (self.e * self.ep)
        self.routes.gather(x, routing, out=sorted_x, inverse=inverse)
        # Compaction changes capacity, not the established global-budget tiling
        # policy. Local means can cross a BM threshold at each layer and cause
        # substantially more weight decoding on the hot experts.
        key = (routing.metadata.data_ptr(), rows, bound, mean)
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
                tiling_rows=slots,
            )
            # Router metadata can be newly allocated in eager execution. Keep
            # only this layer's latest binding; captured bindings live with the
            # arena generation and cannot retain later eager prefills.
            arena.bindings[self.local_experts] = (key, invoke)
        if torch.npu.is_current_stream_capturing():
            arena.graph_bindings[self.local_experts, key] = invoke
        invoke()
        from sglang.srt.runtime_context import get_forward

        if get_forward().npu_defer_route_combine:
            from .peer_collectives import RoutedOutput

            return RoutedOutput(values, inverse, routing.ids, routing.weights)
        return self.routes.combine(values, inverse, routing)


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

        if self._enable_a2a_moe or self.is_hash:
            raise ValueError(
                "resident V4.1 TP/EP uses all-gather and partial sums; A2A/hash routing is unsupported"
            )
        if not hasattr(self, "_dsv41_resident_router"):
            self._dsv41_resident_router = Router(device=hidden_states.device)
        logits = self.gate(hidden_states, gemm_output_zero_allocator).float()
        alternate = getattr(self.gate, "e_score_correction_bias_vl", None)
        image_mask = None
        if alternate is not None and input_ids_global is not None:
            image_mask = (
                (input_ids_global.reshape(-1) == self.config.image_token_id)
                .int()
                .contiguous()
            )
        parallel = get_parallel()
        valid = (
            forward_batch.moe_num_token_non_padded()
            if forward_batch is not None and parallel.attn_dp_size == 1
            else None
        )
        dp_args = {}
        if (
            forward_batch is not None
            and parallel.attn_dp_size > 1
            and forward_batch.dp_padding_mode.is_max_len()
        ):
            from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper

            counts = live_dp_counts(hidden_states.device, parallel.attn_dp_size)
            stride = _DpGatheredBufferWrapper.get_local_dp_buffer_len()
            if stride * counts.numel() != hidden_states.shape[0]:
                raise ValueError(
                    f"resident TP MoE DP chunks: stride={stride}, ranks={counts.numel()}, rows={hidden_states.shape[0]}"
                )
            dp_args = {"dp_counts": counts, "dp_stride": stride}
        routing = self._dsv41_resident_router.route(
            logits,
            self.gate.e_score_correction_bias,
            factor=self.routed_scaling_factor,
            first=method.first_expert,
            experts=method.e,
            valid_rows=valid.reshape(1) if valid is not None else None,
            alternate_bias=alternate if image_mask is not None else None,
            image_mask=image_mask,
            **dp_args,
        )
        routed = method.routed(
            hidden_states, routing, compact=compact_prefill(forward_batch)
        )
        from .peer_collectives import RoutedOutput

        if isinstance(routed, RoutedOutput):
            if not skip_shared_experts or return_moe_output or leave_output_sum:
                raise ValueError(
                    "deferred native route combine requires the SP return boundary"
                )
            return routed
        shared = (
            self._forward_shared_experts(hidden_states, gemm_output_zero_allocator)
            if not skip_shared_experts
            and hasattr(self, "shared_experts")
            and hidden_states.shape[0]
            else None
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
