"""Stream-ordered MoE collective buffers with stable Graph generations.

HCCL records uses on its communication stream. Allocating a new gather/result
at every layer leaves many large buffers awaiting allocator events while the
host runs ahead. The decoder already orders each collective and its consumers
on the compute stream, so serial layers can explicitly reuse these buffers.
"""

from contextvars import ContextVar
from functools import cache, wraps

import torch

from .policy import enabled

_in_moe = ContextVar("dsv41_moe_collectives", default=False)


class RowBuffer:
    def __init__(self, device, width, dtype):
        self.device, self.width, self.dtype = device, width, dtype
        self.tensor, self.captured, self.retired = None, False, []

    def get(self, rows):
        capturing = torch.npu.is_current_stream_capturing()
        shrink = (
            self.tensor is not None
            and not capturing
            and not self.captured
            and 0 < rows * 4 <= self.tensor.shape[0]
        )
        if self.tensor is None or rows > self.tensor.shape[0] or shrink:
            if capturing:
                raise RuntimeError("warm MoE collective buffers before Graph capture")
            if self.tensor is not None and self.captured:
                self.retired.append(self.tensor)
            self.tensor, self.captured = None, False
            self.tensor = torch.empty(
                (rows, self.width), dtype=self.dtype, device=self.device
            )
        self.captured |= capturing
        return self.tensor[:rows]


@cache
def row_buffer(role, device, stream, width, dtype=torch.bfloat16):
    return RowBuffer(device, width, dtype)


def adapt_run(original):
    if not enabled():
        return original

    @wraps(original)
    def run(layer, hidden_states, forward_batch, **kwargs):
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        from sglang.srt.models import deepseek_v4
        from sglang.srt.models.deepseek_common import moe_transport as transport
        from sglang.srt.runtime_context import get_parallel

        if (
            layer.config.model_type != "deepseek_v41"
            or hidden_states.device.type != "npu"
        ):
            return original(layer, hidden_states, forward_batch, **kwargs)
        token = _in_moe.set(True)
        try:
            if (
                _use_moe_sequence_parallel(get_parallel(), forward_batch)
                and get_moe_a2a_backend().is_none()
                and (not deepseek_v4.is_cp_active(forward_batch))
                and (
                    getattr(layer.mlp, "shared_experts", None) is None
                    or (
                        transport.SHARED_EXPERT_LOCAL
                        and getattr(layer.mlp, "_shared_expert_tp1", False)
                    )
                )
            ):
                return _run_sequence_parallel(
                    layer, hidden_states, forward_batch, **kwargs
                )
            return original(layer, hidden_states, forward_batch, **kwargs)
        finally:
            _in_moe.reset(token)

    return run


def adapt_combine(original_combine):
    if not enabled():
        return original_combine

    @wraps(original_combine)
    def combine(routes, values, inverse, routing, *, out=None):
        if out is None and _in_moe.get():
            stream = torch.npu.current_stream(values.device).npu_stream
            out = row_buffer("combine", values.device, stream, values.shape[1]).get(
                routing.ids.shape[0]
            )
        return original_combine(routes, values, inverse, routing, out=out)

    return combine


def collective_buffer(role, rows, width):
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as geometry

    device = geometry.get_dp_device()
    stream = torch.npu.current_stream(device).npu_stream
    return row_buffer(role, device, stream, width, geometry.get_dp_dtype()).get(rows)


def moe_sequence_parallel_enabled(parallel):
    return (
        parallel.tp_size == 8
        and (parallel.attn_dp_size, parallel.attn_tp_size) in ((4, 2), (8, 1))
        and parallel.attn_cp_size == 1
        and parallel.moe_ep_size == 8
        and parallel.moe_tp_size == 1
        and parallel.moe_dp_size == 1
    )


def _use_moe_sequence_parallel(parallel, batch):
    from .peer_collectives import enabled as peer_enabled

    return (
        moe_sequence_parallel_enabled(parallel)
        # Preserve the existing DP8 MoeRowTransport baseline when opted out.
        and (parallel.attn_tp_size == 2 or peer_enabled())
        and batch.dp_padding_mode is not None
        and batch.dp_padding_mode.is_max_len()
    )


def _workspace(role, x, rows):
    stream = torch.npu.current_stream(x.device).npu_stream
    return row_buffer(role, x.device, stream, x.shape[1], x.dtype).get(rows)


def _run_sequence_parallel(
    layer,
    local,
    batch,
    *,
    input_ids=None,
    input_ids_global=None,
    return_moe_output=False,
):
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as geometry
    from sglang.srt.runtime_context import get_forward, get_parallel

    parallel = get_parallel()
    rows = geometry.get_local_dp_buffer_len()
    if local.shape[0] != rows:
        raise ValueError("MoE SP requires the published local token rows")
    global_rows = rows * parallel.attn_dp_size
    if geometry.get_global_dp_buffer_len() != global_rows:
        raise ValueError("MoE SP requires equal padded DP token segments")
    # This is a model output; a later MoE call must not overwrite it.
    if rows == 0:
        return torch.empty_like(local)

    from .peer_collectives import moe_communicator

    peer = moe_communicator(local, parallel.tp_group)

    tp = parallel.attn_tp_size
    shard_rows = (rows + tp - 1) // tp
    padded_rows = shard_rows * tp
    start = parallel.attn_tp_rank * shard_rows
    owned_rows = max(0, min(shard_rows, rows - start))
    shard = local.narrow(0, min(start, rows), owned_rows)
    shard_input = shard
    if owned_rows != shard_rows and peer is None:
        shard_input = _workspace("moe_sp_padded_input", local, shard_rows)
        if owned_rows:
            shard_input[:owned_rows].copy_(shard)
        shard_input[owned_rows:].zero_()
    shared = None
    if owned_rows and getattr(layer.mlp, "shared_experts", None) is not None:
        shared = layer.mlp._forward_shared_experts(shard)

    # Rank order is DP0/TP0, DP0/TP1, DP1/TP0, ...; concatenating
    # these halves preserves the original global DP token order.
    gathered = _workspace("gather", local, global_rows)
    if peer is not None:
        peer.gather(local, gathered)
    elif padded_rows == rows:
        parallel.tp_group.all_gather_into_tensor(gathered, shard_input)
    else:
        # HCCL requires equal payloads. Compact the communication-only holes
        # before routing, retaining the original DP stride and input-ID order.
        padded_gather = _workspace(
            "moe_sp_padded_gather", local, padded_rows * parallel.attn_dp_size
        )
        parallel.tp_group.all_gather_into_tensor(padded_gather, shard_input)
        gathered.view(parallel.attn_dp_size, rows, -1).copy_(
            padded_gather.view(parallel.attn_dp_size, padded_rows, -1)[:, :rows]
        )
    with get_forward().scoped(
        mlp_reduce_scatter=True, npu_defer_route_combine=peer is not None
    ):
        partial = layer.mlp(
            gathered,
            batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
            skip_shared_experts=True,
            return_moe_output=False,
        )
    if peer is not None:
        return peer.finish(partial, shared, rows)
    if padded_rows != rows:
        padded_partial = _workspace(
            "moe_sp_padded_partial", local, padded_rows * parallel.attn_dp_size
        ).view(parallel.attn_dp_size, padded_rows, -1)
        padded_partial[:, :rows].copy_(partial.view(parallel.attn_dp_size, rows, -1))
        padded_partial[:, rows:].zero_()
        partial = padded_partial.view(-1, local.shape[1])
    shard_output = _workspace("moe_sp_shard", local, shard_rows)
    parallel.tp_group.reduce_scatter_tensor(shard_output, partial)
    if shared is not None:
        # EP8 already completed the expert sum. Add the replicated shared
        # expert exactly once on its token shard, before restoring TP copies.
        shard_output[:owned_rows].add_(shared)
    output = torch.empty(
        (padded_rows, local.shape[1]), dtype=local.dtype, device=local.device
    )
    if tp == 1:
        output.copy_(shard_output)
    else:
        parallel.attn_tp_group.all_gather_into_tensor(output, shard_output)
    return output[:rows]
