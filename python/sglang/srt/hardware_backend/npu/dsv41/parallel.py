"""Stream-ordered MoE collective buffers with stable Graph generations.

HCCL records uses on its communication stream. Allocating a new gather/result
at every layer leaves many large buffers awaiting allocator events while the
host runs ahead. The decoder already orders each collective and its consumers
on the compute stream, so serial layers can explicitly reuse these buffers.
"""

from contextvars import ContextVar
from dataclasses import replace
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


def adapt_shared_before_gather(original):
    if not enabled():
        return original

    @wraps(original)
    def before_gather(mlp):
        from .joint_experts import can_join

        return original(mlp) and not can_join(mlp)

    return before_gather


def adapt_run(original):
    if not enabled():
        return original

    @wraps(original)
    def run(layer, hidden_states, forward_batch, **kwargs):
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        from sglang.srt.models import deepseek_v4
        from sglang.srt.models.deepseek_common.moe_transport import (
            shared_expert_runs_locally,
        )
        from sglang.srt.runtime_context import get_forward, get_parallel
        from .joint_experts import can_join

        if (
            layer.config.model_type != "deepseek_v41"
            or hidden_states.device.type != "npu"
        ):
            return original(layer, hidden_states, forward_batch, **kwargs)
        token = _in_moe.set(True)
        try:
            if (
                get_moe_a2a_backend().is_none()
                and not deepseek_v4.is_cp_active(forward_batch)
                and moe_sequence_parallel_enabled(get_parallel())
            ):
                joint = can_join(layer.mlp)
                layout = getattr(forward_batch, "_dsv41_group_moe_rows", None)
                if layout is not None:
                    if layout.split_at_moe:
                        from . import group_cp2_dp2

                        shard, rows = group_cp2_dp2.interleave_shard(
                            hidden_states, role="mixed_moe"
                        )
                        ids = kwargs.get("input_ids")
                        if ids is not None:
                            ids, _ = group_cp2_dp2.interleave_shard(
                                ids, role="mixed_moe_ids"
                            )
                        with get_forward().scoped(npu_defer_moe_post=False):
                            result = _run_group_shards(
                                layer,
                                shard,
                                forward_batch,
                                layout,
                                joint=joint,
                                input_ids=ids,
                            )
                        return group_cp2_dp2.interleave_gather(
                            result, rows, role="mixed_moe_output"
                        )
                    return _run_group_shards(
                        layer,
                        hidden_states,
                        forward_batch,
                        layout,
                        input_ids=kwargs.get("input_ids"),
                        joint=joint,
                    )
                if _use_moe_sequence_parallel(
                    get_parallel(), forward_batch, joint
                ) and (
                    joint
                    or getattr(layer.mlp, "shared_experts", None) is None
                    or shared_expert_runs_locally(layer.mlp)
                ):
                    return _run_sequence_parallel(
                        layer, hidden_states, forward_batch, joint=joint, **kwargs
                    )
                if joint:
                    return _run_dp_joint(layer, hidden_states, forward_batch, **kwargs)
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


def _use_moe_sequence_parallel(parallel, batch, joint):
    from sglang.srt.environ import envs

    from .peer_collectives import enabled as peer_enabled

    return (
        batch.dp_padding_mode is not None
        and batch.dp_padding_mode.is_max_len()
        and (
            parallel.attn_tp_size == 2
            or peer_enabled()
            or (joint and envs.SGLANG_DP_USE_REDUCE_SCATTER.get())
        )
    )


def _joint_dp_rows(mlp, gathered, batch, rows, input_ids, input_ids_global):
    from sglang.srt.layers.dp_attention import dp_slot_in, get_dp_global_num_tokens

    counts = get_dp_global_num_tokens()
    slot = dp_slot_in(counts)
    if counts[slot] != rows or sum(counts) != gathered.shape[0]:
        raise ValueError("joint MoE owner rows disagree with the DP gather layout")
    if not gathered.shape[0]:
        return gathered, None
    first = sum(counts[:slot])
    return mlp(
        gathered,
        batch,
        input_ids=input_ids,
        input_ids_global=input_ids_global,
        skip_shared_experts=True,
        joint_shared_slice=slice(first, first + rows),
    )


def _run_dp_joint(
    layer,
    local,
    batch,
    *,
    input_ids=None,
    input_ids_global=None,
    return_moe_output=False,
):
    """Use the DP transport's published row layout, including unequal and idle DPs."""
    from sglang.srt.layers.moe.utils import post_experts_all_reduce
    from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport

    transport = MoeRowTransport(layer.mlp, local, batch, input_ids, input_ids_global)
    with transport.scope():
        routed, shared = _joint_dp_rows(
            layer.mlp,
            transport.input,
            batch,
            local.shape[0],
            transport.input_ids,
            transport.input_ids_global,
        )
        if not routed.shape[0]:
            return torch.empty_like(local)
        if not transport.mlp_reduce_scatter and layer.mlp.reduce_results:
            routed = post_experts_all_reduce(routed)
    output = transport.return_output(routed)
    return output + shared if shared is not None else output


def adapt_tbo_moe(original):
    if not enabled():
        return original

    @wraps(original)
    def run(layer, state):
        from sglang.srt.runtime_context import get_forward

        from .joint_experts import can_join

        if not can_join(layer.mlp):
            return original(layer, state)
        if state.pop("shared_local") is not None:
            raise RuntimeError("shared target expert ran before the TBO gather")
        batch = state.forward_batch
        ids = batch._tbo_global_input_ids
        with get_forward().scoped(mlp_reduce_scatter=True):
            state.global_expert_out, state.shared_local = _joint_dp_rows(
                layer.mlp,
                state.pop("global_hidden"),
                batch,
                batch.tbo_padded_len,
                ids,
                ids,
            )

    return run


def _workspace(role, x, rows):
    stream = torch.npu.current_stream(x.device).npu_stream
    return row_buffer(role, x.device, stream, x.shape[1], x.dtype).get(rows)


def moe_routed_output(hidden):
    if not _in_moe.get():
        return hidden.new_empty(hidden.shape, dtype=torch.bfloat16)
    stream = torch.npu.current_stream(hidden.device).npu_stream
    return row_buffer("moe_routed_out", hidden.device, stream, hidden.shape[1]).get(
        hidden.shape[0]
    )


def moe_routing_metadata(hidden, experts):
    if not _in_moe.get():
        return None
    stream = torch.npu.current_stream(hidden.device).npu_stream
    # Each layer consumes its plan on this stream before the next route.
    return row_buffer(
        "moe_route_metadata", hidden.device, stream, 4 * (experts + 1), torch.int32
    ).get(1)[0]


def _run_gathered_experts(mlp, gathered, batch, input_ids, global_ids, owner, peer):
    from sglang.srt.runtime_context import get_forward

    joint_args = {} if owner is None else {"joint_shared_slice": owner}
    with get_forward().scoped(
        mlp_reduce_scatter=True, npu_defer_route_combine=peer is not None
    ):
        return mlp(
            gathered,
            batch,
            input_ids=input_ids,
            input_ids_global=global_ids,
            skip_shared_experts=True,
            return_moe_output=False,
            **joint_args,
        )


def _reduce_scatter(group, output, partial, shared, owned_rows):
    group.reduce_scatter_tensor(output, partial)
    if shared is not None:
        # Add the replicated shared expert once on the owned token rows.
        output[:owned_rows].add_(shared)


def _run_compacted_group_shards(mlp, gathered, batch, layout, input_ids, owner, active):
    stride = layout.stride
    begin, end = active[0] * stride, (active[-1] + 1) * stride
    contiguous = end - begin == len(active) * stride
    global_ids = layout.global_input_ids
    if contiguous:
        hidden = gathered[begin:end]
        counts = layout.counts[active[0] : active[-1] + 1]
        if global_ids is not None:
            global_ids = global_ids[begin:end]
    else:
        hidden = _workspace("group_moe_active", gathered, len(active) * stride)
        ids = global_ids
        if ids is not None:
            global_ids = _workspace("group_moe_active_ids", ids, hidden.shape[0])
        for index, rank in enumerate(active):
            src = slice(rank * stride, (rank + 1) * stride)
            dst = slice(index * stride, (index + 1) * stride)
            hidden[dst].copy_(gathered[src])
            if ids is not None:
                global_ids[dst].copy_(ids[src])
        counts = torch.cat([layout.counts[rank : rank + 1] for rank in active])

    active_owner = None
    if owner is not None:
        owner_rank = owner.start // stride
        if owner_rank in active:
            start = active.index(owner_rank) * stride
            active_owner = slice(start, start + owner.stop - owner.start)
        else:
            active_owner = slice(0, 0)

    batch._dsv41_group_moe_rows = replace(
        layout, counts=counts, global_input_ids=global_ids
    )
    try:
        partial = _run_gathered_experts(
            mlp, hidden, batch, input_ids, global_ids, active_owner, None
        )
    finally:
        batch._dsv41_group_moe_rows = layout
    shared = None
    if owner is not None:
        partial, shared = partial

    # Gathered inputs have been consumed on this stream. Reuse their storage
    # for the full reduce-scatter layout, including zeroes for idle ranks.
    if contiguous:
        gathered[:begin].zero_()
        gathered[begin:end].copy_(partial)
        gathered[end:].zero_()
    else:
        index = 0
        for rank in range(layout.counts.numel()):
            block = gathered[rank * stride : (rank + 1) * stride]
            if index < len(active) and rank == active[index]:
                block.copy_(partial[index * stride : (index + 1) * stride])
                index += 1
            else:
                block.zero_()
    return gathered, shared


def _run_group_shards(
    layer,
    local,
    batch,
    layout,
    *,
    input_ids=None,
    joint=False,
):
    from sglang.srt.runtime_context import get_parallel

    from .peer_collectives import moe_communicator

    parallel = get_parallel()
    tp_group = parallel.tp_group
    rows, stride = local.shape[0], layout.stride
    if rows > stride:
        raise ValueError("MoE local rows exceed the EP gather stride")
    if not stride:
        return torch.empty_like(local)

    active = None
    if not batch.dp_padding_mode.is_max_len() and layout.rank_counts_cpu is not None:
        active = [
            rank for rank, count in enumerate(layout.rank_counts_cpu) if count > 0
        ]
        if not active:
            return torch.empty_like(local)
        if len(active) == parallel.tp_size:
            active = None

    shard = local
    if rows < stride:
        shard = _workspace("group_moe_padded_input", local, stride)
        shard[:rows].copy_(local)
        shard[rows:].zero_()
    gathered = _workspace("gather", local, stride * parallel.tp_size)
    peer = (
        moe_communicator(shard, tp_group, attention_tp=1)
        if batch.dp_padding_mode.is_max_len()
        else None
    )
    if peer is None:
        tp_group.all_gather_into_tensor(gathered, shard)
    else:
        peer.gather(shard, gathered)

    if (
        layout.global_input_ids is None
        and getattr(layer.mlp.gate, "e_score_correction_bias_vl", None) is not None
    ):
        if input_ids is None or input_ids.numel() != rows:
            raise ValueError("MoE image routing requires local token IDs")
        ids = input_ids.reshape(rows, 1)
        ids_shard = _workspace("group_moe_id_shard", ids, stride)
        ids_shard[:rows].copy_(ids)
        ids_shard[rows:].zero_()
        global_ids = _workspace("group_moe_ids", ids, gathered.shape[0])
        tp_group.all_gather_into_tensor(global_ids, ids_shard)
        layout.global_input_ids = global_ids

    shared = None
    owner = None
    if joint:
        start = parallel.tp_rank * stride
        owner = slice(start, start + rows)
    elif (
        rows
        and (active is None or parallel.tp_rank in active)
        and getattr(layer.mlp, "shared_experts", None) is not None
    ):
        shared = layer.mlp._forward_shared_experts(local)
    if active is not None:
        partial, joint_shared = _run_compacted_group_shards(
            layer.mlp, gathered, batch, layout, input_ids, owner, active
        )
        if joint:
            shared = joint_shared
    else:
        partial = _run_gathered_experts(
            layer.mlp, gathered, batch, input_ids, layout.global_input_ids, owner, peer
        )
        if joint:
            partial, shared = partial
    if peer is not None:
        return peer.finish(partial, shared, stride)
    output = local.new_empty((stride, local.shape[1]))
    _reduce_scatter(tp_group, output, partial, shared, rows)
    return output[:rows]


def _run_sequence_parallel(
    layer,
    local,
    batch,
    *,
    input_ids=None,
    input_ids_global=None,
    return_moe_output=False,
    joint=False,
):
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as geometry
    from sglang.srt.runtime_context import get_parallel

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
    shard = local if tp == 1 else local.narrow(0, min(start, rows), owned_rows)
    shard_input = shard
    if owned_rows != shard_rows and peer is None:
        shard_input = _workspace("moe_sp_padded_input", local, shard_rows)
        if owned_rows:
            shard_input[:owned_rows].copy_(shard)
        shard_input[owned_rows:].zero_()
    shared = None
    if (
        not joint
        and owned_rows
        and getattr(layer.mlp, "shared_experts", None) is not None
    ):
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
    owner = None
    if joint:
        first = parallel.attn_dp_rank * rows + min(start, rows)
        owner = slice(first, first + owned_rows)
    partial = _run_gathered_experts(
        layer.mlp, gathered, batch, input_ids, input_ids_global, owner, peer
    )
    if joint:
        partial, shared = partial
    if peer is not None:
        return peer.finish(partial, shared, rows)
    if padded_rows != rows:
        padded_partial = _workspace(
            "moe_sp_padded_partial", local, padded_rows * parallel.attn_dp_size
        ).view(parallel.attn_dp_size, padded_rows, -1)
        padded_partial[:, :rows].copy_(partial.view(parallel.attn_dp_size, rows, -1))
        padded_partial[:, rows:].zero_()
        partial = padded_partial.view(-1, local.shape[1])
    shard_output = (
        torch.empty_like(local)
        if tp == 1
        else _workspace("moe_sp_shard", local, shard_rows)
    )
    _reduce_scatter(parallel.tp_group, shard_output, partial, shared, owned_rows)
    if tp == 1:
        return shard_output
    output = torch.empty(
        (padded_rows, local.shape[1]), dtype=local.dtype, device=local.device
    )
    parallel.attn_tp_group.all_gather_into_tensor(output, shard_output)
    return output[:rows]
