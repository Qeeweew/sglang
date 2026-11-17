"""Opt-in DP4/TP2 or DP8/TP1 + EP8 communication for the serialized 910B decode path.

IPC buffers and compilation belong to eager startup/warmup. Device epochs are
shared by all shapes/graphs on the group; replays must use the same collective
order on both ranks. Concurrent model forwards require separate communicators.
"""

import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.npu_peer_collectives import npu_peer_all_reduce
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_buffer,
    get_forward,
    get_parallel,
    get_resources,
)


class PeerCommunicator:
    world = 2

    def __init__(self, cpu_group, device, attention_tp=2):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm peer all-reduce before NPU Graph capture")
        from .kernels.runtime.native import load_library

        if dist.get_world_size(cpu_group) != self.world:
            raise ValueError("peer all-reduce group size does not match communicator")
        self.cpu_group, self.device = cpu_group, torch.device(device)
        self.attention_tp = attention_tp
        self.handle = None
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=cpu_group)
        if any(host != hosts[0] for host in hosts):
            return  # Inter-host groups retain HCCL.
        status = [None] * self.world
        error = None
        try:
            load_library()
            self.handle = torch.classes.dsv41.PeerCommunicator(
                self.device.index, dist.get_rank(cpu_group), self.world, attention_tp
            )
            metadata = (self.handle.key(), self.handle.pid(), None)
        except Exception as exc:
            error = exc
            metadata = (None, None, str(exc))
        dist.all_gather_object(status, metadata, group=cpu_group)
        if any(item[2] for item in status):
            if self.handle is not None:
                self.handle.close()
                self.handle = None
            raise RuntimeError(
                f"peer all-reduce preparation failed: {[s[2] for s in status]}"
            ) from error
        pids = [item[1] for item in status]
        keys = [item[0] for item in status]
        # Propagate setup errors to both ranks. A local import/whitelist failure
        # must not strand the other rank in its next collective.
        for operation in [
            lambda: self.handle.allow(pids),
            lambda: self.handle.connect(keys),
        ]:
            error = None
            try:
                operation()
            except Exception as exc:
                error = str(exc)
            errors = [None] * self.world
            dist.all_gather_object(errors, error, group=cpu_group)
            if any(errors):
                self.handle.disconnect()
                dist.barrier(group=cpu_group)
                self.handle.close()
                self.handle = None
                raise RuntimeError(f"peer all-reduce IPC setup failed: {errors}")

    def close(self):
        """Collective; call after retiring graphs, before destroying cpu_group."""
        if self.handle is None:
            return
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        self.handle.disconnect()
        dist.barrier(group=self.cpu_group)
        self.handle.close()
        self.handle = None


class PeerAllReduce(PeerCommunicator):
    def all_reduce(self, input):
        output = torch.empty_like(input)
        return npu_peer_all_reduce(input, output, self.handle)


def try_all_reduce(input, group):
    # Attention TP ranks have the same padded shape. No local token-count gate:
    # that could send one rank to HCCL while its peer enters the device protocol.
    if (
        group.world_size != 2
        or input.dtype != torch.bfloat16
        or input.ndim != 2
        or input.shape[1] != 5120
        or not 0 < input.shape[0] <= 128
        or not input.is_contiguous()
        or get_forward().multi_stream
    ):
        return None
    key = f"dsv41_peer_all_reduce:{id(group.cpu_group)}:{input.device}"
    comm = get_buffer(key, lambda: PeerAllReduce(group.cpu_group, input.device))
    if comm.handle is None:
        return None
    if get_forward().npu_defer_attn_reduce:
        return DeferredAttention(comm, input)
    return comm.all_reduce(input)


def close_group_peer_all_reduce(cpu_group):
    buffers = get_resources().buffers
    for key, comm in list(buffers.items()):
        if isinstance(comm, PeerCommunicator) and comm.cpu_group is cpu_group:
            comm.close()
            del buffers[key]


def enabled():
    from .policy import enabled as native_enabled

    return native_enabled() and envs.SGLANG_OPT_DSV41_NPU_PEER_COMM.get()


@dataclass(frozen=True)
class RoutedOutput:
    values: torch.Tensor
    inverse: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class DeferredAttention:
    comm: PeerAllReduce
    partial: torch.Tensor


@dataclass(frozen=True)
class DeferredMoE:
    comm: PeerCommunicator
    routed: RoutedOutput
    shared: object
    rows: int


class PeerMoE(PeerCommunicator):
    world = 8

    def gather(self, input, output):
        from sglang.kernels.ops.communication.npu_peer_collectives import (
            npu_peer_moe_gather,
        )

        return npu_peer_moe_gather(input, output, self.handle)

    def finish(self, partial, shared, rows):
        from sglang.kernels.ops.communication.npu_peer_collectives import (
            npu_peer_moe_return,
        )

        if isinstance(partial, RoutedOutput) and get_forward().npu_defer_moe_post:
            return DeferredMoE(self, partial, shared, rows)
        output = torch.empty((rows, 5120), dtype=torch.bfloat16, device=self.device)
        if isinstance(partial, RoutedOutput):
            return npu_peer_moe_return(
                partial.values,
                output,
                self.handle,
                shared=shared,
                inverse=partial.inverse,
                ids=partial.ids,
                weights=partial.weights,
            )
        return npu_peer_moe_return(partial, output, self.handle, shared=shared)


def moe_communicator(input, group):
    if (
        not enabled()
        or input.device.type != "npu"
        or input.dtype != torch.bfloat16
        or not input.is_contiguous()
        or input.ndim != 2
        or input.shape[1] != 5120
        or not 0 < input.shape[0] <= 64 * get_parallel().attn_tp_size
        or get_forward().multi_stream
    ):
        return None
    tp = get_parallel().attn_tp_size
    key = f"dsv41_peer_moe:{id(group.cpu_group)}:{input.device}:tp{tp}"
    comm = get_buffer(key, lambda: PeerMoE(group.cpu_group, input.device, tp))
    return comm if comm.handle is not None else None


def post(deferred, residual, pre, post_mix, comb, weight):
    from sglang.kernels.ops.communication.npu_peer_collectives import npu_peer_post

    updated = torch.empty_like(residual)
    normalized = (
        residual.new_empty((residual.shape[0], 5120)) if weight is not None else None
    )
    if isinstance(deferred, DeferredAttention):
        args = (deferred.partial, None, None, None, None)
    else:
        routed = deferred.routed
        args = (
            routed.values,
            routed.inverse,
            routed.ids,
            routed.weights,
            deferred.shared,
        )
    npu_peer_post(
        args[0],
        residual,
        pre,
        post_mix,
        comb,
        weight,
        updated,
        normalized,
        deferred.comm.handle,
        inverse=args[1],
        ids=args[2],
        weights=args[3],
        shared=args[4],
    )
    return updated, normalized


def try_draft_stage(stage, positions, hidden_states, batch, previous_pre):
    """Close the draft's two mHC boundaries without exposing deferred tensors.

    DSpark deliberately uses its own plain stage runner rather than the target
    HcState runner. Its attention post can carry the FFN input norm; its FFN
    post must still return all four residual streams and the new pre-mix.
    """
    from .parallel import moe_sequence_parallel_enabled

    if (
        not enabled()
        or hidden_states.device.type != "npu"
        or hidden_states.dtype != torch.bfloat16
        or not 0 < hidden_states.shape[0] <= 64 * get_parallel().attn_tp_size
        or not moe_sequence_parallel_enabled(get_parallel())
        or get_forward().multi_stream
    ):
        return None
    from sglang.srt.models import deepseek_v4_mhc as mhc

    from .mhc import _norm_weight, _validate_config, operator

    _validate_config(stage.hc_cfg)
    if stage.hc_stats_stream is not None:
        return None
    x = mhc.combine(stage.attn_hc, mhc.HcState(hidden_states, previous_pre))
    with (
        stage.self_attn.maybe_use_decode_attn_tp(batch),
        get_forward().scoped(npu_defer_attn_reduce=True),
    ):
        y = stage.self_attn(positions, x, batch)
    pre, post_mix, comb = mhc.mix_stats(stage.attn_hc, hidden_states, None)
    weight = _norm_weight(stage.post_attention_layernorm)
    if isinstance(y, DeferredAttention):
        residual, normalized = post(y, hidden_states, pre, post_mix, comb, weight)
    else:
        residual, normalized = operator().post_combine(
            y, hidden_states, pre, post_mix, comb, weight
        )
    with get_forward().scoped(npu_defer_moe_post=True):
        y = stage._run_moe_ffn_dp_sync(
            normalized, batch, input_ids=None, input_ids_global=None
        )
    pre, post_mix, comb = mhc.mix_stats(stage.ffn_hc, residual, None)
    if isinstance(y, DeferredMoE):
        updated, _ = post(y, residual, pre, post_mix, comb, None)
    else:
        updated = operator().post(y, residual, post_mix, comb)
    return updated, pre
