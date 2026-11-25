"""Row ownership within a mirrored 910B attention group."""

from dataclasses import dataclass, field
from math import prod

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

from . import peer_collectives, policy


_REQUIRED_TOPOLOGY = (8, 4, 2, 1, 8, "none")


def _validate_topology(tp, dp, attn_tp, cp, ep, moe_a2a_backend):
    if (tp, dp, attn_tp, cp, ep, moe_a2a_backend) != _REQUIRED_TOPOLOGY:
        raise ValueError(
            "SGLANG_OPT_DSV41_GROUP_CP2_DP2 requires --tp-size 8 "
            "--attn-dp-size 4 --attn-cp-size 1 --ep-size 8 --moe-a2a-backend none"
        )


def _parallel_topology(parallel):
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    return (
        parallel.tp_size,
        parallel.attn_dp_size,
        parallel.attn_tp_size,
        parallel.attn_cp_size,
        parallel.moe_ep_size,
        get_moe_a2a_backend().value,
    )


def enabled(parallel=None):
    if not policy.enabled():
        return False
    flag = envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2
    if flag.get() != "1":
        return False
    if flag.is_set():
        if parallel is not None:
            _validate_topology(*_parallel_topology(parallel))
        return True
    target = get_parallel() if parallel is None else parallel
    if target is None or not hasattr(target, "tp_size"):
        return False
    return _parallel_topology(target) == _REQUIRED_TOPOLOGY


def validate_options(cfg):
    if not policy.enabled():
        return
    flag = envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2
    if flag.get() != "1":
        return
    topology = (
        cfg.tp_size,
        cfg.attn_dp_size,
        cfg.tp_size // max(1, cfg.attn_dp_size * cfg.attn_cp_size),
        cfg.attn_cp_size,
        cfg.ep_size,
        cfg.moe_a2a_backend,
    )
    valid_flags = not cfg.enable_prefill_cp and bool(cfg.enable_dp_lm_head)
    if not flag.is_set():
        if topology == _REQUIRED_TOPOLOGY and not valid_flags:
            flag.set("0")
        return
    _validate_topology(*topology)
    if not valid_flags:
        raise ValueError(
            "SGLANG_OPT_DSV41_GROUP_CP2_DP2 requires --enable-dp-lm-head "
            "and the standard attention groups (--enable-prefill-cp off)"
        )


def group():
    return get_parallel().attn_tp_group


def lane():
    return get_parallel().attn_tp_rank


def _buffer(role, tensor, rows):
    shape = (rows, *tensor.shape[1:])
    if tensor.device.type != "npu":
        return tensor.new_empty(shape)
    from .parallel import row_buffer

    stream = torch.npu.current_stream(tensor.device).npu_stream
    return (
        row_buffer(
            ("group_cp2_dp2", role),
            tensor.device,
            stream,
            prod(tensor.shape[1:]),
            tensor.dtype,
        )
        .get(rows)
        .view(shape)
    )


def _copy_padded(local, rows, pad_value):
    local[: rows.shape[0]].copy_(rows)
    if rows.shape[0] < local.shape[0]:
        local[rows.shape[0] :].fill_(pad_value)


def interleave_shard(tensor, *, role, pad_value=0):
    rows = tensor.shape[0]
    local = _buffer((role, "shard"), tensor, (rows + 1) // 2)
    _copy_padded(local, tensor[lane() :: 2], pad_value)
    return local, rows


def interleave_gather(local, orig_rows, *, role):
    gathered = batch_half_gather(local, 2 * local.shape[0], role=role)
    out = _buffer((role, "interleave"), local, gathered.shape[0])
    out.view(local.shape[0], 2, *local.shape[1:]).copy_(
        gathered.view(2, *local.shape).transpose(0, 1)
    )
    return out[:orig_rows]


def batch_half_slice(tensor, batch_size, rows_per_req, *, role, pad_value=0):
    rows = batch_size * rows_per_req
    shard_rows = ((batch_size + 1) // 2) * rows_per_req
    begin = lane() * shard_rows
    local = _buffer((role, "shard"), tensor, shard_rows)
    _copy_padded(local, tensor[begin : min(begin + shard_rows, rows)], pad_value)
    return local


def batch_half_gather(local, orig_rows, *, role):
    if not local.shape[0]:
        return local
    out = _buffer((role, "gather"), local, 2 * local.shape[0])
    group().all_gather_into_tensor(out, local)
    return out[:orig_rows]


def prefill_step(batch):
    return enabled() and (
        batch.is_extend_in_batch or batch.forward_mode.is_extend_without_speculative()
    )


def decode_step(batch):
    mode = batch.forward_mode
    return (
        enabled()
        and not getattr(batch, "is_extend_in_batch", False)
        and (mode.is_decode() or mode.is_target_verify() or mode.is_idle())
    )


@dataclass(frozen=True)
class DecodeRows:
    batch: int
    width: int

    @property
    def local_batch(self):
        return (self.batch + 1) // 2

    @property
    def rows(self):
        return self.batch * self.width

    @property
    def local_rows(self):
        return self.local_batch * self.width

    @property
    def peer_slice(self):
        return (
            slice(0, self.local_rows) if lane() else slice(self.local_rows, self.rows)
        )

    def slice(self, tensor, *, role, pad_value=0, per_request=False):
        width = 1 if per_request else self.width
        return batch_half_slice(
            tensor, self.batch, width, role=role, pad_value=pad_value
        )

    def gather(self, tensor, *, role):
        return self.gather_pair(tensor, None, role=role)[0]

    def gather_layers(
        self, pending, *, role, comm_role="forward", skip_own_output=False
    ):
        if not self.rows:
            return pending
        layers, _, features = pending.shape
        out = _buffer((role, "layers"), pending[0], layers * self.rows)
        out = out.view(layers, self.rows, features)
        gathered = peer_collectives.try_group_gather(
            pending,
            self.rows,
            group(),
            out=out,
            comm_role=comm_role,
            skip_own_output=skip_own_output,
        )
        if gathered is not None:
            return out

        gathered = batch_half_gather(pending.unsqueeze(0), 2, role=role)
        # HCCL returns lane-major storage. The cache uses layer-major rows.
        even_rows = 2 * self.local_rows
        if self.rows == even_rows:
            out.view(layers, 2, self.local_rows, features).copy_(
                gathered.transpose(0, 1)
            )
        else:
            out[:, : self.local_rows].copy_(gathered[0])
            out[:, self.local_rows :].copy_(
                gathered[1, :, : self.rows - self.local_rows]
            )
        return out

    def gather_pair(self, first, second, *, role, out_first=None, out_second=None):
        if self.rows:
            gathered = peer_collectives.try_group_gather(
                first,
                self.rows,
                group(),
                other=second,
                out=out_first,
                other_out=out_second,
                comm_role="forward",
            )
            if gathered is not None:
                return gathered

        if second is None:
            first = batch_half_gather(first, self.rows, role=role)
        else:
            packed = batch_half_gather(
                torch.cat((first, second), dim=-1), self.rows, role=role
            )
            first, second = packed.split((first.shape[-1], second.shape[-1]), dim=-1)
        first = first.contiguous() if out_first is None else out_first.copy_(first)
        if second is not None:
            second = (
                second.contiguous() if out_second is None else out_second.copy_(second)
            )
        return first, second

    def gather_projection(self, kv, scores, *, out_kv=None, out_scores=None):
        return self.gather_pair(
            kv,
            scores,
            role="decode_compressor",
            out_first=out_kv,
            out_second=out_scores,
        )


def prefill_hash_ids(hasher, batch):
    full_ids = hasher(batch.input_ids, batch)
    return interleave_shard(full_ids, role="engram_ids")[0]


def interleaved_counts(cu, rank):
    lane_offsets = (cu + 1 - rank) // 2
    return torch.diff(lane_offsets).tolist()


@dataclass(frozen=True)
class PrefillSubset:
    """An ordered group subset selected from the preceding interleaved layout."""

    local_indices: torch.Tensor
    gathered_indices: torch.Tensor
    stride: int

    @classmethod
    def from_rows(cls, rows, device):
        parts = (
            (rows % 2 == 0).nonzero().flatten(),
            (rows % 2 == 1).nonzero().flatten(),
        )
        stride = max(part.numel() for part in parts)
        restore = torch.empty(rows.numel(), dtype=torch.int64, device="cpu")
        for rank, part in enumerate(parts):
            restore[part] = torch.arange(part.numel(), device="cpu") + rank * stride
        local = rows[parts[lane()]].long() // 2
        return cls(local.to(device), restore.to(device), stride)

    def select(self, value):
        return value.index_select(0, self.local_indices)

    def gather(self, value, *, role):
        if not self.stride:
            return value
        send = _buffer((role, "subset"), value, self.stride)
        _copy_padded(send, value, 0)
        gathered = batch_half_gather(send, 2 * self.stride, role=role)
        return gathered.index_select(0, self.gathered_indices)

    def redistribute(self, value, *, role):
        if value is None:
            return None
        full = self.gather(self.select(value), role=role)
        return interleave_shard(full, role=role)[0]


def rank_counts(counts, shard_rows, *, rows_per_request=None, out):
    """Expand live DP counts in DP0/lane0, DP0/lane1, DP1/lane0, ... order."""
    lanes = out.view(-1, 2)
    if rows_per_request is None:
        first = (counts + 1) // 2
    else:
        first = counts.clamp_max(shard_rows)
    lanes[:, 0].copy_(first)
    lanes[:, 1].copy_(counts - first)


@dataclass
class MoeRows:
    stride: int
    counts: torch.Tensor
    split_at_moe: bool
    global_input_ids: torch.Tensor | None = None
    rank_counts_cpu: tuple[int, ...] | None = None
    # Compacted rows use dataclasses.replace and must derive their own mask.
    image_mask: torch.Tensor | None = field(default=None, init=False)


def prepare_moe_rows(
    batch,
    local,
    *,
    rows_per_request=None,
    counts=None,
    split_at_moe=False,
):
    from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

    from .moe import live_dp_counts

    parallel = get_parallel()
    stride = (local.shape[0] + 1) // 2 if split_at_moe else local.shape[0]
    rank_counts_cpu = None
    if not batch.dp_padding_mode.is_max_len():
        dp_counts_cpu = list(get_dp_global_num_tokens())
        group_rows = max(dp_counts_cpu)
        width = rows_per_request or 1
        stride = ((group_rows + 2 * width - 1) // (2 * width)) * width
        # Host geometry may include an odd lane's padding token. Use the
        # published real counts to identify idle ranks when available.
        real_counts = getattr(batch, "global_num_tokens_cpu", None) or dp_counts_cpu
        lanes = []
        for count in real_counts:
            first = (count + 1) // 2 if rows_per_request is None else min(count, stride)
            lanes.extend((first, count - first))
        rank_counts_cpu = tuple(lanes)
    live = live_dp_counts(local.device, parallel.attn_dp_size)
    counts = live if counts is None else counts.masked_fill(live == 0, 0)
    out = live_dp_counts(local.device, parallel.tp_size)
    rank_counts(counts, stride, rows_per_request=rows_per_request, out=out)
    batch._dsv41_group_moe_rows = MoeRows(
        stride, out, split_at_moe, rank_counts_cpu=rank_counts_cpu
    )
