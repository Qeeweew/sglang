"""V4.1 Engram on 910B: compressed host tables and graph-safe device operators.

Only the explicit ``SGLANG_DSV41_910B=1`` integration enables this patch. Host
backing is owned by the model until its graphs have been destroyed; loading
never expands a table, and forward never transfers a complete table to HBM.
"""

from __future__ import annotations

import logging
import os
from functools import cache

import torch
from torch import nn

logger = logging.getLogger(__name__)


@cache
def _ops(device):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import EngramOps

    return EngramOps(device=device)


class _CompressedHostTable:
    layout = "per_rank"

    def __init__(self, rows, device, *, group=None, layer_id=0):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import (
            EngramTable,
            HostBuffer,
        )

        self.group = group
        self.layout = "shared" if group is not None else "per_rank"
        self.inodes = None
        if group is None:
            self.codes = HostBuffer((rows, 256), device=device)
        else:
            fds, owner = [], None
            try:
                if group.rank_in_group == 0:
                    for suffix, width in (("codes", 256), ("scales", 8)):
                        fd = os.memfd_create(
                            f"dsv41-engram-{layer_id}-{suffix}", os.MFD_CLOEXEC
                        )
                        fds.append(fd)
                        os.ftruncate(fd, rows * width)
                    owner = (os.getpid(), fds)
                pid, originals = group.broadcast_object(owner, src=0)
                if group.rank_in_group != 0:
                    for fd in originals:
                        fds.append(os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR))
                self.inodes = tuple(os.fstat(fd).st_ino for fd in fds)
                self.codes = HostBuffer.from_shared_fd(
                    fds[0], (rows, 256), device=device
                )
                self.scales = HostBuffer.from_shared_fd(
                    fds[1], (rows, 8), device=device
                )
                group.barrier()
            finally:
                for fd in fds:
                    os.close(fd)
        try:
            if group is None:
                self.scales = HostBuffer((rows, 8), device=device)
            self.operator = EngramTable(self.codes, self.scales, device=device)
        except BaseException:
            if hasattr(self, "scales"):
                self.scales.close()
            self.codes.close()
            raise
        self.dirty = False
        self.loaded = set()

    def close(self):
        """Call only after destroying every graph referencing this table."""
        self.codes.close()
        self.scales.close()


def _embedding_init(self, num_embeddings, dim, layer_id):
    from sglang.srt.runtime_context import get_parallel

    nn.Module.__init__(self)
    if dim != 256:
        raise ValueError("V4.1 Engram requires 256-byte FP8 rows")
    parallel = get_parallel()
    self.dim, self.tp_size = dim, parallel.tp_size
    self.row_start = num_embeddings * parallel.tp_rank // self.tp_size
    self.rows = num_embeddings * (parallel.tp_rank + 1) // self.tp_size - self.row_start
    self.host_table = _CompressedHostTable(
        num_embeddings,
        torch.npu.current_device(),
        layer_id=layer_id,
        group=parallel.tp_group if self.tp_size > 1 else None,
    )
    self.weight = nn.Parameter(
        self.host_table.codes.tensor.view(torch.float8_e4m3fn), requires_grad=False
    )
    self.scale = nn.Parameter(
        self.host_table.scales.tensor.view(torch.float8_e8m0fnu), requires_grad=False
    )
    self.weight.weight_loader = self._load_rows
    self.scale.weight_loader = self._load_rows
    if parallel.tp_rank == 0:
        logger.info(
            "DeepSeek-V4.1 Engram layer %d: %.2f GiB compressed %s host table.",
            layer_id,
            num_embeddings * 264 / 2**30,
            self.host_table.layout,
        )
    logger.debug(
        "Engram layer %d: %.3f GiB, layout=%s, rank=%d/%d, inodes=%s",
        layer_id,
        num_embeddings * 264 / 2**30,
        self.host_table.layout,
        parallel.tp_rank,
        self.tp_size,
        self.host_table.inodes,
    )


def _load_rows(self, param, loaded_weight):
    if loaded_weight.device.type != "cpu":
        raise ValueError("Engram checkpoint bytes must be loaded directly from CPU")
    if param.shape != loaded_weight.shape:
        raise ValueError(
            f"Engram table shape {tuple(loaded_weight.shape)} != {tuple(param.shape)}"
        )
    # copy_ across FP8 dtypes could convert values. This contract copies bytes.
    rows = slice(self.row_start, self.row_start + self.rows)
    param.data[rows].view(torch.uint8).copy_(
        loaded_weight[rows].contiguous().view(torch.uint8)
    )
    self.host_table.loaded.add("weight" if param is self.weight else "scale")
    self.host_table.dirty = True


def _finish_load(self, label):
    if self.host_table.loaded != {"weight", "scale"}:
        raise RuntimeError(f"incomplete Engram table {label}: {self.host_table.loaded}")
    if self.host_table.group is not None:
        self.host_table.group.barrier()
    self.host_table.dirty = False


def _embedding_forward(self, indices, forward_batch=None, *, cp_all_tokens=False):
    return self.host_table.operator.gather(indices.contiguous())


def _init_history(self, num_req_slots, device):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import HashParameters

    if self.max_ngram_size != 4:
        raise ValueError("910B V4.1 Engram hash requires max_ngram_size=4")
    device = torch.device(device)
    if device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    self._dsv41_history = torch.zeros(
        (num_req_slots + 1, 3), device=device, dtype=torch.int32
    )
    self.history = self._dsv41_history
    self.pad_row = num_req_slots
    self._dsv41_parameters = HashParameters(
        self.token_map.cpu(),
        self.multipliers.cpu(),
        self.primes.cpu(),
        self.offsets.cpu(),
        pad_id=self.pad_id,
        image_id=self.image_token_id if self.image_token_id is not None else -1,
        device=device,
    )
    self._dsv41_ops = _ops(str(device))


def _hasher_forward(self, input_ids, batch):
    assert self.history is not None, "EngramHasher.init_history was not called"
    count, device = input_ids.numel(), input_ids.device
    if count == 0:
        return torch.empty((0, 2, 24), device=device, dtype=torch.int64)
    ids = input_ids.to(torch.int32).contiguous()
    positions = batch.positions.to(torch.int64).contiguous()
    slots = batch.req_pool_indices.to(torch.int32).contiguous()
    requests = slots.numel()
    mode = batch.forward_mode
    loc = batch.out_cache_loc
    active = (
        torch.ones(count, device=device, dtype=torch.int32)
        if loc is None
        else (loc != 0).to(torch.int32)
    )
    if mode.is_decode():
        if count != requests:
            raise ValueError("decode Engram expects one input token per request")
        cu = torch.arange(requests + 1, device=device, dtype=torch.int32)
        slots = torch.where(active != 0, slots, -1).contiguous()
        accepted = active
    elif mode.is_target_verify():
        block = int(batch.spec_info.draft_token_num)
        if count != requests * block:
            raise ValueError("verify Engram expects an equal block per request")
        cu = torch.arange(requests + 1, device=device, dtype=torch.int32) * block
        accepted = None
    elif mode.is_extend():
        lengths = batch.extend_seq_lens.to(torch.int32)
        # The terminal cu excludes graph padding; no device-to-host sum.
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(torch.int32)))
        slots = torch.where(lengths > 0, slots, -1).contiguous()
        supplied = batch.engram_history
        if supplied is not None:
            # Scheduler supplies exact predecessor rows after prefix hits or
            # slot reuse. Use a spare row for zero-length/padded requests.
            destinations = torch.where(slots >= 0, slots, self.pad_row).long()
            self.history[destinations] = supplied.to(torch.int32)
        active = active * (torch.arange(count, device=device) < cu[-1]).to(torch.int32)
        accepted = lengths
        if loc is not None:
            last = (cu[1:] - 1).clamp(0, count - 1).long()
            accepted = accepted * (loc[last] != 0).to(torch.int32)
    else:
        raise ValueError(f"unsupported Engram forward mode: {mode}")
    result = self._dsv41_ops.hash(
        ids,
        positions,
        cu.contiguous(),
        slots,
        self._dsv41_history,
        active.contiguous(),
        self._dsv41_parameters,
    )
    if accepted is not None:
        self._dsv41_ops.commit(
            ids, cu.contiguous(), slots, accepted.contiguous(), self._dsv41_history
        )
    # Layer-major native output makes the subsequent [:,layer] a contiguous
    # view. No transpose copy of the hash matrix is required.
    return result.permute(1, 0, 2)


def _commit_after_verify(self, verify_ids_2d, req_pool_indices, commit_lens):
    requests, block = verify_ids_2d.shape
    ids = verify_ids_2d.reshape(-1).to(torch.int32).contiguous()
    cu = torch.arange(requests + 1, device=ids.device, dtype=torch.int32) * block
    self._dsv41_ops.commit(
        ids,
        cu,
        req_pool_indices.to(torch.int32).contiguous(),
        commit_lens.to(torch.int32).contiguous(),
        self._dsv41_history,
    )


class EngramEmbedding(nn.Module):
    # Native registered host reads do not use the CUDA scheduler stager.
    shared_host_table_without_ats = False

    def __init__(self, num_embeddings, dim, layer_id, column_bounds, *, mxfp4=False):
        if mxfp4:
            raise ValueError("910B Engram requires the FP8/E8M0 checkpoint table")
        self.staged_by_scheduler = False
        self.mxfp4 = False
        _embedding_init(self, num_embeddings, dim, layer_id)

    _load_rows = _load_rows
    finish_load = _finish_load
    forward = _embedding_forward

    def _apply(self, fn, recurse=True):
        # Registered compressed host tables stay host-owned across model.to().
        return self


def gate(
    x, kv, q_weight, k_weight, eps, clamp_value, input_ids=None, image_token_id=None
):
    if eps != 1e-20 or clamp_value != 1e-6:
        raise ValueError("910B V4.1 Engram gate requires eps=1e-20, clamp=1e-6")
    active = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
    if input_ids is not None and image_token_id is not None:
        active = (input_ids != image_token_id).int().contiguous()
    return _ops(str(x.device)).gate(
        x.contiguous(),
        kv.contiguous(),
        q_weight.contiguous(),
        k_weight.contiguous(),
        active,
    )
