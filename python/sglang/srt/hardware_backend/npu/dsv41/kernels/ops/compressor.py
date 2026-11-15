"""V4.1 projected C1/C2 compressor with provisional output and explicit commit.

No framework registration, projection, RoPE or cache write is performed here.
Every layer/source owns its state; callers serialize operations on that state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    check_tensor,
    load_library,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    require_disjoint as _disjoint,
)


@dataclass
class CompressorState:
    """pending[slots,2,width] FP32 and committed_lengths[slots] INT64.

    At odd committed length the pending row is the final even-position token's
    (KV, score). At even length its payload is inactive and may retain old data.
    Seed both fields explicitly when restoring a nonempty prefix.
    """

    pending: torch.Tensor
    committed_lengths: torch.Tensor

    @classmethod
    def allocate(cls, slots, *, width=512, device="npu:0"):
        if type(slots) is not int or slots < 1:
            raise ValueError("state slot capacity must be positive")
        if type(width) is not int or width <= 0 or width > 4096 or width % 16:
            raise ValueError(
                "width must be a positive multiple of16 within UB capacity4096"
            )
        return cls(
            torch.zeros((slots, 2, width), dtype=torch.float32, device=device),
            torch.zeros(slots, dtype=torch.int64, device=device),
        )


@dataclass
class Provisional:
    """Token-aligned outputs. positions=-1 excludes incomplete pairs.

    C2 positions are original pair-completion positions (odd); logical compressed
    row is positions//2. rope_positions are pair starts (even), matching SGLang
    and vllm-ascend. Neither is a physical cache slot. Keep all inputs and this
    object alive while a graph using them is alive.
    """

    latent: torch.Tensor
    pooled: torch.Tensor
    positions: torch.Tensor
    rope_positions: torch.Tensor
    starts: torch.Tensor
    kv: torch.Tensor
    scores: torch.Tensor | None
    cu: torch.Tensor
    request_ids: torch.Tensor
    owner: object

    @property
    def completion_positions(self):
        return self.positions


class ProjectedCompressor:
    """C1(BF16 projection) / C2(FP32 projection+score), bind-stable width and FP32 gamma.

    prepare is read-only with respect to persistent state. commit accepts a
    prefix length per request and publishes only that prefix. No device metadata
    is copied to the host. Cu must be a partition of all T rows, request IDs must
    be unique, and state/output lifetimes are caller-owned across streams/graphs.
    """

    def __init__(self, ratio, gamma, state, *, library=None):
        if ratio not in (1, 2):
            raise ValueError("V4.1 compressor ratio must be 1 or 2")
        if not isinstance(state, CompressorState):
            raise TypeError("state must be CompressorState")
        self.ratio, self.gamma, self.state = ratio, gamma, state
        self.device = gamma.device
        if self.device.type != "npu":
            raise ValueError("compressor requires NPU tensors")
        self.slots = state.committed_lengths.numel()
        if (
            gamma.ndim != 1
            or gamma.numel() <= 0
            or gamma.numel() > 4096
            or gamma.numel() % 16
        ):
            raise ValueError("gamma width must be aligned within UB capacity4096")
        self.width = gamma.numel()
        check_tensor(gamma, torch.float32, (self.width,), self.device, "gamma")
        check_tensor(
            state.pending,
            torch.float32,
            (self.slots, 2, self.width),
            self.device,
            "pending",
        )
        check_tensor(
            state.committed_lengths,
            torch.int64,
            (self.slots,),
            self.device,
            "committed_lengths",
        )
        _disjoint((gamma, state.pending, state.committed_lengths))
        self.library_path = Path(native_library_path(library)).resolve()
        self.library = load_library(self.library_path)
        self._worker_budget = torch.npu.get_device_properties(
            self.device
        ).vector_core_num
        self._prepared_workers = set()

    def prepare(
        self,
        kv,
        scores,
        cu,
        request_ids,
        *,
        out=None,
        pooled=None,
        positions=None,
        rope_positions=None,
        starts=None,
    ):
        tokens, batches = kv.shape[0], request_ids.numel()
        if out is None:
            out = torch.empty(
                (tokens, self.width), device=self.device, dtype=torch.bfloat16
            )
        if pooled is None:
            pooled = torch.empty(
                (tokens, self.width), device=self.device, dtype=torch.float32
            )
        if positions is None:
            positions = torch.empty(tokens, device=self.device, dtype=torch.int64)
        if rope_positions is None:
            rope_positions = torch.empty(tokens, device=self.device, dtype=torch.int64)
        if starts is None:
            starts = torch.empty(batches, device=self.device, dtype=torch.int64)
        tensors = (
            kv,
            scores,
            self.gamma,
            cu,
            request_ids,
            self.state.pending,
            self.state.committed_lengths,
            pooled,
            out,
            positions,
            rope_positions,
            starts,
        )
        workers = min(self._worker_budget, max(tokens, batches))
        if workers and (False, workers) not in self._prepared_workers:
            self.library.prepare_compressor_prepare(*tensors, self.ratio)
            self._prepared_workers.add((False, workers))
        self.library.compressor_prepare(*tensors, self.ratio)
        return Provisional(
            out,
            pooled,
            positions,
            rope_positions,
            starts,
            kv,
            scores,
            cu,
            request_ids,
            self,
        )

    def commit(
        self,
        provisional,
        accepted_prefix,
        *,
        positions=None,
        rope_positions=None,
        status=None,
    ):
        """Return accepted_completion_positions[T], accepted_rope_positions[T], status[B].

        status 0=committed, 1=invalid accepted length, 2=stale start snapshot,
        3=invalid request/state ID. An error leaves that request's state untouched
        and masks all its accepted positions. Check statuses at a graph boundary;
        the forward path never synchronizes the device to inspect them.
        """
        if not isinstance(provisional, Provisional) or provisional.owner is not self:
            raise ValueError("provisional result belongs to another compressor")
        p = provisional
        tokens, batches = p.kv.shape[0], p.request_ids.numel()
        if positions is None:
            positions = torch.empty(tokens, device=self.device, dtype=torch.int64)
        if rope_positions is None:
            rope_positions = torch.empty(tokens, device=self.device, dtype=torch.int64)
        if status is None:
            status = torch.empty(batches, device=self.device, dtype=torch.int32)
        tensors = (
            p.kv,
            p.scores,
            p.cu,
            p.request_ids,
            p.starts,
            accepted_prefix,
            self.state.pending,
            self.state.committed_lengths,
            positions,
            rope_positions,
            status,
        )
        _disjoint((*tensors, p.latent, p.pooled, p.positions, p.rope_positions))
        workers = min(self._worker_budget, batches)
        if workers and (True, workers) not in self._prepared_workers:
            self.library.prepare_compressor_commit(*tensors, self.ratio)
            self._prepared_workers.add((True, workers))
        self.library.compressor_commit(*tensors, self.ratio)
        return positions, rope_positions, status
