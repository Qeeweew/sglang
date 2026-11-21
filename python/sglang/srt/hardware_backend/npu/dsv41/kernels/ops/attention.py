"""Causal C0/C1/C2 and noncausal DSpark attention over paged BF16 KV."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    check_tensor as _tensor,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    load_library,
    require_disjoint,
)


class _NativeAttention:
    """Three DeepJIT dataflows specialized to the actual bound geometry.

    Each bind prepares the entry; launch borrows scratch from the NPU allocator.
    Replay may update every device metadata tensor.
    """

    def __init__(
        self,
        heads,
        mode,
        *,
        ratio=1,
        capacity=128,
        ring=False,
        window_rows=128,
        cores=0,
    ):
        if type(heads) is not int or heads <= 0 or heads > 64 or heads % 16:
            raise ValueError("heads must be a positive multiple of16 within capacity64")
        if capacity <= 0 or capacity % 128 or capacity > 4096:
            raise ValueError("capacity must be a positive multiple of128 within4096")
        if type(window_rows) is not int or window_rows < 1 or window_rows > capacity:
            raise ValueError("window_rows must fit attention capacity")
        if mode == 1 and window_rows > 128:
            raise ValueError("sparse direct window exceeds its128-row buffer capacity")
        if type(cores) is not int or cores < 0:
            raise ValueError("cores must be nonnegative;0 selects device capacity")
        self.heads, self.mode, self.ratio = heads, mode, ratio
        self.capacity, self.ring = capacity, bool(ring)
        self.window_rows, self.cores = window_rows, cores

    def empty(self, q, out):
        if q.ndim != 3 or q.shape[1] != self.heads or q.device.type != "npu":
            raise ValueError("empty Q must match BF16[0,H,D] on NPU")
        if not 0 < q.shape[2] <= 512 or q.shape[2] % 16:
            raise ValueError(
                "width must be a positive multiple of16 within capacity512"
            )
        _tensor(q, torch.bfloat16, q.shape, q.device, "Q")
        _tensor(out, q.dtype, q.shape, q.device, "output")

    def bind(
        self,
        q,
        kv,
        compressed,
        indices,
        table,
        compressed_table,
        cu,
        lengths,
        compressed_lengths,
        residual,
        window,
        sinks,
        out,
        *,
        verify_ring=None,
    ):
        if q.ndim != 3 or q.device.type != "npu":
            raise ValueError("requires BF16 Q[T,H,D] on NPU")
        device, tokens, batch, width = q.device, q.shape[0], table.shape[0], q.shape[2]
        if not 0 < width <= 512 or width % 16:
            raise ValueError(
                "width must be a positive multiple of16 within capacity512"
            )
        if kv.ndim != 4 or kv.shape[1] < 16 or kv.shape[1] % 16:
            raise ValueError("paged KV requires positive DMA-aligned page rows")
        page_rows = kv.shape[1]
        _tensor(q, torch.bfloat16, (tokens, self.heads, width), device, "Q")
        _tensor(out, q.dtype, q.shape, device, "output")
        _tensor(kv, q.dtype, (kv.shape[0], page_rows, 1, width), device, "KV")
        if kv.shape[0] < 1:
            raise ValueError(
                "nonempty attention requires a positive physical page pool"
            )
        _tensor(table, torch.int32, (batch, table.shape[1]), device, "page table")
        _tensor(cu, torch.int32, (batch + 1,), device, "Cu")
        _tensor(lengths, torch.int32, (batch,), device, "lengths")
        _tensor(sinks, torch.float32, (self.heads,), device, "sinks")
        if window is not None:
            _tensor(window, torch.int32, (tokens, 1), device, "window lengths")
        columns, selected_rows, capacity = 0, 0, self.capacity
        if self.mode == 1:
            columns, selected_rows = compressed_table.shape[1], indices.shape[-1]
            if selected_rows < 128 or selected_rows > 512 or selected_rows % 128:
                raise ValueError("selected rows must be aligned within capacity512")
            capacity = 128 + selected_rows
            _tensor(
                compressed,
                q.dtype,
                (compressed.shape[0], page_rows // self.ratio, 1, width),
                device,
                "compressed KV",
            )
            if compressed.shape[0] < 1:
                raise ValueError("requires positive compressed physical page pool")
            _tensor(indices, torch.int32, (tokens, 1, selected_rows), device, "indices")
            _tensor(
                compressed_table,
                torch.int32,
                (batch, columns),
                device,
                "compressed table",
            )
            _tensor(
                compressed_lengths, torch.int32, (batch,), device, "compressed lengths"
            )
            _tensor(residual, torch.int32, (batch,), device, "compressed residual")
        inputs = (
            q,
            kv,
            compressed,
            indices,
            table,
            compressed_table,
            cu,
            lengths,
            compressed_lengths,
            residual,
            window,
            sinks,
        )
        require_disjoint((out,), inputs)
        native = load_library()
        args = (
            *inputs,
            out,
            None,
            capacity,
            self.ratio,
            self.mode,
            self.ring,
            self.window_rows,
            self.cores,
            verify_ring,
        )
        native.prepare_attention(*args)
        launch = native.attention
        return lambda: launch(*args)


class WindowAttention:
    """BF16 causal C0/SWA128, FP32 sink/statistics, shared KV, TND queries."""

    def __init__(self, *, heads=64, window_rows=128, cores=0):
        capacity = (window_rows + 127) // 128 * 128
        self.kernel = _NativeAttention(
            heads, 0, capacity=capacity, window_rows=window_rows, cores=cores
        )

    def bind(
        self,
        q,
        kv,
        block_table,
        cu_seqlens_q,
        lengths,
        sinks,
        window_lengths,
        *,
        out,
        verify_ring=None,
    ):
        if q.shape[0] == 0:
            self.kernel.empty(q, out)
            return lambda: None
        if block_table.ndim != 2 or block_table.shape[1] < 1:
            raise ValueError(
                "nonempty attention requires a positive page-table capacity"
            )
        if cu_seqlens_q.numel() != block_table.shape[0] + 1:
            raise ValueError(
                "Cu must contain one boundary per request plus the terminal boundary"
            )
        return self.kernel.bind(
            q,
            kv,
            None,
            None,
            block_table,
            None,
            cu_seqlens_q,
            lengths,
            None,
            None,
            window_lengths,
            sinks,
            out,
            verify_ring=verify_ring,
        )

    def __call__(
        self,
        q,
        kv,
        block_table,
        cu_seqlens_q,
        lengths,
        sinks,
        window_lengths,
        *,
        out=None,
        verify_ring=None,
    ):
        out = q.new_empty(q.shape) if out is None else out
        self.bind(
            q,
            kv,
            block_table,
            cu_seqlens_q,
            lengths,
            sinks,
            window_lengths,
            out=out,
            verify_ring=verify_ring,
        )()
        return out


class SparseAttention:
    """BF16 C1/C2 + SWA128, joint softmax with one FP32 sink per head.

    Each instance fixes its ratio; C1 and C2 share one precompiled kernel.
    All compressed inputs remain explicit alongside SWA and sink tensors.
    """

    def __init__(
        self,
        *,
        ratio=1,
        window_reuse=False,
        heads=64,
        window_rows=128,
        cores=0,
    ):
        if ratio not in {1, 2}:
            raise ValueError("compressed attention ratio must be 1 or 2")
        self.kernel = _NativeAttention(
            heads,
            1,
            ratio=ratio,
            capacity=640,
            ring=window_reuse,
            window_rows=window_rows,
            cores=cores,
        )
        self.window_reuse = window_reuse

    def bind(
        self,
        q,
        kv,
        compressed_kv,
        indices,
        block_table,
        compressed_table,
        cu_seqlens_q,
        lengths,
        compressed_lengths,
        residual,
        window_lengths,
        sinks,
        *,
        out,
        verify_ring=None,
    ):
        if q.shape[0] == 0:
            self.kernel.empty(q, out)
            return lambda: None
        if self.window_reuse and block_table.shape[0] != 1:
            raise ValueError("window reuse requires a single request")
        if any(
            table.ndim != 2 or table.shape[1] < 1
            for table in (block_table, compressed_table)
        ):
            raise ValueError(
                "nonempty attention requires positive original/compressed page-table capacities"
            )
        if cu_seqlens_q.numel() != block_table.shape[0] + 1:
            raise ValueError(
                "Cu must contain one boundary per request plus the terminal boundary"
            )
        return self.kernel.bind(
            q,
            kv,
            compressed_kv,
            indices,
            block_table,
            compressed_table,
            cu_seqlens_q,
            lengths,
            compressed_lengths,
            residual,
            window_lengths,
            sinks,
            out,
            verify_ring=verify_ring,
        )

    def __call__(
        self,
        q,
        kv,
        compressed_kv,
        indices,
        block_table,
        compressed_table,
        cu_seqlens_q,
        lengths,
        compressed_lengths,
        residual,
        window_lengths,
        sinks,
        *,
        out=None,
        verify_ring=None,
    ):
        out = q.new_empty(q.shape) if out is None else out
        self.bind(
            q,
            kv,
            compressed_kv,
            indices,
            block_table,
            compressed_table,
            cu_seqlens_q,
            lengths,
            compressed_lengths,
            residual,
            window_lengths,
            sinks,
            out=out,
            verify_ring=verify_ring,
        )()
        return out


class DraftAttention:
    """BF16 noncausal DSpark attention, directly reading paged KV."""

    def __init__(self, *, capacity=256, heads=64, window_rows=128, cores=0):
        self.kernel = _NativeAttention(
            heads, 2, capacity=capacity, window_rows=window_rows, cores=cores
        )

    def bind(self, q, kv, table, cu, lengths, sinks, *, out):
        if q.shape[0] == 0:
            self.kernel.empty(q, out)
            return lambda: None
        if table.ndim != 2 or table.shape[1] < 1 or cu.numel() != table.shape[0] + 1:
            raise ValueError(
                "draft attention requires exact request boundaries and a positive page table"
            )
        return self.kernel.bind(
            q, kv, None, None, table, None, cu, lengths, None, None, None, sinks, out
        )

    def __call__(self, q, kv, table, cu, lengths, sinks, *, out=None):
        out = torch.empty_like(q) if out is None else out
        self.bind(q, kv, table, cu, lengths, sinks, out=out)()
        return out
