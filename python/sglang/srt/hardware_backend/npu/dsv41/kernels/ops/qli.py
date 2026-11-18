"""910B INT8 query/key lightning indexer with explicit plan-time DeepJIT preparation.

The plan owns fixed-address outputs, not launch scratch. It can upload CPU sequence
metadata or borrow the serving backend's live device tensors. Query ranges
are derived inside the kernel; there is no host scheduling table. Keep the
plan and captured input tensors alive with the Graph.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    load_library,
    record_inputs,
)


def _cpu_i32(value, name):
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError(f"{name} must originate on CPU")
    result = torch.as_tensor(value, dtype=torch.int64, device="cpu")
    if result.numel() and (
        int(result.min()) < -2147483648 or int(result.max()) > 2147483647
    ):
        raise ValueError(f"{name} exceeds INT32")
    return result.to(torch.int32).contiguous()


class QLIPlan:
    """A fixed-address TND/paged-K plan with bind-time Cube geometry.

    ``pages`` is a CPU INT32 [requests,pages_per_request] logical-to-physical
    page table. ``lengths`` counts *compressed* K tokens; residual is 0..ratio-1.
    Query row j sees floor((length*ratio+residual-q_rows+j+1)/ratio) keys.
    candidate_mode is 'off', 'source' (2048 block IDs per row), or 'consumer'.
    Scale and head weight tensors are FP16, with explicit FP16 rounding before FP32 head accumulation.
    Plans are not reentrant: use one plan per concurrently executing stream.
    """

    def __init__(
        self,
        cu_q,
        lengths,
        pages,
        *,
        ratio=1,
        residual=None,
        page_size=128,
        candidate_mode="off",
        cores=None,
        heads=32,
        depth=128,
        topk=512,
        device="npu:0",
        library_dir=None,
    ):
        if ratio not in (1, 2):
            raise ValueError("ratio must be 1 or 2")
        if page_size < 16 or page_size % 16:
            raise ValueError("page_size must be a positive multiple of 16")
        if candidate_mode not in ("off", "source", "consumer"):
            raise ValueError("unknown candidate_mode")
        if heads <= 0 or heads % 16 or depth <= 0 or depth % 32:
            raise ValueError("heads/depth violate MMAD or INT8 C0 alignment")
        if topk <= 0 or topk > 2048 or topk % 8:
            raise ValueError("topk must be a multiple of 8 in 1..2048")
        self.heads, self.depth, self.topk = heads, depth, topk
        self.ratio, self.page_size = ratio, page_size
        self.mode = {"source": 1, "consumer": 2, "off": 3}[candidate_mode]
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("QLI requires an NPU device")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        props = torch.npu.get_device_properties(self.device)
        available = props.cube_core_num
        self.cores = int(available if cores is None else cores)
        if self.cores < 1 or self.cores > available:
            raise ValueError(f"cores must be in 1..{available}")
        self._library = load_library(native_library_path(directory=library_dir))
        self._library.qli_geometry(heads, depth)
        self._launch = self._library.qli
        self._pages_shape = tuple(_cpu_i32(pages, "pages").shape)
        if len(self._pages_shape) != 2 or self._pages_shape[1] == 0:
            raise ValueError("pages must have shape [requests, positive page capacity]")
        self.batch = self._pages_shape[0]
        initial_cu = _cpu_i32(cu_q, "cu_q")
        if initial_cu.ndim != 1 or initial_cu.numel() != self.batch + 1:
            raise ValueError("cu_q must have requests+1 entries")
        self.rows = int(initial_cu[-1])
        if self.rows < 0:
            raise ValueError("negative query rows")
        self.query_tile_rows = self._library.qli_query_tile_rows(
            self.rows, self.cores, self.heads
        )
        self.key_splits = self._library.qli_key_splits(
            self.rows, self.cores, self.heads, self._pages_shape[1] * self.page_size
        )
        # Physical K/scale strides are known only at actual tensor binding.
        self._prepared_layouts = set()
        # Keep captured addresses fixed while uploading all sequence metadata
        # in one transfer. Align each view to 128 bytes for the native ABI.
        shapes = {
            "cu_q": (self.batch + 1,),
            "lengths": (self.batch,),
            "residual": (self.batch,),
            "pages": self._pages_shape,
        }
        offsets, size = {}, 0
        for name, shape in shapes.items():
            size = (size + 31) // 32 * 32
            count = 1
            for dimension in shape:
                count *= dimension
            offsets[name] = (size, count)
            size += count
        self._upload_cpu = torch.zeros(size, dtype=torch.int32, pin_memory=True)
        self._upload_npu = torch.empty(size, dtype=torch.int32, device=self.device)
        self._host_metadata = {}
        for name, shape in shapes.items():
            begin, count = offsets[name]
            self._host_metadata[name] = self._upload_cpu[begin : begin + count].view(
                shape
            )
            setattr(self, name, self._upload_npu[begin : begin + count].view(shape))
        self._control_size = offsets["pages"][0]
        self._bound_metadata = False
        self.indices = torch.empty(
            (self.rows, self.topk), dtype=torch.int32, device=self.device
        )
        self.candidate_out = (
            torch.empty((self.rows, 2048), dtype=torch.int32, device=self.device)
            if self.mode == 1
            else None
        )
        self.workspace = None
        self.workspace_bytes = self._library.qli_workspace_bytes(
            self.device.index,
            self.cores,
            self.heads,
            self.rows,
            self.topk,
            self.mode,
            self._pages_shape[1] * self.page_size,
        )
        self._last_inputs = ()
        self.update(cu_q, lengths, residual=residual, pages=pages)

    def update(self, cu_q, lengths, *, residual=None, pages=None):
        """Copy CPU metadata to stable NPU storage, outside capture.

        One packed copy is blocking on the calling stream. The plan owns its
        pinned staging buffer. All previous runs/replays reading this metadata
        must have completed before this call. Synchronization between
        independent streams is the caller's job; no global sync is inserted.
        ``rows`` and request count stay fixed; request row partitions may change.
        """
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("update metadata outside Graph capture")
        if self._bound_metadata:
            raise RuntimeError("bound device metadata must be updated by its owner")
        cu = _cpu_i32(cu_q, "cu_q")
        kl = _cpu_i32(lengths, "lengths")
        rs = (
            torch.zeros_like(kl) if residual is None else _cpu_i32(residual, "residual")
        )
        if (
            cu.shape != (self.batch + 1,)
            or kl.shape != (self.batch,)
            or rs.shape != kl.shape
        ):
            raise ValueError("metadata shape mismatch")
        if (
            int(cu[0]) != 0
            or int(cu[-1]) != self.rows
            or bool((cu[1:] < cu[:-1]).any())
        ):
            raise ValueError(
                "cu_q must start at 0, be monotonic and keep total rows fixed"
            )
        if bool((kl < 0).any()) or bool(((rs < 0) | (rs >= self.ratio)).any()):
            raise ValueError("invalid K lengths or compression residual")
        if bool((kl > self._pages_shape[1] * self.page_size).any()):
            raise ValueError("K length exceeds page table capacity")
        if bool((cu[1:] - cu[:-1] > kl.to(torch.int64) * self.ratio + rs).any()):
            raise ValueError("query length exceeds original K length")
        pg = self._pages_cpu if pages is None else _cpu_i32(pages, "pages")
        if tuple(pg.shape) != self._pages_shape:
            raise ValueError("page table shape changed")
        # Only used entries need valid physical IDs; unused table cells may be -1.
        used = []
        for b, length in enumerate(kl.tolist()):
            needed = (length + self.page_size - 1) // self.page_size
            row = pg[b, :needed]
            if bool((row < 0).any()):
                raise ValueError("negative physical page ID in used table")
            used.extend(row.tolist())
        self._max_page = max(used, default=-1)
        self._pages_cpu = pg.clone()
        self._lengths_cpu, self._cu_cpu, self._residual_cpu = (
            kl.clone(),
            cu.clone(),
            rs.clone(),
        )
        self._max_k = int(kl.max()) if self.batch else 0
        self._max_q = int((cu[1:] - cu[:-1]).max()) if self.batch else 0
        for name, value in (("cu_q", cu), ("lengths", kl), ("residual", rs)):
            self._host_metadata[name].copy_(value)
        if pages is not None:
            self._host_metadata["pages"].copy_(pg)
        count = self._upload_cpu.numel() if pages is not None else self._control_size
        self._upload_npu[:count].copy_(self._upload_cpu[:count], non_blocking=False)

    def bind_metadata(
        self, cu_q, lengths, pages, *, residual, page_count, max_key_length=None
    ):
        """Use caller-owned device metadata, without host reads or a schedule.

        Bind before capture and keep these tensors alive with the plan. The
        owner updates them in stream order and guarantees valid Cu, causal
        lengths, residuals and page IDs in [0, page_count). No data-dependent
        host validation is performed; the serving allocator owns those bounds.
        """
        if torch.npu.is_current_stream_capturing() or self._last_inputs:
            raise RuntimeError("bind device metadata before Graph capture")
        if type(page_count) is not int or page_count < 1:
            raise ValueError("page_count must be positive")
        for value, shape, name in (
            (cu_q, (self.batch + 1,), "cu_q"),
            (lengths, (self.batch,), "lengths"),
            (residual, (self.batch,), "residual"),
            (pages, self._pages_shape, "pages"),
        ):
            if (
                value.device != self.device
                or value.dtype != torch.int32
                or tuple(value.shape) != shape
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous INT32 {shape} on {self.device}"
                )
        self.cu_q, self.lengths, self.residual, self.pages = (
            cu_q,
            lengths,
            residual,
            pages,
        )
        self._max_page = page_count - 1
        self._max_q = -1  # C++ derives the safe dynamic query bound from Q rows.
        self._max_k = -1 if max_key_length is None else max_key_length
        self._bound_metadata = True

    def bind(
        self,
        q,
        k,
        weights,
        q_scale,
        k_scale,
        *,
        candidates=None,
        out=None,
        candidate_out=None,
    ):
        """Prepare observed physical K/scale strides without executing the kernel."""
        output = self.indices if out is None else out
        candidate_output = (
            self.candidate_out if candidate_out is None else candidate_out
        )
        args = (
            q,
            k,
            weights,
            q_scale,
            k_scale,
            self.cu_q,
            self.lengths,
            self.residual,
            self.pages,
            candidates,
            output,
            candidate_output,
            self.workspace,
        )
        if self.rows:
            self._library.prepare_run_qli(
                *args,
                self.ratio,
                self.mode,
                self._max_q,
                self._max_k,
                self._max_page,
                self.cores,
            )
            self._prepared_layouts.add(
                (k.shape[1], self.pages.shape[1], k.stride(0), k_scale.stride(0))
            )
        return self

    def run(
        self,
        q,
        k,
        weights,
        q_scale,
        k_scale,
        *,
        candidates=None,
        out=None,
        candidate_out=None,
    ):
        """Enqueue INT8 QK/ReLU/head-reduction/topk on the current NPU stream.

        k: [physical_pages,page_size,depth]; scales k: [physical_pages,page_size].
        Only physical page stride may be padded; q, weights and q_scale are
        contiguous [rows,heads,depth] / [rows,heads]. Returns (indices,candidates_out).
        """
        output = self.indices if out is None else out
        candidate_output = (
            self.candidate_out if candidate_out is None else candidate_out
        )
        stream = torch.npu.current_stream(self.device)
        args = (
            q,
            k,
            weights,
            q_scale,
            k_scale,
            self.cu_q,
            self.lengths,
            self.residual,
            self.pages,
            candidates,
            output,
            candidate_output,
            self.workspace,
        )
        layout = (k.shape[1], self.pages.shape[1], k.stride(0), k_scale.stride(0))
        if self.rows and layout not in self._prepared_layouts:
            self._library.prepare_run_qli(
                *args,
                self.ratio,
                self.mode,
                self._max_q,
                self._max_k,
                self._max_page,
                self.cores,
            )
            self._prepared_layouts.add(layout)
        # Captured tensors must live as long as the Graph; eager stream
        # lifetimes are recorded inside the extension.
        record_inputs(self, args, stream, attribute="_last_inputs")
        if self.rows:
            self._launch(
                *args,
                self.ratio,
                self.mode,
                self._max_q,
                self._max_k,
                self._max_page,
                self.cores,
            )
        return output, candidate_output
