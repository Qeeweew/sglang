"""Standalone per-request CED tail maps and byte-exact typed row movement."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
)


class TailMetadata(NamedTuple):
    row_map: torch.Tensor
    tail_cu: torch.Tensor
    request_ids: torch.Tensor
    live_count: torch.Tensor


def tail_capacity(cu_cpu, window=128):
    """Host scheduler helper; reads CPU metadata only, never device .item()."""
    if isinstance(cu_cpu, torch.Tensor) and cu_cpu.device.type != "cpu":
        raise ValueError("requires CPU cu")
    cu = torch.as_tensor(cu_cpu, dtype=torch.int64, device="cpu")
    if (
        cu.ndim != 1
        or not cu.numel()
        or int(cu[0]) != 0
        or bool((cu[1:] < cu[:-1]).any())
    ):
        raise ValueError("invalid cu")
    if window < 1 or int(cu[-1]) > 2147483647:
        raise ValueError("window/count exceeds INT32 contract")
    return int((cu[1:] - cu[:-1]).clamp_max(window).sum())


class CED(NativeOperator):
    def __init__(self, *, device="npu:0", library_dir=None):
        super().__init__(device)
        self._library = load_library(native_library_path(directory=library_dir))

    def bind_metadata(self, cu, *, window=128, capacity=None, out):
        from functools import partial

        capacity = out.row_map.numel() if capacity is None else capacity
        self._library.prepare_ced_metadata(cu, window, capacity, list(out))
        return partial(self.metadata, cu, window=window, capacity=capacity, out=out)

    def bind_gather(self, source, row_map, *, out, fill=0):
        from functools import partial

        self._library.prepare_ced_copy(source, row_map, out, False, fill)
        return partial(self.gather, source, row_map, out=out, fill=fill)

    def bind_scatter(self, updates, row_map, out):
        from functools import partial

        self._library.prepare_ced_copy(updates, row_map, out, True, 0)
        return partial(self.scatter, updates, row_map, out)

    def metadata(self, cu, *, window=128, capacity=None, out=None):
        """Build tail maps without reading device counts; spare rows are -1."""
        if out is not None and not isinstance(out, TailMetadata):
            raise ValueError("out must be TailMetadata")
        values = self._library.ced_metadata(
            cu,
            window,
            -1 if capacity is None else capacity,
            [] if out is None else list(out),
        )
        return TailMetadata(*values)

    def gather(self, source, row_map, *, out=None, fill=0):
        return self._library.ced_copy(source, row_map, out, False, fill)

    def scatter(self, updates, row_map, out):
        """Repeated destinations use the final update; other rows persist."""
        return self._library.ced_copy(updates, row_map, out, True, 0)
