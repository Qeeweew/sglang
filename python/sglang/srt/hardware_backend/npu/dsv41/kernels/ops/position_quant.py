"""910B BF16 tail64 RoPE, Indexer INT8 quantization and paged cache writes.

Build explicitly. DeepJIT prepares finite entries before capture; warm calls retain prepared handles.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
    record_inputs,
)


class RopeTables(NamedTuple):
    cos: torch.Tensor
    sin: torch.Tensor
    valid: torch.Tensor
    slots: torch.Tensor | None


class PositionQuant(NativeOperator):
    def __init__(self, *, rotary_dim=64, device="npu:0", library_dir=None):
        super().__init__(device)
        self._library = load_library(native_library_path(directory=library_dir))
        if (
            type(rotary_dim) is not int
            or rotary_dim <= 0
            or rotary_dim > 64
            or rotary_dim % 16
        ):
            raise ValueError(
                "rotary_dim must be a positive multiple of16 within capacity64"
            )
        self.rotary_dim = rotary_dim
        self._prepared = set()
        self._run = self._library.position_quant
        self._scatter = self._library.cache_scatter
        offsets = torch.cat(
            (
                ((torch.arange(rotary_dim, dtype=torch.int32, device="cpu") ^ 1) * 4),
                ((torch.arange(rotary_dim, dtype=torch.int32, device="cpu") // 2) * 4),
            )
        )
        signs = torch.tensor(
            [-1.0, 1.0] * (rotary_dim // 2), dtype=torch.float32, device="cpu"
        )
        self.tables = torch.cat((offsets, signs.view(torch.int32))).to(self.device)
        self._last = ()

    def prepare(self, *args, operation=0, dtype=0):
        """Prepare actual tensor geometry and the host-selected workers before capture."""
        if not args:
            raise ValueError("prepare requires the actual operation tensors")
        # Request extents distinguish host preparation calls only. The C++
        # device key contains fixed layout plus actual selected workers, not rows.
        geometry = tuple(
            (value.dtype, tuple(value.shape), tuple(value.stride()))
            if isinstance(value, torch.Tensor)
            else value
            for value in args
        )
        alias = operation == 0 and args[0].data_ptr() == args[3].data_ptr()
        key = (operation, dtype, geometry, alias)
        if key not in self._prepared:
            prepare = (
                self._library.prepare_position_quant
                if operation in (0, 1)
                else self._library.prepare_cache_scatter
                if operation == 2
                else self._library.prepare_rope_cache_gather
            )
            with torch.npu.device(self.device):
                prepare(*args)
            self._prepared.add(key)

    def precompile(self, **configuration):
        """Compile a supplied actual layout/worker plan without launching."""
        with torch.npu.device(self.device):
            self._library.precompile_position_quant(**configuration)

    def _enqueue(self, args, fn, operation=0, dtype=0):
        self.prepare(*args, operation=operation, dtype=dtype)
        record_inputs(self, args, torch.npu.current_stream(self.device))
        fn(*args)

    def rope(self, x, cos, sin, *, inverse=False, out=None):
        """Rotate tail64 adjacent pairs; in-place out=x is supported."""
        out = torch.empty_like(x) if out is None else out
        self._enqueue((x, cos, sin, out, None, self.tables, 0, inverse), self._run)
        return out

    def gather_cos_sin(self, cos_cache, sin_cache, positions, *, slots=None, out=None):
        """Device positions -> compact RoPE rows, validity and safe cache slots."""
        if out is None:
            rows = positions.numel()
            out = RopeTables(
                cos_cache.new_empty((rows, cos_cache.shape[1])),
                sin_cache.new_empty((rows, sin_cache.shape[1])),
                positions.new_empty(rows, dtype=torch.int32),
                torch.empty_like(positions) if slots is not None else None,
            )
        if not isinstance(out, RopeTables):
            raise ValueError("out must be RopeTables")
        self._enqueue(
            (cos_cache, sin_cache, positions, slots, *out),
            self._library.rope_cache_gather,
            operation=3,
        )
        return out

    def _quantize(self, x, cos, sin, query, out, scale_out):
        out = torch.empty_like(x, dtype=torch.int8) if out is None else out
        if scale_out is None:
            scale_out = x.new_empty(
                x.shape[:-1], dtype=torch.float16 if query else torch.float32
            )
        self._enqueue(
            (x, cos, sin, out, scale_out, self.tables, 1 if query else 2, False),
            self._run,
            operation=1,
        )
        return out, scale_out

    def quantize_query(self, x, *, cos=None, sin=None, out=None, scale_out=None):
        return self._quantize(x, cos, sin, True, out, scale_out)

    def quantize_key(self, x, *, cos=None, sin=None, out=None, scale_out=None):
        return self._quantize(x, cos, sin, False, out, scale_out)

    def scatter_bf16(self, values, slots, cache):
        self._check(values, torch.bfloat16, name="values")
        self._enqueue(
            (values, None, slots, cache, None), self._scatter, operation=2, dtype=1
        )
        return cache

    def scatter_index(self, codes, scales, slots, key_cache, scale_cache):
        self._check(codes, torch.int8, name="codes")
        self._enqueue(
            (codes, scales, slots, key_cache, scale_cache),
            self._scatter,
            operation=2,
            dtype=0,
        )
        return key_cache
