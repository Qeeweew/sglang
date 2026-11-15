"""Startup NPU repack into final P/S2 (FP16 or sign-tagged uint8 high bytes)."""

from __future__ import annotations

from math import gcd

import numpy as np
import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
    NativeOperator,
    load_library,
    record_inputs,
)


class MXFP4WeightPacker(NativeOperator):
    def __init__(self, k, n, *, panel_k=None, device="npu:0", compact_scales=True):
        super().__init__(device)
        # Startup UB capacity only; resident NZ order is independent of this chunk.
        if panel_k is None:
            panel_k = k if k <= 512 else gcd(k, 512)
        if (
            k <= 0
            or n <= 0
            or k % 32
            or not 32 <= panel_k <= 512
            or panel_k % 32
            or k % panel_k
            or n % 64
        ):
            raise ValueError(
                "packing requires exact group32 panels and complete N blocks"
            )
        self.k, self.n, self.pk = k, n, panel_k
        self.compact = bool(compact_scales)
        self.library = load_library()
        # Tiny shape-dependent UB gather tables; checkpoint bytes stay compressed.
        rank = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 15, 14, 13, 12, 11, 10, 9], dtype=np.uint16
        )
        byte = np.arange(256, dtype=np.uint16)
        rank_lut = rank[byte & 15] | (rank[byte >> 4] << 4)
        exponent = np.arange(256, dtype=np.int16)
        scale_lut = np.zeros(256, dtype=np.float16)
        scale_lut[104:144] = np.ldexp(np.ones(40), exponent[104:144] - 128)
        row_bytes = ((panel_k // 2 + 31) // 32) * 32
        count = panel_k // 2 * 4
        at = np.arange(count, dtype=np.uint32)
        source_ids = ((at % 4) * 4 * row_bytes + at // 4) * 2
        at = np.arange(panel_k * 4, dtype=np.uint32)
        row, col = at // 4, at % 4
        out_ids = ((row % 2 * count + row // 2 * 4 + col) * 2).astype(np.uint32)
        at = np.arange(panel_k // 32 * 16, dtype=np.uint32)
        column = at % 16
        scale_ids = (column * 32 + at // 16) * 2
        tables = np.concatenate(
            (
                rank_lut.view(np.uint8),
                scale_lut.view(np.uint8),
                source_ids.view(np.uint8),
                out_ids.view(np.uint8),
                scale_ids.view(np.uint8),
            )
        )
        self.tables = torch.from_numpy(tables).to(self.device)
        self.codes = torch.empty((n, k // 2), dtype=torch.uint8, device=self.device)
        self.scales = torch.empty((n, k // 32), dtype=torch.uint8, device=self.device)
        self.host_codes = torch.empty(
            self.codes.shape, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.host_scales = torch.empty(
            self.scales.shape, dtype=torch.uint8, device="cpu", pin_memory=True
        )

    def bind(self, codes, scales, *, packed, s2):
        """Prepare the bound packing layout before Graph capture."""
        from functools import partial

        args = (codes, scales, self.tables, packed, s2, self.pk)
        self.library.prepare_mxfp4_pack(*args)
        return partial(self.library.mxfp4_pack, *args)

    def __call__(self, codes, scales, *, packed, s2):
        if s2.dtype != (torch.uint8 if self.compact else torch.float16):
            raise ValueError("scale output dtype differs from startup packing layout")
        args = (codes, scales, self.tables, packed, s2)
        record_inputs(self, args, torch.npu.current_stream(self.device))
        self.library.mxfp4_pack(*args, self.pk)

    def load_cpu(self, codes, scales, *, packed, s2):
        """Synchronize before reusing pinned CPU staging; final outputs stay resident.

        Scale validation is on the bounded raw CPU shard, not expanded weights.
        No inference-time call or graph capture is performed by this loader.
        """
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("checkpoint repacking is startup-only")
        for tensor, shape in (
            (codes, self.host_codes.shape),
            (scales, self.host_scales.shape),
        ):
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8
                or tuple(tensor.shape) != tuple(shape)
            ):
                raise ValueError("load_cpu expects raw byte CPU checkpoint shards")
        exponents = scales.numpy()
        if exponents.min() < 104 or exponents.max() > 143:
            raise ValueError("every scale must admit exact finite nonzero FP16 S/2")
        self.host_codes.copy_(codes)
        self.host_scales.copy_(scales)
        self.codes.copy_(self.host_codes, non_blocking=True)
        self.scales.copy_(self.host_scales, non_blocking=True)
        self(self.codes, self.scales, packed=packed, s2=s2)
        torch.npu.current_stream(self.device).synchronize()
