"""V4.1 dense E4M3FN weights, 32x32 E8M0 scales, BF16 NPU storage.

This layout differs from Engram's per-row group-32 scales. The explicit loader
copies bounded compressed chunks, decodes directly into the final BF16 weight,
and synchronizes the caller stream before releasing mapped checkpoint storage.
"""

from __future__ import annotations

import json
import mmap
import struct
from functools import cache
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


def _bytes(tensor, kind):
    allowed = (
        {torch.uint8, torch.float8_e4m3fn}
        if kind == "codes"
        else {torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8)}
    )
    if tensor.dtype not in allowed or tensor.ndim != 2 or not tensor.is_contiguous():
        raise ValueError(
            f"{kind} must be a contiguous 2-D raw-byte or checkpoint FP8 tensor"
        )
    return tensor.view(torch.uint8)


@cache
def _tables(device):
    codes = (
        torch.arange(256, dtype=torch.uint8, device="cpu")
        .view(torch.float8_e4m3fn)
        .float()
    )
    scales = torch.tensor(
        [2.0 ** (i - 127) for i in range(255)] + [float("nan")], device="cpu"
    )
    with torch.npu.device(device):
        lut = torch.cat([codes, scales]).to(device)
        groups = (torch.arange(128, dtype=torch.int32, device="cpu") // 32 * 4).to(
            device
        )
        torch.npu.synchronize()
    return lut, groups


def _validate_out(out, shape, device):
    if (
        out.shape != shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.device != device
    ):
        raise ValueError(f"output must be contiguous BF16 [N,K] on {device}")


class FP8Weight:
    def __init__(self, library=None, *, device=None):
        self.lib = load_library(library)
        self.device = (
            torch.device("npu", torch.npu.current_device())
            if device is None
            else torch.device(device)
        )
        if self.device.type != "npu":
            raise ValueError("FP8 decoding requires a selected NPU")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.lut, self.groups = _tables(self.device)
        self.last_load = None

    def bind(self, codes, scales, *, out):
        """Prepare real shard geometry before capture and retain fixed tensors."""
        from functools import partial

        codes, scales = _bytes(codes, "codes"), _bytes(scales, "scales")
        args = (codes, scales, out, self.lut, self.groups)
        self.lib.prepare_fp8_weight(*args)
        return partial(self.lib.fp8_weight, *args)

    def convert(self, codes, scales, *, out=None):
        """Asynchronous current-stream decode; inputs are compressed NPU data."""
        codes, scales = _bytes(codes, "codes"), _bytes(scales, "scales")
        out = codes.new_empty(codes.shape, dtype=torch.bfloat16) if out is None else out
        self.lib.fp8_weight(codes, scales, out, self.lut, self.groups)
        return out

    def load_cpu(
        self, codes, scales, *, out=None, chunk_rows=None, staging_bytes=64 << 20
    ):
        """Startup-only bounded loader; returns a complete, ready BF16 weight.

        staging_bytes bounds explicit compressed NPU staging, excluding the
        final BF16 parameter and the tiny LUT. chunk_rows must align to 32.
        """
        codes, scales = _bytes(codes, "codes"), _bytes(scales, "scales")
        if codes.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("load_cpu requires CPU tensors")
        rows, cols = codes.shape
        if scales.shape != ((rows + 31) // 32, (cols + 31) // 32):
            raise ValueError("dense scales use 32x32 blocks, not Engram per-row groups")
        out = (
            torch.empty((rows, cols), dtype=torch.bfloat16, device=self.device)
            if out is None
            else out
        )
        _validate_out(out, codes.shape, self.device)
        if not out.numel():
            self.last_load = {
                "rows": rows,
                "cols": cols,
                "chunks": 0,
                "staging_bytes": 0,
            }
            return out
        scale_cols = (cols + 31) // 32
        if chunk_rows is None:
            groups = int(staging_bytes) // (32 * cols + scale_cols)
            if groups < 1:
                raise ValueError(
                    "staging budget cannot hold one 32-row compressed block"
                )
            chunk_rows = 32 * groups
        if chunk_rows <= 0 or chunk_rows % 32:
            raise ValueError("chunk_rows must be a positive multiple of 32")
        stage_rows = min(rows, int(chunk_rows))
        actual_bytes = stage_rows * cols + ((stage_rows + 31) // 32) * scale_cols
        if actual_bytes > staging_bytes:
            raise ValueError("requested chunk_rows exceeds staging budget")
        code_stage = torch.empty(
            (stage_rows, cols), dtype=torch.uint8, device=out.device
        )
        scale_stage = torch.empty(
            ((stage_rows + 31) // 32, scale_cols), dtype=torch.uint8, device=out.device
        )
        chunks = 0
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            count = stop - start
            sf_start, sf_stop = start // 32, (stop + 31) // 32
            code_stage[:count].copy_(codes[start:stop])
            scale_stage[: sf_stop - sf_start].copy_(scales[sf_start:sf_stop])
            self.convert(
                code_stage[:count],
                scale_stage[: sf_stop - sf_start],
                out=out[start:stop],
            )
            chunks += 1
        # Startup lifetime boundary: mmap/CPU views can be released on return.
        torch.npu.current_stream(self.device).synchronize()
        self.last_load = {
            "rows": rows,
            "cols": cols,
            "chunks": chunks,
            "chunk_rows": chunk_rows,
            "staging_bytes": actual_bytes,
        }
        return out

    def load_safetensors(self, directory, weight_key, scale_key=None, **options):
        """Load one real indexed checkpoint pair without full-payload CPU copies.

        Private writable mmap supplies zero-copy raw-byte CPU tensor views;
        the loader never mutates them or the checkpoint files. Different weight
        and scale shards are supported. No SGLang parameter-name translation.
        """
        directory = Path(directory).resolve()
        scale_key = (
            weight_key.removesuffix(".weight") + ".scale"
            if scale_key is None
            else scale_key
        )
        index = _index(directory / "model.safetensors.index.json")
        weight_path, scale_path = (
            directory / index[weight_key],
            directory / index[scale_key],
        )
        weight_header, weight_start = _header(weight_path)
        scale_header, scale_start = _header(scale_path)
        ws, ss = weight_header[weight_key], scale_header[scale_key]
        if ws["dtype"] != "F8_E4M3" or ss["dtype"] != "F8_E8M0":
            raise ValueError("expected checkpoint F8_E4M3 weight and F8_E8M0 scale")
        mappings = {}
        codes = scales = None
        try:
            for path in {weight_path, scale_path}:
                with path.open("rb") as source:
                    mappings[path] = mmap.mmap(
                        source.fileno(), 0, access=mmap.ACCESS_COPY
                    )
            codes = torch.frombuffer(
                mappings[weight_path],
                dtype=torch.uint8,
                count=ws["data_offsets"][1] - ws["data_offsets"][0],
                offset=weight_start + ws["data_offsets"][0],
            ).reshape(ws["shape"])
            scales = torch.frombuffer(
                mappings[scale_path],
                dtype=torch.uint8,
                count=ss["data_offsets"][1] - ss["data_offsets"][0],
                offset=scale_start + ss["data_offsets"][0],
            ).reshape(ss["shape"])
            return self.load_cpu(codes, scales, **options)
        finally:
            del codes, scales
            for mapping in mappings.values():
                mapping.close()


@cache
def _index(path):
    return json.loads(Path(path).read_text())["weight_map"]


@cache
def _header(path):
    with Path(path).open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(length))
    return header, 8 + length
