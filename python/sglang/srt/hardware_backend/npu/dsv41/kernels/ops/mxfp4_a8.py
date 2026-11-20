"""Ascend 910B grouped GEMM for INT8 tokens and resident four-bit experts."""

from functools import cache

import numpy as np
import torch


@cache
def _byte_tables():
    magnitudes = np.array([0, 1, 2, 3, 4, 6, 8, 12])
    nibbles = np.arange(256)
    tables, changes = [], []
    for shift in range(5):
        reduced = magnitudes >> shift
        ranks = np.searchsorted(magnitudes, reduced)
        codes = np.concatenate((ranks, ranks | ((reduced != 0) << 3))).astype(np.uint8)
        tables.append(codes[nibbles & 15] | (codes[nibbles >> 4] << 4))
        changed = np.tile((reduced << shift) != magnitudes, 2)
        changes.append(changed[nibbles & 15] | changed[nibbles >> 4])
    return np.stack(tables), np.stack(changes)


def pack_expert(codes, scales, *, allow_rounding=True):
    """Pack one CPU checkpoint [N,K/2], [N,K/32] (raw E2M1/E8M0 bytes).

    Returns three CPU tensors, to stack on the expert axis. For under-range
    groups, truncate magnitude toward zero on the row's integer grid. Set
    allow_rounding=False to reject any non-exact result instead.
    """
    for value in (codes, scales):
        if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 2:
            raise ValueError("expected raw 2-D CPU uint8 checkpoint tensors")
    width, packed_k = codes.shape
    depth = packed_k * 2
    if (
        not 256 <= depth <= 8192
        or depth % 32
        or not 0 < width <= 65535
        or width % 64
        or scales.shape != (width, depth // 32)
    ):
        raise ValueError("unsupported checkpoint geometry")
    codes, scales = codes.numpy(), scales.numpy()
    stripe = next(size for size in (256, 192, 128, 64) if width % size == 0)
    packed = np.empty((width // stripe, depth // 32, stripe, 16), dtype=np.uint8)
    factors = np.empty(packed.shape[:2] + (stripe // 2,), dtype=np.uint8)
    channel = np.empty(width, dtype=np.float32)
    table, changes = _byte_tables()
    for start in range(0, width, 32):
        tile, offset = divmod(start, stripe)
        exps = scales[start : start + 32].astype(np.int16)
        if np.any(exps == 255):
            raise ValueError("NaN E8M0 exponent")
        base = np.maximum(exps.min(1), exps.max(1) - 3)
        if np.any(base < 2):
            raise ValueError("subnormal FP32 channel coefficient is not supported")
        channel[start : start + 32] = np.ldexp(np.float32(1), base - 128)
        delta = exps - base[:, None]
        raw = codes[start : start + 32].reshape(32, -1, 16)
        ranked = raw
        underflow = delta < 0
        if underflow.any():
            shift = np.minimum(-delta[underflow], 4)[:, None]
            affected = raw[underflow]
            if not allow_rounding and changes[shift, affected].any():
                raise ValueError(
                    f"non-exact INT8 embedding in channels {start}:{start + 32}"
                )
            ranked = raw.copy()
            ranked[underflow] = table[shift, affected]
        even = ranked[::2].transpose(1, 2, 0)
        odd = ranked[1::2].transpose(1, 2, 0)
        slab = packed[tile, :, offset : offset + 32]
        slab[:, ::2] = (even & 15) | (odd << 4)
        slab[:, 1::2] = (even >> 4) | (odd & 0xF0)
        neg = (-(1 << delta.clip(min=0))).astype(np.uint8).T
        factors[tile, :, offset // 2 : offset // 2 + 16] = (neg[:, ::2] & 15) | (
            neg[:, 1::2] << 4
        )
    panel = next(
        v for v in range(131072 // stripe // 32 * 32, 0, -32) if depth % v == 0
    )
    local = panel * stripe // 2
    quantum = next(v for v in range(16384, 0, -1024) if local % v == 0)
    chunks = packed.reshape(-1, 4, quantum // 8)
    byte = np.arange(256, dtype=np.uint32)
    digits = (byte & 15) | ((byte >> 4) << 16)
    terms = [
        ((digits & 0x70007) << mag) | ((digits & 0x80008) << (sign - 3))
        for mag, sign in ((0, 6), (3, 7), (8, 14), (11, 15))
    ]
    for start in range(0, len(chunks), 32):
        part = chunks[start : start + 32]
        word = (
            terms[0][part[:, 0]]
            | terms[1][part[:, 1]]
            | terms[2][part[:, 2]]
            | terms[3][part[:, 3]]
        )
        part.reshape(len(part), -1)[:] = word.view(np.uint8)
    return tuple(
        torch.from_numpy(value.reshape(-1)) for value in (packed, factors, channel)
    )


def quantize_tokens(activation):
    """Quantize finite FP16/BF16 [M,K] rows; returns INT8 codes and FP32 scales."""
    import torch_npu

    if (
        activation.device.type != "npu"
        or activation.ndim != 2
        or activation.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError("expected an FP16 or BF16 NPU token matrix")
    return torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)


def pack_g32(codes, metadata):
    width, packed = codes.shape
    depth = 2 * packed
    if (
        codes.device.type != "cpu"
        or metadata.device.type != "cpu"
        or codes.dtype != torch.uint8
        or metadata.dtype != torch.uint8
        or depth % 256
        or width % 64
        or metadata.shape != (width, depth // 64 + 4)
    ):
        raise ValueError("invalid signed INT4 G32 checkpoint")
    stripe = next(
        size
        for size in range(128 if depth < width else 256, 0, -64)
        if width % size == 0
    )
    groups = depth // 32
    panels = (
        codes.numpy().reshape(-1, stripe // 32, 32, groups, 16).transpose(0, 3, 1, 4, 2)
    )
    even, odd = panels[..., ::2], panels[..., 1::2]
    slab = np.empty(even.shape[:-1] + (2, 16), np.uint8)
    slab[..., 0, :] = (even & 15) | (odd << 4)
    slab[..., 1, :] = (even >> 4) | (odd & 240)
    data = metadata.numpy()
    nibbles = data[:, :-4].reshape(-1, stripe, groups // 2).transpose(0, 2, 1)
    even, odd = nibbles[..., ::2], nibbles[..., 1::2]
    factors = np.empty(even.shape[:-1] + (2, stripe // 2), np.uint8)
    factors[..., 0, :] = (even & 15) | (odd << 4)
    factors[..., 1, :] = (even >> 4) | (odd & 240)
    channel = data[:, -4:].copy().view("<f4").reshape(-1)
    if not np.all(np.isfinite(channel) & (channel > 0)):
        raise ValueError("invalid INT4 channel scale")
    return tuple(
        torch.from_numpy(value.reshape(-1)) for value in (slab, factors, channel)
    )


def h32_quant(x, interleaved, *, count=None, shared_rows=0):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )
    from sglang.srt.runtime_context import get_buffer

    rows, width = x.shape

    def indices():
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm H32 before capture")
        offsets = torch.arange(width, dtype=torch.int32, device="cpu") * 4
        return torch.stack(
            (
                offsets.reshape(-1, 8, 32).transpose(1, 2).reshape(-1),
                offsets.reshape(-1, 32, 8).transpose(1, 2).reshape(-1),
            )
        ).to(x.device)

    perm = get_buffer(f"dsv41.h32.{x.device}.{width}", indices)
    codes = x.new_empty((rows, width), dtype=torch.int8)
    scale = x.new_empty(rows, dtype=torch.float32)
    load_library().h32_quant(
        x.contiguous(), perm, codes, scale, interleaved, count, shared_rows
    )
    return codes, scale


class A8GroupedGemm:
    """Grouped INT8 Cube projection with prequantized and standalone inputs."""

    def __init__(self, k, n, *, experts, device, g32=False, interleaved=False):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
            load_library,
        )

        self.k, self.n, self.experts, self.device = k, n, experts, device
        library = load_library()
        self.op = library.int4_g32_a8 if g32 else library.mxfp4_a8
        self.g32, self.interleaved = g32, interleaved
        self.scratch_bytes = library.mxfp4_a8_scratch(k, n, device.index)
        self.prepared = False

    def _workspace(self, x):
        from sglang.srt.runtime_context import get_buffer

        stream = torch.npu.current_stream(self.device)
        return get_buffer(
            f"dsv41.a8.{self.device}.{stream.npu_stream}.{self.scratch_bytes}",
            lambda: x.new_empty(self.scratch_bytes, dtype=torch.uint8),
        )

    def from_quantized(self, quantized, scale, weights, routing, *, out):
        self.op(quantized, *weights, scale, routing, out, self._workspace(quantized))

    def bind(self, x, codes, factors, channel, routing, *, out, **tiling_hints):
        if channel.shape != (self.experts, self.n) or x.shape[1] != self.k:
            raise ValueError("A8 expert/input geometry mismatch")
        workspace = self._workspace(x)
        if not self.prepared:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm A8 expert projections before capture")
            self.op(
                x.new_empty((1, self.k), dtype=torch.int8),
                codes,
                factors,
                channel,
                x.new_empty(1, dtype=torch.float32),
                routing,
                out[:1],
                workspace,
                True,
            )
            self.prepared = True

        def invoke():
            quantized, token_scale = (
                h32_quant(x, self.interleaved) if self.g32 else quantize_tokens(x)
            )
            self.op(
                quantized, codes, factors, channel, token_scale, routing, out, workspace
            )
            return out

        return invoke


def bind(activation, weights, token_scale, routing, *, out, workspace=None, g32=False):
    """Bind outside capture. Route records are the same as NativeGroupedGemm.

    Each concurrently executing call needs its own workspace and output.
    Only routed rows are overwritten; the caller owns all graph addresses.
    """
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    if activation.device.type != "npu" or activation.ndim != 2 or out.ndim != 2:
        raise ValueError("expected NPU activation and output matrices")
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("bind mxfp4_a8 before graph capture")
    lib = load_library()
    if workspace is None:
        size = lib.mxfp4_a8_scratch(
            activation.shape[1], out.shape[1], activation.device.index
        )
        workspace = activation.new_empty(size, dtype=torch.uint8)
    args = (activation, *weights, token_scale, routing, out, workspace)
    op = lib.int4_g32_a8 if g32 else lib.mxfp4_a8
    op(*args, True)

    def invoke():
        op(*args)
        return out

    return invoke
