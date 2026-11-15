"""Independent CPU references for compressed INT4/MXFP4 native weight layouts.

Rank encoding originated in AscendFlow d594f0c format v2. These NumPy-only
references generate expected payloads; production packing uses Ascend C.
The resident NZ order is independent of the kernel compute tile and startup chunk.
"""

import numpy as np

_RANK = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 15, 14, 13, 12, 11, 10, 9], dtype=np.uint8)


_BYTE_RANK = _RANK[np.arange(256) & 15] | (_RANK[np.arange(256) >> 4] << 4)


def _native_expert(codes, exponents, *, k, compact_scales=False):
    """Fixed FP16 NZ C0=16; only compressed N16-sized temporaries."""
    codes, exponents = np.asarray(codes), np.asarray(exponents)
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != k // 2:
        raise ValueError("uint8 [N,K/2] checkpoint codes required")
    n = codes.shape[0]
    if (
        n <= 0
        or n % 64
        or k <= 0
        or k % 32
        or exponents.dtype != np.uint8
        or exponents.shape != (n, k // 32)
    ):
        raise ValueError("complete K32/N64 and uint8 [N,K/32] scales required")
    if np.any((exponents < 104) | (exponents > 143)):
        raise ValueError("every scale must admit exact finite nonzero FP16 S/2")
    words = np.empty(n * k // 8, dtype=np.int32)
    payload = words.view(np.uint8).reshape(n // 16, k, 8)
    scales = np.empty(
        (n // 16, k // 32, 16), dtype=np.uint8 if compact_scales else np.float16
    )
    for block in range(n // 16):
        first = block * 16
        rank = _BYTE_RANK[codes[first : first + 16]]
        payload[block, 0::2] = ((rank[0::2] & 15) | ((rank[1::2] & 15) << 4)).T
        payload[block, 1::2] = ((rank[0::2] >> 4) | (rank[1::2] & 0xF0)).T
        exponent = exponents[first : first + 16].T.astype(np.int16)
        if compact_scales:
            high = (4 * (exponent - 113) + np.where(exponent < 114, 168, 0)).astype(
                np.uint8
            )
            scales[block] = high
        else:
            scales[block] = np.exp2(exponent - 128).astype(np.float16)
    return words, scales.reshape(-1)


def pack_experts(experts, *, n, k, count, compact_scales=False):
    """Canonical compressed [E,N/16,K,8], scale [E,N/16,K/32,16].

    No compute tile, packed panel or N-stripe parameter changes this format.
    compact_scales=True stores natural-order FP16 high bytes in each N16 scale row.
    """
    if min(n, k, count) <= 0 or k % 32 or n % 64:
        raise ValueError("positive experts, complete K32 and N64 required")
    words = np.empty((count, n * k // 8), dtype=np.int32)
    scales = np.empty(
        (count, n * k // 32), dtype=np.uint8 if compact_scales else np.float16
    )
    seen = 0
    for index, (codes, exponents) in enumerate(experts):
        if (
            index >= count
            or np.shape(codes) != (n, k // 2)
            or np.shape(exponents) != (n, k // 32)
        ):
            raise ValueError(
                "expert count or checkpoint shape differs from local geometry"
            )
        words[index], scales[index] = _native_expert(
            codes, exponents, k=k, compact_scales=compact_scales
        )
        seen += 1
    if seen != count:
        raise ValueError("expert iterator ended before the local expert count")
    return words, scales


def shard_expert(w13, s13, w2, s2, *, tp_rank=0, tp_size=1):
    """Select a TP intermediate shard BEFORE repacking checkpoint bytes.

    w13=[gate,up] has [2I,H/2] bytes, w2=[H,I/2]. Scales have
    [2I,H/32] and [H,I/32] E8M0 bytes. EP selects experts in the caller.
    This partitions expert intermediate dimensions across NPU ranks.
    """
    w13, s13, w2, s2 = (np.asarray(x) for x in (w13, s13, w2, s2))
    if any(x.dtype != np.uint8 or x.ndim != 2 for x in (w13, s13, w2, s2)):
        raise ValueError("checkpoint codes and E8M0 scales must be uint8 matrices")
    intermediate, hidden = w2.shape[1] * 2, w2.shape[0]
    if (
        hidden % 32
        or w13.shape != (2 * intermediate, hidden // 2)
        or s13.shape != (2 * intermediate, hidden // 32)
        or s2.shape != (hidden, intermediate // 32)
    ):
        raise ValueError("inconsistent gate/up/down checkpoint shapes")
    if tp_size <= 0 or not 0 <= tp_rank < tp_size or intermediate % (32 * tp_size):
        raise ValueError("TP must split I into complete 32-element scale groups")
    width = intermediate // tp_size
    start, end = tp_rank * width, (tp_rank + 1) * width
    # Slicing the concatenated 2I axis contiguously would mix gate and up ranks.
    gate_up = np.concatenate(
        (w13[start:end], w13[intermediate + start : intermediate + end])
    )
    gate_up_scale = np.concatenate(
        (s13[start:end], s13[intermediate + start : intermediate + end])
    )
    return (
        (gate_up, gate_up_scale),
        (w2[:, start // 2 : end // 2], s2[:, start // 32 : end // 32]),
    )


def pack_int4_group32(codes, scales):
    """Signed nibble checkpoint -> native FP16 NZ C0=16, with FP16 scales."""
    codes, scales = np.asarray(codes), np.asarray(scales)
    if codes.dtype != np.uint8 or codes.ndim != 2:
        raise ValueError("codes must be uint8 [N,K/2]")
    n, k = codes.shape[0], codes.shape[1] * 2
    if not n or not k or n % 64 or k % 32:
        raise ValueError("complete K32/N64 required")
    if (
        scales.dtype != np.float16
        or scales.shape != (n, k // 32)
        or not np.all(np.isfinite(scales) & (scales >= 0) & (scales <= 8188))
    ):
        raise ValueError("FP16 [N,K/32] scales in [0,8188] required")
    words = np.empty(n * k // 8, dtype=np.int32)
    payload = words.view(np.uint8).reshape(n // 16, k, 8)
    for block in range(n // 16):
        raw = codes[block * 16 : (block + 1) * 16]
        payload[block, 0::2] = ((raw[0::2] & 15) | ((raw[1::2] & 15) << 4)).T
        payload[block, 1::2] = ((raw[0::2] >> 4) | (raw[1::2] & 0xF0)).T
    factors = scales.reshape(n // 16, 16, k // 32).transpose(0, 2, 1).copy().reshape(-1)
    return words, factors
