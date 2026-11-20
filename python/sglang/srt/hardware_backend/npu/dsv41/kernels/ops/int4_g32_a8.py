"""Signed INT4 G32 expert weights and INT8 grouped GEMM on Ascend 910B."""

import numpy as np
import torch


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


class G32GroupedGemm:
    """Routed projection using the token codes shared with the TP1 expert."""

    def __init__(self, k, n, *, device):
        from ..runtime.native import load_library

        self.device = device
        library = load_library()
        self.op = library.int4_g32_a8
        self.scratch_bytes = library.int4_g32_a8_scratch(k, n, device.index)

    def _workspace(self, x):
        from sglang.srt.runtime_context import get_buffer

        stream = torch.npu.current_stream(self.device)
        return get_buffer(
            f"dsv41.int4_g32_a8.{self.device}.{stream.npu_stream}.{self.scratch_bytes}",
            lambda: x.new_empty(self.scratch_bytes, dtype=torch.uint8),
        )

    def from_quantized(self, quantized, scale, weights, routing, *, out):
        self.op(quantized, *weights, scale, routing, out, self._workspace(quantized))


def bind(activation, weights, token_scale, routing, *, out, workspace=None):
    """Bind outside capture. Route records are the same as NativeGroupedGemm.

    Each concurrently executing call needs its own workspace and output.
    Only routed rows are overwritten; the caller owns all graph addresses.
    """
    from ..runtime.native import load_library

    if activation.device.type != "npu" or activation.ndim != 2 or out.ndim != 2:
        raise ValueError("expected NPU activation and output matrices")
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("bind int4_g32_a8 before graph capture")
    lib = load_library()
    if workspace is None:
        size = lib.int4_g32_a8_scratch(
            activation.shape[1], out.shape[1], activation.device.index
        )
        workspace = activation.new_empty(size, dtype=torch.uint8)
    args = (activation, *weights, token_scale, routing, out, workspace)
    op = lib.int4_g32_a8
    op(*args, True)

    def invoke():
        op(*args)
        return out

    return invoke
