"""INT8 G256 weights with BF16/FP16 activations and whole-K FP32 dots."""

from dataclasses import dataclass

BRIDGE_SLOTS = 2


def workspace_elements(workers, bn, bk):
    return workers * BRIDGE_SLOTS * bn * bk


def quantize_bf16_weight(weight, out, scales):
    from ..runtime.native import load_library

    load_library().quantize_g256_weight(weight, out, scales)


def repack_weight(weight, scales):
    """Startup only: ND INT8 [N,K], raw FP16 [K/256,N] -> N16 NZ streams.

    The result is ordinary ND storage with explicit axes.
    """
    import torch

    if weight.ndim != 2 or weight.dtype != torch.int8:
        raise ValueError("INT8 [N,K] weight required")
    n, k = weight.shape
    if n % 128 or k % 256 or not n or not k:
        raise ValueError("N128 and K256 alignment required")
    if scales.dtype != torch.float16 or tuple(scales.shape) != (k // 256, n):
        raise ValueError("raw FP16 [K/256,N] scales required")
    if (
        not bool(torch.isfinite(scales).all())
        or bool((scales < 0).any())
        or bool((scales > 65504 / 128).any())
    ):
        raise ValueError(
            "nonnegative finite scales with representable FP16 decoded weights required"
        )
    return (
        weight.reshape(n // 16, 16, k).permute(0, 2, 1).contiguous(),
        scales.reshape(k // 256, n // 16, 16).permute(1, 0, 2).contiguous(),
    )


@dataclass
class PreparedW8A16:
    input: object
    weight: object
    scales: object
    output: object
    workspace: object
    plan: tuple
    native: object

    def __call__(self):
        self.native.w8a16(
            self.input,
            self.weight,
            self.scales,
            self.output,
            self.workspace,
            *self.plan,
        )
        return self.output


class W8A16:
    """Bind before Graph capture. No activation quantization or folded scales."""

    def bind(
        self,
        activation,
        weight,
        scales,
        *,
        out=None,
        workspace=None,
        block_m=None,
        block_n=None,
        block_k=256,
    ):
        import torch
        from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
            load_library,
        )

        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("bind W8A16 and warm up before Graph capture")
        if activation.ndim not in (2, 3):
            raise ValueError("activation must be [M,K] or [M,B,K]")
        m = activation.shape[0]
        batches = activation.shape[1] if activation.ndim == 3 else 1
        if weight.ndim != 3 or weight.shape[0] % batches:
            raise ValueError("weight must be repacked [B*N/16,K,16]")
        n = weight.shape[0] // batches * 16
        if block_m is None:
            block_m = 16 if m <= 16 else 64 if m <= 64 else 256
        workers = torch.npu.get_device_properties(activation.device).cube_core_num
        if block_n is None:
            # Prefer the larger panel that still fills a first wave. BM256
            # needs BN128 to fit L0C; BK512 needs BN128 to fit double L1.
            columns = (256, 128, 64) if block_m <= 128 and block_k <= 256 else (128, 64)
            row_tiles = max(1, (m + block_m - 1) // block_m)
            block_n = next(
                (
                    bn
                    for bn in columns
                    if n % bn == 0 and row_tiles * batches * (n // bn) >= workers
                ),
                64,
            )
        plan = (block_m, block_n, block_k)
        if out is None:
            out = torch.empty(
                (m, batches, n) if activation.ndim == 3 else (m, n),
                device=activation.device,
                dtype=activation.dtype,
            )
        if workspace is None:
            workspace = torch.empty(
                workspace_elements(workers, block_n, block_k),
                device=activation.device,
                dtype=activation.dtype,
            )
        native = load_library()
        native.prepare_w8a16(activation, weight, scales, out, workspace, *plan)
        return PreparedW8A16(activation, weight, scales, out, workspace, plan, native)
