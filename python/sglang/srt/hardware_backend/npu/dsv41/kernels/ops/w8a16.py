"""INT8 G256 weights with BF16/FP16 activations and whole-K FP32 dots."""

from dataclasses import dataclass


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
        block_k=None,
    ):
        from ..runtime.native import load_library

        native = load_library()
        out, bm, bn, bk = native.prepare_w8a16(
            activation, weight, scales, out, workspace, block_m, block_n, block_k
        )
        plan = (bm, bn, bk)
        return PreparedW8A16(activation, weight, scales, out, workspace, plan, native)
