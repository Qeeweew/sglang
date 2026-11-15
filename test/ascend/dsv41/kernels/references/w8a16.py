"""Independent original-NK CPU dequantization and FP64 dot oracle."""

import torch


def dequantize(weight, scales, dtype):
    n, k = weight.shape
    # Use FP64 for the exact integer-times-half product, then explicitly model
    # the required FP16 storage boundary before the optional BF16 conversion.
    product = (
        weight.cpu().double().reshape(n, k // 256, 256)
        * scales.cpu().T.double()[..., None]
    )
    return product.half().reshape(n, k).to(dtype)


def reference(activation, weight, scales):
    x = activation.cpu()
    batched = x.ndim == 3
    batches = x.shape[1] if batched else 1
    n = weight.shape[0] // batches
    decoded = (
        dequantize(weight, scales, x.dtype).double().reshape(batches, n, x.shape[-1])
    )
    if not batched:
        return (x.double() @ decoded[0].T).to(x.dtype)
    return torch.stack(
        [x[:, batch].double() @ decoded[batch].T for batch in range(batches)], dim=1
    ).to(x.dtype)


def torch_npu_reference(activation, weight, scales):
    """BF16 scale and product, independently of the native FP16 boundary."""
    n, k = weight.shape
    product = (
        weight.cpu().double().reshape(n, k // 256, 256)
        * scales.cpu().T.double()[..., None]
    )
    decoded = product.bfloat16().reshape(n, k)
    return (activation.cpu().double() @ decoded.double().T).bfloat16()


def quantize_g256(x):
    """Absmax, FP16 stored scales, then FP32 RNE saturated to [-128,127].

    Scale is clamped to the minimum positive FP16 subnormal. Finite source
    groups must have absmax/127 <= 65504 so the requested FP16 scale is finite.
    """
    group_size = 256
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError("expected [rows,K] with K divisible by 256")
    values = x.detach().cpu().float().reshape(x.shape[0], -1, group_size)
    scale = (values.abs().amax(-1) / 127.0).clamp(min=2.0**-24).half()
    if not torch.isfinite(values).all() or not torch.isfinite(scale).all():
        raise ValueError("group scale must be finite and representable in FP16")
    quant = (values / scale.float()[..., None]).round().clamp(-128, 127).to(torch.int8)
    return quant.reshape(x.shape).contiguous(), scale.T.contiguous()
