"""V4.1/910B checkpoint decoding: BF16 loading buffers, compressed experts.

This patch is enabled only by ``SGLANG_DSV41_910B=1``.  Keep the original FP8
configuration for checkpoint identification and expert routing, but construct
ordinary checkpoint linears through the G256 W8A16 method. Decode each pair on the
NPU before the upstream loader consumes it.  This bounds transient device
storage to one decoded matrix and 64 MiB of compressed staging; neither routed
experts nor the host Engram embedding tables enter this conversion path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from functools import wraps

import torch

from .policy import enabled

logger = logging.getLogger(__name__)
_MARKER = "_dsv41_bf16_dense"


def _is_v41(config) -> bool:
    return (
        getattr(config, "model_type", None) == "deepseek_v41"
        or getattr(getattr(config, "text_config", None), "model_type", None)
        == "deepseek_v41"
    )


def _is_compressed_host_weight(name: str) -> bool:
    return ".experts." in name or ".engram.embed." in name


def _scale_prefix(name: str) -> str | None:
    for suffix in (".scale", ".weight_scale_inv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def prepare_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    decoder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream matched FP8 pairs, preserving original names for HF remapping.

    The source may emit scale or weight first, including pairs in different
    safetensors shards.  Unrelated tensors pass through without copies.  A
    missing scale is an error instead of silently interpreting FP8 as BF16.
    ``decoder`` is injectable for CPU-only loader contract tests.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    converter = None
    converted = 0
    for name, tensor in weights:
        if _is_compressed_host_weight(name):
            yield name, tensor
            continue
        if name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
            prefix, kind = name[: -len(".weight")], "weight"
        elif (
            prefix := _scale_prefix(name)
        ) is not None and tensor.dtype == torch.float8_e8m0fnu:
            kind = "scale"
        else:
            yield name, tensor
            continue
        bucket = pending.setdefault(prefix, {})
        if kind in bucket:
            raise ValueError(f"duplicate V4.1 FP8 {kind}: {prefix}")
        # A streaming loader can reuse its CPU buffer after advancing.  Only
        # pending compressed tensors need ownership across iterator steps.
        if getattr(tensor, "_sglang_runai_streamer_tensor", False):
            tensor = tensor.clone()
        bucket[kind] = tensor
        if len(bucket) != 2:
            continue
        weight, scale = bucket["weight"], bucket["scale"]
        if weight.ndim != 2 or scale.shape != (
            (weight.shape[0] + 31) // 32,
            (weight.shape[1] + 31) // 32,
        ):
            raise ValueError(f"invalid V4.1 32x32 FP8 scales for {prefix}")
        if decoder is None:
            if converter is None:
                from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import FP8Weight

                converter = FP8Weight()
            decoded = converter.load_cpu(weight.contiguous(), scale.contiguous())
        else:
            decoded = decoder(weight, scale)
        if decoded.dtype != torch.bfloat16 or decoded.shape != weight.shape:
            raise ValueError(
                f"V4.1 FP8 decoder returned an invalid BF16 matrix: {prefix}"
            )
        del pending[prefix]
        converted += 1
        yield prefix + ".weight", decoded
    if pending:
        missing = [
            f"{name} ({'scale' if 'weight' in pair else 'weight'} missing)"
            for name, pair in pending.items()
        ]
        raise ValueError(
            "incomplete V4.1 FP8 checkpoint pairs: " + ", ".join(missing[:8])
        )
    logger.debug(
        "V4.1/910B decoded %d FP8 dense/shared checkpoint matrices to BF16 loading buffers",
        converted,
    )


def adapt_get_config(original_get_config):
    if not enabled():
        return original_get_config

    @wraps(original_get_config)
    def get_config(model_config, load_config):
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        config = original_get_config(model_config, load_config)
        if enabled() and _is_v41(model_config.hf_config):
            if not isinstance(config, Fp8Config) or config.weight_block_size != [
                32,
                32,
            ]:
                raise ValueError(
                    "V4.1/910B requires the source FP8 checkpoint with 32x32 scales"
                )
            if model_config.dtype != torch.bfloat16:
                raise ValueError(
                    "V4.1/910B dense/shared computation requires --dtype bfloat16"
                )
            setattr(config, _MARKER, True)
            config._dsv41_is_draft = model_config.is_draft_model
            text = getattr(
                model_config.hf_config, "text_config", model_config.hf_config
            )
            fmt = getattr(text, "dsv41_target_moe_format", None)
            if fmt not in (None, "quarot_h32_int4_g32_v1"):
                raise ValueError(f"unrecognized target MoE format: {fmt}")
            config._dsv41_g32 = bool(fmt) and not config._dsv41_is_draft
        return config

    return get_config


def adapt_get_method(original_get_method):
    if not enabled():
        return original_get_method

    @wraps(original_get_method)
    def get_method(config, layer, prefix):
        from sglang.srt.layers.linear import LinearBase

        if not getattr(config, _MARKER, False):
            return original_get_method(config, layer, prefix)

        if isinstance(layer, LinearBase):
            from sglang.srt.hardware_backend.npu.dsv41.linear import (
                SharedW8A8Method,
                W8A16LinearMethod,
            )

            if getattr(config, "_dsv41_g32", False) and ".shared_experts." in prefix:
                return SharedW8A8Method(interleaved=prefix.endswith(".down_proj"))
            return W8A16LinearMethod()
        if enabled():
            from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

            if isinstance(layer, FusedMoE):
                from sglang.srt.environ import envs
                from sglang.srt.hardware_backend.npu.dsv41.moe import (
                    ResidentMXFP4Method,
                )

                return ResidentMXFP4Method(
                    is_draft=config._dsv41_is_draft,
                    g32=getattr(config, "_dsv41_g32", False),
                    a8=envs.SGLANG_OPT_DSV41_NPU_TARGET_MOE_A8.get()
                    and not config._dsv41_is_draft,
                )
        return original_get_method(config, layer, prefix)

    return get_method


def adapt_prepare(original_prepare):
    if not enabled():
        return original_prepare

    @wraps(original_prepare)
    def prepare(weights, quant_config):
        if getattr(quant_config, _MARKER, False):
            return prepare_weights(weights)
        return original_prepare(weights, quant_config)

    return prepare
