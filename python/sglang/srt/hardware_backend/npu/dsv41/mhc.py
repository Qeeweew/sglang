"""Run SGLang's V4.1 mHC state machine with offline 910B kernels."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache, wraps

import torch

from sglang.srt.hardware_backend.npu.dsv41.dense_weights import enabled
from sglang.srt.hardware_backend.npu.dsv41.kernels.config import V41

logger = logging.getLogger(__name__)
_prefill_fusions = ContextVar("dsv41_prefill_fusions", default=None)


@contextmanager
def prefill_fusions(enabled):
    """Enable prefill fusions; tensors keep independent output storage."""
    token = _prefill_fusions.set({} if enabled else None)
    try:
        yield
    finally:
        _prefill_fusions.reset(token)


def prefill_fusions_enabled():
    return _prefill_fusions.get() is not None


def prefill_fusion_state():
    return _prefill_fusions.get()


def _head_input(model, residual):
    if (
        enabled()
        and model.config.model_type == "deepseek_v41"
        and residual.device.type == "npu"
    ):
        return residual
    return residual.float()


def _head_combine(model, residual, pre, mult, dtype):
    from sglang.kernels.ops.layernorm.mhc import mhc_combine

    if prefill_fusions_enabled() and residual.device.type == "npu":
        # Fuse collapse and normalization, preserving the BF16 handoff.
        residual = residual.view(-1, mult, V41.hidden)
        return operator().combine(
            residual,
            pre,
            _norm_weight(model.norm),
        )
    return mhc_combine(residual, pre, mult, dtype)


def _head_norm(model, value):
    if prefill_fusions_enabled() and value.device.type == "npu":
        return value
    return model.norm(value)


@cache
def operator():
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import MHC

    return MHC()


def _supported(cfg, residual):
    return enabled() and residual.device.type == "npu" and cfg.pre_from_prev


def _validate_config(cfg):
    if (cfg.mult, cfg.hidden, cfg.sinkhorn_iters, cfg.eps, cfg.rms_eps) != (
        4,
        V41.hidden,
        20,
        1e-6,
        1e-20,
    ):
        raise ValueError(
            "V4.1/910B mHC requires H=4,D=5120,20 Sinkhorn iterations,eps=1e-6,rms_eps=1e-20"
        )


def cache_norm_weight(norm):
    """Refresh after checkpoint load; keep the BF16 parameter name unchanged."""
    if norm.variance_epsilon != 1e-20:
        raise ValueError("V4.1/910B mHC main RMSNorm requires epsilon=1e-20")
    norm._dsv41_mhc_norm_fp32 = norm.weight.detach().float().contiguous()
    return norm._dsv41_mhc_norm_fp32


def _norm_weight(norm):
    weight = getattr(norm, "_dsv41_mhc_norm_fp32", None)
    if weight is None:
        weight = cache_norm_weight(norm)
    return weight


def adapt_stats(original_stats):
    if not enabled():
        return original_stats

    @wraps(original_stats)
    def stats(hc, x):
        if not _supported(hc.cfg, x):
            return original_stats(hc, x)
        _validate_config(hc.cfg)
        return tuple(operator().stats(x.contiguous(), hc.fn, hc.scale, hc.base))

    return stats


def adapt_combine(original_combine):
    if not enabled():
        return original_combine

    @wraps(original_combine)
    def combine(hc, state, quantized):
        residual = state.residual
        if not _supported(hc.cfg, residual):
            return original_combine(hc, state, quantized)
        _validate_config(hc.cfg)
        if quantized is not None:
            raise ValueError("V4.1/910B mHC does not produce FP8 activations")
        return operator().combine(
            residual.contiguous(), state.pre, _norm_weight(hc.norm)
        )

    return combine


def adapt_post(original_post):
    if not enabled():
        return original_post

    @wraps(original_post)
    def post(cfg, x, residual, post_mix, comb):
        if not _supported(cfg, residual):
            return original_post(cfg, x, residual, post_mix, comb)
        _validate_config(cfg)
        return operator().post(
            x.contiguous(),
            residual.contiguous(),
            post_mix.contiguous(),
            comb.contiguous(),
        )

    return post


def adapt_post_fusion(original_post_fusion):
    if not enabled():
        return original_post_fusion

    @wraps(original_post_fusion)
    def post_fusion(hc, y, residual, coefficients, next):
        from sglang.srt.models import deepseek_v4_mhc as mhc

        if (
            _supported(hc.cfg, residual)
            and next is not None
            and (next.hc is not None)
            and next.hc.cfg.pre_from_prev
            and next.norm_fusable
            and (y.shape[0] >= 16)
        ):
            _validate_config(hc.cfg)
            _validate_config(next.hc.cfg)
            pre, post_mix, comb = coefficients
            updated, normalized = operator().post_combine(
                y.contiguous(),
                residual.contiguous(),
                pre.contiguous(),
                post_mix.contiguous(),
                comb.contiguous(),
                _norm_weight(next.norm),
            )
            return mhc.HcState(updated, pre, mhc.HcNormed(normalized))
        return original_post_fusion(hc, y, residual, coefficients, next)

    return post_fusion


def adapt_boundary(original_boundary):
    if not enabled():
        return original_boundary

    @wraps(original_boundary)
    def boundary(
        norm,
        *,
        accepts_mxfp8,
        hc=None,
        accepts_mxfp8_deepgemm=False,
        mega_moe_input=None,
        extend=False,
    ):
        if norm.weight.device.type == "npu" and (hc is None or hc.cfg.pre_from_prev):
            return original_boundary(
                norm,
                accepts_mxfp8=False,
                hc=hc,
                accepts_mxfp8_deepgemm=False,
                mega_moe_input=None,
                extend=extend,
            )
        return original_boundary(
            norm,
            accepts_mxfp8=accepts_mxfp8,
            hc=hc,
            accepts_mxfp8_deepgemm=accepts_mxfp8_deepgemm,
            mega_moe_input=mega_moe_input,
            extend=extend,
        )

    return boundary


def adapt_refresh(original_refresh):
    if not enabled():
        return original_refresh

    @wraps(original_refresh)
    def refresh(layer):
        result = original_refresh(layer)
        if (
            layer.hc_pre_from_prev_sublayer
            and layer.input_layernorm.weight.device.type == "npu"
        ):
            cache_norm_weight(layer.input_layernorm)
            cache_norm_weight(layer.post_attention_layernorm)
        return result

    return refresh


def adapt_head_collapse(original_head_collapse):
    if not enabled():
        return original_head_collapse

    @wraps(original_head_collapse)
    def head_collapse(x_flat, pre, hc, out_dtype):
        if (
            enabled()
            and x_flat.device.type == "npu"
            and (hc == 4)
            and (x_flat.ndim == 2)
            and (x_flat.shape[1] == 20480)
        ):
            if out_dtype != torch.bfloat16:
                raise ValueError("V4.1/910B mHC head output must be BF16")
            return operator().collapse(x_flat.contiguous(), pre.contiguous())
        return original_head_collapse(x_flat, pre, hc, out_dtype)

    return head_collapse


def adapt_broadcast(original_broadcast):
    if not enabled():
        return original_broadcast

    @wraps(original_broadcast)
    def broadcast(x, hc_mult):
        if (
            enabled()
            and x.device.type == "npu"
            and (hc_mult == 4)
            and (x.shape[-1] == V41.hidden)
        ):
            return x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
        return original_broadcast(x, hc_mult)

    return broadcast


def adapt_post_runner(original):
    if not enabled():
        return original

    @wraps(original)
    def run(hc, out, residual, *, stats_stream, next, world_size, precomputed=None):
        from sglang.srt.models import deepseek_v4_mhc as mhc

        from .peer_collectives import DeferredAttention, DeferredMoE
        from .peer_collectives import post as peer_post

        if isinstance(out, (DeferredAttention, DeferredMoE)):
            _validate_config(hc.cfg)
            pre, post_mix, comb = mhc._compute_triplet(
                hc, residual, stats_stream, precomputed
            )
            norm_weight = None
            if (
                next is not None
                and next.hc is not None
                and next.hc.cfg.pre_from_prev
                and next.norm_fusable
            ):
                _validate_config(next.hc.cfg)
                norm_weight = _norm_weight(next.norm)
            updated, normalized = peer_post(
                out, residual, pre, post_mix, comb, norm_weight
            )
            return mhc.HcState(
                updated,
                pre,
                mhc.HcNormed(normalized) if normalized is not None else None,
            )

        if (
            _supported(hc.cfg, residual)
            and isinstance(out, torch.Tensor)
            and (precomputed is None)
            and (0 < residual.shape[0] <= 10)
        ):
            _validate_config(hc.cfg)
            updated, triplet = operator().stats_post(
                out.contiguous(), residual.contiguous(), hc.fn, hc.scale, hc.base
            )
            return mhc.HcState(updated, triplet.pre)
        return original(
            hc,
            out,
            residual,
            stats_stream=stats_stream,
            next=next,
            world_size=world_size,
            precomputed=precomputed,
        )

    return run
