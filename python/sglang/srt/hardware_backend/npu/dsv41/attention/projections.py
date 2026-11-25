"""V4.1/910B BF16 Q/K preparation and FP32 compressor projections.

The C2 GEMMs consume startup-repacked BF16 weights and publish FP32 directly.
JIT specializations are prepared before Graph capture; repacking occurs at load time.
"""

from __future__ import annotations

import logging
from functools import cache, wraps

import torch
from sglang.srt.hardware_backend.npu.dsv41.dense_weights import enabled
from sglang.srt.hardware_backend.npu.dsv41.kernels.config import V41

logger = logging.getLogger(__name__)


def _padded_rope_tables(freqs_cis):
    cached = getattr(freqs_cis, "_dsv41_cos_sin_padded", None)
    if cached is not None:
        return cached
    if freqs_cis.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("warm padded RoPE table before Graph capture")
    table = torch.view_as_real(freqs_cis).flatten(-2)
    limit = table.shape[0]
    half = table.shape[1] // 2
    cos = torch.cat((table[:, 0::2], table.new_ones((1, half))), dim=0).contiguous()
    sin = torch.cat((table[:, 1::2], table.new_zeros((1, half))), dim=0).contiguous()
    cached = (cos, sin, limit)
    freqs_cis._dsv41_cos_sin_padded = cached
    return cached


def gather_rope(freqs_cis, positions):
    """Gather from CUDA's compact complex64 table without full-length copies."""
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.position_quant import (
        RopeTables,
    )

    cos_pad, sin_pad, limit = _padded_rope_tables(freqs_cis)
    pos = positions.long()
    valid = (pos >= 0) & (pos < limit)
    idx = torch.where(valid, pos, limit)
    cos = cos_pad.index_select(0, idx)
    sin = sin_pad.index_select(0, idx)
    return RopeTables(cos, sin, valid.int(), None)


def expanded_rope(trig):
    """Share upstream's repeated lanes and the native compact inputs per forward."""
    cos = trig.cos.repeat_interleave(2, dim=-1).view(-1, 1, 1, 64)
    sin = trig.sin.repeat_interleave(2, dim=-1).view(-1, 1, 1, 64)
    inverse_sin = -sin
    # The native rotary consumes compact FP32 lanes. Preserve those references
    # through the upstream 64-lane interface instead of slicing/copying them
    # back at every layer. Negation stays inside the native inverse kernel.
    sin._dsv41_compact_rope = (trig.cos, trig.sin, False)
    inverse_sin._dsv41_compact_rope = (trig.cos, trig.sin, True)
    return (cos, sin), (cos, inverse_sin)


def prime_rope(attentions, forward_batch, positions):
    memo = {}
    for attn in attentions:
        key = id(attn.freqs_cis)
        if key not in memo:
            trig = gather_rope(attn.freqs_cis, positions.long())
            memo[key] = (positions, trig, expanded_rope(trig))
    forward_batch._dsv41_rope_memo = memo


def cached_rope(freqs_cis, positions, forward_batch):
    """Reuse only this forward's memo with the same position tensor view."""
    entry = getattr(forward_batch, "_dsv41_rope_memo", {}).get(id(freqs_cis))
    if entry is not None and entry[0] is positions:
        return entry[1]
    return gather_rope(freqs_cis, positions)


def compact_rope_storage(model):
    """Drop NPU adapter/legacy full-table copies after model construction."""
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

    before = torch.npu.memory_allocated()
    seen = set()
    for module in model.modules():
        freqs = getattr(module, "freqs_cis", None)
        if not isinstance(freqs, torch.Tensor) or freqs.device.type != "npu":
            continue
        seen.add(freqs.untyped_storage().data_ptr())
        owner = getattr(module, "rotary_emb", None)
        if owner is not None:
            for name in tuple(owner._buffers):
                if "cos_sin_cache" in name or name.startswith("_npu_interleaved_rope_"):
                    delattr(owner, name)
            for name in ("cos_cached_total", "sin_cached_total"):
                if hasattr(owner, name):
                    setattr(owner, name, None)
        cached = Dsv4NpuRoPE._instances.get(id(freqs))
        if cached is not None:
            cached._real_imag = None
            cached._tables.clear()
        for name in ("_dsv41_cos", "_dsv41_sin"):
            if name in module._buffers:
                delattr(module, name)
    logger.debug(
        "V4.1 compact RoPE: %d shared frequency tables; released %.3f GiB full-table copies",
        len(seen),
        (before - torch.npu.memory_allocated()) / 2**30,
    )


@cache
def compressor_projection():
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import DenseProjection

    return DenseProjection(k=V41.hidden, columns=V41.head_dim, out_dtype="f32")


@cache
def wo_a_operator(groups=V41.wo_groups):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import WoAProjection

    return WoAProjection(groups=groups)


@cache
def position_operator(device):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import PositionQuant

    return PositionQuant(device=device)


def prepare_compressor(module):
    """Load-time repack only; keep source parameters for checkpoint bookkeeping."""
    if module.compress_ratio != 2:
        return
    if module.use_fused_gate:
        raise ValueError(
            "V4.1/910B requires separate C2 wkv and wgate checkpoint weights"
        )
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import repack_linear

    for suffix in ("wkv", "wgate"):
        weight = getattr(module, suffix).weight
        if (
            weight.shape != (V41.head_dim, V41.hidden)
            or weight.dtype != torch.bfloat16
            or weight.device.type != "npu"
        ):
            raise ValueError(
                "V4.1/910B compressor weights must be BF16 [512,5120] on NPU"
            )
        value = repack_linear(weight.detach())
        name = "_dsv41_packed_" + suffix
        if name in module._buffers:
            setattr(module, name, value)
        else:
            module.register_buffer(name, value, persistent=False)


def adapt_kernel_heads(original_kernel_heads):
    if not enabled():
        return original_kernel_heads

    @wraps(original_kernel_heads)
    def kernel_heads(self, num_tokens):
        if enabled() and self.is_dsv41:
            if self.n_local_heads not in (16, 32, 64):
                raise ValueError("V4.1/910B requires attention TP1/2/4")
            return self.n_local_heads
        return original_kernel_heads(self, num_tokens)

    return kernel_heads


def adapt_wo_a(original_wo_a):
    if not enabled():
        return original_wo_a

    @wraps(original_wo_a)
    def wo_a(o, weight, *args, **kwargs):
        if enabled() and weight.dtype == torch.int8:
            from sglang.srt.hardware_backend.npu.dsv41.linear import project_wo_a

            return project_wo_a(o, weight)
        if (
            enabled()
            and o.device.type == "npu"
            and (o.dtype == weight.dtype == torch.bfloat16)
            and (o.ndim == 3)
            and (o.shape[1] in (2, 4, 8))
            and (o.shape[2] == V41.wo_k)
            and (weight.shape == (o.shape[1], V41.wo_rank, V41.wo_k))
        ):
            if kwargs.get("fuse_mxfp8_quant") or kwargs.get("emit_fp8"):
                raise ValueError("V4.1/910B wo_a requires BF16 output")
            return wo_a_operator(o.shape[1])(o, weight)
        return original_wo_a(o, weight, *args, **kwargs)

    return wo_a


@cache
def _split_rms_operator():
    from ..kernels.runtime.native import load_library

    return getattr(load_library(), "split_rms_norm", None)


def _normalized_qkv(layer, x):
    import torch_npu

    if not layer.fuse_wqa_wkv:
        return layer.q_norm(layer.wq_a(x)[0]), layer.kv_norm(layer.wkv(x)[0])

    packed, _ = layer.wqkv_a(x)
    first, second = layer.q_norm.weight, layer.kv_norm.weight
    split_norm = _split_rms_operator()
    if (
        split_norm is not None
        and packed.ndim == 2
        and packed.size(0) <= 2048
        and packed.stride(1) == 1
        and packed.stride(0) >= packed.size(1)
        and first.is_contiguous()
        and second.is_contiguous()
        and all(0 < w.numel() <= 4096 and w.numel() % 64 == 0 for w in (first, second))
        and all(
            t.dtype == torch.bfloat16 and torch_npu.get_npu_format(t) in (0, 2)
            for t in (packed, first, second)
        )
    ):
        return split_norm(
            packed,
            first,
            second,
            layer.q_norm.variance_epsilon,
            layer.kv_norm.variance_epsilon,
        )
    return layer.q_norm(packed[..., : layer.q_lora_rank]), layer.kv_norm(
        packed[..., layer.q_lora_rank :]
    )


def adapt_prepare(original_prepare):
    if not enabled():
        return original_prepare

    @wraps(original_prepare)
    def prepare(
        self,
        x,
        positions,
        forward_batch,
        attn_backend,
        q_out=None,
        x_quant=None,
        q_rope_out=None,
        k_nope_out=None,
        k_rope_out=None,
    ):
        import torch_npu
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        if not (enabled() and self.is_dsv41 and (x.device.type == "npu")):
            return original_prepare(
                self,
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out,
                x_quant=x_quant,
                q_rope_out=q_rope_out,
                k_nope_out=k_nope_out,
                k_rope_out=k_rope_out,
            )
        if x_quant is not None or any(
            (t is not None for t in (q_rope_out, k_nope_out, k_rope_out))
        ):
            raise ValueError(
                "V4.1/910B preparation uses BF16 projections and a BF16 KV cache"
            )
        if x.dtype != torch.bfloat16:
            raise ValueError("V4.1/910B attention input must be BF16")
        source = attn_backend.forward_metadata.tail_source
        if source is not None and self.layer_id == source.layer_id:
            q_lora, kv = source.q_lora, None
        else:
            q_lora, kv = _normalized_qkv(self, x)
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.q_head_norm:
            q = torch_npu.npu_rms_norm(q, self.q_rms_norm_ones, self.eps)[0]
        cos, sin = self._get_npu_rope_position_cache(
            forward_batch, positions, q.dtype, inverse=False
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            q,
            None if kv is None else kv.unsqueeze(1),
            cos,
            sin,
            qk_nope_dim=self.qk_nope_head_dim,
        )
        if kv is not None:
            attn_backend.store_cache(
                layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch
            )
        if q_out is not None:
            if q_out.dtype != torch.bfloat16:
                raise ValueError("V4.1/910B Q storage must be BF16")
            q_out.copy_(q)
        if self.compress_ratio in (1, 2) and (
            self.compressor is not None or self.indexer is not None
        ):
            attn_backend.forward_low_ratio_sources(
                layer=self,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
            )
        return (q, None)

    return prepare


def adapt_rope_positions(original_rope_positions):
    if not enabled():
        return original_rope_positions

    @wraps(original_rope_positions)
    def rope_positions(self, forward_batch, positions, dtype, inverse=False):
        if enabled() and self.is_dsv41 and (positions.device.type == "npu"):
            memo = getattr(forward_batch, "_dsv41_rope_memo", {})
            entry = memo.get(id(self.freqs_cis))
            if entry is not None and entry[0] is positions:
                return entry[2][int(inverse)]
            return expanded_rope(gather_rope(self.freqs_cis, positions.long()))[
                int(inverse)
            ]
        return original_rope_positions(self, forward_batch, positions, dtype, inverse)

    return rope_positions


def adapt_multi_prepare(original_multi_prepare):
    if not enabled():
        return original_multi_prepare

    @wraps(original_multi_prepare)
    def multi_prepare(
        self, x, positions, forward_batch, attn_backend, q_out=None, x_quant=None
    ):
        if enabled() and self.is_dsv41 and (x.device.type == "npu"):
            return self._forward_prepare(
                x,
                positions,
                forward_batch,
                attn_backend,
                q_out=q_out,
                x_quant=x_quant,
            )[0]
        return original_multi_prepare(
            self, x, positions, forward_batch, attn_backend, q_out, x_quant=x_quant
        )

    return multi_prepare


def adapt_project(original_project):
    if not enabled():
        return original_project

    @wraps(original_project)
    def project(self, x):
        if not enabled() or x.device.type != "npu":
            return original_project(self, x)
        if self.compress_ratio == 1:
            return (self.wkv(x), None)
        if not hasattr(self, "_dsv41_packed_wkv"):
            raise RuntimeError(
                "C2 weights must be repacked with prepare_compressor after checkpoint load"
            )
        op = compressor_projection()
        return (op(x, self._dsv41_packed_wkv), op(x, self._dsv41_packed_wgate))

    return project


def adapt_post_load(original_post_load):
    if not enabled():
        return original_post_load

    @wraps(original_post_load)
    def post_load(model, *args, **kwargs):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor

        result = original_post_load(model, *args, **kwargs)
        if getattr(model.config, "model_type", None) == "deepseek_v41":
            for module in model.modules():
                if (
                    isinstance(module, DeepseekV41Compressor)
                    and module.norm.weight.device.type == "npu"
                ):
                    prepare_compressor(module)
            compact_rope_storage(model)
        return result

    return post_load


def adapt_rotary(original_rotary):
    if not enabled():
        return original_rotary

    @wraps(original_rotary)
    def rotary(q_rope, kv_rope, cos4, sin4, qk_nope_dim=0):
        if not enabled() or q_rope.device.type != "npu":
            return original_rotary(q_rope, kv_rope, cos4, sin4, qk_nope_dim)
        if q_rope.shape[-1] - qk_nope_dim != 64 or cos4.shape[-1] != 64:
            raise ValueError(
                "V4.1/910B rotary adapter requires a tail of 64 adjacent-pair dimensions"
            )
        rows = q_rope.shape[0]
        compact = getattr(sin4, "_dsv41_compact_rope", None)
        if compact is None:
            cos = cos4.reshape(rows, 64)[:, ::2].float().contiguous()
            sin = sin4.reshape(rows, 64)[:, ::2].float().contiguous()
            inverse = False
        else:
            cos, sin, inverse = compact
        op = position_operator(q_rope.device)
        for value in (q_rope, kv_rope):
            if value is None:
                continue
            target = value if value.is_contiguous() else value.contiguous()
            op.rope(target, cos, sin, inverse=inverse, out=target)
            if target is not value:
                value.copy_(target)

    return rotary
