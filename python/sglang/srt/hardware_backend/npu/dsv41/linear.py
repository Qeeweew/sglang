"""Target/draft dense projections and shared experts -> G256 W8A16.

Weights quantize as each local checkpoint matrix finishes loading. Only one
BF16 local matrix is staged; persistent parameters are INT8 from construction.
"""

from __future__ import annotations

import logging
import weakref
from collections import Counter
from functools import wraps

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from .policy import enabled

logger = logging.getLogger(__name__)

_WO_A = weakref.WeakValueDictionary()


def enabled_for(quant_config):
    return getattr(quant_config, "_dsv41_bf16_dense", False)


class SharedW8A8Method(UnquantizedLinearMethod):
    def __init__(self, interleaved):
        self.interleaved = interleaved

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attrs,
    ):
        if input_size_per_partition % 256 or (
            not self.interleaved and any(n % 256 for n in output_partition_sizes)
        ):
            raise ValueError("shared W8A8 TP cuts an interleaved H32 panel")
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            torch.int8,
            **attrs,
        )
        scales = torch.nn.Parameter(
            layer.weight.new_empty(
                (sum(output_partition_sizes), 1), dtype=torch.float32
            ),
            requires_grad=False,
        )
        scales.output_dim = 0
        scales.weight_loader = layer.weight.weight_loader
        layer.register_parameter("weight_scale_inv", scales)

    def process_weights_after_loading(self, layer):
        from sglang.srt.hardware_backend.npu.utils import npu_format_cast

        layer.weight.data = npu_format_cast(layer.weight.T.contiguous())
        layer.weight_scale_inv.data = layer.weight_scale_inv.flatten()

    def apply(self, layer, x, bias=None):
        from .kernels.ops.h32_quant import h32_quant

        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], layer.weight_scale_inv.numel()))
        quantized, scale = h32_quant(x.reshape(-1, x.shape[-1]), self.interleaved)
        out = self.from_quantized(layer, quantized, scale).reshape(*x.shape[:-1], -1)
        return out if bias is None else out + bias

    @staticmethod
    def from_quantized(layer, quantized, scale, *, output_dtype=torch.bfloat16):
        return torch.ops.npu.npu_quant_matmul(
            quantized,
            layer.weight,
            layer.weight_scale_inv,
            pertoken_scale=scale,
            output_dtype=output_dtype,
        )

    def apply_into(self, layer, x, output, bias=None):
        return output.copy_(self.apply(layer, x, bias))

    def apply_with_addend(self, layer, x, addend, bias=None):
        return self.apply(layer, x, bias).add_(addend)


class W8A16LinearMethod(UnquantizedLinearMethod):
    """Keep SGLang's checkpoint sharding and collectives around the new GEMM."""

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **attrs,
    ):
        torch.npu.config.allow_internal_format = True
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            torch.int8,
            **attrs,
        )
        self.setup(layer, output_partition_sizes)

    def setup(self, layer, partitions):
        self.n, self.k = layer.weight.shape
        if self.k % 256 or self.n % 128:
            raise ValueError(f"W8A16 G256 requires K256/N128, got N{self.n}/K{self.k}")
        self.parts = tuple(partitions)
        self.loaded, self.pending = set(), None
        self.batches = 1
        self.finalized = False
        self.plans = {}
        layer.register_buffer(
            "w8a16_scale",
            torch.empty(
                (self.k // 256, self.n), dtype=torch.float16, device=layer.weight.device
            ),
            persistent=False,
        )
        original = layer.weight.weight_loader

        def load(param, value, *args, **kwargs):
            if self.finalized:
                raise RuntimeError("load W8A16 checkpoint weights before repacking")
            if self.pending is None:
                self.pending = torch.nn.Parameter(
                    torch.empty(
                        (self.n, self.k), device=param.device, dtype=torch.bfloat16
                    ),
                    requires_grad=False,
                )
                self.pending.__dict__.update(param.__dict__)
            result = original(self.pending, value, *args, **kwargs)
            shard = args[0] if args else kwargs.get("loaded_shard_id")
            if shard is None:
                self.loaded.update(range(len(self.parts)))
            else:
                if not isinstance(shard, int) or not 0 <= shard < len(self.parts):
                    raise ValueError(f"unknown W8A16 merged checkpoint shard: {shard}")
                self.loaded.add(shard)
            if len(self.loaded) == len(self.parts):
                from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
                    quantize_bf16_weight,
                )

                quantize_bf16_weight(self.pending, param, layer.w8a16_scale)
                self.pending = None
            return result

        layer.weight.weight_loader = load

    def process_weights_after_loading(self, layer):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
            W8A16,
            repack_weight,
        )

        if self.finalized:
            return
        if self.pending is not None or len(self.loaded) != len(self.parts):
            raise RuntimeError("incomplete W8A16 checkpoint matrix")
        weight, scales = repack_weight(layer.weight.detach(), layer.w8a16_scale)
        # Retain the upstream NK parameter shape and loader attributes. Its
        # storage now follows N16/K/16 and only our W8A16 path may consume it.
        layer.weight.data = weight.view(self.n, self.k)
        layer.w8a16_scale = scales
        self.weight = layer.weight.view(self.n // 16, self.k, 16)
        if self.batches > 1:
            _WO_A[layer.weight.data_ptr()] = layer
        self.op = W8A16()
        self.finalized = True

    def _launch(self, layer, x, output):
        geometry = (x.device, x.dtype, output.dtype, tuple(x.shape))
        prepared = self.plans.get(geometry)
        if prepared is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm the actual W8A16 shape before Graph capture")
            call = self.op.bind(x, self.weight, layer.w8a16_scale, out=output)
            prepared = call.native, call.plan
            self.plans[geometry] = prepared
        native, plan = prepared
        native.w8a16(x, self.weight, layer.w8a16_scale, output, None, *plan)
        return output

    def apply(self, layer, x, bias=None, *, output_dtype=torch.bfloat16):
        shape = x.shape[:-1] + (self.n,)
        flat = x.reshape(-1, self.k)
        out = torch.empty(shape, dtype=output_dtype, device=x.device)
        return self.apply_into(layer, flat, out.view(-1, self.n), bias).view(shape)

    def apply_into(self, layer, x, output, bias=None):
        if x.numel() == 0:
            return output
        flat = x.reshape(-1, self.k)
        self._launch(layer, flat, output.view(-1, self.n))
        if bias is not None:
            output.add_(bias)
        return output

    def apply_with_addend(self, layer, x, addend, bias=None):
        return self.apply(layer, x, bias).add_(addend)

    def batched(self, layer, x):
        out = x.new_empty((x.shape[0], self.batches, self.n // self.batches))
        return self._launch(layer, x, out) if x.shape[0] else out


def project_wo_a(x, weight):
    layer = _WO_A.get(weight.data_ptr())
    if layer is None:
        raise RuntimeError(
            "W8A16 wo_a weight was not finalized after checkpoint loading"
        )
    return layer.quant_method.batched(layer, x)


def adapt_init(original):
    if not enabled():
        return original

    @wraps(original)
    def init(self, config, layer_id, quant_config, prefix, **kwargs):
        from . import group_cp2_dp2

        if (
            getattr(config, "model_type", None) == "deepseek_v41"
            and group_cp2_dp2.enabled()
        ):
            from sglang.srt.runtime_context import get_device

            if get_device().device == "npu":
                kwargs["attention_projection_group"] = "replicated"
        original(self, config, layer_id, quant_config, prefix, **kwargs)
        if self.is_dsv41 and enabled_for(quant_config):
            layer = self.wo_a
            layer.weight.data = torch.empty_like(layer.weight, dtype=torch.int8)
            method = W8A16LinearMethod()
            method.setup(layer, (layer.weight.shape[0],))
            method.batches = self.n_local_groups
            layer.quant_method = method

    return init


def adapt_post_load(original_post_load):
    if not enabled():
        return original_post_load

    @wraps(original_post_load)
    def post_load(model, *args, **kwargs):
        result = original_post_load(model, *args, **kwargs)
        totals, seen = (Counter(), set())
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            if tensor.device.type != "npu":
                continue
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            category = (
                "routed_four_bit"
                if any(
                    (
                        f".experts.{field}" in name
                        for field in (
                            "gate_up_packed",
                            "gate_up_scales",
                            "gate_up_channel",
                            "down_packed",
                            "down_scales",
                            "down_channel",
                        )
                    )
                )
                else "g256_scales"
                if name.endswith("w8a16_scale")
                else "dense_int8"
                if tensor.dtype == torch.int8
                else str(tensor.dtype)
            )
            totals[category] += storage.nbytes()
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().tp_rank == 0:
            logger.info(
                "DeepSeek-V4.1 resident weights: %.2f GiB (four-bit experts %.2f GiB, INT8 dense/shared %.2f GiB).",
                sum(totals.values()) / 2**30,
                totals["routed_four_bit"] / 2**30,
                totals["dense_int8"] / 2**30,
            )
        logger.debug(
            "W8A16 resident weights GiB by storage: %s; total=%.3f; allocated=%.3f reserved=%.3f",
            {k: round(v / 2**30, 4) for k, v in totals.items()},
            sum(totals.values()) / 2**30,
            torch.npu.memory_allocated() / 2**30,
            torch.npu.memory_reserved() / 2**30,
        )
        return result

    return post_load
