from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sgl_kernel_npu.quantization.repack import repack_int4_npu

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class _NPULinearMethodBase(LinearMethodBase):

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW4A16LinearMethod(_NPULinearMethodBase):
    """Linear method for Ascend NPU W4A16 quantization.

    Uses npu_weight_quant_batchmatmul for W4A16 computation.
    Weight is packed int4 stored in int32 format.
    """

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
        group_size: int = 128,
    ):
        super().__init__(quant_config)
        self.num_bits = 4
        self.pack_factor = 8  # 32 // 4 = 8
        self.group_size = group_size

    def _unpack_from_int32(
        self,
        weight: torch.Tensor,
        shape: torch.Size,
        num_bits: int,
        packed_dim: int = 1,
    ) -> torch.Tensor:
        """
        Unpacks quantized weights from int32 format back to original bits.

        :param weight: The packed int32 tensor containing quantized weights
        :param shape: Original shape to restore
        :param num_bits: The number of bits used for quantization (<= 8)
        :param packed_dim: Dimension along which weights are packed (0 or 1)
        :return: Unpacked tensor with int8 dtype after applying offset correction
        """
        assert weight.dtype == torch.int32, f"Expecting `weight.dtype` is torch.int32 but got {weight.dtype}."
        assert num_bits <= 8, f"Expecting `num_bits` should not be larger than 8 but got {num_bits}."

        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked_weight = torch.zeros(
                (weight.shape[0], weight.shape[1] * pack_factor),
                device=weight.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked_weight[:, i::pack_factor] = (weight >>
                                                      (num_bits * i)) & mask
            original_row_size = int(shape[1])
            unpacked_weight = unpacked_weight[:, :original_row_size]
        else:
            unpacked_weight = torch.zeros(
                (weight.shape[0] * pack_factor, weight.shape[1]),
                device=weight.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked_weight[i::pack_factor, :] = (weight >>
                                                      (num_bits * i)) & mask
            original_row_size = int(shape[0])
            unpacked_weight = unpacked_weight[:original_row_size, :]

        offset = pow(2, num_bits) // 2
        unpacked_weight = (unpacked_weight - offset).to(torch.int8)

        return unpacked_weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Convert from Marlin/compressed-tensors format to Ascend NPU format.
        Uses optimized Triton kernel for efficient int4 repacking (unpack + transpose + repack).
        Only symmetric quantization is supported.
        """
        # weight shape from checkpoint: [output_size, input_size // pack_factor] = [N, K//8]
        # After loading, we need to convert to [K, N//8] for NPU format

        weight_shape = layer.weight_packed.data.shape
        output_size = weight_shape[0]  # N
        packed_input_size = weight_shape[1]  # K//8
        input_size = packed_input_size * self.pack_factor  # K

        # Current weight shape: [N, K//8] (int32 packed)
        # Step 1: Transpose to [K//8, N] for the optimized kernel
        weight_t = layer.weight_packed.data.transpose(0, 1).contiguous()

        # Step 2: Use optimized Triton kernel to fuse unpack + transpose + repack
        # Input: [K//8, N], Output: [K, N//8]
        layer.weight_packed.data = repack_int4_npu(weight_t)

        # Transpose scale from [output_size, num_groups] to [num_groups, output_size]
        layer.weight_scale.data = layer.weight_scale.data.transpose(
            0, 1).contiguous()

        # For symmetric quantization, create zero offset (antiquant_offset must not be None)
        # Create zeros_like weight_scale for antiquant_offset
        layer.weight_offset = torch.nn.Parameter(
            torch.zeros_like(layer.weight_scale.data),
            requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply W4A16 linear transformation.

        Uses batch_gemm_w4a16_small_bs for batch_size <= 4,
        otherwise uses npu_weight_quant_batchmatmul for W4A16 computation.
        The weight is already packed by process_weights_after_loading.
        Only symmetric quantization is supported (weight_offset should be zeros).
        """
        # Get batch size (first dimension of input x)
        batch_size = x.shape[0]

        if batch_size <= 4:
            # Use custom batch_gemm_w4a16_small_bs for small batch sizes
            output = torch.ops.npu.batch_gemm_w4a16_small_bs(
                x,
                layer.weight_packed,
                layer.weight_scale,
            )
            # Add bias if provided (batch_gemm_w4a16_small_bs doesn't handle bias)
            if bias is not None:
                output = output + bias
        else:
            # Use npu_weight_quant_batchmatmul for larger batch sizes
            output = torch.ops.npu.npu_weight_quant_batchmatmul(
                x=x,
                weight=layer.weight_packed,
                antiquant_scale=layer.weight_scale,
                antiquant_offset=layer.weight_offset,
                antiquant_group_size=self.group_size,
                bias=bias,
            )
        return output


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

        expanding_factor = layer.weight.data.shape[0]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if original_dtype != torch.int8:
            x = torch.ops.npu.npu_quantize(
                x,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if isinstance(layer, RowParallelLinear) and layer.tp_rank > 0:
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        return torch.ops.npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer: torch.nn.Module):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = npu_format_cast(layer.weight.data)

        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # Compressed-tensors format doesn't have this field
        if hasattr(layer, "weight_offset"):
            layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(x, tuple):
            """dynamic_scale is calculated in malprolog kernel"""
            original_dtype = torch.bfloat16
            quant_out, dynamic_scale = x
        else:
            original_dtype = x.dtype
            quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=original_dtype,
        )


class NPU_W4A4DynamicLinearMethod(_NPULinearMethodBase):

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
            layer.weight.data.to(torch.int32)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=original_dtype,
        )
