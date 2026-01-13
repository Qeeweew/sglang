from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import torch
import threading
import sys

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import get_bool_env_var
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
)

logger = logging.getLogger(__name__)

try:
    import torch_npu
    import nanovllm_ext
except Exception:
    raise ImportError("nanovllm_ext or torch_npu is not installed.")

from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, StandardDispatchOutput


@dataclass
class NpuEpConfig:
    enabled: bool = False
    fully_offload: bool = True

@dataclass
class NpuEpConfig:
    enabled: bool = False
    fully_offload: bool = True
    layer_idx: int = -1


def create_arm_ep_config_from_server_args(
    server_args, layer_idx: int
) -> Optional[NpuEpConfig]:
    """
    根据 ServerArgs 判断当前层是否应该 Offload 到 ARM/NPU。
    参数约定 (通过 getattr 获取以兼容旧版 server_args):
      - enable_arm_ep: bool
      - arm_ep_start_layer: int (默认 0)
    """
    # 1. 检查是否全局启用
    # 优先使用 server_args，如果没有则回退到环境变量
    enable_args = getattr(server_args, "enable_arm_ep", False)
    
    if not enable_args:
        return None

    # 2. 检查 Layer 范围
    # 如果指定了 start_layer，则小于该 index 的层运行在 GPU 上，大于等于的运行在 NPU 上
    start_layer = getattr(server_args, "arm_ep_start_layer", 0)
    
    # 如果 layer_idx 小于起始层，返回 None (表示该层保持原样/GPU运行)
    if layer_idx < start_layer:
        return None

    logger.info(f"[ARM EP] Layer {layer_idx} will be offloaded to CPU (Start Layer: {start_layer})")
    
    return NpuEpConfig(
        enabled=True, 
        fully_offload=True,
        layer_idx=layer_idx
    )


# -------------------------
# 全进程缓存：按 (device_id, stream_ptr) 维度存 NpuCallbackManager
# -------------------------
_GLOBAL_CB_LOCK = threading.Lock()
_GLOBAL_CB_MANAGERS: dict[tuple[int, int], Any] = {}

def _get_or_create_global_callback_manager(stream_ptr: int) -> Any:
    """
    同一进程内：同一个 (device_id, stream_ptr) 只创建一次。
    若 stream_ptr 不同：创建新的 manager 并缓存，不再报错。
    """
    device_id = int(torch_npu.npu.current_device())
    key = (device_id, int(stream_ptr))

    with _GLOBAL_CB_LOCK:
        mgr = _GLOBAL_CB_MANAGERS.get(key)
        if mgr is not None:
            return mgr

        mgr = torch.classes.nanovllm.NpuCallbackManager(int(stream_ptr), device_id)
        _GLOBAL_CB_MANAGERS[key] = mgr
        logger.info(f"[NPU EP] Created NpuCallbackManager. device_id={device_id}, stream_ptr={stream_ptr}")
        return mgr

class ArmEpWrapperMethod(FusedMoEMethodBase):
    """
    方案B：load 时不创建大权重，不把权重存成 layer 参数。
    checkpoint 每读到一块 expert 权重 -> 立刻切分(TP/EP) -> quantize_and_store 到 CPU handle。
    """

    def __init__(self, npu_method: FusedMoEMethodBase, config: NpuEpConfig):
        self.npu_method = npu_method
        self.config = config

        self.moe_infer_handle: Optional[Any] = None

        # graph ctx: key=(num_tokens, top_k, dtype_int)
        self.graph_contexts: Dict[Tuple[int, int, int], Any] = {}

        # meta
        self.num_experts: Optional[int] = None            # local experts count
        self.hidden_size: Optional[int] = None
        self.intermediate_size: Optional[int] = None      # per TP partition
        self.params_dtype: Optional[torch.dtype] = None
        self.is_gated: bool = True

    # -------------------------
    # 方案B：create_weights 不分配大权重，只做占位 + 初始化 handle
    # -------------------------
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # 假设 tp-size = 1
        if get_tensor_model_parallel_rank() != 0:
            return 

        self.num_experts = int(num_experts)  # 注意：这里传进来的是 num_local_experts
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size_per_partition)
        self.params_dtype = params_dtype
        self.is_gated = bool(getattr(layer, "moe_runner_config", None).is_gated) if hasattr(layer, "moe_runner_config") else True

        # 提前创建 CPU 量化缓存 handle（不需要等 process_weights_after_loading）
        if self.moe_infer_handle is None:
            self.moe_infer_handle = torch.classes.nanovllm.MoEInfer(
                self.num_experts,
                self.hidden_size,
                self.intermediate_size,
            )

        # 创建 very small dummy params 仅用于让 load_weights 找到 params_dict[name]
        # 不要创建真实 shape，否则 CPU 也会很大
        layer.w13_weight = torch.nn.Parameter(torch.empty(1, device="cpu", dtype=params_dtype), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.empty(1, device="cpu", dtype=params_dtype), requires_grad=False)

        layer.w13_weight.weight_loader = lambda param, loaded_weight, weight_name, shard_id, expert_id: self._stream_quant_weight(
            layer=layer,
            loaded_weight=loaded_weight,
            shard_id=shard_id,
            expert_id=int(expert_id),
        )
        layer.w2_weight.weight_loader = layer.w13_weight.weight_loader

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"):
        self.moe_runner_config = moe_runner_config
        # 仍然让底层 method 建 runner（但不要让它 create_weights）
        self.npu_method.create_moe_runner(layer, moe_runner_config)

    def _stream_quant_weight(self, layer: torch.nn.Module, loaded_weight: torch.Tensor, shard_id: str, expert_id: int) -> None:
        assert self.moe_infer_handle is not None
        assert self.hidden_size is not None

        # loaded_weight 通常在 CPU；确保在 CPU 上处理
        w = loaded_weight.detach()
        if w.device.type != "cpu":
            w = w.cpu()

        if shard_id == "w1":
            self.moe_infer_handle.quantize_and_store_expert(expert_id, "gate_proj", w)
        elif shard_id == "w3":
            self.moe_infer_handle.quantize_and_store_expert(expert_id, "up_proj", w)
        elif shard_id == "w2":
            self.moe_infer_handle.quantize_and_store_expert(expert_id, "down_proj", w)
        else:
            raise ValueError(f"Unsupported shard_id={shard_id}")

    # -------------------------
    # load 完后的 hook：方案B下不再需要把 w13/w2 从 NPU/CPU 搬来搬去
    # -------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 方案B：已经在 weight_loader 时量化并存储完毕
        # 这里可选：释放 dummy param（不建议完全删掉，避免后续有代码访问属性）
        try:
            torch_npu.npu.empty_cache()
        except Exception:
            pass

    # -------------------------
    # forward
    # -------------------------
    def apply(self, layer: torch.nn.Module, dispatch_output: "StandardDispatchOutput") -> "CombineInput":
        # -------------------------
        # capture：graph ctx + callback manager（stream 强绑定）
        # -------------------------
        stream_ptr = int(torch_npu.npu.current_stream().npu_stream)
        _get_or_create_global_callback_manager(stream_ptr)

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if not self.config.fully_offload:
            return self.npu_method.apply(layer, dispatch_output)

        assert self.moe_infer_handle is not None, "moe_infer_handle is not initialized"

        hidden_states = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        num_tokens = int(hidden_states.shape[0])
        top_k = int(topk_ids.shape[1])

        # dtype normalize (keep on NPU)
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        # -------------------------
        # 非 capture (eager)：直接走 C++ 的 moe_forward_npu_stream
        # -------------------------
        if not get_is_capture_mode():
            out = torch.ops.nanovllm.moe_forward_npu_stream(
                hidden_states,
                topk_ids,
                topk_weights,
                self.moe_infer_handle,
            )
            return StandardCombineInput(hidden_states=out)

        dtype_int = 1 if hidden_states.dtype == torch.bfloat16 else 0  # 0: fp16, 1: bf16
        ctx_key = (num_tokens, top_k, dtype_int)
        ctx = self.graph_contexts.get(ctx_key)
        if ctx is None:
            ctx = torch.classes.nanovllm.MoEGraphContext(
                self.moe_infer_handle,
                num_tokens,
                top_k,
                dtype_int,
            )
            self.graph_contexts[ctx_key] = ctx

        out = torch.empty_like(hidden_states)
        torch.ops.nanovllm.moe_forward_npu_graph_out(hidden_states, topk_ids, topk_weights, self.moe_infer_handle, ctx, out)
        return StandardCombineInput(hidden_states=out)