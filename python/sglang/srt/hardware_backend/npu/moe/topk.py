from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.layers.moe.topk import StandardTopKOutput

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":
    """NPU fused TopK implementation using Ascend optimized kernels."""

    scoring_func = topk_config.scoring_func
    use_softmax_kernel = scoring_func == "softmax" and topk_config.correction_bias is None

    if use_softmax_kernel:
        # Optimized path: softmax without bias
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=topk_config.top_k,
        )
        if topk_config.renormalize:
            weights_to_norm = (
                topk_weights
                if topk_config.num_fused_shared_experts == 0
                else topk_weights[:, :-1]
            )
            topk_weights = l1_norm(weights_to_norm)
    else:
        # Universal path: supports sigmoid/softmax with/without bias, grouped/non-grouped
        x = router_logits.to(torch.float32)
        k = topk_config.top_k

        kernel_kwargs = {
            "renorm": 0,  # NPU kernel only supports renorm=0
            "norm_type": 1 if scoring_func == "sigmoid" else 0,
            "routed_scaling_factor": 1.0,
            "eps": float(1e-20),
        }

        # Handle bias and grouping
        if topk_config.correction_bias is not None:
            kernel_kwargs["bias"] = topk_config.correction_bias.to(torch.float32)
        if topk_config.use_grouped_topk:
            kernel_kwargs["k_group"] = topk_config.topk_group
            kernel_kwargs["group_count"] = topk_config.num_expert_group
        kernel_kwargs["group_select_mode"] = 1

        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            x, k, **kernel_kwargs
        )

        if topk_config.renormalize:
            topk_weights = l1_norm(topk_weights)

    topk_weights = topk_weights.to(torch.float32)

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    get_global_experts_capturer().capture(
        layer_id=layer_id,
        topk_ids=topk_ids,
    )

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
