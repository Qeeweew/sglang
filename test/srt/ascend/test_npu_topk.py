#!/usr/bin/env python3
"""
Test script for NPU MoE TopK implementation.
Compares NPU fused_topk_npu with native PyTorch implementation.
"""

import torch
import sys
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class TopKConfig:
    top_k: int
    use_grouped_topk: bool = False
    topk_group: Optional[int] = None
    num_expert_group: Optional[int] = None
    renormalize: bool = True
    num_fused_shared_experts: int = 0
    custom_routing_function: Optional[Callable] = None
    correction_bias: Optional[torch.Tensor] = None
    torch_native: bool = False
    routed_scaling_factor: Optional[float] = None
    apply_routed_scaling_factor_on_output: bool = False
    fused_shared_experts_scaling_factor: Optional[float] = None
    scoring_func: str = "softmax"


def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor = None,
    scoring_func: str = "softmax",
):
    """Native PyTorch implementation (ground truth)."""
    def scoring_func_impl(gating_output: torch.Tensor) -> torch.Tensor:
        if scoring_func == "softmax":
            return gating_output.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            return gating_output.sigmoid()
        else:
            raise ValueError(f"Invalid scoring function: {scoring_func}")

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = scoring_func_impl(gating_output)
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert hidden_states.shape[0] == gating_output.shape[0]
        topk_weights = scoring_func_impl(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def run_test(name, hidden_states, router_logits, correction_bias, top_k, renormalize, scoring_func, use_grouped_topk=False):
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")

    # Run native on CPU
    native_weights, native_ids = fused_topk_torch_native(
        hidden_states=hidden_states.cpu(),
        gating_output=router_logits.cpu(),
        topk=top_k,
        renormalize=renormalize,
        correction_bias=correction_bias.cpu() if correction_bias is not None else None,
        scoring_func=scoring_func,
    )

    # Run NPU
    topk_config = TopKConfig(
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        correction_bias=correction_bias,
        scoring_func=scoring_func,
    )

    from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu

    npu_output = fused_topk_npu(
        hidden_states=hidden_states,
        router_logits=router_logits,
        topk_config=topk_config,
        layer_id=0,
    )
    npu_weights = npu_output.topk_weights.cpu()
    npu_ids = npu_output.topk_ids.cpu()

    # Print sample
    print(f"Native topk_ids[0]:    {native_ids[0].tolist()}")
    print(f"NPU topk_ids[0]:       {npu_ids[0].tolist()}")

    # Compare: check if the same set of experts are selected with correct weights
    # TopK may return different order, so we need to match by expert ID
    all_match = True
    max_diff = 0.0

    for i in range(native_ids.shape[0]):
        native_set = set(native_ids[i].tolist())
        npu_set = set(npu_ids[i].tolist())

        # Check if same experts are selected
        if native_set != npu_set:
            print(f"Token {i}: Expert set mismatch!")
            print(f"  Native: {sorted(native_set)}")
            print(f"  NPU:    {sorted(npu_set)}")
            all_match = False
            continue

        # For matching experts, compare weights
        for j, expert_id in enumerate(native_ids[i].tolist()):
            # Find this expert in NPU result
            npu_idx_list = (npu_ids[i] == expert_id).nonzero(as_tuple=True)[0]
            if len(npu_idx_list) == 0:
                print(f"Token {i}: Expert {expert_id} not found in NPU result!")
                all_match = False
                continue

            npu_idx = npu_idx_list[0].item()
            native_w = native_weights[i][j].item()
            npu_w = npu_weights[i][npu_idx].item()
            diff = abs(native_w - npu_w)
            max_diff = max(max_diff, diff)

            if diff > 1e-3:
                print(f"Token {i}, Expert {expert_id}: weight mismatch! Native={native_w:.6f}, NPU={npu_w:.6f}, diff={diff:.2e}")
                all_match = False

    print(f"\nAll tokens match: {all_match}")
    print(f"Max weight difference: {max_diff:.2e}")

    status = "PASS" if all_match else "FAIL"
    print(f"Result: {status}")

    return all_match


def main():
    if not torch.npu.is_available():
        print("ERROR: NPU is not available!")
        return 1

    print(f"NPU Device: {torch.npu.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")

    results = []

    # Test 1: MiniMax M2 style - sigmoid + correction_bias
    torch.manual_seed(42)
    results.append(run_test(
        "MiniMax M2 (sigmoid + bias)",
        hidden_states=torch.randn(16, 3072, dtype=torch.float16, device='npu'),
        router_logits=torch.randn(16, 256, dtype=torch.float32, device='npu'),
        correction_bias=torch.randn(256, dtype=torch.float32, device='npu') * 0.1,
        top_k=8,
        renormalize=True,
        scoring_func="sigmoid",
    ))

    # Test 2: Softmax without bias
    torch.manual_seed(42)
    results.append(run_test(
        "Softmax without bias",
        hidden_states=torch.randn(16, 2048, dtype=torch.float16, device='npu'),
        router_logits=torch.randn(16, 128, dtype=torch.float32, device='npu'),
        correction_bias=None,
        top_k=4,
        renormalize=True,
        scoring_func="softmax",
    ))

    # Test 3: Sigmoid without bias
    torch.manual_seed(42)
    results.append(run_test(
        "Sigmoid without bias",
        hidden_states=torch.randn(16, 3072, dtype=torch.float16, device='npu'),
        router_logits=torch.randn(16, 256, dtype=torch.float32, device='npu'),
        correction_bias=None,
        top_k=8,
        renormalize=True,
        scoring_func="sigmoid",
    ))

    # Test 4: Different batch sizes
    print(f"\n{'='*60}")
    print("Test: Different batch sizes")
    print(f"{'='*60}")
    all_pass = True
    for batch_size in [1, 4, 8, 32]:
        torch.manual_seed(42)
        result = run_test(
            f"Batch size {batch_size}",
            hidden_states=torch.randn(batch_size, 3072, dtype=torch.float16, device='npu'),
            router_logits=torch.randn(batch_size, 256, dtype=torch.float32, device='npu'),
            correction_bias=torch.randn(256, dtype=torch.float32, device='npu') * 0.1,
            top_k=8,
            renormalize=True,
            scoring_func="sigmoid",
        )
        all_pass = all_pass and result
    results.append(all_pass)

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    test_names = ["MiniMax M2", "Softmax no bias", "Sigmoid no bias", "Batch sizes"]
    for name, passed in zip(test_names, results):
        print(f"  {name:25s}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results)
    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
