"""Independent numerical references and fixtures for projection."""

from __future__ import annotations

import torch


def compare(actual, expected, label, records, fp32=False):
    got = actual.cpu().float()
    torch.testing.assert_close(
        got,
        expected.float(),
        atol=3e-5 if fp32 else 0.005,
        rtol=3e-5 if fp32 else 0.02,
        msg=label,
    )
    records.append(
        {
            "case": label,
            "max_abs": (got - expected.float()).abs().max().item()
            if got.numel()
            else 0.0,
        }
    )


def output_canary(shape, dtype):
    count = 1
    for d in shape:
        count *= d
    allocation = torch.full((count + 64,), 7.0, dtype=dtype, device="npu")
    return allocation[32 : 32 + count].view(shape), allocation


def bf16_dot_envelope(actual, inputs, weight_nt, label):
    """BF16 products are exact in FP32; bound arbitrary FP32 summation order.

    For these finite moderate test inputs there is no under/overflow. A K-term
    FP32 reduction has error <= gamma_(K-1) * sum(abs(products)). Round both
    endpoints to BF16 instead of hiding cancellation with a global atol.
    """
    x, w = inputs.double(), weight_nt.double()
    dot = x @ w.T
    unit_roundoff = 2.0**-24
    k = inputs.shape[-1]
    gamma = (k - 1) * unit_roundoff / (1 - (k - 1) * unit_roundoff)
    budget = gamma * (x.abs() @ w.abs().T)
    lower, upper = (dot - budget).bfloat16().float(), (dot + budget).bfloat16().float()
    got = actual.cpu().float()
    assert ((got >= lower) & (got <= upper)).all(), (
        label + "/FP32-dot+BF16-rounding-envelope"
    )
    return {
        "fp32_accumulation_budget_max": budget.max().item() if budget.numel() else 0,
        "bf16_rounding_envelope_violations": 0,
    }


def independent_mlp(x, wg, wu, wd, dtype):
    """Independent full CPU chain, with BF16 handoffs at all three stages."""
    gate = (x.to(dtype) @ wg.to(dtype).T).bfloat16().to(dtype).clamp(max=10)
    up = (x.to(dtype) @ wu.to(dtype).T).bfloat16().to(dtype).clamp(-10, 10)
    activation = (torch.nn.functional.silu(gate) * up).bfloat16()
    return (activation.to(dtype) @ wd.to(dtype).T).bfloat16()


def shared_reference_checks(actual, x, wg, wu, wd, label, records):
    # Keep strict individual operator checks, including a derived per-element
    # FP32 accumulation/BF16 rounding envelope for each GEMM.
    gate = (x.float() @ wg.float().T).bfloat16()
    up = (x.float() @ wu.float().T).bfloat16()
    compare(actual["gate"], gate, label + "/gate", records)
    compare(actual["up"], up, label + "/up", records)
    details = bf16_dot_envelope(actual["gate"], x, wg, label + "/gate")
    details.update(
        {
            "up_" + k: v
            for k, v in bf16_dot_envelope(actual["up"], x, wu, label + "/up").items()
        }
    )
    actual_gate, actual_up = actual["gate"].cpu(), actual["up"].cpu()
    activation = (
        torch.nn.functional.silu(actual_gate.float().clamp(max=10))
        * actual_up.float().clamp(-10, 10)
    ).bfloat16()
    torch.testing.assert_close(
        actual["activation"].cpu().float(),
        activation.float(),
        atol=0.0001,
        rtol=0.008,
        msg=label + "/swiglu",
    )
    actual_activation = actual["activation"].cpu()
    down = (actual_activation.float() @ wd.float().T).bfloat16()
    compare(actual["out"], down, label + "/down", records)
    details.update(
        {
            "down_" + k: v
            for k, v in bf16_dot_envelope(
                actual["out"], actual_activation, wd, label + "/down"
            ).items()
        }
    )
    got = actual["out"].cpu().float()
    for dtype in (torch.float32, torch.float64):
        reference = independent_mlp(x, wg, wu, wd, dtype).float()
        denominator = reference.norm().item()
        error = (
            (got - reference).norm().item() / denominator
            if denominator
            else (got - reference).norm().item()
        )
        # Declared qualification for this fixed BF16 test matrix; not a bound
        # for arbitrary model inputs or a model-quality claim.
        assert error <= 5e-3, (
            label,
            dtype,
            "independent full-chain relative L2",
            error,
        )
        details[str(dtype) + "_full_chain_rel_l2"] = error
    records.append({"case": label + "/independent-full-chain", **details})
