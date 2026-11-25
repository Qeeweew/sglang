"""Fixed-budget MoE windows preserve routing masks."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv41 import group_cp2_dp2, moe, parallel
from sglang.srt.runtime_context import get_forward, get_parallel


@pytest.mark.parametrize(
    "rows,stride,counts,valid",
    [
        (16384, None, None, None),
        (32768, None, None, 27111),
        (1025, None, None, 903),
        (1000, 200, [123, 0, 200, 1, 175], None),
        (1035, 345, [258, 100, 344], None),
    ],
)
def test_window_output_slices_and_masks(monkeypatch, rows, stride, counts, valid):
    budget = 256
    monkeypatch.setattr(moe, "NO_SYNC_PREFILL_TOKENS", budget)
    hidden = (torch.arange(rows) % 63).to(torch.bfloat16).reshape(-1, 1)
    image_mask = (torch.arange(rows) % 7 == 0).int()
    seen = []

    def run(x, dp_counts, dp_stride, valid_rows, mask, *, out, window_budget):
        size = x.shape[0]
        assert 0 < size <= window_budget <= budget
        seen.append((out.untyped_storage().data_ptr(), out.storage_offset(), size))
        live = torch.arange(size)
        if dp_counts is not None:
            assert size == dp_stride * dp_counts.numel()
            live = live % dp_stride < dp_counts.repeat_interleave(dp_stride)
        elif valid_rows is not None:
            live = live < valid_rows
        else:
            live = live >= 0
        out.copy_(torch.where(live[:, None], x + mask[:, None], 0))
        return out

    with patch.object(
        torch.Tensor, "item", side_effect=AssertionError("read device scalar")
    ):
        routed = moe._run_expert_windows(
            hidden,
            run,
            counts=torch.tensor(counts, dtype=torch.int64)
            if counts is not None
            else None,
            stride=stride,
            valid=torch.tensor([valid], dtype=torch.int32)
            if valid is not None
            else None,
            image_mask=image_mask,
        )
    expected_live = torch.ones(rows, dtype=torch.bool)
    if counts is not None:
        for rank, count in enumerate(counts):
            expected_live[rank * stride + count : (rank + 1) * stride] = False
    elif valid is not None:
        expected_live[valid:] = False
    torch.testing.assert_close(
        routed, torch.where(expected_live[:, None], hidden + image_mask[:, None], 0)
    )
    start = 0
    for storage, offset, size in seen:
        assert storage == routed.data_ptr() and offset == start
        start += size
    assert start == rows


def test_layout_change_refreshes_mask_for_reused_ids():
    """After CED switches to tail rows, routing follows the new IDs in the same buffer."""
    image_id = 3
    ids = torch.tensor([[image_id], [5]])
    hidden = torch.zeros(2, 1)
    counts = torch.tensor([2])
    method = moe.ResidentMXFP4Method()
    method.first_expert, method.e = 0, 1
    method.routed = lambda x, mask, **kwargs: mask

    def gate(x, _):
        return x

    gate.e_score_correction_bias = torch.zeros(1)
    gate.e_score_correction_bias_vl = torch.ones(1)
    layer = SimpleNamespace(
        experts=SimpleNamespace(quant_method=method),
        gate=gate,
        config=SimpleNamespace(image_token_id=image_id),
        _dsv41_resident_router=SimpleNamespace(
            route=lambda *args, image_mask, **kwargs: image_mask
        ),
        _enable_a2a_moe=False,
        is_hash=False,
        routed_scaling_factor=1.0,
        reduce_results=False,
    )
    batch = SimpleNamespace()
    with (
        patch.dict(os.environ, {"SGLANG_DSV41_910B": "1"}),
        get_parallel().override(attn_dp_size=4),
    ):
        forward = moe.adapt_forward(lambda *args, **kwargs: pytest.fail("fallback"))
        for live_ids, expected in (([image_id, 5], [1, 0]), ([5, image_id], [0, 1])):
            ids[:, 0].copy_(torch.tensor(live_ids))
            batch._dsv41_group_moe_rows = group_cp2_dp2.MoeRows(
                stride=2, counts=counts, split_at_moe=False, global_input_ids=ids
            )
            mask = forward(layer, hidden, batch, input_ids_global=ids)
            assert mask.tolist() == expected


def _resident_experts(g32):
    from dsv41.kernels.references.mxfp4_grouped import fixture
    from dsv41.kernels.references.quant_packing import pack_experts

    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.int4_g32_a8 import pack_g32
    from sglang.srt.hardware_backend.npu.dsv41.linear import (
        SharedW8A8Method,
        W8A16LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

    method = moe.ResidentMXFP4Method(g32=g32)
    method.ep, method.first_expert, method.e, method.h, method.i = 8, 0, 48, 256, 256
    method.limit = 10.0
    routed, shared = [], []

    for n in (2 * method.i, method.h):
        k = method.h
        if g32:
            channel = torch.full((n,), 1.0 / k)
            factors = torch.full((n, k // 64), 0x88, dtype=torch.uint8)
            scales = torch.cat((factors, channel.view(torch.uint8).view(n, 4)), dim=1)
            codes = torch.full((n, k // 2), 0x11, dtype=torch.uint8)
            packed = pack_g32(codes, scales)
            routed.append(tuple(t.repeat(method.e, 1).npu() for t in packed))
            shared.append(
                SimpleNamespace(
                    weight=npu_format_cast(torch.ones(k, n, dtype=torch.int8).npu()),
                    weight_scale_inv=channel.npu(),
                    quant_method=SharedW8A8Method(interleaved=False),
                    bias=None,
                )
            )
        else:
            codes, scales = fixture(method.e, n, k)
            packed = pack_experts(
                zip(codes, scales), n=n, k=k, count=method.e, compact_scales=True
            )
            routed.append(tuple(torch.from_numpy(t).npu() for t in packed))
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(
                torch.empty(n, k, dtype=torch.int8, device="npu"), requires_grad=False
            )
            layer.weight.weight_loader = lambda param, value: param.copy_(value)
            layer.bias = None
            quant = layer.quant_method = W8A16LinearMethod()
            quant.setup(layer, (method.i, method.i) if n == 2 * method.i else (n,))
            layer.weight.weight_loader(
                layer.weight, torch.full((n, k), 1 / k).bfloat16().npu()
            )
            quant.process_weights_after_loading(layer)
            shared.append(layer)
    method.local_experts = method.bind_local_experts(*routed)

    def shared_down(x):
        layer = shared[1]
        return layer.quant_method.apply(layer, x), None

    shared_experts = SimpleNamespace(
        gate_up_proj=shared[0],
        down_proj=shared[1] if g32 else shared_down,
        swiglu_limit=method.limit,
    )
    if not g32:
        shared_down.bias = None
        shared_down.quant_method = shared[1].quant_method
    return method, shared_experts


class _Gate:
    def __init__(self):
        self.e_score_correction_bias = torch.full((384,), -100.0, device="npu")
        self.e_score_correction_bias[:6].fill_(100.0)
        self.e_score_correction_bias_vl = self.e_score_correction_bias.roll(6)

    def __call__(self, x, allocator):
        return x.new_zeros((x.shape[0], 384), dtype=torch.float32)


@pytest.mark.skipif(os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B")
@pytest.mark.parametrize(
    "g32,rows,budget,stride,counts,valid",
    [
        (False, 16384, 8192, None, None, None),
        (False, 32768, 8192, None, None, None),
        (False, 1025, 256, None, None, 778),
        (True, 1040, 256, 208, [208, 1, 0, 207, 200], None),
        (True, 1035, 256, 345, [300, 42, 344], None),
    ],
)
def test_native_windowed_moe_matches_unwindowed(
    monkeypatch, g32, rows, budget, stride, counts, valid
):
    from sglang.srt import runtime_context
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router

    torch.npu.set_device(0)
    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(attn_dp_size=1 if counts is None else 4),
    )
    method, shared = _resident_experts(g32)
    router = Router()
    plans = []

    def route(*args, **kwargs):
        plan = router.route(*args, **kwargs)
        plans.append((plan.ids, plan.metadata[1:2].clone()))
        return plan

    layer = SimpleNamespace(
        experts=SimpleNamespace(quant_method=method),
        shared_experts=shared,
        gate=_Gate(),
        config=SimpleNamespace(image_token_id=3),
        _dsv41_resident_router=SimpleNamespace(route=route),
        _shared_expert_tp1=True,
        _enable_a2a_moe=False,
        is_hash=False,
        routed_scaling_factor=1.5,
    )
    rng = torch.Generator().manual_seed(7901)
    hidden = (torch.randn(rows, method.h, generator=rng) * 0.02).bfloat16().npu()
    ids = (torch.arange(rows) % 7).npu()
    batch = SimpleNamespace(
        moe_num_token_non_padded=lambda: (
            torch.tensor(valid, dtype=torch.int32, device=hidden.device)
            if valid is not None
            else None
        )
    )
    if counts is not None:
        batch._dsv41_group_moe_rows = SimpleNamespace(
            counts=torch.tensor(counts, dtype=torch.int64, device=hidden.device),
            stride=stride,
        )

    forward = moe.adapt_forward(
        lambda *args, **kwargs: pytest.fail("unexpected fallback")
    )

    def run():
        result = forward(
            layer,
            hidden,
            batch,
            input_ids_global=ids,
            skip_shared_experts=True,
            return_moe_output=True,
        )
        return (result.routed, result.shared)

    token = parallel._in_moe.set(True)
    try:
        monkeypatch.setattr(moe, "NO_SYNC_PREFILL_TOKENS", rows)
        reference = run()
        plans.clear()
        monkeypatch.setattr(moe, "NO_SYNC_PREFILL_TOKENS", budget)
        with patch.object(
            torch.Tensor, "item", side_effect=AssertionError("read device scalar")
        ):
            actual = run()
    finally:
        parallel._in_moe.reset(token)
    assert len(plans) > 1 and all(ids.shape[0] <= budget for ids, _ in plans)
    seen = torch.cat([ids for ids, _ in plans]).cpu()
    real = torch.ones(rows, dtype=torch.bool)
    if counts is not None:
        for rank, count in enumerate(counts):
            real[rank * stride + count : (rank + 1) * stride] = False
    elif valid is not None:
        real[valid:] = False
    assert (seen[real] >= 0).all() and (seen[~real] == -1).all()
    assert sum(int(count.cpu()[0]) for _, count in plans) == 6 * int(real.sum())
    for value, expected in zip(actual, reference, strict=True):
        if expected is None:
            assert value is None
        else:
            torch.testing.assert_close(value.cpu(), expected.cpu(), atol=0, rtol=0)


def test_peer_combine_keeps_single_window_fast_path():
    hidden = torch.empty((12, 1))
    deferred = object()
    with get_forward().scoped(npu_defer_route_combine=True):
        assert (
            moe._run_expert_windows(
                hidden,
                lambda *args: deferred,
                counts=None,
                stride=None,
                valid=None,
                image_mask=None,
            )
            is deferred
        )
