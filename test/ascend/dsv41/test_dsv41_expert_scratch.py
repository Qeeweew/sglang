"""Prefill arena growth must fit one eager generation and preserve Graph storage."""

import weakref
from unittest.mock import patch

import pytest
import torch
from sglang.srt.hardware_backend.npu.dsv41.parallel import RowBuffer
from sglang.srt.hardware_backend.npu.dsv41.moe import ExpertScratch


@pytest.mark.parametrize("hidden,width", [(8, 4), (4, 8), (8, 3)])
def test_stage_reuse_preserves_projection_inputs_and_results(hidden, width):
    arena = ExpertScratch(torch.device("cpu"), hidden, width)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        x, inverse, gate_up, activated, values = arena.get(17)
    assert inverse.shape == (17,)
    storages = {t.untyped_storage().data_ptr() for t in (x, gate_up, activated, values)}
    assert len(storages) == 1
    assert x.untyped_storage().nbytes() == 2 * 17 * (max(hidden, width) + 2 * width)
    assert all(t.is_contiguous() for t in (x, gate_up, activated, values))
    torch.manual_seed(19)
    original = torch.randn_like(x)
    up = torch.randn(2 * width, hidden).half()
    down = torch.randn(hidden, width).half()
    expected_gu = original @ up.T
    gate, linear = expected_gu.float().chunk(2, -1)
    expected_activation = (
        torch.nn.functional.silu(gate.clamp(max=10)) * linear.clamp(-10, 10)
    ).half()
    expected = expected_activation @ down.T
    x.copy_(original)
    torch.mm(x, up.T, out=gate_up)
    gate, linear = gate_up.float().chunk(2, -1)
    activated.copy_(
        (torch.nn.functional.silu(gate.clamp(max=10)) * linear.clamp(-10, 10)).half()
    )
    torch.mm(activated, down.T, out=values)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)


def test_empty_and_negative_row_requests():
    arena = ExpertScratch(torch.device("cpu"), 8, 4)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        empty = arena.get(0, reserve=12)
        assert empty[0].shape == (0, 8) and empty[1].shape == (0,)
        assert arena.capacity == 12
        with pytest.raises(ValueError, match="non-negative"):
            arena.get(-1)


@pytest.mark.parametrize("local_rows", [0, 3, 30])
def test_resident_scratch_keeps_all_slots_for_eager_and_capture(local_rows):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sglang.srt.hardware_backend.npu.dsv41 import moe

    method = moe.ResidentMXFP4Method()
    method.ep, method.first_expert, method.e, method.h, method.i = 8, 0, 48, 8, 4
    plan = SimpleNamespace(
        ids=torch.zeros((5, 6), dtype=torch.int32),
        metadata=torch.tensor([1, local_rows, 0, 1]),
    )
    method.routes = SimpleNamespace(
        gather=Mock(), combine=Mock(return_value="combined")
    )
    method.local_experts = Mock()
    arena = moe.ExpertScratch(torch.device("cpu"), 8, 4)
    x = torch.empty((5, 8), dtype=torch.bfloat16)
    with (
        patch.object(
            torch.npu, "is_current_stream_capturing", return_value=False
        ) as capture,
        patch.object(
            torch.npu, "current_stream", return_value=SimpleNamespace(npu_stream=1)
        ),
        patch.object(moe, "expert_scratch", return_value=arena),
        patch.object(
            torch.Tensor, "item", side_effect=AssertionError("read device scalar")
        ),
    ):
        assert method.routed(x, plan) == "combined"
        capture.return_value = True
        assert method.routed(x, plan) == "combined"
    assert arena.capacity == 30
    gather = method.routes.gather.call_args.kwargs
    assert gather["out"].shape == (30, 8) and gather["inverse"].shape == (30,)
    bounds = method.local_experts.bind.call_args.kwargs
    assert bounds["max_expert_rows"] == 5
    assert bounds["mean_expert_rows"] == 30 / (48 * 8)
    assert bounds["tiling_rows"] == 30
    method.local_experts.bind.assert_called_once()


def test_prefill_budget_survives_short_window_and_releases_for_ced_tail():
    from sglang.srt.hardware_backend.npu.dsv41.moe import NO_SYNC_PREFILL_TOKENS

    arena = ExpertScratch(torch.device("cpu"), 5120, 2304)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        budget = NO_SYNC_PREFILL_TOKENS * 6
        full = arena.get(budget)
        assert arena.capacity == 49152
        size = full[0].untyped_storage().nbytes() + full[1].nbytes
        assert size / 2**20 == 912.1875
        pointer = full[0].data_ptr()
        old = weakref.ref(arena.buffers[0])
        del full
        short = arena.get(6, reserve=budget)
        assert short[0].shape == (6, 5120) and short[0].data_ptr() == pointer
        del short
        arena.get(560 * 6)
        assert arena.capacity == 3360 and old() is None


def test_eager_growth_releases_previous_generation_before_allocation():
    arena = ExpertScratch(torch.device("cpu"), 8, 4)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(16)
        arena.bindings["layer"] = arena.buffers
        old = [weakref.ref(t) for t in arena.buffers]
        allocate = torch.empty

        def bounded_allocate(shape, **kwargs):
            assert all(ref() is None for ref in old), "old eager arena is still live"
            expected = 17 * (8 + 2 * 4) if kwargs["dtype"] == torch.float16 else 17
            assert shape == (expected,), "growth reserves unused activation rows"
            return allocate(shape, **kwargs)

        with patch.object(torch, "empty", side_effect=bounded_allocate):
            result = arena.get(17)
    assert result[0].shape == (17, 8)
    assert arena.capacity == 17 and not arena.retired


def test_growth_preserves_graph_addresses_but_not_later_eager_generations():
    arena = ExpertScratch(torch.device("cpu"), 8, 4)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(8)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=True):
        graph = arena.get(8)
        arena.graph_bindings["layer"] = graph
        pointers = [t.data_ptr() for t in graph]
        graph[0].fill_(3)
        with pytest.raises(RuntimeError, match="before Graph capture"):
            arena.get(9)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(17)
        eager = [weakref.ref(t) for t in arena.buffers]
        arena.get(33)
    assert all(ref() is None for ref in eager)
    assert len(arena.retired) == 1
    assert arena.retired_bindings[0]["layer"] is graph
    assert not arena.bindings and not arena.graph_bindings
    assert [t.data_ptr() for t in graph] == pointers
    assert [t.data_ptr() for t in arena.retired[0]] == pointers
    assert torch.all(graph[0] == 3)


@pytest.mark.parametrize("kind", ["expert", "collective"])
def test_large_dp_arena_is_released_before_returning_to_single_request(kind):
    arena = (
        ExpertScratch(torch.device("cpu"), 8, 4)
        if kind == "expert"
        else RowBuffer(torch.device("cpu"), 8, torch.bfloat16)
    )
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(8)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=True):
        graph = arena.get(8)
    graph_value = graph[0] if kind == "expert" else graph
    graph_value.fill_(11)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(64)
        old = weakref.ref(arena.buffers[0] if kind == "expert" else arena.tensor)
        result = arena.get(16)
        value = result[0] if kind == "expert" else result
        assert old() is None
        assert value.shape == (16, 8)
        assert (arena.capacity if kind == "expert" else arena.tensor.shape[0]) == 16
    assert len(arena.retired) == 1
    assert torch.all(graph_value == 11)


def test_collective_buffer_reuses_rows_and_preserves_captured_generations():
    arena = RowBuffer(torch.device("cpu"), 8, torch.bfloat16)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(8)
        first = arena.tensor.data_ptr()
        assert arena.get(3).data_ptr() == first
        old = weakref.ref(arena.tensor)
        arena.get(9)
        assert old() is None and not arena.retired
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=True):
        graph = arena.get(6)
        graph.fill_(11)
        with pytest.raises(RuntimeError, match="before Graph capture"):
            arena.get(10)
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
        arena.get(10)
        eager = weakref.ref(arena.tensor)
        arena.get(11)
    assert eager() is None and len(arena.retired) == 1
    assert arena.retired[0].data_ptr() == graph.data_ptr()
    assert torch.all(graph == 11)


def test_prefill_fusion_scope_restores_state_without_output_pool():
    from sglang.srt.hardware_backend.npu.dsv41.mhc import (
        prefill_fusion_state,
        prefill_fusions,
        prefill_fusions_enabled,
    )

    assert not prefill_fusions_enabled()
    with pytest.raises(ValueError, match="restore"), prefill_fusions(True):
        first = prefill_fusion_state()
        first["aux"] = object()
        with prefill_fusions(True):
            assert prefill_fusion_state() is not first
            assert not prefill_fusion_state()
        assert prefill_fusion_state() is first
        raise ValueError("restore")
    assert prefill_fusion_state() is None


def test_w8a16_projection_outputs_are_independently_owned():
    from types import SimpleNamespace

    from sglang.srt.hardware_backend.npu.dsv41.linear import W8A16LinearMethod

    method = SimpleNamespace(
        k=256, n=25600, apply_into=lambda layer, x, out, bias: out.fill_(7)
    )
    x = torch.zeros(6, 256, dtype=torch.bfloat16)
    output = W8A16LinearMethod.apply(method, object(), x)
    again = W8A16LinearMethod.apply(method, object(), x)
    again.fill_(9)
    assert output.shape == (6, 25600) and torch.all(output == 7)
    assert output.data_ptr() != again.data_ptr()
