"""Current SGLang boundary and graph contracts used by the 910B adapters."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("g32", [False, True], ids=["mxfp4", "w4a8"])
@pytest.mark.parametrize(
    "counts,slot,max_len,rs,gatherv",
    [
        ([3] * 8, 3, True, True, False),
        ([3] * 8, 3, True, False, False),
        ([2, 0, 1, 5, 0, 3, 2, 0], 3, False, False, False),
        ([2, 0, 1, 5, 0, 3, 2, 0], 1, False, False, True),
        ([0, 5, 3, 1], 2, False, False, False),
        ([0] * 8, 7, False, False, False),
    ],
)
def test_joint_experts_dp_transport(
    monkeypatch, g32, counts, slot, max_len, rs, gatherv
):
    from inspect import unwrap

    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.dsv41 import (
        joint_experts,
        parallel,
        peer_collectives,
    )
    from sglang.srt.hardware_backend.npu.dsv41.linear import (
        SharedW8A8Method,
        W8A16LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.dsv41.moe import ResidentMXFP4Method
    from sglang.srt.layers import dp_attention
    from sglang.srt.layers.moe import utils
    from sglang.srt.models.deepseek_common import moe_transport as transport
    from sglang.srt.runtime_context import get_forward, get_parallel

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    monkeypatch.setattr(peer_collectives, "enabled", lambda: False)
    batch = SimpleNamespace(dp_padding_mode=SimpleNamespace(is_max_len=lambda: max_len))
    global_rows = torch.arange(2 * sum(counts), dtype=torch.bfloat16).reshape(-1, 2)
    first = sum(counts[:slot])
    owner = slice(first, first + counts[slot])
    local = global_rows[owner].clone()
    calls = []

    def gather(out, source, context):
        assert source is local and context is batch and out.dtype == torch.bfloat16
        calls.append("gather")
        out.copy_(global_rows)

    def reduce(out, partial):
        calls.append("rs")
        out.copy_(partial[owner] + 3)

    def reducev(partial, *, output, sizes):
        assert sizes == counts
        reduce(output, partial)

    def all_reduce(partial):
        calls.append("sum")
        return partial + 3

    def scatter(out, reduced, context):
        assert context is batch
        calls.append("scatter")
        out.copy_(reduced[owner])

    def mlp(gathered, context, **kwargs):
        calls.append("experts")
        assert kwargs["joint_shared_slice"] == owner and kwargs["skip_shared_experts"]
        assert torch.equal(gathered, global_rows)
        assert get_forward().mlp_reduce_scatter == (rs or gatherv)
        assert kwargs["input_ids_global"] is ids
        return gathered, gathered[owner] * 2 if counts[slot] else None

    ids = torch.arange(sum(counts))
    method = ResidentMXFP4Method(g32=g32)
    method.i, method.limit = 8, 10.0
    gate, down = (
        (SharedW8A8Method(False), SharedW8A8Method(True))
        if g32
        else (W8A16LinearMethod(), W8A16LinearMethod())
    )
    gate.parts, down.k = (8, 8), 8
    mlp._shared_expert_tp1 = mlp.reduce_results = True
    mlp.shared_experts = SimpleNamespace(
        swiglu_limit=method.limit,
        gate_up_proj=SimpleNamespace(
            quant_method=gate, bias=None, weight_scale_inv=torch.empty(16)
        ),
        down_proj=SimpleNamespace(quant_method=down, bias=None),
    )
    mlp.experts = SimpleNamespace(quant_method=method)
    monkeypatch.setattr(transport, "SHARED_EXPERT_LOCAL", True)
    monkeypatch.setattr(
        transport,
        "shared_expert_runs_locally",
        parallel.adapt_shared_before_gather(
            unwrap(transport.shared_expert_runs_locally)
        ),
    )
    assert joint_experts.can_join(mlp) and not transport.shared_expert_runs_locally(mlp)
    method.is_draft = True
    assert transport.shared_expert_runs_locally(mlp)
    method.is_draft = False
    with monkeypatch.context() as invalid:
        invalid.setattr(mlp.shared_experts, "swiglu_limit", 0)
        with pytest.raises(ValueError, match="matching clipping"):
            transport.shared_expert_runs_locally(mlp)
    monkeypatch.setattr(dp_attention, "get_dp_global_num_tokens", lambda: counts)
    monkeypatch.setattr(transport, "get_dp_global_num_tokens", lambda: counts)
    monkeypatch.setattr(
        transport, "get_global_dp_buffer", lambda group: torch.empty_like(global_rows)
    )
    monkeypatch.setattr(
        transport, "get_local_dp_buffer", lambda group: torch.empty_like(local)
    )
    monkeypatch.setattr(
        transport, "get_moe_a2a_backend", lambda: SimpleNamespace(is_none=lambda: True)
    )
    monkeypatch.setattr(transport, "is_cp_active", lambda batch: False)
    monkeypatch.setattr(transport, "is_dp_gatherv_active", lambda: gatherv)
    monkeypatch.setattr(transport, "should_use_dp_reduce_scatterv", lambda: False)
    monkeypatch.setattr(transport, "get_is_extend_in_batch", lambda: False)
    monkeypatch.setattr(transport, "dp_gather_replicate", gather)
    monkeypatch.setattr(transport, "dp_reduce_scatter_tensor", reduce)
    monkeypatch.setattr(transport, "dp_scatter", scatter)
    monkeypatch.setattr(utils, "post_experts_all_reduce", all_reduce)
    geometry = dp_attention._DpGatheredBufferWrapper
    monkeypatch.setattr(geometry, "get_local_dp_buffer_len", lambda: len(local))
    monkeypatch.setattr(geometry, "get_global_dp_buffer_len", lambda: sum(counts))
    monkeypatch.setattr(peer_collectives, "moe_communicator", lambda *_: None)
    monkeypatch.setattr(
        parallel, "_workspace", lambda role, src, rows: src.new_empty(rows, 2)
    )
    with (
        envs.SGLANG_DP_USE_REDUCE_SCATTER.override(rs),
        get_parallel().override(
            tp_size=8,
            attn_tp_size=8 // len(counts),
            attn_tp_rank=0,
            attn_dp_size=len(counts),
            attn_dp_rank=slot,
            attn_cp_size=1,
            moe_ep_size=8,
            moe_tp_size=1,
            moe_dp_size=1,
            tp_group=SimpleNamespace(
                reduce_scatterv=reducev,
                reduce_scatter_tensor=reduce,
                all_gather_into_tensor=lambda out, src: gather(out, src, batch),
            ),
        ),
    ):
        layer = SimpleNamespace(mlp=mlp)
        use_sp = parallel._use_moe_sequence_parallel(get_parallel(), batch, joint=True)
        assert use_sp == (max_len and rs)
        if use_sp:
            out = parallel._run_sequence_parallel(
                layer, local, batch, joint=True, input_ids_global=ids
            )
        else:
            out = parallel._run_dp_joint(layer, local, batch, input_ids_global=ids)
    expected_calls = ["gather"]
    if sum(counts):
        expected_calls += (
            ["experts", "rs"] if rs or gatherv else ["experts", "sum", "scatter"]
        )
    assert calls == expected_calls
    assert torch.equal(out, local * 3 + 3)


def test_joint_experts_tbo_keeps_shared_on_its_microbatch(monkeypatch):
    from sglang.srt.batch_overlap.operations import _StateDict
    from sglang.srt.hardware_backend.npu.dsv41 import joint_experts, parallel
    from sglang.srt.layers import dp_attention
    from sglang.srt.runtime_context import get_forward, get_parallel

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    monkeypatch.setattr(joint_experts, "can_join", lambda mlp: True)
    run = parallel.adapt_tbo_moe(lambda *_: pytest.fail("split experts"))
    states = []

    def mlp(
        hidden,
        batch,
        *,
        joint_shared_slice,
        skip_shared_experts,
        input_ids,
        input_ids_global,
    ):
        assert get_forward().mlp_reduce_scatter and skip_shared_experts
        assert input_ids is input_ids_global is batch._tbo_global_input_ids
        return hidden * 2, hidden[joint_shared_slice] + 1

    with get_parallel().override(attn_dp_size=4, attn_dp_rank=2):
        for counts in ([1, 3, 2, 1], [0, 2, 1, 1]):
            state = _StateDict()
            state.forward_batch = SimpleNamespace(
                tbo_padded_len=counts[2],
                _tbo_global_input_ids=torch.arange(sum(counts)),
            )
            hidden = torch.arange(sum(counts))[:, None].bfloat16()
            state.global_hidden, state.shared_local = hidden, None
            monkeypatch.setattr(
                dp_attention, "get_dp_global_num_tokens", lambda: counts
            )
            run(SimpleNamespace(mlp=mlp), state)
            states.append((state, hidden, counts))
    for state, hidden, counts in states:
        torch.testing.assert_close(state.global_expert_out, hidden * 2)
        torch.testing.assert_close(
            state.shared_local, hidden[sum(counts[:2]) : sum(counts[:3])] + 1
        )


@pytest.mark.parametrize(
    "tp,rows,owner", [(1, 3, slice(6, 9)), (2, 3, slice(8, 9)), (2, 1, slice(3, 3))]
)
def test_joint_experts_peer_owner(monkeypatch, tp, rows, owner):
    from sglang.srt.hardware_backend.npu.dsv41 import parallel, peer_collectives
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as geometry
    from sglang.srt.runtime_context import get_forward, get_parallel

    local = torch.ones(rows, 2, dtype=torch.bfloat16)
    global_input = torch.arange(rows * (8 // tp) * 2).reshape(-1, 2).bfloat16()
    batch = SimpleNamespace(dp_padding_mode=SimpleNamespace(is_max_len=lambda: True))
    calls = []

    def gather(source, destination):
        assert source is local and source.dtype == destination.dtype == torch.bfloat16
        calls.append("gather")
        destination.copy_(global_input)

    def mlp(gathered, context, **kwargs):
        assert context is batch and kwargs["joint_shared_slice"] == owner
        assert get_forward().npu_defer_route_combine and kwargs["skip_shared_experts"]
        calls.append("joint")
        return gathered, gathered[owner] if owner.start < owner.stop else None

    def finish(partial, shared, output_rows):
        assert output_rows == rows and torch.equal(partial, global_input)
        if owner.start < owner.stop:
            assert torch.equal(shared, global_input[owner])
        else:
            assert shared is None
        calls.append("finish")
        return local

    peer = SimpleNamespace(gather=gather, finish=finish)
    monkeypatch.setattr(peer_collectives, "enabled", lambda: True)
    monkeypatch.setattr(peer_collectives, "moe_communicator", lambda *_: peer)
    monkeypatch.setattr(geometry, "get_local_dp_buffer_len", lambda: rows)
    monkeypatch.setattr(geometry, "get_global_dp_buffer_len", lambda: len(global_input))
    monkeypatch.setattr(
        parallel, "_workspace", lambda *args: torch.empty_like(global_input)
    )
    with get_parallel().override(
        attn_dp_size=8 // tp,
        attn_dp_rank=2,
        attn_tp_size=tp,
        attn_tp_rank=tp - 1,
        tp_group=None,
    ):
        assert parallel._use_moe_sequence_parallel(get_parallel(), batch, joint=True)
        assert (
            parallel._run_sequence_parallel(
                SimpleNamespace(mlp=mlp), local, batch, joint=True
            )
            is local
        )
    assert calls == ["gather", "joint", "finish"]


def test_boundary_keeps_new_options_on_other_devices(monkeypatch):
    from sglang.srt.hardware_backend.npu.dsv41.mhc import adapt_boundary
    from sglang.srt.models.deepseek_v4_mhc import make_boundary

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    # Use the real boundary implementation, independent of import-time opt-in.
    from inspect import unwrap

    boundary = adapt_boundary(unwrap(make_boundary))
    norm = SimpleNamespace(
        weight=torch.ones(5120, dtype=torch.bfloat16),
        cast_x_before_out_mul=False,
        variance_size_override=None,
    )
    target = object()
    result = boundary(
        norm,
        accepts_mxfp8=True,
        accepts_mxfp8_deepgemm=True,
        mega_moe_input=target,
        extend=True,
    )
    assert result.accepts_mxfp8 and result.accepts_mxfp8_deepgemm
    assert result.mega_moe_input is target and result.extend


@pytest.mark.parametrize("ready", [False, True])
def test_dspark_replay_loads_complete_batch_and_restores_marker(monkeypatch, ready):
    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(attn_dp_size=1),
    )
    batch = SimpleNamespace(forward_metadata_ready=ready)
    loads = []
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
        ),
        load_batch=lambda current: loads.append(current.forward_metadata_ready),
    )
    backend = object.__new__(NPUCudaGraphBackend)
    backend._decode_runner = runner
    assert NPUCudaGraphBackend.prepare_replay(backend, batch)
    assert loads == [False]
    assert batch.forward_metadata_ready is ready


@pytest.mark.parametrize("page_size", [128, 64])
def test_radix_init_preserves_current_typed_slots(monkeypatch, page_size):
    from sglang.srt.hardware_backend.npu.dsv41.prefill.cache import adapt_radix_init
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TREE_WINDOW, TreeSlot

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    pool = SimpleNamespace(_dsv41_910b=True, compression_ratios=(1, 2))
    slots = (
        TreeSlot(component="FULL", family=TREE_TOKEN),
        TreeSlot(component="SWA", family=TREE_WINDOW, window=128),
    )
    overrides = {TREE_WINDOW: object}
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        page_size=page_size,
        tree_slots=slots,
        component_registry_override=overrides,
    )
    calls = []
    init = adapt_radix_init(lambda cache, args: calls.append(args))
    if page_size != 128:
        with pytest.raises(ValueError, match="128-token pages"):
            init(object(), params)
        assert not calls
        return
    init(object(), params)
    assert calls == [params]
    assert params.tree_slots is slots
    assert params.component_registry_override is overrides
    assert tuple(component.name for component in params.tree_components) == (
        "FULL",
        "SWA",
    )


def test_native_tail_preserves_current_output_contract(monkeypatch):
    from dataclasses import fields

    from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import TailRows
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
        LogitsMetadata,
    )

    assert set(LateLayerTail.__struct_fields__) <= {f.name for f in fields(TailRows)}
    selected = torch.tensor([2, 3])
    tail = TailRows(
        token_indices=selected,
        positions=selected,
        extend_seq_lens=torch.tensor([2]),
        extend_seq_lens_cpu=[2],
        swa_out_cache_loc=selected,
        row_map=selected.int(),
        movement=SimpleNamespace(gather=lambda x, rows: x[rows.long()]),
    )
    monkeypatch.setattr(
        LogitsMetadata, "from_forward_batch", lambda batch: SimpleNamespace()
    )
    calls = []

    def logits(ids, hidden, head, metadata, aux, **kwargs):
        calls.append((ids, metadata.extend_seq_lens_cpu))
        return SimpleNamespace()

    model = SimpleNamespace(
        capture_aux_hidden_states=True,
        _get_dspark_prefill_tail=lambda batch: tail,
        logits_processor=logits,
        lm_head=None,
    )
    hidden = torch.zeros(2, 4)
    result = DeepseekV4ForCausalLM._postprocess_output(
        model,
        ((hidden, hidden), hidden),
        torch.arange(4),
        SimpleNamespace(return_logprob=False),
    )
    torch.testing.assert_close(calls[0][0], selected)
    assert calls[0][1] == [2]
    torch.testing.assert_close(result.hidden_states_token_indices, selected)


@pytest.mark.parametrize("skip", [False, True])
def test_ced_empty_chunk_guards_current_prefill_entry(monkeypatch, skip):
    from inspect import unwrap

    from sglang.srt.hardware_backend.npu.dsv41.dspark.kv_inject import adapt_inject
    from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
        PrefillPlacement,
        TargetHiddenKvInjector,
    )

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    calls = []
    injector = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(_dsv41_skip_prefill_injection=skip)
        ),
        _dsv41_in_prefill=True,
        _inject=lambda **kwargs: calls.append(kwargs),
    )
    inject = adapt_inject(unwrap(TargetHiddenKvInjector.inject_prefill))
    rows = torch.tensor([0])
    inject(
        injector,
        PrefillPlacement(rows, rows, None, None, rows),
        target_hidden=torch.ones(1, 4),
        token_indices=None,
        target_hidden_is_projected=False,
    )
    assert len(calls) == (0 if skip else 1)


def test_native_greedy_sampling_reaches_torch_argmax(monkeypatch):
    from sglang.kernels.ops.sampling import argmax

    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    logits = SimpleNamespace(device=SimpleNamespace(type="npu"))
    expected = torch.tensor([3])
    calls = []

    def torch_argmax(value, dim):
        calls.append((value, dim))
        return expected

    monkeypatch.setattr(torch, "argmax", torch_argmax)
    assert argmax(logits) is expected
    assert calls == [(logits, -1)]


def test_dspark_helpers_ignore_transfer_to_npu_cuda_flag(monkeypatch):
    from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        BuildOutTokens,
    )

    class NPUTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

        @property
        def is_cuda(self):
            return True

    value = torch.Tensor._make_subclass(NPUTensor, torch.ones(1))
    monkeypatch.setenv("SGLANG_DSV41_910B", "0")
    assert inputs_on_cuda(value)
    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    assert not inputs_on_cuda(value)
    result = object()
    monkeypatch.setattr(BuildOutTokens, "torch", lambda **kwargs: result)
    assert BuildOutTokens.execute(draft_tokens=value) is result
