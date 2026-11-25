"""Current SGLang boundary and graph contracts used by the 910B adapters."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("tp,rows,owned_rows", [(1, 3, 3), (2, 3, 1), (2, 1, 0)])
def test_sequence_parallel_hccl_gather_and_peer_finish(
    monkeypatch, tp, rows, owned_rows
):
    from sglang.srt.hardware_backend.npu.dsv41 import parallel, peer_collectives
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as geometry
    from sglang.srt.runtime_context import get_forward, get_parallel

    local = torch.arange(rows * 2, dtype=torch.bfloat16).reshape(rows, 2)
    dp = 8 // tp
    global_input = torch.arange(rows * dp * 2).reshape(-1, 2).bfloat16()
    batch = SimpleNamespace(dp_padding_mode=SimpleNamespace(is_max_len=lambda: True))
    calls = []

    def hccl_gather(destination, source):
        calls.append("hccl_gather")
        if destination.shape[0] == global_input.shape[0]:
            destination.copy_(global_input)
        else:
            padded = destination.view(dp, -1, 2)
            padded[:, :rows].copy_(global_input.view(dp, rows, 2))
            padded[:, rows:].zero_()

    def mlp(gathered, context, **kwargs):
        assert context is batch
        assert get_forward().npu_defer_route_combine and kwargs["skip_shared_experts"]
        assert torch.equal(gathered, global_input)
        calls.append("routed")
        return gathered

    mlp.shared_experts = object()
    mlp._forward_shared_experts = lambda shard: (calls.append("shared"), shard * 2)[1]

    def finish(partial, shared, output_rows):
        assert output_rows == rows and torch.equal(partial, global_input)
        if owned_rows:
            assert torch.equal(shared, local[-owned_rows:] * 2)
        else:
            assert shared is None
        calls.append("finish")
        return local

    peer = SimpleNamespace(finish=finish)
    monkeypatch.setattr(peer_collectives, "enabled", lambda: True)
    monkeypatch.setattr(peer_collectives, "moe_communicator", lambda *_: peer)
    monkeypatch.setattr(geometry, "get_local_dp_buffer_len", lambda: rows)
    monkeypatch.setattr(geometry, "get_global_dp_buffer_len", lambda: len(global_input))
    monkeypatch.setattr(
        parallel, "_workspace", lambda role, x, n: x.new_empty((n, x.shape[1]))
    )
    with get_parallel().override(
        tp_size=8,
        tp_rank=2 * tp + tp - 1,
        moe_ep_size=8,
        attn_dp_size=dp,
        attn_dp_rank=2,
        attn_tp_size=tp,
        attn_tp_rank=tp - 1,
        tp_group=SimpleNamespace(all_gather_into_tensor=hccl_gather),
    ):
        assert parallel._use_moe_sequence_parallel(get_parallel(), batch)
        assert (
            parallel._run_sequence_parallel(SimpleNamespace(mlp=mlp), local, batch)
            is local
        )
    expected = ["hccl_gather"] + (["shared"] if owned_rows else []) + ["routed", "finish"]
    assert calls == expected


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


@pytest.mark.parametrize("contiguous_start", [None, 2])
def test_native_tail_preserves_current_output_contract(monkeypatch, contiguous_start):
    from dataclasses import fields

    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import row_copy
    from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import TailRows
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.models.deepseek_v4 import (
        DeepseekV4ForCausalLM,
        LogitsMetadata,
    )

    assert set(LateLayerTail.__struct_fields__) <= {f.name for f in fields(TailRows)}
    if contiguous_start is None:
        monkeypatch.setattr(
            row_copy, "gather_rows", lambda x, rows, *, out=None: x[rows.long()]
        )
    selected = torch.tensor([2, 3])
    tail = TailRows(
        token_indices=selected,
        positions=selected,
        extend_seq_lens=torch.tensor([2]),
        extend_seq_lens_cpu=[2],
        swa_out_cache_loc=selected,
        row_map=selected.int(),
        contiguous_start=contiguous_start,
    )
    if contiguous_start is not None:
        source = torch.arange(16).view(4, 4).T
        compact = tail.rows(source)
        assert compact.is_contiguous()
        torch.testing.assert_close(compact, source[2:4])
        compact.zero_()
        assert tail.rows(source, out=compact) is compact
        torch.testing.assert_close(
            compact, torch.tensor([[2, 6, 10, 14], [3, 7, 11, 15]])
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
