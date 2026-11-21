"""Eager CED integration; no whole-model loading or compilation in forward.

The upstream model owns HcState selection and DSpark/logits row mapping. This
module supplies its tail interface and failure boundary for the 910B backend.
Encoder replay is merged into the normal prefill; cached main/index KV is read-only.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from functools import wraps

import torch

from ..policy import enabled

CED_KEEP_TOKENS = 140


def ced_enabled():
    from sglang.srt.runtime_context import get_exec

    try:
        return get_exec().features.enable_decoder_swa_bounded_replay
    except ValueError:
        # Standalone pool/workspace tests can precede runtime publication.
        return False


def can_split_tail_source(config):
    """The compact source must regenerate every index its consumers inherit."""
    last = max(config.kv_source_layer_ids)
    return (
        config.compress_ratios[last] == 1
        and last in config.index_source_layer_ids
        and config.candidate_source_layer_id == last
        and last not in config.engram_layer_ids
        and getattr(config, "num_hash_layers", 0) <= last
    )


def validate_ced_config(config):
    sources = list(config.kv_source_layer_ids)
    depth = config.num_hidden_layers
    if not sources or any(i < 0 or i >= depth for i in sources):
        raise ValueError("910B CED requires valid KV source layers")
    cut = max(sources) + 1
    if cut >= depth or not config.hc_pre_from_prev_sublayer:
        raise ValueError("910B CED requires late layers and the HcState model loop")
    if any(r not in (0, 1) for r in config.compress_ratios[cut:depth]):
        raise ValueError("910B CED late layers cannot own a new compressor")
    if getattr(config, "num_hash_layers", 0) > cut or any(
        i >= cut for i in config.engram_layer_ids
    ):
        raise ValueError("910B CED requires hash/Engram layers before the cut")
    if config.sliding_window != 128:
        raise ValueError("910B CED currently requires a 128-token SWA window")
    if config.candidate_source_layer_id not in sources:
        raise ValueError("910B CED requires a full-stage candidate KV source")
    captures = getattr(config, "dspark_target_layer_ids", ())
    if any(i < cut or i >= depth for i in captures):
        raise ValueError("910B CED DSpark capture layers must follow the KV cut")
    return cut - 1 if can_split_tail_source(config) else cut


def ced_tail_window(config):
    """Approximate bounded replay, including DSpark's rollback allowance."""
    validate_ced_config(config)
    return CED_KEEP_TOKENS


def validate_options(cfg):
    """NPU-specific restrictions, in addition to the unchanged upstream checks."""
    if not (
        cfg.enable_decoder_swa_bounded_replay or cfg.enable_encoder_swa_bounded_replay
    ):
        return
    from sglang.srt.model_executor.cuda_graph_config import Backend

    tp, ep = getattr(cfg, "tp_size", 1), getattr(cfg, "ep_size", 1)
    attn_dp = getattr(cfg, "attn_dp_size", 1)
    resident_dp = (
        tp == 8
        and ((attn_dp == 8 and ep in (1, 2, 4, 8)) or (attn_dp == 4 and ep == 8))
        and enabled()
        and getattr(cfg, "moe_a2a_backend", "none") == "none"
    )
    if cfg.enable_encoder_swa_bounded_replay and not (
        tp == ep == attn_dp == 1 or (resident_dp and attn_dp in (4, 8) and ep == 8)
    ):
        raise ValueError(
            "910B encoder replay requires single rank or attention DP4/TP2 or DP8/TP1 + EP8"
        )
    unsupported = (
        ("prefill graphs", cfg.cuda_graph_config.prefill.backend != Backend.DISABLED),
        (
            "unqualified resident topology or DP/CP/PP",
            any(
                getattr(cfg, k, 1) != 1
                for k in (
                    "dp_size",
                    "attn_cp_size",
                    "dcp_size",
                    "pp_size",
                )
            )
            or cfg.enable_prefill_cp
            or not ((tp == ep == attn_dp == 1) or resident_dp),
        ),
        ("PD disaggregation", cfg.disaggregation_mode != "null"),
        ("mixed prefill/decode", cfg.enable_mixed_chunk),
        ("LoRA", cfg.enable_lora),
        ("radix sessions", cfg.enable_session_radix_cache),
        ("external cache linker", cfg.enable_unified_cache_external_linker),
        ("hierarchical cache", getattr(cfg, "enable_hierarchical_cache", False)),
        ("unified memory", cfg.enable_unified_memory),
        ("TRT-LLM attention", getattr(cfg, "dsv4_attn_backend", "auto") == "trtllm"),
        (
            "full prompt hidden states",
            getattr(cfg, "enable_return_hidden_states", False)
            or getattr(cfg, "return_hidden_states_mode", None) == "full",
        ),
    )
    for name, active in unsupported:
        if active:
            raise ValueError(f"910B CED does not support {name} yet")
    if cfg.page_size != 128:
        raise ValueError("910B CED prefix caching requires page_size=128")
    if not cfg.chunked_prefill_size or cfg.chunked_prefill_size < 256:
        raise ValueError("910B bounded replay requires chunked-prefill-size >= 256")


def merged_prefill(batch, runner):
    """A worker-only view: leave scheduler allocation/accounting unchanged.

    The scheduler already charges 128 replay rows on request admission. Cached
    locations are retained for reads; the backend projects/writes sources only
    for the original uncached suffix.
    """
    window = runner.token_to_kv_pool.request_window
    if (
        window is None
        or not batch.forward_mode.is_extend_without_speculative()
        or getattr(batch, "_dsv41_main_prefix", None) is not None
    ):
        return batch
    view = copy(batch)
    prefixes, counts, ids, locations, floors, history = [], [], [], [], [], []
    replayed = False
    offset = 0
    hasher = runner.model.model.engram_hasher
    history_size = hasher.max_ngram_size - 1 if hasher is not None else 0
    resets = batch.encoder_swa_reset
    requests = batch.req_pool_indices_cpu
    for i, (req, prefix, count) in enumerate(
        zip(batch.reqs, batch.prefix_lens, batch.extend_lens)
    ):
        reset = resets[i]
        start = max(0, prefix - 128) if reset else prefix
        if reset and prefix % 128:
            raise ValueError("SWA replay requires a page-aligned cached prefix")
        request = int(requests[i])
        if reset:
            window.valid_starts[request] = start
        floors.append(window.valid_starts.get(request, 0))
        prefixes.append(start)
        counts.append(prefix - start + count)
        if start < prefix:
            replayed = True
            ids.append(
                torch.tensor(
                    req.full_untruncated_fill_ids[start:prefix],
                    dtype=torch.int64,
                    device=runner.device,
                )
            )
            # Replay queries read cache pages through req_to_token. They have
            # no write locations, including for DSpark's draft KV injection.
            locations.append(
                torch.full(
                    (prefix - start,), -1, dtype=torch.int64, device=runner.device
                )
            )
        locations.append(batch.out_cache_loc[offset : offset + count])
        ids.append(batch.input_ids[offset : offset + count])
        offset += count
        if history_size:
            preceding = req.full_untruncated_fill_ids[
                max(0, start - history_size) : start
            ]
            history.append(hasher.history_row(preceding))
    # Preserve the scheduler's resolved device suffix (including future IDs).
    # Only cached replay IDs need a new transfer on this forward stream.
    view.input_ids = torch.cat(ids) if replayed else batch.input_ids
    view.prefill_input_ids_cpu = None
    view.prefix_lens, view.extend_lens = prefixes, counts
    view.extend_num_tokens = sum(counts)
    view.out_cache_loc = torch.cat(locations) if replayed else batch.out_cache_loc
    view.extend_logprob_start_lens = [
        n + p - s
        for n, p, s in zip(batch.extend_logprob_start_lens, batch.prefix_lens, prefixes)
    ]
    view.engram_history = (
        torch.tensor(history, dtype=torch.int32, device=runner.device)
        if history_size
        else None
    )
    view._dsv41_main_prefix = list(batch.prefix_lens)
    view._dsv41_swa_floor = floors
    if replayed:
        from sglang.srt.hardware_backend.npu.dsv41.prefill.inputs import (
            restore_replay_embeddings,
        )

        restore_replay_embeddings(view, runner)
    return view


@dataclass(frozen=True)
class TailRows:
    """The upstream LateLayerTail interface for the 910B backend."""

    token_indices: torch.Tensor
    positions: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    swa_out_cache_loc: torch.Tensor
    row_map: torch.Tensor
    contiguous_start: int | None = None
    pad_rows: int = 0
    cp_metadata: object = None
    local_lens_cpu: list[int] | None = None
    global_token_indices: torch.Tensor | None = None
    req_global: torch.Tensor | None = None
    pos_global: torch.Tensor | None = None
    scatter_indices: object = None
    real_rows_count: object = None

    def rows(self, tensor, *, out=None):
        if tensor is None:
            return None
        if (
            self.contiguous_start is not None
            and self.contiguous_start + self.row_map.numel() <= tensor.shape[0]
        ):
            selected = tensor.narrow(0, self.contiguous_start, self.row_map.numel())
            # The compact HcState must not retain the full prefill allocation.
            return (
                selected.clone(memory_format=torch.contiguous_format)
                if out is None
                else out.copy_(selected)
            )
        from ..kernels.ops.row_copy import gather_rows

        return gather_rows(tensor.contiguous(), self.row_map, out=out)

    real_rows = rows


class WindowCoverage:
    """Bounded host metadata tied to FULL *and* SWA physical ownership.

    A submitted prefill is published only after target forward and (for DSpark)
    draft injection both finish. Free/remap or another write invalidates it.
    Decode/verify do not publish additional boundaries in this first version;
    this can reduce cache hits but cannot admit an uncomputed late-layer row.
    """

    @torch.inference_mode(False)
    def __init__(self, capacity):
        self.swa = torch.full((capacity,), -1, dtype=torch.int64, device="cpu")

    def invalidate(self, full):
        self.swa[full.to(device="cpu", dtype=torch.int64)] = -1

    def publish(self, full, swa):
        full, swa = full.cpu().long(), swa.cpu().long()
        self.swa[full] = torch.where(swa > 0, swa, -1)

    def reusable_prefix(self, full, swa, *, page=128, window=128):
        full, swa = full.cpu().long(), swa.cpu().long()
        valid = (swa > 0) & (self.swa[full] == swa)
        # Each candidate's whole window must be computed, including after a
        # tree split inside a long, sparsely computed CED extend.
        ends = torch.arange(page, len(full) + 1, page, device="cpu")
        missing = torch.cat(
            (torch.zeros(1, dtype=torch.int64, device="cpu"), (~valid).cumsum(0))
        )
        complete = missing[ends] == missing[(ends - window).clamp_min(0)]
        usable = ends[complete]
        return int(usable[-1]) if usable.numel() else 0


def adapt_prepare_sync(original_sync):
    if not enabled():
        return original_sync

    @wraps(original_sync)
    def prepare_sync(fb, runner):
        from sglang.srt.hardware_backend.npu.dsv41.prefill.parallel import (
            prepare_replay_dp,
        )

        if runner.model_config.hf_config.model_type == "deepseek_v41":
            prepare_replay_dp(fb, runner)
        return original_sync(fb, runner)

    return prepare_sync


def adapt_forward(original_forward):
    if not enabled():
        return original_forward

    def forward_impl(model, input_ids, positions, forward_batch, *args, **kwargs):
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.runtime_context import get_exec

        if not (
            model.config.model_type == "deepseek_v41"
            and get_exec().features.enable_decoder_swa_bounded_replay
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            return original_forward(
                model, input_ids, positions, forward_batch, *args, **kwargs
            )
        backend = get_attn_backend()
        try:
            model.model._check_late_layer_tail_readers(
                forward_batch, backend.tail_forward_metadata.late_layer_tail
            )
            result = original_forward(
                model, input_ids, positions, forward_batch, *args, **kwargs
            )
            if not model.capture_aux_hidden_states:
                backend.publish_ced_prefill()
            return result
        except BaseException:
            backend.invalidate_ced_requests()
            raise
        finally:
            saved_dp = getattr(forward_batch, "_dsv41_ced_saved_dp", None)
            if saved_dp is not None:
                exit_dp(forward_batch, saved_dp)
            backend.restore_late_layer_tail(forward_batch)

    def exit_dp(fb, saved):
        from inspect import unwrap

        from sglang.srt.models import deepseek_v4

        original_exit_dp = unwrap(deepseek_v4.exit_late_layer_dp)
        original_exit_dp(fb, saved)
        fb._dsv41_ced_saved_dp = None

    @wraps(original_forward)
    def forward(model, input_ids, positions, forward_batch, *args, **kwargs):
        from sglang.srt.hardware_backend.npu.dsv41.mhc import prefill_fusions
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
        )

        fuse_outputs = (
            model.config.model_type == "deepseek_v41"
            and input_ids.device.type == "npu"
            and forward_batch.forward_mode.is_extend_without_speculative()
            and (
                model.capture_aux_hidden_states
                and model.model.dspark_layers_to_capture is not None
                or (
                    not model.capture_aux_hidden_states
                    and forward_batch.capture_hidden_mode == CaptureHiddenMode.NULL
                )
            )
        )
        with prefill_fusions(fuse_outputs):
            return forward_impl(
                model, input_ids, positions, forward_batch, *args, **kwargs
            )

    return forward


def adapt_scatter_tail(original_scatter):
    if not enabled():
        return original_scatter

    @wraps(original_scatter)
    def scatter_tail(tail, rows, num_tokens, zero=False):
        if not isinstance(tail, TailRows):
            return original_scatter(tail, rows, num_tokens, zero=zero)
        from ..kernels.ops.row_copy import scatter_rows

        output = rows.new_zeros((num_tokens, *rows.shape[1:]))
        if (
            tail.contiguous_start is not None
            and rows.shape[0] == tail.row_map.numel()
            and tail.contiguous_start + rows.shape[0] <= num_tokens
        ):
            output.narrow(0, tail.contiguous_start, rows.shape[0]).copy_(rows)
            return output
        return scatter_rows(rows.contiguous(), tail.row_map, output)

    return scatter_tail


def adapt_prefill(original_prefill):
    if not enabled():
        return original_prefill

    @wraps(original_prefill)
    def prefill(worker, *args, **kwargs):
        backend = worker.model_runner.attn_backend
        if backend.pool.request_window is not None:
            args = (merged_prefill(args[0], worker.model_runner), *args[1:])
        if not getattr(backend, "ced_enabled", False):
            return original_prefill(worker, *args, **kwargs)
        injector = worker._kv_injector
        previous = getattr(injector, "_dsv41_in_prefill", False)
        injector._dsv41_in_prefill = True
        try:
            result = original_prefill(worker, *args, **kwargs)
            backend.publish_ced_prefill()
            return result
        except BaseException:
            backend.invalidate_ced_requests()
            raise
        finally:
            injector._dsv41_in_prefill = previous

    return prefill


def adapt_prepare(original_prepare):
    if not enabled():
        return original_prepare

    @wraps(original_prepare)
    def prepare(batch):
        from sglang.srt.runtime_context import get_exec

        if not get_exec().features.enable_encoder_swa_bounded_replay:
            return original_prepare(batch)
        resets = [r.kv.req_pool_idx is None or r.is_retracted for r in batch.reqs]
        for req in batch.reqs:
            if req.return_logprob and req.logprob_start_len not in (
                -1,
                len(req.origin_input_ids),
            ):
                raise ValueError(
                    "encoder SWA replay cannot return cached prompt logprobs"
                )
        with get_exec().features.override(enable_encoder_swa_bounded_replay=False):
            result = original_prepare(batch)
        batch.encoder_swa_reset = resets
        return result

    return prepare


def adapt_cap(original_cap):
    if not enabled():
        return original_cap

    @wraps(original_cap)
    def cap(req, length):
        from sglang.srt.runtime_context import get_exec

        limit = original_cap(req, length)
        if get_exec().features.enable_encoder_swa_bounded_replay:
            limit = min(limit, max(0, length - CED_KEEP_TOKENS))
        return limit

    return cap


def adapt_worker_forward(original_worker):
    if not enabled():
        return original_worker

    @wraps(original_worker)
    def worker_forward(worker, batch=None, *args, **kwargs):
        if (
            batch is not None
            and worker.model_runner.token_to_kv_pool.request_window is not None
        ):
            batch = merged_prefill(batch, worker.model_runner)
        return original_worker(worker, batch, *args, **kwargs)

    return worker_forward


def adapt_replay(original_replay):
    if not enabled():
        return original_replay

    @wraps(original_replay)
    def replay(worker, batch):
        if worker.model_runner.token_to_kv_pool.request_window is None:
            return original_replay(worker, batch)

    return replay


def adapt_dummy(original_dummy):
    if not enabled():
        return original_dummy

    @wraps(original_dummy)
    def dummy(runner, *args, **kwargs):
        from sglang.srt.runtime_context import get_exec

        if runner.model_runner.is_draft_worker:
            with get_exec().features.override(enable_encoder_swa_bounded_replay=False):
                return original_dummy(runner, *args, **kwargs)
        return original_dummy(runner, *args, **kwargs)

    return dummy


def adapt_init_batch(original_batch):
    if not enabled():
        return original_batch

    @wraps(original_batch)
    def init_batch(cls, batch, runner, *args, **kwargs):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            get_server_return_hidden_states_mode,
        )

        if (
            ced_enabled()
            and not runner.is_draft_worker
            and runner.model_config.hf_config.model_type == "deepseek_v41"
            and batch.forward_mode.is_extend_without_speculative()
            and max(
                batch.return_hidden_states_mode, get_server_return_hidden_states_mode()
            )
            == CaptureHiddenMode.FULL
        ):
            # DSpark's capture_hidden_mode=FULL override is internal. A user's
            # full-prompt request lives on ScheduleBatch and must still fail.
            raise ValueError(
                "910B CED cannot return hidden states of all prompt tokens"
            )
        batch_type = (
            ForwardBatch
            if not runner.is_draft_worker
            and runner.model_config.hf_config.model_type == "deepseek_v41"
            and batch.forward_mode.is_extend_without_speculative()
            else cls
        )
        fb = original_batch(batch_type, batch, runner, *args, **kwargs)
        for name in ("_dsv41_main_prefix", "_dsv41_swa_floor"):
            setattr(fb, name, getattr(batch, name, None))
        fb._dsv41_prompt_lengths = [
            max(r.extend_end, len(r.origin_input_ids)) for r in batch.reqs
        ]
        return fb

    return init_batch
