"""V4.1 attention adapter over the independently built 910B ops.

Prefill uses scheduler metadata; decode/verify build device metadata in Graph.
Buffers retain their addresses across replay; forward never reads the host.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import NamedTuple

import torch

from sglang.srt.hardware_backend.npu.dsv41 import group_cp2_dp2
from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
    _normalized_qkv,
    cached_rope,
    gather_rope,
)
from sglang.srt.hardware_backend.npu.dsv41.dspark import verify as _dsv41_verify
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger(__name__)


class _TailSource(NamedTuple):
    layer_id: int
    cache: torch.Tensor
    table: torch.Tensor
    window: torch.Tensor
    q_lora: torch.Tensor


class DSV41AttentionBackend(AttentionBackend):
    # Scheduler CPU lengths remain available for prefill. Decode and verify use
    # device positions, so FutureMap must not pull a per-round CPU mirror.
    needs_cpu_seq_lens = False

    def commit_after_verify(self, lengths):
        from sglang.srt.hardware_backend.npu.dsv41.dspark.verify import _commit

        return _commit(self, lengths)

    def __init__(self, runner):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import PositionQuant

        self.config = runner.model_config.hf_config
        self.pool = runner.token_to_kv_pool
        self.req_pool = runner.req_to_token_pool
        self.token_to_kv_pool = self.pool
        self.req_to_token_pool = self.req_pool
        self.device = torch.device(runner.device)
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.data_type = self.kv_cache_dtype = torch.bfloat16
        self.position = PositionQuant(device=self.device)
        self.context_len = runner.model_config.context_len
        # The request pool includes speculative/overlap lanes after the user
        # context limit. QLI must also see their final page: a six-row verify
        # can cross the limit even when the accepted output is within it.
        token_table = getattr(self.req_pool, "req_to_token", None)
        table_capacity = (
            self.context_len if token_table is None else token_table.shape[1]
        )
        self.max_seq = ((table_capacity + 127) // 128) * 128
        self.page_columns = torch.arange(0, self.max_seq, 128, device=self.device)
        args = runner.server_args
        if getattr(args, "attn_cp_size", 1) != 1:
            raise ValueError("V4.1/910B attention CP is not qualified")
        attn_tp = args.tp_size // args.attn_dp_size
        if args.tp_size % args.attn_dp_size or attn_tp not in (1, 2, 4):
            raise ValueError(
                "V4.1/910B supports attention TP1/2/4 with complete DP groups"
            )
        self.local_heads = self.config.num_attention_heads // (
            1 if group_cp2_dp2.enabled() else attn_tp
        )
        self.batch_capacity = args.max_running_requests
        chunk = args.chunked_prefill_size
        if (
            self.batch_capacity is None
            or chunk is None
            or min(self.batch_capacity, chunk) <= 0
        ):
            raise ValueError(
                "V4.1 attention requires explicit positive max_running_requests and chunked_prefill_size"
            )
        self.buckets = {}
        self.operators = {}
        self.sources = {}
        self.request_prefixes = {}
        self.pool.register_request_state_owner(self)
        self.request_window = self.pool.request_window
        self.forward_metadata = None
        self.low_ratio_prefill_graph = False
        self.tail_forward_metadata = None
        from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import ced_enabled

        self.ced_enabled = ced_enabled()
        self._tail_saved = None
        self._ced_pending = None
        self.invalid_requests = set()
        if self.ced_enabled:
            from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import (
                ced_tail_window,
                validate_ced_config,
            )

            self.ced_cut = validate_ced_config(self.config)
            self.ced_window = ced_tail_window(self.config)

    def clear_request_state(self, request):
        self.request_prefixes.pop(request, None)
        if self.request_window is not None:
            self.request_window.valid_starts.pop(request, None)
        self.invalid_requests.discard(request)
        for source in self.sources.values():
            source.state.committed_lengths[request].zero_()
            source.state.pending[request].zero_()

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @torch.inference_mode(False)
    def _bucket(self, batch, rows, decode, *, with_queries=True):
        # Scheduler metadata can outlive the inference-mode scope that first
        # requested it (e.g. Graph capture followed by eager preparation).
        key = (batch, rows, decode, with_queries)
        if key in self.buckets:
            return self.buckets[key]
        if min(batch, rows) <= 0 or max(batch, rows) >= 2**31:
            raise ValueError("attention requires positive INT32 request/token counts")
        eager = decode is False or decode in ("tail", "group_prefill", "group_tail")
        if eager:
            # Keep one eager metadata generation. Decode/verify metadata owns
            # captured addresses and stays pinned; kernels are shared below.
            for old_key in tuple(self.buckets):
                if old_key[2] == decode:
                    old = self.buckets.pop(old_key)
                    if self.forward_metadata is old:
                        self.forward_metadata = None
                    del old

        def empty(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        meta = SimpleNamespace(
            batch=batch,
            rows=rows,
            cu=empty((batch + 1,), torch.int32),
            lengths=empty((batch,), torch.int32),
            requests=empty((batch,), torch.int32),
            accepted=empty((batch,), torch.int32),
            table=empty((batch, self.max_seq // 128), torch.int32),
            compressed_table=empty((batch, self.max_seq // 128), torch.int32),
            cmp_lengths={r: empty((batch,), torch.int32) for r in (1, 2)},
            residual={r: empty((batch,), torch.int32) for r in (1, 2)},
            window=empty((rows, 1), torch.int32),
            swa_slots=empty((rows,), torch.int64),
            full_slots=empty((rows,), torch.int64),
            starts=empty((batch,), torch.int64),
            late_layer_tail=None,
            verify_window=(
                self.request_window
                if decode in ("verify", "decode_group_query", "verify_group_query")
                else None
            ),
            tail_source=None,
        )
        if with_queries:
            self._init_query_resources(meta, decode)
        self.buckets[key] = meta
        return meta

    def _init_query_resources(self, meta, decode):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import (
            QLIPlan,
            SparseAttention,
            WindowAttention,
        )

        batch, rows = meta.batch, meta.rows
        window_reuse = batch == 1 and decode not in ("group_prefill", "group_tail")
        if window_reuse not in self.operators:
            self.operators[window_reuse] = SimpleNamespace(
                swa=WindowAttention(heads=self.local_heads),
                sparse={
                    r: SparseAttention(
                        ratio=r, window_reuse=window_reuse, heads=self.local_heads
                    )
                    for r in (1, 2)
                },
            )
        operators = self.operators[window_reuse]
        meta.swa, meta.sparse = operators.swa, operators.sparse
        meta.indices = {
            r: torch.full((rows, 1, 512), -1, dtype=torch.int32, device=self.device)
            for r in (1, 2)
        }
        meta.plans, meta.qli_plans, meta.candidates = {}, {}, None
        # QLI and attention use the same exact request and query extents.
        cu = torch.tensor([0] * batch + [rows], dtype=torch.int32, device="cpu")
        lengths = torch.full((batch,), rows, dtype=torch.int32, device="cpu")
        pages = torch.zeros(
            (batch, self.max_seq // 128), dtype=torch.int32, device="cpu"
        )
        for layer in self.config.index_source_layer_ids:
            if self.ced_enabled and decode is False and layer >= self.ced_cut:
                # These Indexer sources run only with the bounded tail metadata.
                continue
            r = self.config.compress_ratios[layer]
            mode = (
                "source"
                if layer == self.config.candidate_source_layer_id
                else "consumer"
                if layer > self.config.candidate_source_layer_id >= 0
                else "off"
            )
            # Indexer layers execute serially and already share meta.indices.
            # Their sequence metadata, schedule and scratch depend on ratio
            # and candidate mode, not the layer's weights or K cache.
            if (r, mode) not in meta.qli_plans:
                meta.qli_plans[r, mode] = QLIPlan(
                    cu,
                    lengths // r,
                    pages,
                    ratio=r,
                    residual=lengths % r,
                    page_size=128 // r,
                    candidate_mode=mode,
                    device=self.device,
                )
                plan = meta.qli_plans[r, mode]
                key_storage = self.pool.index_pools[r].index_k_buffer[0]
                plan.bind_metadata(
                    meta.cu[: batch + 1],
                    meta.cmp_lengths[r][:batch],
                    meta.compressed_table[:batch],
                    residual=meta.residual[r][:batch],
                    page_count=key_storage.numel() // ((128 // r) * 128),
                    max_key_length=self.max_seq // r,
                )
            plan = meta.qli_plans[r, mode]
            meta.plans[layer] = plan
            if mode == "source":
                meta.candidates = plan.candidate_out

    def _group_prefill_queries(self, full, cu, positions, *, role):
        valid = (full.accepted > 0).repeat_interleave(
            torch.diff(cu).to(self.device), output_size=full.rows
        )
        positions = positions.masked_fill(~valid, -1)
        local_positions, _ = group_cp2_dp2.interleave_shard(
            positions, role=(role, "positions"), pad_value=-1
        )
        local = self._bucket(full.batch, local_positions.numel(), role)
        counts = group_cp2_dp2.interleaved_counts(cu, group_cp2_dp2.lane())
        counts[-1] += local.rows - sum(counts)
        local.cu.copy_(
            torch.tensor([0, *counts], dtype=torch.int32, device="cpu").cumsum(0).int()
        )
        for field in ("lengths", "table", "compressed_table"):
            getattr(local, field).copy_(getattr(full, field))
        for r in (1, 2):
            local.cmp_lengths[r].copy_(full.cmp_lengths[r])
            local.residual[r].copy_(full.residual[r])
        local.window.copy_(
            group_cp2_dp2.interleave_shard(
                full.window, role=(role, "window"), pad_value=1
            )[0]
        )
        local.positions = local_positions
        local.query_positions = local_positions.int()
        return local

    @_dsv41_verify.adapt_metadata
    def init_forward_metadata_out_graph(self, fb, in_capture=False):
        self._dsv41_skip_prefill_injection = False
        if fb.forward_mode.is_idle():
            return
        self.tail_forward_metadata = None
        self._ced_pending = None
        if fb.forward_mode.is_decode():
            return self._stage_decode_metadata(fb)
        if not fb.forward_mode.is_extend_without_speculative():
            raise ValueError(
                "V4.1 910B integration currently accepts ordinary prefill/decode"
            )
        batch, rows = fb.batch_size, fb.input_ids.numel()
        counts = list(fb.extend_seq_lens_cpu)
        if len(counts) != batch or any(n < 0 for n in counts):
            raise ValueError(
                "V4.1 prefill requires one nonnegative query count per request"
            )
        lengths = fb.seq_lens_cpu[:batch].to(device="cpu", dtype=torch.int32)
        if bool(((lengths < 0) | (lengths > self.context_len)).any()):
            raise ValueError("V4.1 request exceeds deployment context capacity")
        requests = fb.req_pool_indices[:batch].to(device="cpu", dtype=torch.int64)
        if self.invalid_requests.intersection(requests.tolist()):
            raise RuntimeError(
                "V4.1 request state is invalid after a failed CED forward; release the request slot"
            )
        padding = rows - sum(counts)
        if padding < 0 or (padding and getattr(fb, "dp_padding_mode", None) is None):
            raise ValueError(
                "V4.1 attention query rows disagree with scheduler metadata"
            )
        token_table = self.req_pool.req_to_token
        page_indices = (
            fb.req_pool_indices[:batch].long()[:, None] * token_table.shape[1]
            + self.page_columns[None, :]
        )
        full_pages = (
            token_table.view(-1).index_select(0, page_indices.flatten()).view(batch, -1)
        )
        pages = (full_pages // 128).int()
        # FULL and compressed pools share page IDs. SWA pages have a separate
        # allocator, so translate the first token of every FULL page.
        swa_pages = None
        if self.request_window is None:
            swa_pages = (
                self.pool.translate_loc_from_full_to_swa(full_pages.long()) // 128
            ).int()

        if padding:
            # MLP sync aligns token rows to attention TP (or the MAX_LEN DP
            # stride), without adding a request to scheduler metadata. Keep Cu
            # a complete partition: a private dummy request owns the suffix,
            # sees no KV, writes no cache and commits no compressor state.
            counts.append(padding)
            lengths = torch.cat((lengths, lengths.new_zeros(1)))
            requests = torch.cat((requests, requests.new_zeros(1)))
            pages = torch.cat((pages, torch.zeros_like(pages[:1])))
            if swa_pages is not None:
                swa_pages = torch.cat((swa_pages, torch.zeros_like(swa_pages[:1])))
            batch += 1
        grouped = group_cp2_dp2.prefill_step(fb)
        meta = self._bucket(batch, rows, False, with_queries=not grouped)
        cu = torch.tensor([0] + counts, dtype=torch.int32, device="cpu").cumsum(0).int()
        meta.cu.copy_(cu)
        meta.lengths.copy_(lengths)
        # NPU req pool reserves row zero for Graph padding. Give every padded
        # request a private state slot, and never commit its fabricated query.
        dummy = requests == 0
        for request, length, count in zip(requests.tolist(), lengths.tolist(), counts):
            if request == 0:
                continue
            start = length - count
            if start % 2 and self.request_prefixes.get(request) != start:
                # Speculative commits are device-only. An eager continuation
                # can inspect the existing source state at this rare boundary.
                pair_sources = [s for s in self.sources.values() if s.ratio == 2]
                if not pair_sources or any(
                    int(s.state.committed_lengths[request].item()) != start
                    for s in pair_sources
                ):
                    raise ValueError(
                        "an odd V4.1 continuation requires its committed C2 pending pair"
                    )
            self.request_prefixes[request] = length
        state_ids = torch.where(
            dummy, self.pool.num_req_slots + torch.arange(batch, device="cpu"), requests
        )
        meta.requests.copy_(state_ids.int())
        meta.accepted.copy_(
            torch.where(dummy, 0, torch.tensor(counts, dtype=torch.int32, device="cpu"))
        )
        meta.starts.copy_(
            torch.where(
                dummy,
                0,
                lengths.long() - torch.tensor(counts, dtype=torch.int64, device="cpu"),
            )
        )
        if swa_pages is not None:
            meta.table.copy_(swa_pages)
        meta.compressed_table.copy_(pages)
        # Replay views use captured query buffers but retain the runtime's
        # unpadded cache locations. An idle DP rank therefore has no locations.
        # Keep dummy queries in the collective without letting them write KV.
        slots = fb.out_cache_loc.long()
        slot_rows = slots.numel()
        if slot_rows > rows:
            raise ValueError("V4.1 cache locations exceed captured query rows")
        meta.full_slots.fill_(-1)
        meta.swa_slots.fill_(-1)
        if slot_rows:
            live = torch.repeat_interleave(
                ~dummy, torch.tensor(counts, dtype=torch.int64, device="cpu")
            )[:slot_rows].to(self.device)
            meta.full_slots[:slot_rows].copy_(torch.where(live, slots, -1))
            if self.request_window is None:
                translated = self.pool.translate_loc_from_full_to_swa(slots)
                meta.swa_slots[:slot_rows].copy_(torch.where(live, translated, -1))
        for r in (1, 2):
            meta.cmp_lengths[r].copy_(lengths // r)
            meta.residual[r].copy_(lengths % r)
        self.forward_metadata = meta
        meta.group_queries = meta.group_source = None
        meta.window.zero_()
        meta.source_rows = None
        if self.request_window is not None:
            # SGLang can append a private TP/DP padding request. It owns no
            # main KV, replay history or source-compressor rows.
            floors_cpu = list(
                getattr(fb, "_dsv41_swa_floor", None) or [0] * fb.batch_size
            )
            main_prefix = list(
                getattr(fb, "_dsv41_main_prefix", None)
                or (
                    lengths[: fb.batch_size].long()
                    - torch.tensor(counts[: fb.batch_size])
                ).tolist()
            )
            floors_cpu.extend([0] * (batch - len(floors_cpu)))
            main_prefix.extend([0] * (batch - len(main_prefix)))
            floors = torch.tensor(floors_cpu, dtype=torch.int64, device=self.device)
            self.request_window.prepare(meta, counts=counts, floors=floors)
            source_counts = [
                0 if grouped and is_dummy else int(length) - prefix
                for length, prefix, is_dummy in zip(lengths, main_prefix, dummy)
            ]
            source_rows = torch.cat(
                [
                    torch.arange(int(cu[i + 1]) - count, int(cu[i + 1]), device="cpu")
                    for i, count in enumerate(source_counts)
                ]
            )
            # A cold, unpadded chunk projects every row. Keep that ordinary
            # view instead of copying the full activation at each KV owner.
            meta.source_rows = (
                None if source_counts == counts else source_rows.to(self.device)
            )
            if grouped and meta.source_rows is not None:
                meta.group_source = group_cp2_dp2.PrefillSubset.from_rows(
                    source_rows, self.device
                )
            meta.source_cu = (
                torch.tensor([0] + source_counts, dtype=torch.int32)
                .cumsum(0)
                .int()
                .to(self.device)
            )
            meta.source_starts = torch.tensor(
                main_prefix, dtype=torch.int64, device=self.device
            )
            meta.source_counts = torch.tensor(
                source_counts, dtype=torch.int32, device=self.device
            )
            meta.source_slots = (
                meta.full_slots
                if meta.source_rows is None
                else meta.full_slots.index_select(0, meta.source_rows)
            )
            meta.window[:, 0].copy_(
                (
                    fb.positions
                    - floors.repeat_interleave(
                        torch.tensor(counts, device=self.device), output_size=rows
                    )
                    + 1
                )
                .clamp(1, 128)
                .int()
            )
        if grouped:
            meta.group_queries = self._group_prefill_queries(
                meta, cu, fb.positions, role="group_prefill"
            )
        if self.ced_enabled:
            # CPU coverage bookkeeping happens outside decode/verify capture.
            coverage = self.pool._dsv41_ced_coverage
            if coverage is not None:
                full_cpu = meta.full_slots.cpu()
                coverage.invalidate(full_cpu)
            self._init_tail_metadata(fb, meta, cu, counts, lengths)

    def _stage_decode_metadata(self, fb):
        """Stage only device inputs; positions carry the actual decode prefix.

        The replay view uses captured rows even on an idle attention DP rank.
        Its request zero must neither write KV nor advance a real compressor.
        Cache locations can contain fewer rows than the captured batch.
        """
        batch = fb.batch_size
        if fb.input_ids.numel() != batch or fb.positions.numel() != batch:
            raise ValueError("ordinary decode requires one input row per request")
        grouped = group_cp2_dp2.decode_step(fb)
        meta = self._bucket(batch, batch, True, with_queries=not grouped)
        if not hasattr(meta, "decode_requests"):
            meta.decode_requests = torch.empty(
                batch, dtype=torch.int64, device=self.device
            )
        if self.invalid_requests:
            requests = fb.req_pool_indices[:batch].cpu().tolist()
            if self.invalid_requests.intersection(requests):
                raise RuntimeError(
                    "release V4.1 request slots invalidated by a failed CED forward"
                )
        slots = fb.out_cache_loc
        if slots.numel() > batch:
            raise ValueError("decode cache locations exceed captured query rows")
        meta.full_slots.fill_(-1)
        meta.full_slots[: slots.numel()].copy_(slots)
        meta.decode_requests.copy_(fb.req_pool_indices[:batch])
        meta.starts.copy_(fb.positions)
        if grouped:
            self._stage_group_decode(meta, 1, "decode_group_query")
        else:
            meta.group_decode = meta.group_queries = None
        self.forward_metadata = meta
        # Allocator free/remap invalidates CED coverage for the new suffix.
        # As with verify, no CPU coverage readback is needed here.

    @_dsv41_verify.adapt_metadata_in_graph
    def init_forward_metadata_in_graph(self, fb):
        if not fb.forward_mode.is_decode():
            return
        meta = self.forward_metadata
        batch, requests = meta.batch, meta.decode_requests
        live = requests != 0
        prefix = torch.where(live, meta.starts, 0)
        total = (prefix + 1).int()
        meta.cu.copy_(torch.arange(batch + 1, dtype=torch.int32, device=self.device))
        meta.lengths.copy_(total)
        meta.requests.copy_(
            torch.where(
                live,
                requests,
                self.pool.num_req_slots + torch.arange(batch, device=self.device),
            ).int()
        )
        meta.accepted.copy_(live.int())
        meta.starts.copy_(prefix)
        token_table = self.req_pool.req_to_token
        indices = requests[:, None] * token_table.shape[1] + self.page_columns[None, :]
        full_pages = (
            token_table.view(-1).index_select(0, indices.flatten()).view(batch, -1)
        )
        meta.compressed_table.copy_((full_pages // 128).int())
        if self.request_window is None:
            meta.table.copy_(
                (
                    self.pool.translate_loc_from_full_to_swa(full_pages.long()) // 128
                ).int()
            )
        meta.full_slots.masked_fill_(~live, -1)
        if self.request_window is None:
            translated = self.pool.translate_loc_from_full_to_swa(meta.full_slots)
            meta.swa_slots.copy_(torch.where(live, translated, -1))
        for ratio in (1, 2):
            meta.cmp_lengths[ratio].copy_(total // ratio)
            meta.residual[ratio].copy_(total % ratio)
        if meta.group_decode is not None:
            self._prepare_group_decode(meta, requests)
        elif self.request_window is not None:
            self.request_window.prepare(meta)

    def _stage_group_decode(self, meta, width, role):
        shape = group_cp2_dp2.DecodeRows(meta.batch, width)
        local = self._bucket(shape.local_batch, shape.local_rows, role)
        if not hasattr(local, "swa_pending"):
            layers = len(self.pool.swa_layer_ids())
            local.swa_pending = torch.empty(
                (layers, shape.local_rows, 512),
                dtype=torch.bfloat16,
                device=self.device,
            )
            local.verify_requests = torch.empty(
                shape.local_batch, dtype=torch.int64, device=self.device
            )
        meta.group_decode, meta.group_queries = shape, local
        if not hasattr(meta, "verify_lanes"):
            meta.verify_lanes = torch.arange(width, device=self.device)

    def _prepare_group_decode(self, meta, requests):
        shape, local = meta.group_decode, meta.group_queries
        local.verify_requests.copy_(
            shape.slice(requests, role="target_requests", per_request=True)
        )
        live = local.verify_requests != 0
        local.cu.copy_(
            torch.arange(local.batch + 1, dtype=torch.int32, device=self.device)
            * shape.width
        )
        local.lengths.copy_(
            shape.slice(meta.lengths, role="target_lengths", per_request=True)
        ).masked_fill_(~live, 0)
        for ratio in (1, 2):
            local.cmp_lengths[ratio].copy_(local.lengths // ratio)
            local.residual[ratio].copy_(local.lengths % ratio)
        local.compressed_table.copy_(
            shape.slice(
                meta.compressed_table, role="target_compressed_table", per_request=True
            )
        )
        if local.verify_window is None:
            local.table.copy_(
                shape.slice(meta.table, role="target_table", per_request=True)
            )
            local.swa_slots.copy_(
                shape.slice(meta.swa_slots, role="target_slots", pad_value=-1)
            )

    def finish_group_decode(self, *, verify):
        meta = self.forward_metadata
        if self.request_window is None:
            pending = meta.group_decode.gather_layers(
                meta.group_queries.swa_pending, role="target_swa_paged"
            )
            for layer, values in zip(self.pool.swa_layer_ids(), pending):
                self.position.scatter_bf16(
                    values, meta.swa_slots, self.pool.get_swa_buffer(layer).squeeze(2)
                )
        elif not verify:
            self.request_window.commit_verify(
                meta, requests=meta.decode_requests, accepted=meta.accepted
            )

    def _init_tail_metadata(self, fb, full, cu, counts, lengths):
        from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import TailRows

        prompt_lengths = getattr(fb, "_dsv41_prompt_lengths", None)
        if prompt_lengths is not None:
            prompt_lengths = list(prompt_lengths) + [0] * (
                full.batch - len(prompt_lengths)
            )
        cutoffs = [
            max(0, p - self.ced_window)
            for p in (
                prompt_lengths if prompt_lengths is not None else lengths.tolist()
            )
        ]
        tail_counts = [
            max(1, min(n, int(length) - cutoff)) if n and length > 0 else 0
            for n, length, cutoff in zip(counts, lengths, cutoffs)
        ]
        fb._dsv41_ced_empty_chunk = prompt_lengths is not None and not any(
            n > 0 and length > cutoff
            for n, length, cutoff in zip(counts, lengths, cutoffs)
        )
        row_map_cpu = torch.cat(
            [
                torch.arange(
                    int(cu[i + 1]) - n, int(cu[i + 1]), dtype=torch.int32, device="cpu"
                )
                for i, n in enumerate(tail_counts)
            ]
        )
        rows = row_map_cpu.numel()
        # Request suffixes are ordered and disjoint. The endpoints identify
        # a single contiguous range without reading any device metadata.
        start = int(row_map_cpu[0]) if rows else 0
        contiguous = rows > 0 and int(row_map_cpu[-1]) - start + 1 == rows
        row_map = row_map_cpu.to(self.device)
        grouped = getattr(full, "group_queries", None) is not None
        tail = self._bucket(full.batch, rows, "tail", with_queries=not grouped)
        tail_cu = (
            torch.tensor([0] + tail_counts, dtype=torch.int32, device="cpu")
            .cumsum(0)
            .int()
        )
        tail.cu.copy_(tail_cu)
        for name in ("lengths", "compressed_table"):
            target = getattr(tail, name)
            target.copy_(getattr(full, name))
        tail.requests.copy_(full.requests)
        tail.accepted.copy_(torch.tensor(tail_counts, dtype=torch.int32, device="cpu"))
        tail.accepted.masked_fill_(full.accepted == 0, 0)
        tail.starts.copy_(lengths.long() - torch.tensor(tail_counts, device="cpu"))
        tail.starts.masked_fill_(full.accepted == 0, 0)
        for r in (1, 2):
            tail.cmp_lengths[r].copy_(full.cmp_lengths[r])
            tail.residual[r].copy_(full.residual[r])
        selection = TailRows(
            token_indices=row_map.long(),
            positions=fb.positions.new_empty((rows,), dtype=torch.int64),
            extend_seq_lens=torch.tensor(
                tail_counts[: fb.batch_size], dtype=torch.int32, device=self.device
            ),
            extend_seq_lens_cpu=tail_counts[: fb.batch_size],
            swa_out_cache_loc=tail.swa_slots,
            row_map=row_map,
            contiguous_start=start if contiguous else None,
        )
        selection.rows(fb.positions.long(), out=selection.positions)
        selection.rows(full.full_slots, out=tail.full_slots)
        tail.late_layer_tail = selection
        tail.fb_scalars = {
            name: value.new_tensor(rows)
            for name in ("num_token_non_padded", "global_num_token_non_padded")
            if (value := getattr(fb, name, None)) is not None
        }
        self.tail_forward_metadata = tail
        tail.group_queries = tail.group_selection = None

        def group_queries():
            if grouped:
                tail.group_queries = self._group_prefill_queries(
                    tail, tail_cu, selection.positions, role="group_tail"
                )
                tail.group_selection = group_cp2_dp2.PrefillSubset.from_rows(
                    row_map_cpu, self.device
                )

        if self.request_window is not None:
            floors = list(getattr(fb, "_dsv41_swa_floor", None) or [0] * fb.batch_size)
            floors.extend([0] * (full.batch - len(floors)))
            floor = torch.maximum(
                torch.tensor(floors, device=self.device),
                torch.tensor(cutoffs, device=self.device),
            )
            plan = self.request_window.prepare(tail, counts=tail_counts, floors=floor)
            row_floor = floor.repeat_interleave(
                torch.tensor(tail_counts, device=self.device), output_size=rows
            )
            valid = plan.positions >= row_floor
            tail.swa_slots.masked_fill_(~valid, -1)
            plan.commit_dst.masked_fill_(~valid, -1)
            tail.window[:, 0].copy_(
                (plan.positions - row_floor + 1).clamp(1, 128).int()
            )
            self.request_window.activate(full.ring_plan)
            group_queries()
            return
        tail.table.copy_(full.table)
        selection.rows(full.swa_slots, out=tail.swa_slots)
        floor_lengths = torch.cat(
            [
                # A whole extend retains the preceding chunk's valid SWA KV.
                # A truncated extend must mask its uncomputed local prefix.
                torch.zeros(n, dtype=torch.int32, device="cpu")
                if n == count
                else torch.arange(1, n + 1, dtype=torch.int32, device="cpu")
                for n, count in zip(tail_counts, counts)
            ]
        )
        tail.window[:, 0].copy_(floor_lengths)
        # No device readback in the layer loop or the completion hooks.
        self._ced_pending = (
            full.full_slots.cpu()[row_map_cpu],
            full.swa_slots.cpu()[row_map_cpu],
        )
        group_queries()

    def enter_late_layer_tail(self, fb):
        if self._tail_saved is not None or self.tail_forward_metadata is None:
            raise RuntimeError("invalid CED full/tail metadata transition")
        full, tail = self.forward_metadata, self.tail_forward_metadata
        rows = tail.late_layer_tail
        if tail.tail_source is None and not getattr(fb, "_dsv41_ced_skip_all", False):
            if tail.group_queries is not None:
                subset = tail.group_selection
                for r in (1, 2):
                    tail.group_queries.indices[r].copy_(
                        subset.redistribute(
                            full.group_queries.indices[r], role=("tail_indices", r)
                        )
                    )
                if full.group_queries.candidates is not None:
                    tail.group_queries.candidates.copy_(
                        subset.redistribute(
                            full.group_queries.candidates, role="tail_candidates"
                        )
                    )
            else:
                for r in (1, 2):
                    rows.rows(full.indices[r], out=tail.indices[r])
                if full.candidates is not None:
                    rows.rows(full.candidates, out=tail.candidates)
        # Preserve the exact original objects, including optional scalar tensors.
        updates = {
            "input_ids": rows.rows(fb.input_ids),
            "positions": rows.positions,
            "out_cache_loc": tail.full_slots,
            "extend_seq_lens": rows.extend_seq_lens,
            "extend_seq_lens_cpu": rows.extend_seq_lens_cpu,
            "extend_start_loc": tail.cu[: tail.batch],
            "extend_num_tokens": tail.rows,
            "global_num_token_non_padded_cpu": tail.rows,
        }
        for name in ("num_token_non_padded", "global_num_token_non_padded"):
            value = getattr(fb, name, None)
            if value is not None:
                updates[name] = tail.fb_scalars[name]
        memo = getattr(fb, "_dsv41_rope_memo", None)
        if memo and tail.group_queries is None:
            from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
                expanded_rope,
            )
            from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.position_quant import (
                RopeTables,
            )

            narrowed = {}
            for key, (_, trig, _) in memo.items():
                selected = RopeTables(
                    rows.rows(trig.cos),
                    rows.rows(trig.sin),
                    rows.rows(trig.valid),
                    None,
                )
                narrowed[key] = (rows.positions, selected, expanded_rope(selected))
            updates["_dsv41_rope_memo"] = narrowed
        elif memo:
            updates["_dsv41_rope_memo"] = {}
        saved = (full, {name: getattr(fb, name) for name in updates})
        self._tail_saved = saved
        for name, value in updates.items():
            setattr(fb, name, value)
        self.forward_metadata = tail
        if self.request_window is not None:
            self.request_window.activate(tail.ring_plan)
        return saved

    def exit_late_layer_tail(self, saved, fb):
        if self._tail_saved is not saved:
            raise RuntimeError("CED exit does not match its full metadata")
        self.restore_late_layer_tail(fb)

    def restore_late_layer_tail(self, fb):
        if self.tail_forward_metadata is not None:
            self.tail_forward_metadata.tail_source = None
        if self._tail_saved is not None:
            self.forward_metadata, fields = self._tail_saved
            for name, value in fields.items():
                setattr(fb, name, value)
            self._tail_saved = None
            if self.request_window is not None:
                self.request_window.activate(self.forward_metadata.ring_plan)

    def publish_ced_prefill(self):
        if self._ced_pending is not None:
            self.pool._dsv41_ced_coverage.publish(*self._ced_pending)
            self._ced_pending = None

    def invalidate_ced_requests(self):
        self._ced_pending = None
        if self.forward_metadata is not None:
            requests = self.forward_metadata.requests.cpu().tolist()
            self.invalid_requests.update(r for r in requests if r > 0)

    def get_swa_out_cache_loc(self, forward_batch):
        return self.forward_metadata.swa_slots

    def store_cache(self, *, layer_id, swa_k, forward_batch):
        meta = self.forward_metadata
        if getattr(meta, "group_decode", None) is not None:
            local = meta.group_queries
            values = swa_k.view(-1, 512).contiguous()
            local.swa_pending[self.pool._swa_local_layer_id(layer_id)].copy_(values)
            if local.verify_window is None:
                self.position.scatter_bf16(
                    values,
                    local.swa_slots,
                    self.pool.get_swa_buffer(layer_id).squeeze(2),
                )
            return
        if getattr(meta, "group_queries", None) is not None:
            swa_k = group_cp2_dp2.interleave_gather(swa_k, meta.rows, role="swa")
        if meta.verify_window is not None:
            local = self.pool._swa_local_layer_id(layer_id)
            meta.swa_pending[local].copy_(swa_k.view(-1, 512))
            return
        self.position.scatter_bf16(
            swa_k.contiguous(),
            meta.swa_slots,
            self.pool.get_swa_buffer(layer_id).squeeze(2),
        )
        if self.request_window is not None:
            self.request_window.commit(layer_id, swa_k)

    def _source(self, layer):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import (
            CompressorState,
            ProjectedCompressor,
        )

        if layer.layer_id not in self.sources:
            state = CompressorState.allocate(
                self.pool.num_req_slots + self.batch_capacity + 1,
                device=self.device,
            )
            gamma = layer.compressor.norm.weight.float().contiguous()
            self.sources[layer.layer_id] = ProjectedCompressor(
                layer.compress_ratio, gamma, state
            )
        return self.sources[layer.layer_id]

    def _write_source_cache(self, layer, x):
        meta, r = self.forward_metadata, layer.compress_ratio
        source = self._source(layer)
        # The scheduler owns the committed prefix; warmup/capture do not
        # advance it. A Radix hit restores an even, 128-aligned prefix;
        # its complete C1/C2 rows are already retained with FULL pages and
        # no pending-pair payload is needed. Subsequent odd chunk/decode
        # boundaries use this request's committed pending pair.
        selected = getattr(meta, "source_rows", None)
        starts = meta.starts if selected is None else meta.source_starts
        cu = meta.cu if selected is None else meta.source_cu
        accepted = meta.accepted if selected is None else meta.source_counts
        full_slots = meta.full_slots if selected is None else meta.source_slots
        source.state.committed_lengths[meta.requests.long()] = starts
        group_decode = getattr(meta, "group_decode", None)
        if group_decode is not None:
            kv, scores = group_decode.gather_projection(*layer.compressor.project(x))
        elif getattr(meta, "group_queries", None) is None:
            source_x = x if selected is None else x.index_select(0, selected)
            kv, scores = layer.compressor.project(source_x)
        else:
            subset = meta.group_source
            kv, scores = layer.compressor.project(
                x if subset is None else subset.select(x)
            )
            projection = kv if scores is None else torch.cat((kv, scores), dim=-1)
            projection = (
                group_cp2_dp2.interleave_gather(
                    projection, meta.rows, role="compressor"
                )
                if subset is None
                else subset.gather(projection, role="compressor")
            )
            if scores is None:
                kv = projection
            else:
                kv, scores = projection.split((kv.shape[-1], scores.shape[-1]), dim=-1)
        if group_decode is None:
            kv = kv.contiguous()
            scores = None if scores is None else scores.contiguous()
        prepared = source.prepare(
            kv,
            scores,
            cu[: meta.batch + 1],
            meta.requests,
        )
        completion, rope_positions, _status = source.commit(prepared, accepted)
        slots = torch.where(completion >= 0, full_slots // r, -1).long()
        self.store_compressed_kv(layer, prepared.latent, rope_positions, slots)

    def store_compressed_kv(self, layer, latent, rope_positions, slots):
        trig = gather_rope(layer.freqs_cis, rope_positions)
        if layer.indexer is not None and layer.indexer.owns_k:
            import torch_npu

            index = layer.indexer
            key_raw = index.wk(latent)
            if hasattr(index.k_norm, "variance_epsilon"):
                key = torch_npu.npu_rms_norm(
                    key_raw, index.k_norm.weight, index.k_norm.variance_epsilon
                )[0]
            else:
                key = index.k_norm(key_raw)
            codes, scale = self.position.quantize_key(
                key.contiguous(), cos=trig.cos, sin=trig.sin
            )
            pool = self.pool.index_pools[layer.compress_ratio]
            local = self.pool.layer_mapping[layer.layer_id].compress_layer_id
            self.position.scatter_index(
                codes,
                scale,
                slots,
                pool.index_k_buffer[local].view(-1, 128),
                pool.index_scale_buffer[local].view(-1),
            )
        values = self.position.rope(latent, trig.cos, trig.sin)
        self.position.scatter_bf16(
            values, slots, self.pool.get_key_buffer(layer.layer_id).squeeze(2)
        )

    def write_tail_source_kv(self, *, layer, x, positions, forward_batch):
        """Publish this source's KV before narrowing its attention/Indexer queries."""
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        full, tail = self.forward_metadata, self.tail_forward_metadata
        self._write_source_cache(layer, x)
        q_lora, kv = _normalized_qkv(layer, x)
        cos, sin = layer._get_npu_rope_position_cache(
            forward_batch, positions, kv.dtype, inverse=False
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            kv.unsqueeze(1), None, cos, sin, qk_nope_dim=layer.qk_nope_head_dim
        )
        self.store_cache(layer_id=layer.layer_id, swa_k=kv, forward_batch=forward_batch)
        if getattr(forward_batch, "_dsv41_ced_skip_all", False):
            return
        # The source was computed across the entire chunk. Its compact queries
        # still see the full history, with the original page bases and SWA floors.
        # The next layer can reuse the workspace once this attention has run.
        cache = self.pool.get_swa_buffer(layer.layer_id)
        window = full.window
        if getattr(full, "group_queries", None) is not None:
            window, _ = group_cp2_dp2.interleave_shard(
                tail.late_layer_tail.rows(window),
                role="tail_source_window",
                pad_value=1,
            )
            q_lora = tail.group_selection.redistribute(q_lora, role="tail_q_lora")
        elif tail.rows != full.rows:
            rows = tail.late_layer_tail
            window = rows.rows(window)
            q_lora = rows.rows(q_lora)
        tail.tail_source = _TailSource(
            layer.layer_id, cache, full.table, window, q_lora
        )

    def forward_low_ratio_sources(self, *, layer, x, q_lora, positions, forward_batch):
        meta, r = self.forward_metadata, layer.compress_ratio
        verify = forward_batch.forward_mode.is_target_verify()
        source = meta.tail_source
        if layer.compressor is not None:
            if verify:
                _dsv41_verify.write_provisional_source_kv(self, layer, x)
            elif meta.late_layer_tail is None:
                self._write_source_cache(layer, x)
            elif source is None or layer.layer_id != source.layer_id:
                raise RuntimeError(
                    "CED late queries cannot advance source compressor state"
                )
        if layer.indexer is not None:
            meta = getattr(meta, "group_queries", None) or meta
            index = layer.indexer
            q, _ = index.wq_b(q_lora)
            q = q.view(-1, 32, 128).contiguous()
            pos = positions.long()
            if verify:
                vmemo = getattr(forward_batch, "_dsv41_verify_rope_memo", None)
                if vmemo is None:
                    vmemo = {}
                    forward_batch._dsv41_verify_rope_memo = vmemo
                vkey = (id(layer.freqs_cis), positions.data_ptr(), positions.shape)
                trig = vmemo.get(vkey)
                if trig is None:
                    trig = gather_rope(layer.freqs_cis, pos)
                    vmemo[vkey] = trig
            else:
                trig = cached_rope(layer.freqs_cis, pos, forward_batch)
            codes, qscale = self.position.quantize_query(q, cos=trig.cos, sin=trig.sin)
            weight, _ = index.weights_proj(x)
            weight = (weight.float() * index.head_weight_scale).half().contiguous()
            pool = self.pool.index_pools[r]
            local = self.pool.layer_mapping[layer.layer_id].compress_layer_id
            plan = meta.plans[layer.layer_id]
            plan.run(
                codes,
                pool.index_k_buffer[local].view(-1, 128 // r, 128),
                weight,
                qscale,
                pool.index_scale_buffer[local].view(-1, 128 // r),
                candidates=meta.candidates if plan.mode == 2 else None,
                out=meta.indices[r][: meta.rows, 0],
                positions=getattr(meta, "query_positions", None),
            )

    def forward(
        self, q, k, v, layer, forward_batch, *, compress_ratio, attn_sink, **kwargs
    ):
        m = self.forward_metadata
        source = m.tail_source
        m = getattr(m, "group_queries", None) or m
        if q.shape[0] != m.rows:
            raise ValueError("attention Q rows must match the runtime metadata")
        if q.shape[1:] != (self.local_heads, 512):
            raise ValueError("attention Q must contain exactly the local TP heads")
        table, window = m.table, m.window
        verify_ring = None
        positions = getattr(m, "query_positions", None)
        if source is not None and layer.layer_id == source.layer_id:
            cache, table, window = source.cache, source.table, source.window
            self.forward_metadata.tail_source = None
        elif m.verify_window is not None:
            cache, verify_ring = m.verify_window.verify_inputs(
                m, self.pool._swa_local_layer_id(layer.layer_id)
            )
        else:
            cache = self.pool.get_swa_buffer(layer.layer_id)
        sinks = attn_sink.contiguous()
        if compress_ratio == 0:
            return m.swa(
                q,
                cache,
                table,
                m.cu,
                m.lengths,
                sinks,
                window,
                verify_ring=verify_ring,
                positions=positions,
            )
        r = compress_ratio
        return m.sparse[r](
            q,
            cache,
            self.pool.get_key_buffer(layer.layer_id),
            m.indices[r],
            table,
            m.compressed_table,
            m.cu,
            m.lengths,
            m.cmp_lengths[r],
            m.residual[r],
            window,
            sinks,
            verify_ring=verify_ring,
            positions=positions,
        )
