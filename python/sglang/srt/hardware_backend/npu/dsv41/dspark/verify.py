"""DSpark linear target verification with provisional C1/C2 state.

Verify writes tentative cache rows for causal attention, but never publishes
compressor state. The sampler subsequently commits exactly anchor+accepted
draft inputs; the sampled bonus has no KV/state entry until the next verify.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

import torch
from sglang.srt.hardware_backend.npu.dsv41.attention.projections import gather_rope

from ..policy import enabled

logger = logging.getLogger(__name__)


def _metadata(self, fb, in_capture=False):
    batch, rows = fb.batch_size, fb.input_ids.numel()
    info = fb.spec_info
    width = int(getattr(info, "draft_token_num", 6))
    if width != 6 or rows != batch * width:
        raise ValueError(
            "910B DSpark target verify requires six linear input rows per request"
        )
    if getattr(info, "ragged_verify_layout", None) is not None:
        raise ValueError(
            "910B DSpark target verify currently uses the static six-row schedule"
        )
    meta = self._bucket(batch, rows, "verify")
    if not hasattr(meta, "verify_sources"):
        meta.verify_sources = {}
        meta.verify_token_mask = torch.empty(rows, dtype=torch.bool, device=self.device)
        meta.verify_commit_lens = torch.empty(
            batch, dtype=torch.int32, device=self.device
        )
        meta.verify_requests = torch.empty(batch, dtype=torch.int64, device=self.device)
        meta.verify_generation = 0
        meta.verify_committed_generation = -1
    meta.verify_generation += 1
    if self.invalid_requests:
        requests = fb.req_pool_indices[:batch].cpu().tolist()
        if self.invalid_requests.intersection(requests):
            raise RuntimeError(
                "release V4.1 request slots invalidated by a failed CED forward"
            )
    # Replay's view retains the unpadded locations. Stage only this input;
    # requests and positions already live in the runner's static buffers.
    meta.full_slots.zero_()
    slots = fb.out_cache_loc[:rows]
    meta.full_slots[: slots.numel()].copy_(slots)
    meta.verify_requests.copy_(fb.req_pool_indices[:batch])
    meta.starts.copy_(fb.positions.view(batch, width)[:, 0])
    meta.verify_width = width
    self.forward_metadata = self._dsv41_verify_meta = meta
    if self.ced_enabled:
        self.tail_forward_metadata = None
        self._ced_pending = None
        # Allocator free/remap already invalidates CED coverage. Verify only
        # writes the reserved suffix, whose coverage has never been published.


def _metadata_in_graph(self, fb):
    meta = self.forward_metadata
    batch, width = meta.batch, meta.verify_width
    requests = meta.verify_requests
    live = requests != 0
    # Positions are authoritative for both upstream prefix-only seq_lens and
    # standalone expanded-length batches. They also avoid a CPU length mirror.
    prefix = torch.where(live, meta.starts, 0)
    total = (prefix + width).int()
    cu = torch.arange(batch + 1, dtype=torch.int32, device=self.device) * width
    token_table = self.req_pool.req_to_token
    page_indices = requests[:, None] * token_table.shape[1] + self.page_columns[None, :]
    full_pages = (
        token_table.view(-1).index_select(0, page_indices.flatten()).view(batch, -1)
    )
    pages = (full_pages // 128).int()

    meta.cu.copy_(cu)
    meta.lengths.copy_(total)
    state_ids = torch.where(
        live,
        requests,
        self.pool.num_req_slots + torch.arange(batch, device=self.device),
    )
    meta.requests.copy_(state_ids.int())
    meta.accepted.copy_(live.int() * width)
    meta.starts.copy_(prefix)
    meta.compressed_table.copy_(pages)
    if self.request_window is None:
        # Private request windows populate both outputs below. Translating the
        # full context table first would read the paged-SWA mapping only to
        # overwrite it, even though target attention never consumes it.
        swa_pages = (
            self.pool.translate_loc_from_full_to_swa(full_pages.long()) // 128
        ).int()
        meta.table.copy_(swa_pages)
        meta.swa_slots.copy_(self.pool.translate_loc_from_full_to_swa(meta.full_slots))
    meta.verify_token_mask.copy_(live.repeat_interleave(width))
    for ratio in (1, 2):
        meta.cmp_lengths[ratio].copy_(total // ratio)
        meta.residual[ratio].copy_(total % ratio)
    if self.request_window is not None:
        meta.ring_window = self.request_window
        self.request_window.prepare(meta, width=width)


def _record(self, layer, source):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import (
        CompressorState,
        ProjectedCompressor,
    )

    meta, ratio = self.forward_metadata, layer.compress_ratio
    records = meta.verify_sources
    if layer.layer_id not in records:
        rows, batch = meta.rows, meta.batch

        def allocate(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=self.device)

        state = CompressorState.allocate(source.slots, device=self.device)
        records[layer.layer_id] = SimpleNamespace(
            source=source,
            shadow=ProjectedCompressor(ratio, source.gamma, state),
            kv=allocate((rows, 512), torch.bfloat16 if ratio == 1 else torch.float32),
            scores=allocate((rows, 512), torch.float32) if ratio == 2 else None,
            latent=allocate((rows, 512), torch.bfloat16),
            pooled=allocate((rows, 512), torch.float32),
            positions=allocate((rows,), torch.int64),
            rope=allocate((rows,), torch.int64),
            starts=allocate((batch,), torch.int64),
            committed_positions=allocate((rows,), torch.int64),
            committed_rope=allocate((rows,), torch.int64),
            status=allocate((batch,), torch.int32),
            provisional=None,
        )
    return records[layer.layer_id]


def _sources(self, *, layer, x, q_lora, positions, forward_batch):
    meta, ratio = self.forward_metadata, layer.compress_ratio
    if layer.compressor is not None:
        source = self._source(layer)
        record = _record(self, layer, source)
        # Copy only to a bucket-owned shadow. Rejected drafts never become the
        # next call's pending partner, including replayed all-reject verifies.
        record.shadow.state.pending.copy_(source.state.pending)
        record.shadow.state.committed_lengths.copy_(source.state.committed_lengths)
        record.shadow.state.committed_lengths[meta.requests.long()] = meta.starts
        kv, scores = layer.compressor.project(x)
        record.kv.copy_(kv)
        if scores is not None:
            record.scores.copy_(scores)
        prepared = record.shadow.prepare(
            record.kv,
            record.scores,
            meta.cu[: meta.batch + 1],
            meta.requests,
            out=record.latent,
            pooled=record.pooled,
            positions=record.positions,
            rope_positions=record.rope,
            starts=record.starts,
        )
        # Every tensor in this object has a stable bucket-owned address. It is
        # safe to retain across Graph replay and an intervening eager verify.
        record.provisional = replace(prepared, owner=source)
        slots = torch.where(
            (prepared.positions >= 0) & meta.verify_token_mask,
            meta.full_slots // ratio,
            -1,
        ).long()
        trig = gather_rope(layer.freqs_cis, prepared.rope_positions)
        if layer.indexer is not None and layer.indexer.owns_k:
            index = layer.indexer
            key = index.k_norm(index.wk(prepared.latent))
            codes, scale = self.position.quantize_key(
                key.contiguous(), cos=trig.cos, sin=trig.sin
            )
            pool = self.pool.index_pools[ratio]
            local = self.pool.layer_mapping[layer.layer_id].compress_layer_id
            self.position.scatter_index(
                codes,
                scale,
                slots,
                pool.index_k_buffer[local].view(-1, 128),
                pool.index_scale_buffer[local].view(-1),
            )
        values = self.position.rope(prepared.latent, trig.cos, trig.sin)
        self.position.scatter_bf16(
            values, slots, self.pool.get_key_buffer(layer.layer_id).squeeze(2)
        )
    if layer.indexer is not None:
        index = layer.indexer
        query, _ = index.wq_b(q_lora)
        query = query.view(-1, 32, 128).contiguous()
        trig = gather_rope(layer.freqs_cis, positions.long())
        codes, qscale = self.position.quantize_query(query, cos=trig.cos, sin=trig.sin)
        weight, _ = index.weights_proj(x)
        weight = (weight.float() * index.head_weight_scale).half().contiguous()
        pool = self.pool.index_pools[ratio]
        local = self.pool.layer_mapping[layer.layer_id].compress_layer_id
        plan = meta.plans[layer.layer_id]
        indices, _ = plan.run(
            codes,
            pool.index_k_buffer[local].view(-1, 128 // ratio, 128),
            weight,
            qscale,
            pool.index_scale_buffer[local].view(-1, 128 // ratio),
            candidates=meta.candidates if plan.mode == 2 else None,
        )
        meta.indices[ratio].fill_(-1)
        meta.indices[ratio][: meta.rows, 0].copy_(indices)


def _commit_sources(meta):
    requests = meta.requests.long()
    for record in meta.verify_sources.values():
        record.source.state.committed_lengths[requests] = meta.starts
        record.source.commit(
            record.provisional,
            meta.verify_commit_lens,
            positions=record.committed_positions,
            rope_positions=record.committed_rope,
            status=record.status,
        )
    if hasattr(meta, "ring_window"):
        meta.ring_window.commit_verify(meta)


def _commit_graph(meta, device):
    """Capture once per verify bucket, using a zero-length warmup commit.

    Zero accepts leave pending payloads untouched. Prefix lengths are seeded
    exactly as in an ordinary commit; live accepts are uploaded after capture.
    Provisional objects are replaced on each forward, but all their tensors
    and every source state have bucket-owned stable addresses.
    """
    sources = tuple(meta.verify_sources)
    if getattr(meta, "commit_graph_sources", None) != sources:
        meta.verify_commit_lens.zero_()
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            _commit_sources(meta)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            _commit_sources(meta)
        stream.synchronize()
        meta.commit_graph, meta.commit_graph_sources = graph, sources
    return meta.commit_graph


def _commit(self, commit_lens, req_pool_indices=None, *, folded=False):
    """Publish verified input prefix, including anchor and excluding bonus.

    Device accepts never round-trip through the host. Device value/order errors
    are reported in the compressor status, checked at a result boundary; CPU
    inputs can be validated immediately. ``folded`` accounts for a commit
    already recorded in the target epilogue, without submitting it twice.
    """
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("DSpark acceptance commit runs outside target Graph capture")
    meta = getattr(self, "_dsv41_verify_meta", None)
    if meta is None:
        raise RuntimeError("no target-verify metadata is available to commit")
    if meta.verify_committed_generation == meta.verify_generation:
        raise RuntimeError("target-verify state was already committed")
    counts = commit_lens.detach().flatten()
    if counts.numel() > meta.batch or (
        counts.device.type == "cpu"
        and bool(((counts < 0) | (counts > meta.verify_width)).any())
    ):
        raise ValueError(
            "commit_lens must count 0..6 verified inputs, never the sampled bonus"
        )
    if not folded and meta.verify_sources:
        graph = _commit_graph(meta, self.device)
        accepted = meta.verify_commit_lens
        accepted.zero_()
        accepted[: counts.numel()].copy_(counts, non_blocking=True)
        if req_pool_indices is not None:
            if req_pool_indices.numel() != counts.numel():
                raise ValueError(
                    "verify commit requests and lengths have different sizes"
                )
            request_ids = req_pool_indices.to(self.device, non_blocking=True).flatten()
            accepted[: counts.numel()].masked_fill_(
                request_ids != meta.verify_requests[: counts.numel()], -1
            )
        accepted.masked_fill_(meta.verify_requests == 0, 0)
        graph.replay()
    statuses = {number: record.status for number, record in meta.verify_sources.items()}
    meta.verify_committed_generation = meta.verify_generation
    return statuses


def adapt_metadata(original_meta):
    if not enabled():
        return original_meta

    @wraps(original_meta)
    def metadata(self, fb, in_capture=False):
        if fb.forward_mode.is_target_verify():
            return _metadata(self, fb, in_capture)
        return original_meta(self, fb, in_capture)

    return metadata


def adapt_metadata_in_graph(original_in_graph):
    if not enabled():
        return original_in_graph

    @wraps(original_in_graph)
    def metadata_in_graph(self, fb):
        if fb.forward_mode.is_target_verify():
            return _metadata_in_graph(self, fb)
        return original_in_graph(self, fb)

    return metadata_in_graph


def adapt_sources(original_sources):
    if not enabled():
        return original_sources

    @wraps(original_sources)
    def sources(self, *, layer, x, q_lora, positions, forward_batch):
        if forward_batch.forward_mode.is_target_verify():
            return _sources(
                self,
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
            )
        return original_sources(
            self,
            layer=layer,
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
        )

    return sources
