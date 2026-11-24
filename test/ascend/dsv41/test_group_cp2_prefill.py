"""Two-rank 910B prefill with paged SWA or replay, CED, and C1/C2 pools.

Run with torch.distributed.run --standalone --nproc-per-node=2 on two idle cards.
"""

import os
from functools import partial
from itertools import accumulate
from types import SimpleNamespace

import torch
import torch.distributed as dist
from dsv41.test_ced_tail_source import _prepared_backend, _source_layer

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv41 import engram, group_cp2_dp2
from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
    adapt_prepare,
    prime_rope,
)
from sglang.srt.runtime_context import get_parallel


class Group:
    world_size = 2

    def all_gather_into_tensor(self, output, value):
        dist.all_gather_into_tensor(output, value)


def _cold_batch(backend, fb, counts, ratio):
    device = backend.device
    ends = [0, *accumulate(counts)]
    rows, batch = ends[-1], len(counts)
    columns = (max(counts) + 127) // 128
    pages = 1 + columns * batch
    rng = torch.Generator().manual_seed(2713)
    hidden = torch.randn(rows, 512, generator=rng).bfloat16().to(device)
    req_tables = torch.arange(128, pages * 128, device=device).view(batch, -1)
    backend.req_pool.req_to_token = torch.cat((req_tables[:1] * 0, req_tables))
    backend.pool.num_req_slots = batch + 1
    backend.batch_capacity = batch
    backend.context_len = backend.max_seq = columns * 128
    backend.page_columns = torch.arange(0, backend.max_seq, 128, device=device)
    backend.request_prefixes = {}
    kv = torch.zeros(pages, 128 // ratio, 1, 512, dtype=torch.bfloat16, device=device)
    swa = torch.zeros(pages, 128, 1, 512, dtype=torch.bfloat16, device=device)
    key = torch.zeros(pages, 128 // ratio, 1, 128, dtype=torch.int8, device=device)
    scale = torch.zeros(pages, 128 // ratio, dtype=torch.float16, device=device)
    backend.pool.get_key_buffer = lambda _: kv
    backend.pool.get_swa_buffer = lambda _: swa
    backend.pool.index_pools[ratio] = SimpleNamespace(
        index_k_buffer=[key], index_scale_buffer=[scale]
    )
    fb.batch_size = batch
    fb.extend_seq_lens_cpu = list(counts)
    fb.seq_lens_cpu = torch.tensor(counts, dtype=torch.int32)
    fb.extend_seq_lens = fb.seq_lens_cpu.to(device)
    fb.extend_start_loc = torch.tensor(ends[:-1], dtype=torch.int32, device=device)
    fb.req_pool_indices = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    fb.out_cache_loc = torch.cat([t[:n] for t, n in zip(req_tables, counts)])
    fb.positions = torch.cat([torch.arange(n, device=device) for n in counts])
    fb.input_ids = fb.positions.clone()
    fb.extend_num_tokens = fb.global_num_token_non_padded_cpu = rows
    return hidden


def _setup(device, ratio, request_window, grouped, counts):
    # Reuse the CED inputs and weights for a phase-local two-rank backend.
    backend, fb, hidden = _prepared_backend(
        device, bounded=False, request_window=request_window
    )
    staged = backend.forward_metadata
    backend.ced_enabled = True
    backend.context_len = backend.max_seq
    backend.pool._dsv41_ced_coverage = None
    backend.pool.translate_loc_from_full_to_swa = lambda slots: slots
    backend.request_prefixes = {1: 139, 2: 183}
    fb.is_extend_in_batch = True
    fb.engram_history = None
    fb.seq_lens_cpu = staged.lengths.cpu()
    fb.req_pool_indices = staged.requests.long()
    fb.out_cache_loc = fb.out_cache_loc.clone()
    if request_window:
        fb._dsv41_main_prefix = [159, 187]
    backend.config.compress_ratios[0] = ratio
    if ratio == 2:
        kv = backend.pool.get_key_buffer(0)[:, :64].contiguous()
        pool = backend.pool.index_pools[1]
        backend.pool.get_key_buffer = lambda _: kv
        backend.pool.index_pools[2] = SimpleNamespace(
            index_k_buffer=[pool.index_k_buffer[0][:, :64].contiguous()],
            index_scale_buffer=[pool.index_scale_buffer[0][:, :64].contiguous()],
        )
    if counts is not None:
        hidden = _cold_batch(backend, fb, counts, ratio)
    backend.buckets.clear()
    with envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.override(str(int(grouped))):
        backend.init_forward_metadata_out_graph(fb)
    if grouped:
        for meta in (backend.forward_metadata, backend.tail_forward_metadata):
            assert not hasattr(meta, "qli_plans")
            assert meta.group_queries.qli_plans
    return backend, fb, hidden


def _run(device, ratio, request_window, grouped, counts):
    backend, fb, hidden = _setup(device, ratio, request_window, grouped, counts)
    hasher = SimpleNamespace(
        max_ngram_size=4,
        token_map=torch.arange(backend.max_seq),
        multipliers=torch.arange(1, 9).view(2, 4),
        primes=torch.full((2, 3, 8), 97),
        offsets=torch.arange(24).expand(2, 24) * 97,
        pad_id=0,
        image_token_id=None,
    )
    engram._init_history(hasher, backend.pool.num_req_slots, device)
    hash_batch = partial(engram._hasher_forward, hasher)
    if grouped:
        hashes = group_cp2_dp2.prefill_hash_ids(hash_batch, fb)
        hashes = group_cp2_dp2.interleave_gather(
            hashes, fb.input_ids.numel(), role="fixture_hashes"
        )
    else:
        hashes = hash_batch(fb.input_ids, fb)
    layer = _source_layer(device)
    layer.compress_ratio = ratio
    if counts is not None:
        phase = torch.arange(backend.max_seq)[:, None] * torch.linspace(0.01, 0.1, 32)
        layer.freqs_cis = torch.polar(torch.ones_like(phase), phase).to(device)
    if ratio == 2:
        layer.compressor.project = lambda x: (x.float(), x.float().square())
    positions = fb.positions
    if grouped:
        hidden, _ = group_cp2_dp2.interleave_shard(hidden, role="fixture_hidden")
        positions = backend.forward_metadata.group_queries.positions
    prime_rope((layer,), fb, positions)
    backend.write_tail_source_kv(
        layer=layer, x=hidden, positions=positions, forward_batch=fb
    )
    saved = backend.enter_late_layer_tail(fb)
    try:
        tail = backend.forward_metadata
        if grouped:
            hidden = tail.group_selection.redistribute(hidden, role="fixture_tail")
            positions = tail.group_queries.positions
            prime_rope((layer,), fb, positions)
        else:
            hidden, positions = tail.late_layer_tail.rows(hidden), fb.positions
        prepare = adapt_prepare(None)
        q, _ = prepare(layer, hidden, positions, fb, backend)
        output = backend.forward(
            q,
            None,
            None,
            layer,
            fb,
            compress_ratio=ratio,
            attn_sink=torch.zeros(16, dtype=torch.float32, device=device),
        ).flatten(1)
        if grouped:
            output = group_cp2_dp2.interleave_gather(
                output, tail.rows, role="fixture_output"
            )
    finally:
        backend.exit_late_layer_tail(saved, fb)
    pool = backend.pool.index_pools[ratio]
    state = backend.sources[0].state
    return (
        output,
        hashes,
        (
            backend.pool.get_key_buffer(0),
            pool.index_k_buffer[0],
            pool.index_scale_buffer[0],
            backend.request_window.ring
            if request_window
            else backend.pool.get_swa_buffer(0),
            state.pending,
            state.committed_lengths,
            hasher._dsv41_history,
        ),
    )


@torch.inference_mode()
def main():
    import torch_npu  # noqa: F401

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    try:
        with get_parallel().override(attn_tp_rank=rank, attn_tp_group=Group()):
            cases = [(r, window, None) for r in (1, 2) for window in (False, True)]
            cases.extend(
                (2, False, counts)
                for counts in ([8192], [16384], [1, 127, 128, 513, 2048])
            )
            for ratio, request_window, counts in cases:
                expected, expected_hashes, expected_pools = _run(
                    device, ratio, request_window, False, counts
                )
                actual, hashes, pools = _run(
                    device, ratio, request_window, True, counts
                )
                torch.testing.assert_close(
                    actual.cpu(), expected.cpu(), rtol=0.01, atol=0.01
                )
                assert torch.equal(hashes.cpu(), expected_hashes.cpu())
                generator = torch.Generator().manual_seed(47)
                weight = torch.randn(actual.shape[-1], 64, generator=generator)
                logits = (actual.cpu().float() @ weight).bfloat16()
                reference = (expected.cpu().float() @ weight).bfloat16()
                assert (
                    torch.nn.functional.cosine_similarity(
                        logits.float(), reference.float()
                    )
                    > 0.9999
                ).all()
                assert torch.equal(logits.argmax(-1), reference.argmax(-1))
                for index, (actual_pool, expected_pool) in enumerate(
                    zip(pools, expected_pools)
                ):
                    local = actual_pool.contiguous().view(torch.uint8).flatten()
                    mirror = local.new_empty(2 * local.numel())
                    dist.all_gather_into_tensor(mirror, local)
                    first, second = mirror.cpu().chunk(2)
                    assert torch.equal(first, second), (ratio, request_window, counts)
                    # The split QKV norm threshold can select a different
                    # BF16 rounding for SWA when the lane has fewer rows.
                    swa = index == 3
                    torch.testing.assert_close(
                        actual_pool.cpu(),
                        expected_pool.cpu(),
                        rtol=0.008 if swa else 0,
                        atol=0,
                    )
                if rank == 0:
                    print(
                        f"PASS prefill C{ratio} request_window={request_window} counts={counts}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
