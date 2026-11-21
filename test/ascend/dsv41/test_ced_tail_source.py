"""Compare the final CED source's compact queries with the full-row path."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.test.test_utils import CustomTestCase


def _prepared_backend(device, *, bounded, request_window):
    from sglang.srt.hardware_backend.npu.dsv41.attention.backend import (
        DSV41AttentionBackend,
    )
    from sglang.srt.hardware_backend.npu.dsv41.attention.request_window import (
        BF16RequestWindow,
    )
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import PositionQuant
    from sglang.srt.model_executor.forward_mode import ForwardMode

    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)

    rng = torch.Generator().manual_seed(8121)

    def sample(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=rng).to(dtype).to(device)

    backend = DSV41AttentionBackend.__new__(DSV41AttentionBackend)
    backend.device = device
    backend.max_seq = 512
    backend.local_heads = 16
    backend.batch_capacity = 2
    backend.invalid_requests = set()
    backend.page_columns = torch.arange(0, 512, 128, device=device)
    backend.config = SimpleNamespace(
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        compress_ratios=[1],
    )
    backend.ced_enabled, backend.ced_cut, backend.ced_window = bounded, 0, 4
    backend.buckets, backend.operators, backend.sources = {}, {}, {}
    backend.forward_metadata = backend.tail_forward_metadata = backend._tail_saved = (
        None
    )
    backend.position = PositionQuant(device=device)
    window = None
    if request_window:
        window = BF16RequestWindow(
            num_slots=3,
            layers=1,
            capacity=256,
            page_size=128,
            workspace_rows=1024,
        )
        window.ring.copy_(sample(window.ring.shape))
        get_swa_buffer = window.buffer
    else:
        swa = sample((8, 128, 1, 512))

        def get_swa_buffer(layer_id):
            return swa

    backend.request_window = window

    kv = sample((8, 128, 1, 512))
    key = (sample((8, 128, 1, 128)) * 8).to(torch.int8)
    scale = torch.full((8, 128, 1, 1), 0.05, dtype=torch.float16, device=device)
    backend.pool = SimpleNamespace(
        num_req_slots=3,
        _swa_local_layer_id=lambda layer_id: layer_id,
        get_key_buffer=lambda layer_id: kv,
        get_swa_buffer=get_swa_buffer,
        index_pools={
            1: SimpleNamespace(index_k_buffer=[key], index_scale_buffer=[scale])
        },
        layer_mapping={0: SimpleNamespace(compress_layer_id=0)},
    )

    starts, counts = [139, 183], [246, 13]
    cu = torch.tensor([0, 246, 259], dtype=torch.int32)
    lengths = torch.tensor([385, 196], dtype=torch.int32)
    positions = torch.cat(
        [torch.arange(s, s + n, device=device) for s, n in zip(starts, counts)]
    )
    pages = tensor([[1, 5, 2, 6], [3, 7, 0, 0]])
    req_pages = torch.cat((pages.new_zeros((1, 4)), pages))
    backend.req_pool = SimpleNamespace(
        req_to_token=(
            req_pages[:, :, None] * 128 + torch.arange(128, device=device)
        ).reshape(3, 512)
    )
    full_slots = (
        pages.long()[
            torch.repeat_interleave(tensor([0, 1], torch.int64), tensor(counts)),
            positions // 128,
        ]
        * 128
        + positions % 128
    )
    full = backend._bucket(2, 259, False)
    backend.forward_metadata = full
    full.cu.copy_(cu)
    full.lengths.copy_(lengths)
    full.starts.copy_(tensor(starts, torch.int64))
    full.requests.copy_(tensor([1, 2]))
    full.accepted.copy_(tensor(counts))
    full.full_slots.copy_(full_slots)
    full.compressed_table.copy_(pages)
    for ratio in (1, 2):
        full.cmp_lengths[ratio].copy_(lengths // ratio)
        full.residual[ratio].copy_(lengths % ratio)

    if request_window:
        # Both requests carry replay-only rows. Their main-cache prefix remains
        # published with the Radix pages while SWA is rebuilt on these rows.
        full.source_rows = torch.cat(
            (
                torch.arange(20, 246, device=device),
                torch.arange(250, 259, device=device),
            )
        )
        full.full_slots[:20] = -1
        full.full_slots[246:250] = -1
        full.source_starts = tensor([159, 187], torch.int64)
        full.source_cu = tensor([0, 226, 235])
        full.source_counts = tensor([226, 9])
        full.source_slots = full.full_slots.index_select(0, full.source_rows)
        floors = tensor([128, 180], torch.int64)
        window.prepare(full, counts=counts, floors=floors)
        full.window[:, 0].copy_(
            (positions - floors.repeat_interleave(tensor(counts)) + 1)
            .clamp(1, 128)
            .int()
        )
    else:
        full.table.copy_(pages)
        full.swa_slots.copy_(full_slots)
    fb = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.EXTEND,
        input_ids=positions.clone(),
        positions=positions,
        out_cache_loc=full.full_slots,
        extend_seq_lens=tensor(counts),
        extend_seq_lens_cpu=counts,
        extend_start_loc=tensor([0, 246]),
        extend_num_tokens=259,
        global_num_token_non_padded_cpu=259,
        _dsv41_swa_floor=[128, 180],
    )
    if bounded:
        backend._init_tail_metadata(fb, full, cu, counts, lengths)
    return backend, fb, sample((259, 512))


def _source_layer(device):
    from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
        cached_rope,
        expanded_rope,
    )

    phase = torch.arange(512)[:, None] * torch.linspace(0.01, 0.1, 32)
    q_norm, kv_norm = (
        torch.nn.RMSNorm(512, eps=eps, dtype=torch.bfloat16, device=device)
        for eps in (1e-6, 1e-5)
    )
    q_norm.variance_epsilon, kv_norm.variance_epsilon = q_norm.eps, kv_norm.eps
    kv_norm.weight.fill_(1.125)
    layer = SimpleNamespace(
        layer_id=0,
        is_dsv41=True,
        compress_ratio=1,
        fuse_wqa_wkv=True,
        q_head_norm=False,
        n_local_heads=16,
        head_dim=512,
        q_lora_rank=512,
        qk_nope_head_dim=448,
        wqkv_a=lambda x: (torch.cat((x, x.roll(1, dims=-1)), dim=-1), None),
        wq_b=lambda x: (x.repeat(1, 16), None),
        kv_norm=kv_norm,
        q_norm=q_norm,
        compressor=SimpleNamespace(
            norm=SimpleNamespace(weight=torch.ones(512, device=device)),
            project=lambda x: (x, None),
        ),
        indexer=SimpleNamespace(
            owns_k=True,
            wk=lambda x: x[:, :128].contiguous(),
            k_norm=torch.nn.Identity(),
            wq_b=lambda x: (x[:, :128].repeat(1, 32), None),
            weights_proj=lambda x: (x[:, :32].abs() + 0.25, None),
            head_weight_scale=1.0,
        ),
        freqs_cis=torch.polar(torch.ones_like(phase), phase).to(device),
    )
    layer._get_npu_rope_position_cache = lambda fb, positions, dtype, inverse=False: (
        expanded_rope(cached_rope(layer.freqs_cis, positions, fb))[int(inverse)]
    )
    return layer


@unittest.skipUnless(
    os.environ.get("SGLANG_DSV41_TEST_NPU") == "1",
    "requires the native 910B publication and a visible NPU",
)
class TestCEDTailSource(CustomTestCase):
    def test_replay_and_distinct_request_pages_match_full_source_attention(self):
        self._compare_with_full_attention(request_window=True)

    def test_paged_swa_preserves_the_full_source_window(self):
        self._compare_with_full_attention(request_window=False)

    def test_split_and_full_prefill_match_through_direct_verify(self):
        self._compare_with_full_attention(request_window=True, verify=True)

    @torch.inference_mode()
    def _compare_with_full_attention(self, *, request_window, verify=False):
        from sglang.srt.hardware_backend.npu.dsv41.attention.projections import (
            adapt_prepare,
            prime_rope,
        )

        device = torch.device("npu:0")
        torch.npu.set_device(device)
        layer = _source_layer(device)
        sinks = torch.zeros(16, dtype=torch.float32, device=device)
        prepare = adapt_prepare(lambda *_: self.fail("expected native preparation"))

        def run(backend, fb, x):
            q, _ = prepare(layer, x, fb.positions, fb, backend)
            return backend.forward(
                q, None, None, layer, fb, compress_ratio=1, attn_sink=sinks
            )

        baseline, baseline_batch, baseline_x = _prepared_backend(
            device, bounded=False, request_window=request_window
        )
        prime_rope([layer], baseline_batch, baseline_batch.positions)
        baseline_output = run(baseline, baseline_batch, baseline_x)
        reference = baseline.forward_metadata
        optimized, batch, x = _prepared_backend(
            device, bounded=True, request_window=request_window
        )
        prime_rope([layer], batch, batch.positions)
        tail = optimized.tail_forward_metadata
        rows = tail.late_layer_tail.row_map.long()

        self.assertEqual(rows.cpu().tolist(), [242, 243, 244, 245, 255, 256, 257, 258])
        if request_window:
            self.assertFalse(torch.equal(tail.table[1].cpu(), reference.table[1].cpu()))
            self.assertEqual(
                reference.window[rows, 0].cpu().tolist(), [128] * 4 + [13, 14, 15, 16]
            )
        else:
            self.assertEqual(reference.window[rows, 0].cpu().tolist(), [0] * 8)
            self.assertEqual(tail.window[:, 0].cpu().tolist(), [1, 2, 3, 4] * 2)

        compress = Mock(wraps=layer.compressor.project)
        project_qkv = Mock(wraps=layer.wqkv_a)
        project_q = Mock(wraps=layer.wq_b)
        project_weights = Mock(wraps=layer.indexer.weights_proj)
        layer.compressor.project = compress
        layer.wqkv_a, layer.wq_b = project_qkv, project_q
        layer.indexer.weights_proj = project_weights
        optimized.write_tail_source_kv(
            layer=layer, x=x, positions=batch.positions, forward_batch=batch
        )
        saved = optimized.enter_late_layer_tail(batch)
        try:
            output = run(optimized, batch, x.index_select(0, rows))
            torch.testing.assert_close(
                output.cpu(), baseline_output[rows].cpu(), atol=0.01, rtol=0.01
            )
            torch.testing.assert_close(
                tail.indices[1].cpu(), reference.indices[1][rows].cpu()
            )
            torch.testing.assert_close(
                tail.candidates.cpu(), reference.candidates[rows].cpu()
            )
            torch.testing.assert_close(
                tail.candidates[:, 0].cpu(), (batch.positions // 8).int().cpu()
            )
        finally:
            optimized.exit_late_layer_tail(saved, batch)

        if request_window:
            torch.testing.assert_close(
                optimized.request_window.ring.cpu(), baseline.request_window.ring.cpu()
            )
        else:
            torch.testing.assert_close(
                optimized.pool.get_swa_buffer(0).cpu(),
                baseline.pool.get_swa_buffer(0).cpu(),
                rtol=0,
                atol=0,
            )
        self.assertEqual(
            optimized.sources[0].state.committed_lengths[1:3].cpu().tolist(),
            [385, 196],
        )
        self.assertEqual(compress.call_count, 1)
        self.assertEqual(project_qkv.call_count, 1)
        self.assertEqual(project_qkv.call_args.args[0].shape[0], 259)
        self.assertEqual(project_q.call_count, 1)
        self.assertEqual(project_q.call_args.args[0].shape[0], 8)
        self.assertEqual(project_weights.call_count, 1)
        self.assertEqual(project_weights.call_args.args[0].shape[0], 8)
        torch.testing.assert_close(
            optimized.pool.get_key_buffer(0).cpu(),
            baseline.pool.get_key_buffer(0).cpu(),
            rtol=0,
            atol=0,
        )
        for name in ("index_k_buffer", "index_scale_buffer"):
            torch.testing.assert_close(
                getattr(optimized.pool.index_pools[1], name)[0].cpu(),
                getattr(baseline.pool.index_pools[1], name)[0].cpu(),
                rtol=0,
                atol=0,
            )
        if verify:
            self._compare_verify(baseline, optimized, baseline_x, run)

    def _compare_verify(self, baseline, optimized, x, run):
        from sglang.srt.model_executor.forward_mode import ForwardMode

        device = x.device
        for requests, starts, accepted in (
            ([1, 2], [385, 196], [3, 1]),
            ([2], [197], [0]),
        ):
            with self.subTest(requests=requests):
                ids = torch.tensor(requests, dtype=torch.int64, device=device)
                positions = torch.tensor(starts, device=device)[:, None] + torch.arange(
                    6, device=device
                )
                original_ring = baseline.request_window.ring.cpu()
                expected_ring = original_ring.clone()
                outputs = []
                for backend in (baseline, optimized):
                    fb = SimpleNamespace(
                        batch_size=len(requests),
                        forward_mode=ForwardMode.TARGET_VERIFY,
                        spec_info=SimpleNamespace(draft_token_num=6),
                        positions=positions.flatten(),
                        input_ids=positions.flatten(),
                        req_pool_indices=ids,
                        out_cache_loc=backend.req_pool.req_to_token[
                            ids[:, None], positions
                        ].flatten(),
                    )
                    workspace = backend.request_window.workspace.kv_buffer[0]
                    prior_workspace = workspace.cpu()
                    backend.init_forward_metadata_out_graph(fb)
                    backend.init_forward_metadata_in_graph(fb)
                    meta = backend.forward_metadata
                    outputs.append(run(backend, fb, x[: positions.numel()]).cpu())
                    torch.testing.assert_close(
                        workspace.cpu(), prior_workspace, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        backend.request_window.ring.cpu(), original_ring, atol=0, rtol=0
                    )
                    pending = meta.swa_pending[0].cpu().view(len(requests), 6, 512)
                    if backend is baseline:
                        for req, start, count, values in zip(
                            requests, starts, accepted, pending
                        ):
                            cap = backend.request_window.capacity
                            rows = req * cap + (start + torch.arange(count)) % cap
                            expected_ring[0, rows] = values[:count]
                    backend.commit_after_verify(torch.tensor(accepted))
                    torch.testing.assert_close(
                        backend.request_window.ring.cpu(), expected_ring, atol=0, rtol=0
                    )
                    self.assertEqual(
                        backend.sources[0].state.committed_lengths[ids].cpu().tolist(),
                        [start + count for start, count in zip(starts, accepted)],
                    )
                torch.testing.assert_close(outputs[0], outputs[1], atol=0.01, rtol=0.01)


class TestCEDTailSourceConfiguration(CustomTestCase):
    def test_only_a_self_contained_c1_source_moves_into_the_tail(self):
        from sglang.srt.hardware_backend.npu.dsv41.prefill.replay import (
            can_split_tail_source,
            validate_ced_config,
        )

        source_1017 = dict(
            num_hidden_layers=40,
            compress_ratios=[0] * 2 + [2] * 17 + [1] * 18 + [0] * 3,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
            candidate_source_layer_id=20,
            engram_layer_ids=[1, 14],
            hc_pre_from_prev_sublayer=True,
            sliding_window=128,
            dspark_target_layer_ids=[37, 38, 39],
        )
        config = SimpleNamespace(**source_1017)
        self.assertTrue(can_split_tail_source(config))
        self.assertEqual(validate_ced_config(config), 20)
        c2_ratios = list(config.compress_ratios)
        c2_ratios[20] = 2
        for overrides in (
            {"compress_ratios": c2_ratios},
            {"index_source_layer_ids": [2, 8, 14, 24, 28, 32, 36]},
            {"candidate_source_layer_id": 14},
            {"engram_layer_ids": [1, 14, 20]},
            {"num_hash_layers": 21},
        ):
            with self.subTest(overrides=overrides):
                config = SimpleNamespace(**(source_1017 | overrides))
                self.assertFalse(can_split_tail_source(config))
                self.assertEqual(validate_ced_config(config), 21)


if __name__ == "__main__":
    unittest.main()
