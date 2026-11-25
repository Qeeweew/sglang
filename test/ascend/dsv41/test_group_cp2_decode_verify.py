"""Check packed SWA commits across odd batches and request ownership changes."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.dsv41 import group_cp2_dp2
from sglang.srt.hardware_backend.npu.dsv41.attention.request_window import (
    BF16RequestWindow,
)


def test_reused_position_buffer_uses_current_rope():
    """Verify must rotate Indexer Q using the positions written by the latest forward."""
    from sglang.srt.hardware_backend.npu.dsv41.attention.backend import (
        DSV41AttentionBackend,
    )
    from sglang.srt.hardware_backend.npu.dsv41.attention.projections import prime_rope
    from sglang.srt.model_executor.forward_mode import ForwardMode

    positions = torch.tensor([0, 1])
    angles = torch.arange(3, dtype=torch.float32)[:, None].expand(-1, 32)
    layer = SimpleNamespace(
        layer_id=0,
        compress_ratio=1,
        compressor=None,
        freqs_cis=torch.polar(torch.ones_like(angles), angles),
        indexer=SimpleNamespace(
            wq_b=lambda q: (q, None),
            weights_proj=lambda x: (x, None),
            head_weight_scale=1.0,
        ),
    )
    observed = []

    def quantize_query(q, *, cos, sin):
        observed.append(torch.complex(cos, sin))
        return q, None

    index_pool = SimpleNamespace(
        index_k_buffer=[torch.empty(128, 128)],
        index_scale_buffer=[torch.empty(128)],
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            tail_source=None,
            plans={0: SimpleNamespace(mode=3, run=lambda *args, **kwargs: None)},
            indices={1: torch.empty(2, 1, 512, dtype=torch.int32)},
            rows=2,
        ),
        position=SimpleNamespace(quantize_query=quantize_query),
        pool=SimpleNamespace(
            index_pools={1: index_pool},
            layer_mapping={0: SimpleNamespace(compress_layer_id=0)},
        ),
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
    for live_positions in (torch.tensor([0, 1]), torch.tensor([1, 2])):
        positions.copy_(live_positions)
        prime_rope([layer], batch, positions)
        DSV41AttentionBackend.forward_low_ratio_sources(
            backend,
            layer=layer,
            x=torch.zeros(2, 32),
            q_lora=torch.zeros(2, 32 * 128),
            positions=positions,
            forward_batch=batch,
        )
        torch.testing.assert_close(observed[-1], layer.freqs_cis[live_positions])


@pytest.mark.parametrize("width,role", [(1, "forward"), (6, "commit")])
def test_grouped_swa_commit_follows_request_across_lanes(monkeypatch, width, role):
    capacity, layers = 128, 3
    rings = [torch.zeros((layers, 5 * capacity, 512)) for _ in range(2)]
    expected = torch.zeros_like(rings[0])
    parallel = SimpleNamespace(attn_tp_rank=0)
    monkeypatch.setattr(group_cp2_dp2, "get_parallel", lambda: parallel)

    class Group:
        def all_gather_into_tensor(self, output, local):
            assert torch.equal(local[0], pending[parallel.attn_tp_rank])
            output.copy_(torch.stack(pending))

    gather = group_cp2_dp2.peer_collectives.try_group_gather

    def check_role(*args, comm_role, **kwargs):
        assert comm_role == role
        return gather(*args, comm_role=comm_role, **kwargs)

    monkeypatch.setattr(group_cp2_dp2.peer_collectives, "try_group_gather", check_role)

    def scatter(values, slots, ring):
        live = slots >= 0
        ring[slots[live].long()] = values[live]

    parallel.attn_tp_group = Group()
    prefixes = [0, 124, 125, 126, 127]
    for step in range(20):
        requests = torch.tensor(([1, 2], [2, 3, 4], [1, 2, 0], [3])[step % 4])
        shape = group_cp2_dp2.DecodeRows(len(requests), width)
        starts = torch.tensor([prefixes[int(req)] for req in requests])
        accepted = (
            torch.ones_like(requests)
            if width == 1
            else torch.tensor([1 + 2 * ((step + i) % 3) for i in range(shape.batch)])
        )
        values = torch.arange(layers * shape.rows, dtype=torch.float32).view(layers, -1)
        values = (values + 1 + 100 * step).unsqueeze(-1).expand(-1, -1, 512)
        pending = []
        for lane in (0, 1):
            parallel.attn_tp_rank = lane
            pending.append(
                torch.stack(
                    [shape.slice(v, role="test", pad_value=-999) for v in values]
                )
            )
        for lane, ring in enumerate(rings):
            parallel.attn_tp_rank = lane
            window = SimpleNamespace(
                ring=ring,
                capacity=capacity,
                layer_offsets=torch.arange(layers).view(-1, 1, 1) * ring.shape[1],
                position=SimpleNamespace(scatter_bf16=scatter),
            )
            meta = SimpleNamespace(
                verify_requests=requests,
                verify_lanes=torch.arange(width),
                verify_commit_lens=accepted,
                starts=starts,
                group_decode=shape,
                group_queries=SimpleNamespace(swa_pending=pending[lane]),
            )
            BF16RequestWindow.commit_verify(window, meta, comm_role=role)
        for i, (request, count) in enumerate(zip(requests.tolist(), accepted.tolist())):
            if request == 0:
                continue
            for j in range(count):
                slot = request * capacity + (prefixes[request] + j) % capacity
                expected[:, slot] = values[:, i * width + j]
            prefixes[request] += count
        assert torch.equal(rings[0], expected)
        assert torch.equal(rings[1], expected)


def test_request_ring_eviction_only_prices_full_pages(monkeypatch):
    from sglang.srt.hardware_backend.npu.dsv41.attention.kv_pool import (
        DSV41NPUTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.kv.alloc.demand import KVDemand

    allocator = object.__new__(DSV41NPUTokenToKVPoolAllocator)
    allocator._swa_req_ring = True
    allocator.full_available_size = lambda: 1024
    monkeypatch.setattr(group_cp2_dp2, "enabled", lambda: True)
    demands = []
    tree = SimpleNamespace(
        supports_prefix_sharing=lambda: True, evict_for_alloc=demands.append
    )
    allocator.evict_to_free_tokens(tree, 512)
    allocator.evict_to_free_tokens(tree, 2048)
    assert demands == [KVDemand(full_tokens=2048)]
