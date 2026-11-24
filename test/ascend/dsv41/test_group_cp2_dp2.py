"""910B group topology and row ownership at the collective boundary."""

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv41 import group_cp2_dp2 as cp2


def test_opt_in_validates_topology(monkeypatch):
    cfg = SimpleNamespace(
        tp_size=8,
        attn_dp_size=4,
        attn_cp_size=1,
        ep_size=8,
        moe_a2a_backend="none",
        enable_dp_lm_head=True,
        enable_prefill_cp=False,
    )
    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    with envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.override("1"):
        cp2.validate_options(cfg)
        cfg.attn_dp_size = 8
        with pytest.raises(ValueError, match="--attn-dp-size 4"):
            cp2.validate_options(cfg)
        monkeypatch.setenv("SGLANG_DSV41_910B", "0")
        cp2.validate_options(cfg)
    monkeypatch.setenv("SGLANG_DSV41_910B", "1")
    with envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.override("0"):
        cp2.validate_options(cfg)

    monkeypatch.delenv("SGLANG_OPT_DSV41_GROUP_CP2_DP2", raising=False)
    assert envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.get() == "0"
    cp2.validate_options(cfg)
    from sglang.srt.layers.moe import utils as moe_utils

    monkeypatch.setattr(
        moe_utils, "get_moe_a2a_backend", lambda: SimpleNamespace(value="none")
    )
    dp4 = SimpleNamespace(
        tp_size=8, attn_dp_size=4, attn_tp_size=2, attn_cp_size=1, moe_ep_size=8
    )
    dp8 = SimpleNamespace(
        tp_size=8, attn_dp_size=8, attn_tp_size=1, attn_cp_size=1, moe_ep_size=8
    )
    assert cp2.enabled(dp4) is False
    assert cp2.enabled(dp8) is False


@pytest.mark.parametrize("width", [None, 1, 5, 6])
def test_odd_row_ownership_round_trip(width, monkeypatch):
    tokens = torch.arange((3 * width if width is not None else 9) * 2).view(-1, 2)
    parallel = SimpleNamespace(attn_tp_size=2)
    monkeypatch.setattr(cp2, "get_parallel", lambda: parallel)
    shards = []
    for lane in (0, 1):
        parallel.attn_tp_rank = lane
        if width is None:
            local, rows = cp2.interleave_shard(tokens, pad_value=-1, role="test")
            assert rows == len(tokens)
        else:
            local = cp2.batch_half_slice(tokens, 3, width, pad_value=-1, role="test")
        shards.append(local)

    if width is None:
        assert torch.equal(shards[0], tokens[::2])
        assert torch.equal(shards[1][:-1], tokens[1::2])
        padding = shards[1][-1:]
    else:
        assert torch.equal(shards[0], tokens[: 2 * width])
        assert torch.equal(shards[1][:width], tokens[2 * width :])
        padding = shards[1][width:]
    assert (padding == -1).all()

    class Group:
        def all_gather_into_tensor(self, output, local):
            assert local.data_ptr() == shards[1].data_ptr()
            output.copy_(torch.cat(shards))

    parallel.attn_tp_group = Group()
    gather = cp2.interleave_gather if width is None else cp2.batch_half_gather
    assert torch.equal(gather(shards[1], len(tokens), role="test"), tokens)


@pytest.mark.parametrize(
    "width,counts,stride,expected",
    [
        (None, [9, 4, 0, 3], 5, [5, 4, 2, 2, 0, 0, 2, 1]),
        (6, [18, 6, 0, 12], 12, [12, 6, 6, 0, 0, 0, 12, 0]),
    ],
)
def test_rank_counts_mask_padding(width, counts, stride, expected):
    out = torch.empty(8, dtype=torch.int32)
    cp2.rank_counts(torch.tensor(counts), stride, rows_per_request=width, out=out)
    assert out.tolist() == expected


@pytest.mark.parametrize("with_scores", [False, True])
@pytest.mark.parametrize("use_peer", [False, True])
def test_projection_gathers_into_record(monkeypatch, with_scores, use_peer):
    shape = cp2.DecodeRows(3, 1)
    kv = torch.arange(3 * 16).view(3, 16).float()
    scores = kv[:, :8] + 100 if with_scores else None
    out_kv = torch.empty_like(kv)
    out_scores = torch.empty_like(scores) if with_scores else None

    def gather_hccl(out, local):
        assert not use_peer
        full = kv if scores is None else torch.cat((kv, scores), dim=-1)
        assert torch.equal(local, full[:2])
        out[: shape.rows].copy_(full)

    group = SimpleNamespace(all_gather_into_tensor=gather_hccl)
    monkeypatch.setattr(cp2, "group", lambda: group)

    def gather_peer(local, rows, passed_group, *, other, out, other_out, comm_role):
        assert rows == shape.rows and passed_group is group and comm_role == "forward"
        assert out is out_kv and other_out is out_scores
        assert torch.equal(local, kv[:2])
        if with_scores:
            assert torch.equal(other, scores[:2])
        if use_peer:
            out.copy_(kv)
            if other_out is not None:
                other_out.copy_(scores)
            return out, other_out

    monkeypatch.setattr(cp2.peer_collectives, "try_group_gather", gather_peer)
    actual_kv, actual_scores = shape.gather_projection(
        kv[:2],
        None if scores is None else scores[:2],
        out_kv=out_kv,
        out_scores=out_scores,
    )
    assert actual_kv is out_kv and actual_scores is out_scores
    torch.testing.assert_close(actual_kv, kv)
    if with_scores:
        torch.testing.assert_close(actual_scores, scores)


def test_peer_dispatch_roles_and_fallback(monkeypatch):
    peer = cp2.peer_collectives
    cpu_group = object()
    group = SimpleNamespace(world_size=2, cpu_group=cpu_group)
    forward = SimpleNamespace(multi_stream=False)
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="npu"),
        shape=(128, 512),
        nbytes=128 * 512 * 2,
        numel=lambda: 128 * 512,
        element_size=lambda: 2,
        is_contiguous=lambda: True,
    )
    keys = []
    enabled = True
    comm = SimpleNamespace(handle=object(), gather=lambda *args, **kwargs: "ipc")

    def get_buffer(key, factory):
        keys.append(key)
        return comm

    monkeypatch.setattr(peer, "enabled", lambda: enabled)
    monkeypatch.setattr(peer, "get_forward", lambda: forward)
    monkeypatch.setattr(peer, "get_buffer", get_buffer)

    for role in ("forward", "commit", "inject"):
        assert peer.try_group_gather(tensor, 255, group, comm_role=role) == "ipc"
    assert len(set(keys)) == 3

    keys.clear()
    enabled = False
    assert peer.try_group_gather(tensor, 255, group, comm_role="forward") is None
    enabled = True
    forward.multi_stream = True
    assert peer.try_group_gather(tensor, 255, group, comm_role="forward") is None
    forward.multi_stream = False
    tensor.nbytes = peer._GROUP_GATHER_CAPACITY + 32
    assert peer.try_group_gather(tensor, 255, group, comm_role="forward") is None
    assert not keys

    tensor.nbytes = 128 * 512 * 2
    comm.handle = None
    assert peer.try_group_gather(tensor, 255, group, comm_role="forward") is None


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "npu:0",
            marks=pytest.mark.skipif(
                os.getenv("SGLANG_DSV41_TEST_NPU") != "1", reason="opt-in 910B"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "dp_counts,rank_counts,stride",
    [
        ([8192, 0, 0, 0], (4096, 4096, 0, 0, 0, 0, 0, 0), 4096),
        ([0, 8192, 0, 0], (0, 0, 4096, 4096, 0, 0, 0, 0), 4096),
        ([4096, 0, 2049, 0], (2048, 2048, 0, 0, 1025, 1024, 0, 0), 2048),
        ([0, 0, 0, 0], (0,) * 8, 0),
        ([1, 0, 0, 0], (1, 0, 0, 0, 0, 0, 0, 0), 1),
    ],
)
def test_sum_len_compacts_idle_ranks_and_restores_collective_layout(
    monkeypatch, dp_counts, rank_counts, stride, device
):
    from sglang.srt import runtime_context
    from sglang.srt.hardware_backend.npu.dsv41 import moe, parallel
    from sglang.srt.layers import dp_attention

    topology = SimpleNamespace(attn_dp_size=4, tp_size=8, tp_rank=0)
    counts = torch.tensor(dp_counts, dtype=torch.int32, device=device)
    monkeypatch.setattr(cp2, "get_parallel", lambda: topology)
    monkeypatch.setattr(runtime_context, "get_parallel", lambda: topology)
    monkeypatch.setattr(
        dp_attention,
        "get_dp_global_num_tokens",
        lambda: [(x + 1) // 2 * 2 for x in dp_counts],
    )
    monkeypatch.setattr(
        moe,
        "live_dp_counts",
        lambda device, ranks: counts if ranks == 4 else counts.new_empty(ranks),
    )
    monkeypatch.setattr(
        parallel, "_workspace", lambda role, x, rows: x.new_empty((rows, x.shape[1]))
    )
    batch = SimpleNamespace(
        dp_padding_mode=dp_attention.DpPaddingMode.SUM_LEN,
        global_num_tokens_cpu=dp_counts,
    )

    active = [rank for rank, count in enumerate(rank_counts) if count]
    hidden = torch.arange(8 * stride, dtype=torch.float32, device=device).view(
        8, stride, 1
    )
    ids = hidden.to(torch.int64) + 100
    for rank, count in enumerate(rank_counts):
        hidden[rank, count:] = float("nan")

    for local_rank in range(8):
        topology.tp_rank = local_rank
        local = hidden[local_rank, : rank_counts[local_rank]]
        cp2.prepare_moe_rows(batch, local)
        layout = batch._dsv41_group_moe_rows
        assert layout.stride == stride and layout.rank_counts_cpu == rank_counts
        assert layout.counts.tolist() == list(rank_counts)
        layout.global_input_ids = ids.flatten(0, 1)

        calls = []
        gather_ptr = None

        def gather(out, shard):
            nonlocal gather_ptr
            calls.append("gather")
            torch.testing.assert_close(shard[: len(local)], local)
            out.copy_(hidden.flatten(0, 1))
            gather_ptr = out.data_ptr()

        def reduce_scatter(out, partial):
            calls.append("reduce_scatter")
            expected = (hidden + 2).nan_to_num(0)
            torch.testing.assert_close(partial.view_as(hidden), expected)
            out.copy_(expected[local_rank])

        def mlp(x, fb, *, input_ids_global, joint_shared_slice, **kwargs):
            torch.testing.assert_close(x, hidden[active].flatten(0, 1), equal_nan=True)
            assert torch.equal(input_ids_global, ids[active].flatten(0, 1))
            assert fb._dsv41_group_moe_rows.counts.tolist() == [
                rank_counts[r] for r in active
            ]
            if active == list(range(active[0], active[-1] + 1)):
                assert (
                    x.data_ptr() == gather_ptr + active[0] * stride * x.element_size()
                )
            shared = x[joint_shared_slice]
            torch.testing.assert_close(shared, local)
            return (x + 2).nan_to_num(0), shared + 7 if len(shared) else None

        topology.tp_group = SimpleNamespace(
            all_gather_into_tensor=gather, reduce_scatter_tensor=reduce_scatter
        )
        mlp.gate = None
        actual = parallel._run_group_shards(
            SimpleNamespace(mlp=mlp), local, batch, layout, joint=True
        )
        torch.testing.assert_close(actual, 2 * local + 9)
        assert calls == (["gather", "reduce_scatter"] if active else [])
        assert batch._dsv41_group_moe_rows is layout
