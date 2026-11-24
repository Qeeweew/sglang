"""Run with torch.distributed.run --standalone --nproc-per-node=2 on idle NPUs."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.npu_peer_collectives import npu_peer_group_gather
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dsv41 import group_cp2_dp2 as cp2
from sglang.srt.hardware_backend.npu.dsv41 import peer_collectives as peer
from sglang.srt.hardware_backend.npu.dsv41.dspark.draft import adapt_inject
from sglang.srt.runtime_context import get_parallel


class Group:
    world_size = 2

    def __init__(self, cpu_group):
        self.cpu_group = cpu_group

    def all_gather_into_tensor(self, output, value):
        dist.all_gather_into_tensor(output, value)


def _tensor_pair(rows, local_rows, outer, width, dtype, rank, device):
    dims = (rows, width) if outer is None else (outer, rows, width)
    expected = (
        ((torch.arange(torch.Size(dims).numel()) % 113) - 56).view(dims).to(dtype)
    )
    shape = (*dims[:-2], local_rows, width)
    local = torch.full(shape, -123, dtype=dtype)
    own = slice(0, local_rows) if rank == 0 else slice(local_rows, rows)
    local[..., : expected[..., own, :].shape[-2], :].copy_(expected[..., own, :])
    return (
        local.to(device),
        torch.empty(dims, dtype=dtype, device=device),
        expected,
        own,
    )


def _case(shape, tensors, rank, device, *, skip_own=False):
    fixtures = [
        _tensor_pair(shape.rows, shape.local_rows, *spec, rank, device)
        for spec in tensors
    ]
    for _, output, _, _ in fixtures:
        output.fill_(123)

    def launch(handle):
        a, out, _, _ = fixtures[0]
        other, other_out = fixtures[1][:2] if len(fixtures) == 2 else (None, None)
        npu_peer_group_gather(
            a,
            out,
            handle,
            other=other,
            other_output=other_out,
            skip_own_output=skip_own,
        )

    def check(delta):
        for _, output, expected, own in fixtures:
            ref = expected + delta
            if skip_own:
                ref[..., own, :] = 123
            torch.testing.assert_close(output.cpu(), ref, rtol=0, atol=0)

    return launch, check, [x[0] for x in fixtures]


def _gather_cases(comm, rank, device):
    cases = [
        _case(cp2.DecodeRows(batch, width), tensors, rank, device, skip_own=skip_own)
        for batch in (1, 3, 31, 32)
        for width in (1, 5, 6)
        for tensors, skip_own in (
            ([(None, 16, torch.bfloat16)], False),
            ([(None, 8, torch.float32)], False),
            ([(None, 512, torch.float32)] * 2, False),
            ([(None, 512, torch.bfloat16), (None, 512, torch.float32)], False),
            ([(None, 5120, torch.bfloat16), (None, 15360, torch.bfloat16)], False),
            ([(3, 512, torch.bfloat16)], True),
        )
    ]
    cases.extend(
        [
            _case(cp2.DecodeRows(41, 6), [(57, 512, torch.bfloat16)], rank, device),
            _case(cp2.DecodeRows(256, 1), [(57, 512, torch.bfloat16)], rank, device),
            _case(
                cp2.DecodeRows(255, 1),
                [(None, 5120, torch.bfloat16), (None, 15360, torch.bfloat16)],
                rank,
                device,
            ),
        ]
    )
    for launch, check, _ in cases:
        launch(comm.handle)
        check(0)

    # The graph alternates channel-empty and multi-MiB messages.
    replay_cases = (cases[0], cases[-3], cases[2], cases[-2], cases[5], cases[-1])
    draft = _case(
        cp2.DecodeRows(2, 3), [(3, 512, torch.bfloat16)], rank, device, skip_own=True
    )
    draft[0](comm.handle)
    draft[1](0)
    pool = torch.npu.graph_pool_handle()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=pool, auto_dispatch_capture=True):
        for launch, _, _ in replay_cases:
            launch(comm.handle)
    draft_graph = torch.npu.NPUGraph()
    with torch.npu.graph(draft_graph, pool=pool, auto_dispatch_capture=True):
        draft[0](comm.handle)
    for step in range(1, 9):
        for _, _, inputs in replay_cases:
            for input in inputs:
                input.add_(1)
        graph.replay()
        for _, check, _ in replay_cases:
            check(step)
        for input in draft[2]:
            input.add_(1)
        draft_graph.replay()
        draft[1](step)


def _inject_parity(device):
    generator = torch.Generator().manual_seed(2026)

    def matrix(*shape):
        return (torch.randn(shape, generator=generator) * 0.125).bfloat16().to(device)

    projected_rows = []
    proj = matrix(128, 64)

    def project(hidden):
        projected_rows.append(hidden.shape[0])
        return hidden @ proj

    freqs = torch.polar(torch.ones(256, 32), torch.randn(256, 32, generator=generator))
    freqs = freqs.to(device)
    stages = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                layer_id=i,
                wkv=lambda x, w=matrix(64, 512): (x @ w,),
                kv_norm=lambda x: torch.nn.functional.rms_norm(x, (512,)),
                freqs_cis=freqs,
            )
        )
        for i in range(3)
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v41"),
        project_target_hidden=project,
        stages=stages,
    )
    inject = adapt_inject(None)
    for rows in (0, 1, 3, 31, 192):
        hidden = matrix(rows, 128)
        pos = torch.arange(rows, device=device)
        slots = torch.arange(rows)
        slots[1::3] = -1
        slots = slots.to(device)
        actual = hidden.new_full((3, rows + 1, 1, 512), -123)
        expected = torch.full_like(actual, -123)
        args = dict(main_hidden=hidden, swa_loc=slots, positions=pos)
        with envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.override("1"):
            inject(
                model, **args, pool=SimpleNamespace(get_swa_buffer=actual.__getitem__)
            )
        if not rows:
            assert not projected_rows
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
            continue
        with envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.override("0"):
            inject(
                model, **args, pool=SimpleNamespace(get_swa_buffer=expected.__getitem__)
            )
        assert projected_rows == [(rows + 1) // 2, rows]
        projected_rows.clear()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.008, atol=0.008)
        actual.fill_(-123)
        with (
            envs.SGLANG_OPT_DSV41_GROUP_CP2_DP2.override("1"),
            envs.SGLANG_OPT_DSV41_NPU_PEER_COMM.override(False),
        ):
            inject(
                model, **args, pool=SimpleNamespace(get_swa_buffer=actual.__getitem__)
            )
        assert projected_rows == [(rows + 1) // 2]
        projected_rows.clear()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.008, atol=0.008)


@torch.inference_mode()
def main():
    import torch_npu  # noqa: F401

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    cpu_group = dist.new_group(backend="gloo")
    group = Group(cpu_group)
    comm = peer.PeerGroupGather(cpu_group, device)
    try:
        _gather_cases(comm, rank, device)
        if rank == 0:
            print(
                "PASS TP2 peer gather eager and 8 changing-input graph replays",
                flush=True,
            )
        with (
            get_parallel().override(attn_tp_rank=rank, attn_tp_group=group),
            envs.SGLANG_OPT_DSV41_NPU_PEER_COMM.override(True),
        ):
            _inject_parity(device)
        if rank == 0:
            print("PASS sharded draft KV injection parity", flush=True)
    finally:
        peer.close_group_peer_all_reduce(cpu_group)
        comm.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
