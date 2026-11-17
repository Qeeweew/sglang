"""EP8 and fused mHC checks used by bench_npu_peer_all_reduce.py."""

import gc
import time

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.npu_peer_collectives import (
    npu_peer_moe_gather,
    npu_peer_moe_return,
    npu_peer_post,
)
from sglang.srt.hardware_backend.npu.dsv41.peer_collectives import PeerMoE


def validate_ep8(comm2, device, rank, stress, attention_tp=2):
    tp, dp = attention_tp, 8 // attention_tp
    comm = PeerMoE(dist.group.WORLD, device, tp)
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import MHC

    mhc = MHC()

    def exact(out, expected):
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0, equal_nan=True)

    def coefficients(rows):
        g = torch.Generator().manual_seed(1234 + rank // tp)
        residual = torch.randn((rows, 4, 5120), generator=g).bfloat16().to(device)
        pre = torch.rand((rows, 4), generator=g).to(device)
        post = torch.rand((rows, 4), generator=g).to(device)
        comb = torch.rand((rows, 4, 4), generator=g).to(device)
        weight = torch.rand((5120,), generator=g).to(device)
        return residual, pre, post, comb, weight

    def compare_post(
        input, expected_x, coeff, handle, route=None, normalize=True, shared=None
    ):
        residual, pre, post, comb, weight = coeff
        updated = torch.empty_like(residual)
        normalized = (
            residual.new_empty((residual.shape[0], 5120)) if normalize else None
        )
        inverse, ids, weights = route or (None, None, None)
        npu_peer_post(
            input,
            residual,
            pre,
            post,
            comb,
            weight if normalize else None,
            updated,
            normalized,
            handle,
            inverse=inverse,
            ids=ids,
            weights=weights,
            shared=shared,
        )
        if normalize:
            eu, en = mhc.post_combine(
                expected_x.to(device), residual, pre, post, comb, weight
            )
        else:
            eu = mhc.post(expected_x.to(device), residual, post, comb)
        exact(updated, eu.cpu())
        if normalize:
            exact(normalized, en.cpu())

    for rows in [1, 3, 5, 6, 10, 12, 20, 24, 40, 48] + (
        [80, 96, 127, 128] if tp == 2 else [63, 64]
    ):
        base_input = torch.arange(rows * 5120).reshape(rows, 5120) % 97 / 32
        source = (base_input + rank // tp).bfloat16()
        expected = torch.cat([(base_input + d).bfloat16() for d in range(dp)])
        guard = torch.full(
            (dp * rows * 5120 + 32,), 123, dtype=torch.bfloat16, device=device
        )
        gathered = guard[16:-16].view(dp * rows, 5120)
        npu_peer_moe_gather(source.to(device), gathered, comm.handle)
        exact(gathered, expected)
        exact(guard[:16], torch.full((16,), 123, dtype=torch.bfloat16))
        exact(guard[-16:], torch.full((16,), 123, dtype=torch.bfloat16))
        base = (
            (torch.arange(dp * rows * 5120).reshape(dp * rows, 5120) % 17 - 8) / 32
        ).bfloat16()
        partial = (base.float() + rank / 8).bfloat16()
        summed = (base.float() * 8 + 3.5).bfloat16()
        own = rows if tp == 1 else ((rows + 1) // 2 if rank % 2 == 0 else rows // 2)
        shared = torch.full((own, 5120), (rank + 1) / 32, dtype=torch.bfloat16)
        expected_local = summed[rank // tp * rows : (rank // tp + 1) * rows]
        shared_full = torch.cat(
            [
                torch.full(((rows + 1) // 2, 5120), (rank // tp * 2 + 1) / 32),
                torch.full((rows // 2, 5120), (rank // tp * 2 + 2) / 32),
            ]
        ).bfloat16()
        if tp == 1:
            shared_full = shared
        expected_local = (expected_local.float() + shared_full.float()).bfloat16()
        guard = torch.full(
            (rows * 5120 + 32,), 123, dtype=torch.bfloat16, device=device
        )
        out = guard[16:-16].view(rows, 5120)
        npu_peer_moe_return(
            partial.to(device), out, comm.handle, shared=shared.to(device)
        )
        exact(out, expected_local)
        exact(guard[:16], torch.full((16,), 123, dtype=torch.bfloat16))
        exact(guard[-16:], torch.full((16,), 123, dtype=torch.bfloat16))
        # Independent attention operands, including BF16 SUM rounding.
        a = (source.float() + rank % 2 / 8).bfloat16()
        b = (source.float() + (1 - rank % 2) / 8).bfloat16()
        if tp == 2:
            compare_post(
                a.to(device),
                (a.float() + b.float()).bfloat16(),
                coefficients(rows),
                comm2.handle,
            )
    print(
        f"rank {rank}: EP8 TP{tp} gather/owner SUM/shared and applicable attention mHC passed",
        flush=True,
    )

    special = (
        torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, 1e38, -1e38]
        )
        .bfloat16()
        .repeat(640)
        .view(1, 5120)
    )
    partial = special.repeat(dp, 1).to(device)
    out = torch.empty((1, 5120), dtype=torch.bfloat16, device=device)
    total = torch.zeros_like(special, dtype=torch.float32)
    for _ in range(8):
        total += special.float()
    npu_peer_moe_return(partial, out, comm.handle)
    exact(out, total.bfloat16())
    for phase in range(8):
        if rank == phase:
            time.sleep(0.005)
        npu_peer_moe_return(partial, out, comm.handle)
        exact(out, total.bfloat16())

    invalid_rows = 64 * tp + 1
    bad = torch.empty((invalid_rows, 5120), dtype=torch.bfloat16, device=device)
    bad_out = torch.empty(
        (dp * invalid_rows, 5120), dtype=torch.bfloat16, device=device
    )
    try:
        npu_peer_moe_gather(bad, bad_out, comm.handle)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("out-of-capacity EP8 input was accepted")
    alias = torch.empty((dp, 5120), dtype=torch.bfloat16, device=device)
    for op in [
        lambda: npu_peer_moe_gather(alias[:1], alias, comm.handle),
        lambda: npu_peer_moe_return(alias, alias[:1], comm.handle),
    ]:
        try:
            op()
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("aliased EP8 tensors were accepted")
    route_fixtures = []
    for rows in [1, 5, 6, 24, 48 if tp == 1 else 96]:
        topk, total = 6, dp * rows
        generator = torch.Generator().manual_seed(500 + rank)
        values = torch.randn((total * topk, 5120), generator=generator).half()
        inverse = torch.randperm(total * topk, generator=generator).int()
        ids = torch.arange(total * topk).reshape(total, topk).remainder(8)
        ids = torch.where(ids == rank, 0, -1).int()
        weights = (
            torch.arange(total * topk).reshape(total, topk).remainder(7).float() / 8
        )
        partial = torch.zeros((total, 5120))
        for k in range(topk):
            active = ids[:, k] >= 0
            partial[active] += (
                values[inverse.reshape(total, topk)[active, k].long()].float()
                * weights[active, k, None]
            )
        partial = partial.bfloat16().float()
        dist.all_reduce(partial)  # Gloo FP32 reference, outside device collective.
        partial = partial.bfloat16()
        expected = partial[rank // tp * rows : (rank // tp + 1) * rows]
        fixture = tuple(t.to(device) for t in (values, inverse, ids, weights))
        out = torch.empty_like(expected, device=device)
        npu_peer_moe_return(
            fixture[0],
            out,
            comm.handle,
            inverse=fixture[1],
            ids=fixture[2],
            weights=fixture[3],
        )
        exact(out, expected)
        coeff = coefficients(rows)
        compare_post(fixture[0], expected, coeff, comm.handle, route=fixture[1:])
        compare_post(
            fixture[0], expected, coeff, comm.handle, route=fixture[1:], normalize=False
        )
        shared = torch.full(
            (
                rows if tp == 1 else ((rows + 1) // 2 if rank % 2 == 0 else rows // 2),
                5120,
            ),
            0.125,
            dtype=torch.bfloat16,
            device=device,
        )
        compare_post(
            fixture[0],
            (expected.float() + 0.125).bfloat16(),
            coeff,
            comm.handle,
            route=fixture[1:],
            shared=shared,
        )
        route_fixtures.append((fixture, coeff, expected))
    print(f"rank {rank}: route finalize + return + mHC/norm passed", flush=True)

    graphs = []
    stream = torch.npu.Stream()
    for rows in [5, 48 if tp == 1 else 96]:
        local = torch.zeros((rows, 5120), dtype=torch.bfloat16, device=device)
        partial = torch.zeros((dp * rows, 5120), dtype=torch.bfloat16, device=device)
        gathered, returned = torch.empty_like(partial), torch.empty_like(local)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            npu_peer_moe_gather(local, gathered, comm.handle)
            npu_peer_moe_return(partial, returned, comm.handle)
        graphs.append((graph, local, partial, gathered, returned))
    snapshots = []
    for phase in range(max(64, stress // 2)):
        graph, local, partial, gathered, returned = graphs[phase % 2]
        local.fill_(phase % 8 + rank // tp)
        partial.fill_(phase % 8 + rank / 8)
        graph.replay()
        if phase % 17 == 0:
            snapshots.append((phase, gathered.clone(), returned.clone()))
    torch.npu.synchronize()
    for phase, gathered, returned in snapshots:
        rows = returned.shape[0]
        exact(
            gathered,
            torch.cat(
                [
                    torch.full((rows, 5120), phase % 8 + d, dtype=torch.bfloat16)
                    for d in range(dp)
                ]
            ),
        )
        exact(
            returned,
            torch.full((rows, 5120), 8 * (phase % 8) + 3.5, dtype=torch.bfloat16),
        )
    graphs.clear()
    del graph
    fixture, coeff, expected = route_fixtures[1]
    residual, pre, post, comb, weight = coeff
    updated, norm = (
        torch.empty_like(residual),
        torch.empty_like(expected, device=device),
    )
    base_values = fixture[0].clone()
    attn_input = torch.zeros_like(norm)
    attn_updated, attn_norm = torch.empty_like(updated), torch.empty_like(norm)
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        if tp == 2:
            npu_peer_post(
                attn_input,
                residual,
                pre,
                post,
                comb,
                weight,
                attn_updated,
                attn_norm,
                comm2.handle,
            )
        npu_peer_post(
            fixture[0],
            residual,
            pre,
            post,
            comb,
            weight,
            updated,
            norm,
            comm.handle,
            inverse=fixture[1],
            ids=fixture[2],
            weights=fixture[3],
        )
    for phase in range(32):
        sign = -1 if phase % 2 else 1
        fixture[0].copy_(base_values * sign)
        attn_input.fill_(phase % 8 + rank / 8)
        residual.fill_((phase % 13 - 6) / 16)
        pre.fill_((phase % 4 + 1) / 8)
        post.fill_((phase % 5 + 1) / 8)
        graph.replay()
        eu, en = mhc.post_combine(
            (expected * sign).to(device), residual, pre, post, comb, weight
        )
        exact(updated, eu.cpu())
        exact(norm, en.cpu())
        attn_sum = torch.full_like(expected, 2 * (phase % 8) + (rank + (rank ^ 1)) / 8)
        eu, en = mhc.post_combine(
            attn_sum.to(device), residual, pre, post, comb, weight
        )
        if tp == 2:
            exact(attn_updated, eu.cpu())
            exact(attn_norm, en.cpu())
    torch.npu.synchronize()
    del graph
    gc.collect()
    comm.close()
    print(
        f"rank {rank}: EP8 mixed-shape replay and fused changed-input replay passed",
        flush=True,
    )
