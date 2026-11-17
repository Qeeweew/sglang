"""Matched Graph benchmarks for the complete 910B MoE communication paths."""

import gc
import statistics

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.npu_peer_collectives import (
    npu_peer_moe_gather,
    npu_peer_moe_return,
    npu_peer_post,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import MHC
from sglang.srt.hardware_backend.npu.dsv41.peer_collectives import PeerMoE


def benchmark_ep8(comm2, pair_group, device, rank, iterations, attention_tp=2):
    tp, dp = attention_tp, 8 // attention_tp
    world_group = dist.new_group(list(range(8)), backend="hccl")
    comm = PeerMoE(dist.group.WORLD, device, tp)
    mhc = MHC()
    stream = torch.npu.Stream()
    measurements = []

    def time_op(op):
        for _ in range(5):
            op()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            for _ in range(iterations):
                op()
        samples = []
        for _ in range(7):
            dist.barrier()
            start, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / iterations)
        del graph
        return statistics.median(samples)

    for rows in [5, 6, 10, 12, 20, 24, 40, 48] + ([80, 96] if tp == 2 else [64]):
        shards = (rows + tp - 1) // tp
        owned = min(rows - rank % tp * shards, shards)
        local = torch.zeros((rows, 5120), dtype=torch.bfloat16, device=device)
        gathered = torch.empty((dp * rows, 5120), dtype=torch.bfloat16, device=device)
        shard = torch.empty((shards, 5120), dtype=torch.bfloat16, device=device)
        padded_gather = torch.empty(
            (8 * shards, 5120), dtype=torch.bfloat16, device=device
        )

        def hccl_gather():
            if rows % tp:
                shard[:owned].copy_(
                    local[rank % tp * shards : rank % tp * shards + owned]
                )
                shard[owned:].zero_()
                dist.all_gather_into_tensor(padded_gather, shard, group=world_group)
                gathered.view(dp, rows, 5120).copy_(
                    padded_gather.view(dp, tp * shards, 5120)[:, :rows]
                )
            else:
                dist.all_gather_into_tensor(
                    gathered,
                    local[rank % tp * shards : (rank % tp + 1) * shards],
                    group=world_group,
                )

        record = {"rows": rows}
        record["gather_peer_us"] = time_op(
            lambda: npu_peer_moe_gather(local, gathered, comm.handle)
        )
        record["gather_hccl_us"] = time_op(hccl_gather)
        partial = torch.zeros_like(gathered)
        padded_partial = torch.zeros_like(padded_gather)
        shard_output, output = torch.empty_like(shard), torch.empty_like(local)
        padded_output = torch.empty(
            (tp * shards, 5120), dtype=torch.bfloat16, device=device
        )
        shared = torch.ones((owned, 5120), dtype=torch.bfloat16, device=device)

        def hccl_return():
            if rows % tp:
                padded_partial.view(dp, tp * shards, 5120)[:, :rows].copy_(
                    partial.view(dp, rows, 5120)
                )
                padded_partial.view(dp, tp * shards, 5120)[:, rows:].zero_()
                source = padded_partial
            else:
                source = partial
            dist.reduce_scatter_tensor(shard_output, source, group=world_group)
            shard_output[:owned].add_(shared)
            if tp == 2:
                dist.all_gather_into_tensor(
                    padded_output, shard_output, group=pair_group
                )
            else:
                padded_output.copy_(shard_output)

        record["return_peer_us"] = time_op(
            lambda: npu_peer_moe_return(partial, output, comm.handle, shared=shared)
        )
        record["return_hccl_us"] = time_op(hccl_return)
        total, topk = dp * rows, 6
        values = torch.zeros((total * topk, 5120), dtype=torch.float16, device=device)
        inverse = torch.arange(total * topk, dtype=torch.int32, device=device)
        ids_cpu = torch.arange(total * topk).reshape(total, topk).remainder(8)
        ids = torch.where(ids_cpu == rank, 0, -1).int().to(device)
        weights = torch.full((total, topk), 1 / 8, dtype=torch.float32, device=device)
        residual = torch.ones((rows, 4, 5120), dtype=torch.bfloat16, device=device)
        pre = torch.full((rows, 4), 0.25, device=device)
        post = torch.full((rows, 4), 0.5, device=device)
        comb = torch.eye(4, device=device).expand(rows, 4, 4).contiguous()
        weight = torch.ones((5120,), device=device)
        updated, normalized = torch.empty_like(residual), torch.empty_like(local)

        def peer_finalized_post():
            npu_peer_post(
                values,
                residual,
                pre,
                post,
                comb,
                weight,
                updated,
                normalized,
                comm.handle,
                inverse=inverse,
                ids=ids,
                weights=weights,
                shared=shared,
            )

        def hccl_finalized_post():
            torch.ops.dsv41.route_combine(values, inverse, ids, weights, partial)
            hccl_return()
            mhc.post_combine(
                padded_output[:rows],
                residual,
                pre,
                post,
                comb,
                weight,
                updated=updated,
                out=normalized,
            )

        record["finalize_return_post_peer_us"] = time_op(peer_finalized_post)
        record["finalize_return_post_hccl_us"] = time_op(hccl_finalized_post)

        def peer_attention_post():
            npu_peer_post(
                local,
                residual,
                pre,
                post,
                comb,
                weight,
                updated,
                normalized,
                comm2.handle,
            )

        def hccl_attention_post():
            dist.all_reduce(local, group=pair_group)
            mhc.post_combine(
                local,
                residual,
                pre,
                post,
                comb,
                weight,
                updated=updated,
                out=normalized,
            )

        if tp == 2:
            record["attention_post_peer_us"] = time_op(peer_attention_post)
            record["attention_post_hccl_us"] = time_op(hccl_attention_post)
        measurements.append(record)
        print(f"rank {rank}: EP8/mHC benchmark rows={rows} complete", flush=True)
    torch.npu.synchronize()
    gc.collect()
    comm.close()
    dist.destroy_process_group(world_group)
    return measurements
