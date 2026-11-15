"""Fused local buckets: stable order, duplicate slots, canaries and changed Graph."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical):
        raise ValueError("explicit physical NPU binding required")
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import (
        Router,
        RoutingPlan,
    )
    from dsv41.kernels.references.route_bucket import compare, reference

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(48173)
    operator = Router()
    records, preparations = [], []
    for experts, topk, local_experts in (
        (384, 6, 48),
        (384, 6, 96),
        (384, 6, 192),
        (384, 6, 384),
        (128, 3, 16),
        (128, 3, 128),
    ):
        for rows in (
            0,
            1,
            6,
            10,
            11,
            21,
            22,
            40,
            48,
            85,
            86,
            170,
            171,
            320,
            384,
            4096,
            8193,
            32768,
            131073,
        ):
            if rows > 32768 and local_experts not in (16, 384):
                continue
            ids = torch.full((rows, topk), -1, dtype=torch.int32, device="npu")
            weights = torch.rand((rows, topk), device="npu")
            storages = []
            guard, sentinel = 32, -918273

            def guarded(shape, dtype):
                count = 1
                for extent in shape:
                    count *= extent
                storage = torch.full(
                    (count + 2 * guard,), sentinel, dtype=dtype, device="npu"
                )
                storages.append(storage)
                return storage[guard:-guard].view(shape)

            out = RoutingPlan(
                guarded((rows, topk), torch.int32),
                weights,
                guarded((local_experts,), torch.int32),
                guarded((local_experts + 1,), torch.int32),
                guarded((rows * topk,), torch.int32),
                guarded((4 * (local_experts + 1),), torch.int32),
            )
            workspace = guarded(
                (
                    min(operator.cores, local_experts),
                    (local_experts + 15) // 16 * 16,
                ),
                torch.int32,
            )
            addresses = [tensor.data_ptr() for tensor in (ids, *out, workspace)]
            for first in sorted({0, experts - local_experts}):
                started = time.perf_counter()
                invoke = operator.bind_bucket(
                    ids, weights, first, local_experts, out=out, workspace=workspace
                )
                if rows == 0 and first == 0:
                    preparations.append(
                        dict(
                            local_experts=local_experts,
                            cold_seconds=time.perf_counter() - started,
                        )
                    )
                invoke()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    invoke()
                for mode in ("uniform", "owned", "outside", "holes", "empty"):
                    host = torch.randint(
                        experts, (rows, topk), generator=rng, dtype=torch.int32
                    )
                    if mode == "owned":
                        host.remainder_(local_experts).add_(first)
                    elif mode == "outside":
                        host.fill_(experts - 1 if first == 0 else 0)
                    elif mode == "holes":
                        host[::2] = -1
                        if rows:
                            host[-1, :3] = torch.tensor(
                                [first, first + local_experts - 1, first]
                            )
                    elif mode == "empty":
                        host.fill_(-1)
                    ids.copy_(host)
                    expected = reference(host, first, local_experts)
                    for execute in (invoke, graph.replay):
                        for storage in storages:
                            storage.fill_(sentinel)
                        execute()
                        torch.npu.synchronize()
                        try:
                            compare(out, expected)
                        except Exception as error:
                            error.add_note(
                                f"rows={rows} experts={experts} first={first} mode={mode}"
                            )
                            raise
                        assert bool((out.order[len(expected[3]) :] == sentinel).all())
                        assert bool(
                            (out.metadata[len(expected[4]) :] == sentinel).all()
                        )
                        for storage in storages:
                            assert bool((storage[:guard] == sentinel).all())
                            assert bool((storage[-guard:] == sentinel).all())
                        assert addresses == [
                            tensor.data_ptr() for tensor in (ids, *out, workspace)
                        ]
                        assert out.weights.data_ptr() == weights.data_ptr()
                    records.append(
                        dict(rows=rows, experts=experts, first=first, mode=mode)
                    )
                del graph
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(
            dict(
                status="passed",
                checks=len(records) * 2,
                cases=records,
                preparations=preparations,
            ),
            output,
            indent=2,
        )
        output.write("\n")
    print(
        f"PASS {len(records) * 2} eager/changed-route Graph bucket checks", flush=True
    )


if __name__ == "__main__":
    main()
