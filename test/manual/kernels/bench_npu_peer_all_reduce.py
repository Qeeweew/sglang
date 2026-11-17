"""910B TP2/EP8 correctness, replay stress and HCCL comparison.

Run with TASK_QUEUE_ENABLE=0 and a freshly built DSV41_NATIVE_DIR:
    torchrun --standalone --nproc-per-node=8 \
        test/manual/kernels/bench_npu_peer_all_reduce.py --output result.json

The eight processes exercise four TP2 pairs, EP8, and fused mHC. This is a
local hardware test, outside CI because it requires the CANN 9.1 native build.
"""

import argparse
import datetime
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from sglang.kernels.ops.communication.npu_peer_collectives import npu_peer_all_reduce
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.peer_collectives import PeerAllReduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--attention-tp", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
    cpu_group = device_group = None
    for first in range(0, world, 2):
        ranks = [first, first + 1]
        cpu = dist.new_group(ranks, backend="gloo")
        npu = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            cpu_group, device_group = cpu, npu
    device = torch.device("npu", torch.npu.current_device())
    comm = PeerAllReduce(cpu_group, device)
    assert comm.handle is not None
    print(f"rank {rank}: prepared", flush=True)
    pair = rank // 2

    def data(rows, phase):
        generator = torch.Generator().manual_seed(phase * 97 + pair)
        a = torch.randn((rows, 5120), generator=generator).bfloat16()
        b = torch.randn((rows, 5120), generator=generator).bfloat16()
        return (a if rank % 2 == 0 else b), (a.float() + b.float()).bfloat16()

    def check(actual, expected):
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=0, atol=0, equal_nan=True
        )

    rows_cases = [1, 3, 5, 6, 10, 12, 20, 24, 40, 48, 80, 96, 127, 128]
    for phase, rows in enumerate(rows_cases):
        x, expected = data(rows, phase)
        x = x.to(device)
        # Separate canaries catch either a channel-boundary or message-tail DMA.
        storage = torch.full(
            (rows * 5120 + 32,), 123, dtype=torch.bfloat16, device=device
        )
        out = storage[16:-16].view(rows, 5120)
        npu_peer_all_reduce(x, out, comm.handle)
        check(out, expected)
        check(storage[:16], torch.full((16,), 123, dtype=torch.bfloat16))
        check(storage[-16:], torch.full((16,), 123, dtype=torch.bfloat16))
    special = (
        torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, 1e38, -1e38]
        )
        .bfloat16()
        .repeat(640)
        .view(1, 5120)
    )
    remote_special = special.flip(1)
    x = (special if rank % 2 == 0 else remote_special).to(device)
    expected = (special.float() + remote_special.float()).bfloat16()
    check(comm.all_reduce(x), expected)
    for invalid in [
        torch.empty((129, 5120), dtype=torch.bfloat16, device=device),
        torch.empty((1, 5120), dtype=torch.float32, device=device),
        torch.empty((1, 5119), dtype=torch.bfloat16, device=device),
    ]:
        try:
            comm.all_reduce(invalid)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("invalid input was accepted")
    try:
        npu_peer_all_reduce(x, x, comm.handle)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("aliased output was accepted")
    print(f"rank {rank}: numerical and canaries passed", flush=True)

    # Enqueue many calls without synchronizing between them. Keep outputs until
    # checking, so a later overwrite cannot hide an earlier protocol failure.
    results = []
    for phase in range(args.stress):
        x = torch.full(
            (1 + phase % 6, 5120),
            (phase % 97) + rank / 8,
            dtype=torch.bfloat16,
            device=device,
        )
        results.append(comm.all_reduce(x))
    for phase, result in enumerate(results):
        a = torch.full(
            result.shape, (phase % 97) + (2 * pair) / 8, dtype=torch.bfloat16
        )
        b = torch.full(
            result.shape, (phase % 97) + (2 * pair + 1) / 8, dtype=torch.bfloat16
        )
        expected = (a.float() + b.float()).bfloat16()
        check(result, expected)
    del results
    if rank % 2:
        time.sleep(0.05)
    x, expected = data(96, 100)
    check(comm.all_reduce(x.to(device)), expected)

    graphs = []
    capture_stream = torch.npu.Stream()
    for rows in [5, 96]:
        x = torch.zeros((rows, 5120), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(x)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture_stream, auto_dispatch_capture=True):
            for _ in range(3):
                npu_peer_all_reduce(x, out, comm.handle)
        graphs.append((graph, x, out))
    for phase in range(64):
        graph, x, out = graphs[phase % 2]
        values, expected = data(x.shape[0], phase + 200)
        x.copy_(values)
        graph.replay()
        check(out, expected)
    with torch.npu.stream(capture_stream):
        x, expected = data(12, 300)
        check(comm.all_reduce(x.to(device)), expected)
    torch.npu.synchronize()
    graphs.clear()
    del graph
    gc.collect()
    print(
        f"rank {rank}: stress, skew, streams and changed-input replay passed",
        flush=True,
    )

    if world == 8:
        from npu_peer_collectives_checks import validate_ep8

        validate_ep8(comm, device, rank, args.stress, args.attention_tp)

    measurements = []
    for rows in [] if args.skip_benchmark else [5, 6, 10, 12, 20, 24, 40, 48, 80, 96]:
        record = {"rows": rows, "bytes": rows * 5120 * 2}
        for backend in ["peer", "hccl"]:
            x = torch.zeros((rows, 5120), dtype=torch.bfloat16, device=device)
            out = torch.empty_like(x)

            def op():
                if backend == "peer":
                    npu_peer_all_reduce(x, out, comm.handle)
                else:
                    dist.all_reduce(x, group=device_group)

            for _ in range(5):
                op()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                graph, stream=capture_stream, auto_dispatch_capture=True
            ):
                for _ in range(args.iterations):
                    op()
            samples = []
            for _ in range(7):
                dist.barrier(group=cpu_group)
                start, end = (
                    torch.npu.Event(enable_timing=True),
                    torch.npu.Event(enable_timing=True),
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / args.iterations)
            record[backend + "_us"] = statistics.median(samples)
            del graph
        measurements.append(record)
    ep8_measurements = None
    if world == 8 and not args.skip_benchmark:
        from npu_peer_collectives_bench import benchmark_ep8

        ep8_measurements = benchmark_ep8(
            comm, device_group, device, rank, args.iterations, args.attention_tp
        )
    all_ranks = [None] * world
    dist.all_gather_object(
        all_ranks,
        {
            "rank": rank,
            "checks": "pass",
            "measurements": measurements,
            "ep8_measurements": ep8_measurements,
        },
    )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "library": str(native_library_path()),
                    "torch": torch.__version__,
                    "torch_npu": torch_npu.__version__,
                    "ranks": all_ranks,
                    "stress_calls": args.stress,
                    "attention_tp": args.attention_tp,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(measurements, indent=2), flush=True)
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
