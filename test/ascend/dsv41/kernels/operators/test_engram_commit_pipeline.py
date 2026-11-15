"""Accepted-history DMA retirement, changed metadata and UB slot capacities."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram_ops import EngramOps
from dsv41.kernels.references.engram_hash import oracle_commit


def repeated_reference(ids, cu, slots, accepted, history, calls):
    """Concatenate the accepted sequence repeatedly, then retain its suffix."""
    result = history.clone()
    width = history.shape[1]
    for request, slot_value in enumerate(slots.tolist()):
        slot, n = int(slot_value), int(accepted[request])
        begin, end = int(cu[request]), int(cu[request + 1])
        if not (0 <= slot < history.shape[0] and 0 < n <= end - begin):
            continue
        sequence = ids[begin : begin + n]
        repeats = min(calls, (width + n - 1) // n)
        result[slot] = torch.cat((history[slot], sequence.repeat(repeats)))[-width:]
    return result


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(846231)
    op = EngramOps()
    cases = 0
    # arch2201 usable UB contract:192KiB less the SDK's256-byte reserve.
    usable_ub_bytes = 192 * 1024 - 256
    tail_capacity = usable_ub_bytes // 4
    double_capacity = (tail_capacity // 2) // 8 * 8
    for requests, width in (
        (1, 3),
        (6, 3),
        (33, 17),
        (41, 1),
        (128, 3),
        (257, 8),
        (41, 257),
        (41, double_capacity),
        (41, double_capacity + 1),
        (41, tail_capacity),
    ):
        # The last three cases straddle two-tail UB capacity and its fallback.
        large = width >= double_capacity
        print("START", requests, width, flush=True)
        dispatches, replays = (2, 2) if large else (20, 20)
        sequence_length = width + 3 if large else max(5, width // 2)
        ids = torch.randint(0, 100003, (requests * sequence_length,), dtype=torch.int32)
        cu = torch.arange(requests + 1, dtype=torch.int32) * sequence_length
        slots = torch.arange(requests, dtype=torch.int32)
        accepted = torch.ones(requests, dtype=torch.int32)
        history = torch.randint(0, 100003, (requests, width), dtype=torch.int32)
        cpu = [ids, cu, slots, accepted, history]
        dev = [t.npu() for t in cpu]
        op.commit(*dev)
        torch.npu.synchronize()
        torch.testing.assert_close(dev[-1].cpu(), oracle_commit(*cpu), rtol=0, atol=0)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            for _ in range(dispatches):
                op.commit(*dev)
        pointers = [t.data_ptr() for t in dev]
        for generation in range(3):
            ids.random_(0, 100003)
            history.random_(0, 100003)
            accepted.random_(-1, sequence_length + 3)
            if large:
                accepted.fill_(width + 1)
            else:
                accepted[0] = 1 + generation % min(width, sequence_length)
            slots.copy_(torch.arange(requests, dtype=torch.int32))
            slots[1::11] = -1
            slots[7::17] = requests + 3
            for target, source in zip(dev, cpu):
                target.copy_(source)
            expected = repeated_reference(*cpu, dispatches * replays)
            torch.npu.synchronize()
            for _ in range(replays):
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(dev[-1].cpu(), expected, rtol=0, atol=0)
            assert pointers == [t.data_ptr() for t in dev]
            cases += 1
        del graph
    print(
        f"PASS {cases} changed-metadata commit Graph cases, single/double/fallback tails",
        flush=True,
    )


if __name__ == "__main__":
    run()
