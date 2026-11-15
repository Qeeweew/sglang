"""Per-request CED tail maps, typed data movement and dummy-safe Graph replay."""

import argparse
import json
import math
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.ced import (
    CED,
    TailMetadata,
    tail_capacity,
)


def reference(cu, window, capacity):
    rows = []
    req = []
    prefix = [0]
    for b, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        n = min(end - start, window)
        rows.extend(range(end - n, end))
        req.extend([b] * n)
        prefix.append(len(rows))
    needed = len(rows)
    if needed > capacity:
        return (
            torch.full((capacity,), -1, dtype=torch.int32),
            torch.zeros(len(cu), dtype=torch.int32),
            torch.full((capacity,), -1, dtype=torch.int32),
            torch.tensor([-needed], dtype=torch.int32),
        )
    rows.extend([-1] * (capacity - needed))
    req.extend([-1] * (capacity - needed))
    return (
        torch.tensor(rows, dtype=torch.int32),
        torch.tensor(prefix, dtype=torch.int32),
        torch.tensor(req, dtype=torch.int32),
        torch.tensor([needed], dtype=torch.int32),
    )


def check_metadata(actual, expected):
    for x, y in zip(actual, expected):
        assert torch.equal(x.cpu(), y), (x.cpu(), y)


def gather_ref(x, row_map, fill=0):
    out = torch.full((len(row_map), *x.shape[1:]), fill, dtype=x.dtype)
    for row, src in enumerate(row_map.tolist()):
        if 0 <= src < x.shape[0]:
            out[row] = x[src]
    return out


def scatter_ref(x, updates, row_map):
    out = x.clone()
    for row, dst in enumerate(row_map.tolist()):
        if 0 <= dst < out.shape[0]:
            out[dst] = updates[row]
    return out


def numeric(op):
    count = 0
    for lengths in (
        [],
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [127],
        [128],
        [129],
        [1025],
        [0, 7, 129, 0, 257],
        [1] * 6,
    ):
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        for window in (1, 128, 257):
            compact = tail_capacity(cu, window)
            for capacity in set((compact, len(lengths) * window, max(0, compact - 1))):
                sizes = (capacity, len(cu), capacity, 1)
                buffers = [
                    torch.full((size + 32,), -1234567, dtype=torch.int32, device="npu")
                    for size in sizes
                ]
                out = TailMetadata(*(buf[:size] for buf, size in zip(buffers, sizes)))
                md = op.metadata(
                    torch.tensor(cu, dtype=torch.int32, device="npu"),
                    window=window,
                    capacity=capacity,
                    out=out,
                )
                check_metadata(md, reference(cu, window, capacity))
                count += 1
                for buf, size in zip(buffers, sizes):
                    assert bool((buf[size:].cpu() == -1234567).all())
    # Byte-exact row features include ragged partial cache-line rows and full
    # hidden/mHC tensors. Ownership groups keep partial writes from racing.
    for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64):
        for shape in ((6,), (3,), (1,), (64,), (5120,), (4, 5120)):
            n = 259
            r = 137
            x = (
                torch.arange(n * math.prod(shape), dtype=torch.int64)
                .reshape(n, *shape)
                .to(dtype)
            )
            row_map = torch.randint(-7, n + 7, (r,), dtype=torch.int32)
            row_map[:6] = torch.tensor([0, 1, 1, -1, n, n - 1])
            fill = -1 if dtype in (torch.int32, torch.int64) else 0
            storage = torch.full(
                (r * math.prod(shape) + 128,), -3, dtype=dtype, device="npu"
            )
            out = storage[: r * math.prod(shape)].view(r, *shape)
            gathered = op.gather(x.npu(), row_map.npu(), out=out, fill=fill)
            assert torch.equal(gathered.cpu(), gather_ref(x, row_map, fill))
            assert bool((storage[r * math.prod(shape) :].cpu() == -3).all())
            original = torch.full_like(x, -7)
            updates = (
                torch.randn((r, *shape)).to(dtype)
                if dtype.is_floating_point
                else torch.randint(-10000, 10000, (r, *shape), dtype=dtype)
            )
            storage = torch.full(
                (n * math.prod(shape) + 128,), -7, dtype=dtype, device="npu"
            )
            out = storage[: n * math.prod(shape)].view(n, *shape)
            op.scatter(updates.npu(), row_map.npu(), out)
            assert torch.equal(out.cpu(), scatter_ref(original, updates, row_map))
            assert bool((storage[n * math.prod(shape) :].cpu() == -7).all())
            count += 1
    print(json.dumps({"numeric_cases": count, "status": "pass"}), flush=True)
    return count


def graph(op):
    capacity = 3 * 128
    cu = torch.tensor([0, 6, 206, 259], dtype=torch.int32, device="npu")
    md = TailMetadata(
        torch.empty(capacity, dtype=torch.int32, device="npu"),
        torch.empty(4, dtype=torch.int32, device="npu"),
        torch.empty(capacity, dtype=torch.int32, device="npu"),
        torch.empty(1, dtype=torch.int32, device="npu"),
    )
    x = torch.randn(259, 4, 5120).bfloat16().npu()
    tail = torch.empty(capacity, 4, 5120, dtype=torch.bfloat16, device="npu")
    dest = torch.full_like(x, -7)
    positions = torch.arange(259, dtype=torch.int64, device="npu")
    tail_positions = torch.empty(capacity, dtype=torch.int64, device="npu")
    state = torch.randn(259, 4, device="npu")
    tail_state = torch.empty(capacity, 4, device="npu")
    state_dest = torch.full_like(state, -13)

    def enqueue():
        op.metadata(cu, capacity=capacity, out=md)
        op.gather(x, md.row_map, out=tail)
        op.gather(positions, md.row_map, out=tail_positions, fill=-1)
        op.gather(state, md.row_map, out=tail_state)
        op.scatter(tail, md.row_map, dest)
        op.scatter(tail_state, md.row_map, state_dest)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            enqueue()
    stream.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        enqueue()
    for host_cu in ([0, 6, 206, 259], [0, 1, 2, 8], [0, 0, 0, 0], [0, 128, 129, 259]):
        x.copy_(torch.randn(x.shape).bfloat16())
        state.copy_(torch.randn(state.shape))
        cu.copy_(torch.tensor(host_cu, dtype=torch.int32))
        dest.fill_(-7)
        state_dest.fill_(-13)
        torch.npu.synchronize()
        g.replay()
        torch.npu.synchronize()
        expected = reference(host_cu, 128, capacity)
        check_metadata(md, expected)
        rm = expected[0]
        assert torch.equal(tail.cpu(), gather_ref(x.cpu(), rm))
        assert torch.equal(tail_positions.cpu(), gather_ref(positions.cpu(), rm, -1))
        assert torch.equal(
            dest.cpu(),
            scatter_ref(torch.full_like(x.cpu(), -7), gather_ref(x.cpu(), rm), rm),
        )
        assert torch.equal(
            state_dest.cpu(),
            scatter_ref(
                torch.full_like(state.cpu(), -13), gather_ref(state.cpu(), rm), rm
            ),
        )
    print(
        json.dumps(
            {
                "graph": "metadata->BF16/mHC/positions gather->scatter; dynamic per-request lengths; dummy no state pollution",
                "status": "pass",
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library-dir", type=Path, default=Path("build/dsv41"))
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(413)
    op = CED(library_dir=a.library_dir.resolve())
    count = numeric(op)
    graph(op)
    print(json.dumps({"cases": count, "status": "pass"}), flush=True)


if __name__ == "__main__":
    main()
