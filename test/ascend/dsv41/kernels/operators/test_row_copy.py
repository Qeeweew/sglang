"""Byte-exact row gather/scatter and dummy-safe Graph replay on 910B."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.row_copy import (
    gather_rows,
    scatter_rows,
)


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.cpu().contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


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


def numeric():
    count = 0
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
            gathered = gather_rows(x.npu(), row_map.npu(), out=out, fill=fill)
            assert_bytes(gathered, gather_ref(x, row_map, fill))
            assert bool((storage[r * math.prod(shape) :].cpu() == -3).all())
            original = torch.full_like(x, -7)
            updates = (
                torch.randn((r, *shape)).to(dtype)
                if dtype.is_floating_point
                else torch.randint(-10000, 10000, (r, *shape), dtype=dtype)
            )
            updates[1].fill_(-5)
            updates[2].fill_(7)
            storage = torch.full(
                (n * math.prod(shape) + 128,), -7, dtype=dtype, device="npu"
            )
            out = storage[: n * math.prod(shape)].view(n, *shape)
            scatter_rows(updates.npu(), row_map.npu(), out)
            assert_bytes(out, scatter_ref(original, updates, row_map))
            assert bool((storage[n * math.prod(shape) :].cpu() == -7).all())
            count += 1
    print(json.dumps({"numeric_cases": count, "status": "pass"}), flush=True)
    return count


def edge_cases():
    for dtype, storage_dtype, bits in (
        (torch.bfloat16, torch.int16, [0, -32768, 0x7F80, 0x7FC1]),
        (torch.float32, torch.int32, [0, -(1 << 31), 0x7F800000, 0x7FC12500]),
    ):
        source = torch.tensor(bits, dtype=storage_dtype).view(dtype)
        row_map = torch.tensor([3, 2, 1, -1, 0, 4, 1], dtype=torch.int32)
        device_source = source.npu()
        mapped = row_map.npu()
        expected = gather_ref(source, row_map)
        gathered = gather_rows(device_source, mapped)
        assert_bytes(gathered, expected)
        out = torch.full_like(device_source, 1)
        scatter_rows(gathered, mapped, out)
        assert_bytes(out, scatter_ref(torch.ones_like(source), expected, row_map))

    empty = torch.empty(0, dtype=torch.int32, device="npu")
    invalid = torch.tensor([-1, 0, 1], dtype=torch.int32, device="npu")
    source = torch.arange(8, dtype=torch.int64, device="npu").view(4, 2)
    assert gather_rows(source, empty).shape == (0, 2)
    assert_bytes(
        gather_rows(source[:0], invalid, fill=-1),
        torch.full((3, 2), -1, dtype=torch.int64),
    )
    original = source.cpu()
    scatter_rows(source[:0], empty, source)
    assert_bytes(source, original)
    no_columns = torch.empty((4, 0), dtype=torch.bfloat16, device="npu")
    assert gather_rows(no_columns, invalid).shape == (3, 0)
    scatter_rows(no_columns[:3], invalid, no_columns)


def tail_map(cu, capacity, window=128):
    rows = []
    for start, end in zip(cu, cu[1:]):
        rows.extend(range(max(start, end - window), end))
    return torch.tensor(rows + [-1] * (capacity - len(rows)), dtype=torch.int32)


def graph():
    capacity = 3 * 128
    row_map = tail_map([0, 6, 206, 259], capacity).npu()
    x = torch.randn(259, 4, 5120).bfloat16().npu()
    tail = torch.empty(capacity, 4, 5120, dtype=torch.bfloat16, device="npu")
    dest = torch.full_like(x, -7)
    positions = torch.arange(259, dtype=torch.int64, device="npu")
    tail_positions = torch.empty(capacity, dtype=torch.int64, device="npu")
    state = torch.randn(259, 4, device="npu")
    tail_state = torch.empty(capacity, 4, device="npu")
    state_dest = torch.full_like(state, -13)

    def enqueue():
        gather_rows(x, row_map, out=tail)
        gather_rows(positions, row_map, out=tail_positions, fill=-1)
        gather_rows(state, row_map, out=tail_state)
        scatter_rows(tail, row_map, dest)
        scatter_rows(tail_state, row_map, state_dest)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            enqueue()
    stream.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        enqueue()
    for cu in ([0, 6, 206, 259], [0, 1, 2, 8], [0, 0, 0, 0], [0, 128, 129, 259]):
        rm = tail_map(cu, capacity)
        host_x = torch.randn(x.shape).bfloat16()
        host_state = torch.randn(state.shape)
        host_positions = torch.arange(259, dtype=torch.int64) + cu[-1]
        if cu[-1]:
            rm[-1] = x.shape[0]
        with torch.npu.stream(stream):
            x.copy_(host_x)
            state.copy_(host_state)
            positions.copy_(host_positions)
            row_map.copy_(rm)
            dest.fill_(-7)
            state_dest.fill_(-13)
            g.replay()
        stream.synchronize()
        gathered_x = gather_ref(host_x, rm)
        gathered_state = gather_ref(host_state, rm)
        assert_bytes(tail, gathered_x)
        assert_bytes(tail_positions, gather_ref(host_positions, rm, -1))
        assert_bytes(tail_state, gathered_state)
        assert_bytes(dest, scatter_ref(torch.full_like(host_x, -7), gathered_x, rm))
        assert_bytes(
            state_dest,
            scatter_ref(torch.full_like(host_state, -13), gathered_state, rm),
        )
    print(
        json.dumps(
            {
                "graph": "changed inputs and maps, padded rows, all-invalid scatter",
                "status": "pass",
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library-dir", type=Path, default=Path("build/dsv41"))
    a = p.parse_args()
    os.environ["DSV41_NATIVE_DIR"] = str(a.library_dir.resolve())
    torch.set_num_threads(8)
    torch.manual_seed(413)
    count = numeric()
    edge_cases()
    graph()
    print(json.dumps({"cases": count, "status": "pass"}), flush=True)


if __name__ == "__main__":
    main()
