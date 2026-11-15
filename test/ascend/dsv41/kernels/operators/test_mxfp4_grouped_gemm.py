"""MXFP4 qualification: startup NPU repack, FP64 dots and Graph updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dsv41.kernels.references.quant_packing import pack_experts
from dsv41.kernels.references.mxfp4_grouped import (
    checkpoint,
    fixture,
    weight,
)


def check(actual, x, weights, ends):
    actual = actual.cpu().numpy()
    assert np.isfinite(actual).all()
    begin, maximum, checked = 0, 0.0, 0
    for w, end in zip(weights, ends):
        # Full outputs for small problems; explicitly sampled rows for real shapes.
        rows = np.arange(begin, end)
        if len(rows) * w.size > 100_000_000:
            rows = np.unique(np.linspace(begin, end - 1, 17, dtype=int))
        if len(rows):
            expected = (x[rows].astype(np.float64) @ w.astype(np.float64).T).astype(
                np.float16
            )
            error = np.abs(
                actual[rows].astype(np.float32) - expected.astype(np.float32)
            )
            maximum = max(maximum, float(error.max()))
            np.testing.assert_allclose(actual[rows], expected, rtol=0.002, atol=0.002)
            checked += len(rows) * w.shape[0]
        begin = end
    np.testing.assert_array_equal(actual[ends[-1] :], 123)
    return {
        "max_abs_error": maximum,
        "checked_elements": checked,
        "full_output": checked == ends[-1] * actual.shape[1],
    }


def run(args):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.health import (
        health_snapshot,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    health = health_snapshot(args.physical, output)
    # Initialize Torch only after the physical mapping and health check.
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mxfp4_weight import (
        MXFP4WeightPacker,
    )
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.quant_grouped_gemm import (
        NativeGroupedGemm,
    )

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(31)
    records = []
    op = NativeGroupedGemm(
        args.k,
        args.n,
        experts=args.experts,
        block_m=args.block_m,
    )
    e, n, k = op.experts, op.n, op.k
    packer = MXFP4WeightPacker(k, n)

    def prepare(seed=17):
        codes, exponents = (
            fixture(e, n, k, seed=seed)
            if args.model is None or seed != 17
            else checkpoint(
                args.model,
                np.linspace(0, 383, e, dtype=int),
                projection=args.projection,
            )
        )
        pp, ss = pack_experts(
            zip(codes, exponents),
            n=n,
            k=k,
            count=e,
            compact_scales=True,
        )
        return (
            pp,
            ss,
            [weight(c, s, k) for c, s in zip(codes, exponents)],
            (codes, exponents),
        )

    def upload(packed, scales, seed=17):
        expected_p, expected_s, decoded, raw = prepare(seed)
        # Production checkpoint loading writes final resident tensors in place.
        # The NumPy pack is only an independent byte-order oracle.
        for expert, (codes, exponents) in enumerate(zip(*raw)):
            packer.load_cpu(
                torch.from_numpy(np.array(codes)),
                torch.from_numpy(np.array(exponents)),
                packed=packed[expert],
                s2=scales[expert],
            )
        np.testing.assert_array_equal(packed.cpu().numpy(), expected_p)
        np.testing.assert_array_equal(scales.cpu().numpy(), expected_s)
        return expected_p, expected_s, decoded

    native_bound = {}
    row_hints = {}

    def compute(x, p, s, gl, out, metadata):
        key = tuple(t.data_ptr() for t in (x, p, s, metadata, out))
        if key not in native_bound:
            native_bound[key] = op.bind(x, p, s, metadata, out=out, **row_hints)
        native_bound[key]()

    def metadata_values(ends):
        records, begin = [], 0
        for expert, end in enumerate(ends):
            if end > begin:
                records += [expert, begin, end - begin, 0]
            begin = end
        values = torch.full((4 * (e + 1),), -1, dtype=torch.int32)
        values[:4] = torch.tensor([len(records) // 4, begin, 0, 1])
        values[4 : 4 + len(records)] = torch.tensor(records, dtype=torch.int32)
        return values

    p = torch.empty((e, n * k // 8), dtype=torch.int32, device="npu:0")
    s = torch.empty(
        (e, n * k // 32),
        dtype=torch.uint8,
        device="npu:0",
    )
    _, _, weights = upload(p, s)
    # Exercise real C++ ownership and stream isolation, replacing the old
    # Python-FFI mock. No device work is submitted by bind itself.
    probe = p.new_zeros((max(k, n),), dtype=torch.float16)
    xx, yy = probe[:k].view(1, k), probe[:n].view(1, n)
    gl = p.new_zeros((e,), dtype=torch.int32)
    metadata = metadata_values([0] * e).npu()
    routing = metadata
    yy = torch.empty_like(yy)
    hints = dict(max_expert_rows=1, mean_expert_rows=0)
    first = op.bind(xx, p, s, routing, out=yy, **hints)
    again = op.bind(xx, p, s, routing, out=yy, **hints)
    assert first.workspace.data_ptr() == again.workspace.data_ptr()
    # A cumulative GroupList is no longer a native routing input.
    for invoke in (
        lambda: op.bind(xx, p, s, gl, out=yy, **hints),
        lambda: torch.ops.dsv41.quant_gmm(
            xx,
            p,
            s,
            gl,
            yy,
            first.workspace,
            first.block_m,
        ),
    ):
        try:
            invoke()
        except (ValueError, RuntimeError) as error:
            assert "compact metadata capacity" in str(error), str(error)
        else:
            raise AssertionError("native GEMM accepted a GroupList")
    other_stream = torch.npu.Stream()
    with torch.npu.stream(other_stream):
        other = op.bind(xx, p, s, routing, out=yy, **hints)
    assert other.workspace.data_ptr() != first.workspace.data_ptr()
    try:
        op.bind(xx, p, s, routing, out=probe[:n].view(1, n), **hints)
    except ValueError:
        pass
    else:
        raise AssertionError("C++ binding accepted overlapping A/C")
    if args.gemv:
        routes = [[0] * e]
        for rows in (1, 2, 6, 15, 16):
            routes += [[rows if j % 3 != 1 else 0 for j in range(e)], [rows] * e]
        routes.append([(j * 7) % 17 for j in range(e)])
        hot = [0] * e
        hot[e // 2] = 16
        routes.append(hot)
        if e >= 6:
            routes.append([6] * 6 + [0] * (e - 6))
        cases = [([0] * e, 0)] + [
            (np.cumsum(counts).tolist(), sum(counts) + 7) for counts in routes
        ]
    else:
        cases = [([0] * e, 0), ([0] * e, 19)]
        for rows in (1, 6, 16, 17, 32, 33, 64, 65, 128, 129, 256, 513, 1537):
            counts = [rows if j % 3 != 1 else 0 for j in range(e)]
            cases.append((np.cumsum(counts).tolist(), sum(counts) + 7))
        # A single hot expert, with empty experts both before and after it.
        counts = [0] * e
        counts[e // 2] = 65
        cases.append((np.cumsum(counts).tolist(), 65))
        if e >= 3:
            counts = [1, 17, 257] + [0] * (e - 3)
            cases.append((np.cumsum(counts).tolist(), sum(counts) + 7))
        # Dense routes exercise several consecutive outputs on each core. Sparse
        # every-third-empty fixtures can hide reuse/retirement failures.
        for rows in (1, 6, 17, 129):
            counts = [rows] * e
            cases.append((np.cumsum(counts).tolist(), sum(counts) + 7))
        if e >= 6:
            counts = [6] * 6 + [0] * (e - 6)
            cases.append((np.cumsum(counts).tolist(), 43))
    for ends, capacity in cases:
        counts = np.diff([0, *ends])
        row_hints = dict(
            max_expert_rows=int(counts.max()),
            mean_expert_rows=ends[-1] / max(1, int(np.count_nonzero(counts))),
        )
        if args.verbose:
            print(
                json.dumps({"phase": "eager", "ends": ends, "capacity": capacity}),
                flush=True,
            )
        source = (rng.standard_normal((capacity, k)) / 8).astype(np.float16)
        x = torch.from_numpy(source).npu()
        gl = torch.tensor(ends, dtype=torch.int32, device="npu:0")
        metadata = gl.new_empty(4 * (e + 1))
        metadata.copy_(metadata_values(ends))
        storage = torch.full(
            (capacity * n + 512,), 123, dtype=torch.float16, device="npu:0"
        )
        out = storage[256:-256].view(capacity, n)
        compute(x, p, s, gl, out, metadata)
        result = check(out, source, weights, ends)
        assert bool((storage[:256] == 123).all()) and bool(
            (storage[-256:] == 123).all()
        )
        np.testing.assert_array_equal(x.cpu(), source)
        records.append({"ends": ends, "capacity": capacity, **result})

    # Capture on a non-default stream; preserve every tensor/storage address.
    stream = torch.npu.Stream()
    capacity = 16 * e + 7 if args.gemv else max(193, 128 * e + 1)
    row_hints = dict(max_expert_rows=capacity, mean_expert_rows=capacity / e)
    with torch.npu.stream(stream):
        x = torch.empty((capacity, k), dtype=torch.float16, device="npu:0")
        gl = torch.zeros(e, dtype=torch.int32, device="npu:0")
        metadata = gl.new_empty(4 * (e + 1))
        metadata.copy_(metadata_values([0] * e))
        storage = torch.full(
            (capacity * n + 512,), 123, dtype=torch.float16, device="npu:0"
        )
        out = storage[256:-256].view(capacity, n)
        x.zero_()
        compute(x, p, s, gl, out, metadata)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            compute(x, p, s, gl, out, metadata)
        addresses = [t.data_ptr() for t in (x, p, s, gl, out)]
        for number in range(7):
            # Packed codes/scales, A and routing boundaries all change.
            ph, sh, weights = upload(p, s, 53 + number)
            source = (rng.standard_normal((capacity, k)) / 8).astype(np.float16)
            x.copy_(torch.from_numpy(source))
            if args.gemv:
                counts = [0] * e
                if number == 1:
                    counts[e // 2] = 6
                elif number in (2, 3):
                    counts = [1 if number == 2 else 16] * e
                elif number == 4:
                    counts = [(j * 7) % 17 for j in range(e)]
                elif number == 5:
                    counts[-1] = 16
                elif number == 6:
                    counts[: min(6, e)] = [6] * min(6, e)
            else:
                counts = [0] * e
                if number < 6:
                    counts[number % e] = (6, 65, 129, capacity, 0, 16)[number]
                else:
                    counts[0] = 1
                    counts[-1] = 17
            ends = np.cumsum(counts).tolist()
            if args.verbose:
                print(
                    json.dumps({"phase": "graph", "ends": ends, "capacity": capacity}),
                    flush=True,
                )
            gl.copy_(torch.tensor(ends, dtype=torch.int32))
            metadata.copy_(metadata_values(ends))
            out.fill_(123)
            graph.replay()
            stream.synchronize()
            result = check(out, source, weights, ends)
            assert [t.data_ptr() for t in (x, p, s, gl, out)] == addresses
            assert bool((storage[:256] == 123).all()) and bool(
                (storage[-256:] == 123).all()
            )
            np.testing.assert_array_equal(p.cpu(), ph)
            np.testing.assert_array_equal(s.cpu(), sh)
            np.testing.assert_array_equal(gl.cpu(), ends)
            records.append(
                {"graph": number, "capacity": capacity, "ends": ends, **result}
            )
    torch.npu.synchronize()
    record = {
        "status": "passed",
        "health": health,
        "policy": op.params,
        "packing": "ascend_c_startup",
        "scope": "per-expert M<=16" if args.gemv else "grouped GEMM",
        "cases": records,
        "checkpoint": str(args.model) if args.model else None,
        "projection": args.projection,
        "graph_weights": "synthetic updates",
        "implementation": "native",
    }
    (output / "device.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"status": "passed", "cases": len(records), "output": str(output)})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--block-m", type=int, default=0, choices=(0, 16, 128, 256))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gemv", action="store_true", help="qualify per-expert M=0..16"
    )
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--projection", choices=("gate_up", "down"), default="gate_up")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
