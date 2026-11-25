"""FP16 SwiGLU storage, FP32 arithmetic, canaries and changing Graph inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest



def run(args):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.health import (
        health_snapshot,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    health = health_snapshot(args.physical, args.output)

    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
        ClippedSwiGLU,
        PackedClippedSwiGLU,
    )

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    rng = np.random.default_rng(352)
    records = []

    def values(rows, d):
        x = (rng.standard_normal((rows, 2 * d)) * 15).astype(np.float16)
        if rows:
            edges = np.array(
                [
                    -65504,
                    -1000,
                    -100,
                    -90,
                    -89,
                    -88,
                    -85,
                    -80,
                    -30,
                    -10,
                    -1,
                    -0.0,
                    0.0,
                    1,
                    10,
                    11,
                    65504,
                ],
                dtype=np.float16,
            )
            x.flat[: min(len(edges), x.size)] = edges[: min(len(edges), x.size)]
        return x

    def check(out, x, guards):
        gate, up = np.split(x.astype(np.float32), 2, axis=-1)
        gate = np.minimum(gate, 10)
        up = np.clip(up, -10, 10)
        # Independent stable FP32 expression.
        e = np.exp(-np.abs(gate))
        sigmoid = np.where(gate >= 0, 1 / (1 + e), e / (1 + e))
        expected = (gate * sigmoid * up).astype(np.float16)
        actual = out.cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.001, atol=2e-6)
        assert all(bool((g == 123).all()) for g in guards)
        error = (
            float(
                np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
            )
            if actual.size
            else 0.0
        )
        records.append({"rows": len(x), "d": out.shape[1], "max_abs_error": error})

    for d in args.dims:
        input_width, output_width = 2 * d, d
        op = PackedClippedSwiGLU(dimension=d, dtype="f16")
        for rows in (0, 1, 6, 33, 2048, *([32768] if d == 2304 else [])):
            source = values(rows, d)
            x = torch.from_numpy(source).npu()
            storage = torch.full(
                (rows * output_width + 64,),
                123,
                dtype=torch.float16,
                device="npu:0",
            )
            out = storage[32:-32].view(rows, output_width)
            op(x, out=out)
            check(out, source, (storage[:32], storage[-32:]))
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            x = torch.empty((33, input_width), dtype=torch.float16, device="npu:0")
            out = torch.empty((33, output_width), dtype=torch.float16, device="npu:0")
            x.zero_()
            call = op.bind(x, out=out)
            call()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                call()
            addresses = [x.data_ptr(), out.data_ptr()]
            for _ in range(3):
                source = values(33, d)
                x.copy_(torch.from_numpy(source))
                graph.replay()
                stream.synchronize()
                check(out, source, ())
                assert [x.data_ptr(), out.data_ptr()] == addresses
        prefix = PackedClippedSwiGLU(dimension=d, dtype="f16", prefix=True)
        source = values(33, d)
        x.copy_(torch.from_numpy(source))
        count = torch.tensor([33], dtype=torch.int32, device="npu")
        with torch.npu.stream(stream):
            prefix(x, rows=count, out=out)
        stream.synchronize()
        with torch.npu.graph(prefix_graph := torch.npu.NPUGraph(), stream=stream):
            prefix(x, rows=count, out=out)
        for valid in (0, 1, 6, 33):
            source = values(33, d)
            source[valid:] = np.nan
            with torch.npu.stream(stream):
                x.copy_(torch.from_numpy(source))
                out.fill_(123)
                count.fill_(valid)
                prefix_graph.replay()
            stream.synchronize()
            check(out[:valid], source[:valid], (out[valid:],))
        del prefix_graph
        if d in (288, 2304):
            separate = ClippedSwiGLU(dimension=d, dtype="f16")
            source = values(33, d)
            gate, up = [
                torch.from_numpy(part.copy()).npu()
                for part in np.split(source, 2, axis=-1)
            ]
            split_call = separate.bind(gate, up, out=out)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                split_call()
            stream.synchronize()
            check(out, source, ())
            with torch.npu.graph(split_graph := torch.npu.NPUGraph(), stream=stream):
                split_call()
            for _ in range(3):
                source = values(33, d)
                with torch.npu.stream(stream):
                    gate.copy_(torch.from_numpy(source[:, :d].copy()))
                    up.copy_(torch.from_numpy(source[:, d:].copy()))
                    split_graph.replay()
                stream.synchronize()
                check(out, source, ())
            del split_graph
    (args.output / "result.json").write_text(
        json.dumps({"health": health, "cases": records}, indent=2) + "\n"
    )
    print(f"PASS: {len(records)} FP16/FP32 SwiGLU cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=(32, 64, 128, 256, 512, 288, 576, 1152, 2304),
    )
    run(parser.parse_args())
