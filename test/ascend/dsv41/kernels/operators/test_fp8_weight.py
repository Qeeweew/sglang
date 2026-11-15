"""Dense 32x32 E8M0 layout, software decode, bounded loader and Graph tests."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.fp8_weight import (
    FP8Weight,
    _header,
    _index,
)
from dsv41.kernels.references.fp8_weight import reference


def check(actual, expected):
    actual = actual.cpu()
    equal = (actual.view(torch.int16) == expected.view(torch.int16)) | (
        actual.isnan() & expected.isnan()
    )
    if not equal.all():
        where = (~equal).nonzero()[0].tolist()
        raise AssertionError(f"BF16 mismatch at {where}; total={(~equal).sum().item()}")


def fake_file(path, key, data, dtype):
    header = json.dumps(
        {
            key: {
                "dtype": dtype,
                "shape": list(data.shape),
                "data_offsets": [0, data.numel()],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data.numpy().tobytes())


def run(args):
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(413232)
    op = FP8Weight(args.library)
    cases = 0
    # Each 32x32 tile contains all 256 FP8 encodings; its single scale sweeps
    # all 256 E8M0 encodings. This catches accidental per-row group32 scaling.
    codes = ((torch.arange(8192)[:, None] % 32) * 32 + torch.arange(32)[None, :]).to(
        torch.uint8
    )
    scales = torch.arange(256, dtype=torch.uint8)[:, None]
    check(op.convert(codes.npu(), scales.npu()), reference(codes, scales))
    cases += 1
    for rows, cols in (
        (0, 32),
        (32, 0),
        (1, 1),
        (2, 17),
        (31, 31),
        (32, 32),
        (33, 33),
        (63, 127),
        (64, 128),
        (65, 129),
        (257, 513),
        (1025, 1025),
    ):
        codes = torch.randint(
            0, 256, (rows, cols), dtype=torch.uint8, generator=generator
        )
        scales = torch.randint(
            0,
            256,
            ((rows + 31) // 32, (cols + 31) // 32),
            dtype=torch.uint8,
            generator=generator,
        )
        nc, ns = codes.npu(), scales.npu()
        guard = torch.full(
            (codes.numel() + 32,), -91.0, dtype=torch.bfloat16, device="npu"
        )
        out = guard[16:-16].view(rows, cols)
        assert op.convert(nc, ns, out=out).data_ptr() == out.data_ptr()
        check(out, reference(codes, scales))
        assert torch.all(guard[:16].cpu() == -91) and torch.all(
            guard[-16:].cpu() == -91
        )
        cases += 1
    codes = torch.randint(0, 255, (65, 257), dtype=torch.uint8, generator=generator)
    scales = torch.randint(115, 130, (3, 9), dtype=torch.uint8, generator=generator)
    expected = reference(codes, scales)
    for rows in (32, 64, 96):
        check(
            op.load_cpu(codes, scales, chunk_rows=rows, staging_bytes=96 * 257 + 27),
            expected,
        )
        assert op.last_load["staging_bytes"] <= 96 * 257 + 27
        cases += 1
    check(
        op.load_cpu(
            codes.view(torch.float8_e4m3fn),
            scales.view(torch.float8_e8m0fnu),
            staging_bytes=32 * 257 + 9,
        ),
        expected,
    )
    assert op.last_load["chunks"] == 3 and op.last_load["staging_bytes"] == 32 * 257 + 9
    cases += 1
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        fake_file(directory / "weight.safetensors", "layer.weight", codes, "F8_E4M3")
        fake_file(directory / "scale.safetensors", "layer.scale", scales, "F8_E8M0")
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layer.weight": "weight.safetensors",
                        "layer.scale": "scale.safetensors",
                    }
                }
            )
        )
        original_weight = (directory / "weight.safetensors").read_bytes()
        check(op.load_safetensors(directory, "layer.weight", chunk_rows=32), expected)
        assert original_weight == (directory / "weight.safetensors").read_bytes()
        cases += 1
    # Nondefault-stream eager and stable-address Graph replay with changed data.
    nc, ns = codes.npu(), scales.npu()
    out = torch.empty(codes.shape, dtype=torch.bfloat16, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        op.convert(nc, ns, out=out)
    stream.synchronize()
    check(out, expected)
    cases += 1
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        op.convert(nc, ns, out=out)
    for _ in range(3):
        codes = torch.randint(0, 256, (65, 257), dtype=torch.uint8, generator=generator)
        scales = torch.randint(0, 256, (3, 9), dtype=torch.uint8, generator=generator)
        with torch.npu.stream(stream):
            nc.copy_(codes)
            ns.copy_(scales)
        graph.replay()
        stream.synchronize()
        check(out, reference(codes, scales))
        cases += 1
    del graph

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("invalid input was accepted")

    rejects(
        lambda: op.convert(nc, torch.empty((65, 9), dtype=torch.uint8, device="npu"))
    )
    rejects(lambda: op.load_cpu(codes, scales, chunk_rows=31))
    rejects(lambda: op.load_cpu(codes, scales, staging_bytes=1))
    alias = torch.empty(2, dtype=torch.uint8, device="npu")
    rejects(
        lambda: op.convert(
            alias[:1].view(1, 1),
            torch.zeros((1, 1), dtype=torch.uint8, device="npu"),
            out=alias.view(torch.bfloat16).view(1, 1),
        )
    )
    cases += 4
    if args.checkpoint:
        root = args.checkpoint.resolve()
        key = "layers.0.ffn.shared_experts.w1.weight"
        scale_key = key.removesuffix(".weight") + ".scale"
        actual = op.load_safetensors(root, key, chunk_rows=256, staging_bytes=2 << 20)
        index = _index(root / "model.safetensors.index.json")
        # Independent file reads into CPU raw bytes, separate from loader mmap.
        raw = []
        for name in (key, scale_key):
            path = root / index[name]
            header, start = _header(path)
            spec = header[name]
            with path.open("rb") as file:
                file.seek(start + spec["data_offsets"][0])
                payload = bytearray(
                    file.read(spec["data_offsets"][1] - spec["data_offsets"][0])
                )
            raw.append(
                torch.frombuffer(payload, dtype=torch.uint8).reshape(spec["shape"])
            )
        check(actual, reference(*raw))
        assert op.last_load["chunks"] == 9 and op.last_load["staging_bytes"] <= 2 << 20
        print(f"real checkpoint {key}: {actual.shape}, {op.last_load}", flush=True)
        cases += 1
    print(
        f"PASS {cases} dense FP8 weight cases (all encodings, 32x32 scales, tails, loader, Graph, aliases)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    run(parser.parse_args())
