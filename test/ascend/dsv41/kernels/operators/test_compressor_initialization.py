"""Fresh C1 initialization after other Vector kernels, including Graph replay.

TBuf.Get() writes its descriptor in the 2201 SDK. Exercise fresh processes and
preceding kernels so an uninitialized descriptor cannot rely on a clean stack.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def worker(width, library):
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.compressor import (
        CompressorState,
        ProjectedCompressor,
    )
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.position_quant import (
        PositionQuant,
    )
    from dsv41.kernels.references.compressor import oracle_prepare
    from dsv41.kernels.support.compressor import validate_prepare

    torch.npu.set_device(0)
    torch.manual_seed(717 + width)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        position = PositionQuant(library_dir=library.parent if library else None)
        x = torch.randn(6, 16, 512, device="npu").bfloat16()
        cosine = torch.randn(6, 32, device="npu")
        sine = torch.randn_like(cosine)
        dirty = [position.rope(x, cosine, sine)]
        dirty.append(
            position.quantize_key(torch.randn(6, 128, device="npu").bfloat16())
        )
        dirty.append(
            position.gather_cos_sin(
                cosine,
                sine,
                torch.tensor([0, 1, -1, 8, 3, 5], dtype=torch.int64, device="npu"),
            )
        )
    stream.synchronize()

    gamma = torch.randn(width)
    kv = torch.randn(6, width).bfloat16()
    cu = torch.tensor([0, 6], dtype=torch.int32)
    ids = torch.tensor([0], dtype=torch.int32)
    pending = torch.zeros(4, 2, width)
    lengths = torch.zeros(4, dtype=torch.int64)
    records = []
    with torch.npu.stream(stream):
        state = CompressorState.allocate(4, width=width)
        op = ProjectedCompressor(1, gamma.npu(), state, library=library)
        device_kv, device_cu, device_ids = kv.npu(), cu.npu(), ids.npu()
        result = op.prepare(device_kv, None, device_cu, device_ids)
    stream.synchronize()

    def check(label):
        expected = oracle_prepare(1, kv, None, gamma, pending, lengths, cu, ids)
        validate_prepare(result, expected, state, pending, lengths, label, records)

    check("fresh")
    graph = torch.npu.NPUGraph()
    outputs = dict(
        out=result.latent,
        pooled=result.pooled,
        positions=result.positions,
        rope_positions=result.rope_positions,
        starts=result.starts,
    )
    pointers = [tensor.data_ptr() for tensor in outputs.values()]
    with torch.npu.graph(graph, stream=stream):
        op.prepare(device_kv, None, device_cu, device_ids, **outputs)
    for generation in range(2):
        kv = torch.randn_like(kv)
        with torch.npu.stream(stream):
            device_kv.copy_(kv)
            result.latent.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        check(f"graph{generation}")
        assert pointers == [tensor.data_ptr() for tensor in outputs.values()]
    assert dirty  # Keep preceding kernel inputs/outputs alive through the probe.
    print(
        json.dumps(
            {
                "width": width,
                "checks": len(records),
                "task_queue": os.environ["TASK_QUEUE_ENABLE"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-width", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_width is not None:
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(args.physical)
        return worker(args.worker_width, args.library)
    records = []
    for width, queue in [(512, "0"), (144, "1"), (4096, "0")]:
        command = [
            sys.executable,
            "-m",
            __spec__.name,
            "--physical",
            str(args.physical),
            "--worker-width",
            str(width),
        ]
        if args.library:
            command += ["--library", str(args.library.resolve())]
        env = os.environ | {
            "ASCEND_RT_VISIBLE_DEVICES": str(args.physical),
            "ASCEND_DEVICE_ID": "0",
            "TASK_QUEUE_ENABLE": queue,
        }
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        print(completed.stdout, end="", flush=True)
        records.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {"physical": args.physical, "fresh_processes": records}, indent=2
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
