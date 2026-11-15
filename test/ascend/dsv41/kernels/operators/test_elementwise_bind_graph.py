"""Bind both schedules before their first dispatch, then capture immediately."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def one(kind: str, reverse: bool):
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
        PackedClippedSwiGLU,
        RMSNorm,
    )

    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(414178)
    rms = kind == "rms"
    dtype = torch.float16 if kind == "fp16" else torch.bfloat16
    width = 5120 if rms else 2304
    operator = (
        RMSNorm()
        if rms
        else PackedClippedSwiGLU(dtype="f16" if kind == "fp16" else "bf16")
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    weight = torch.randn(width).bfloat16()
    device_weight = weight.npu()
    fixtures = []
    for rows in (257, 1) if reverse else (1, 257):
        source = torch.randn(rows, width if rms else 2 * width).to(dtype)
        x = source.npu()
        out = torch.full((rows, width), float("nan"), device="npu", dtype=dtype)
        with torch.npu.stream(stream):
            call = (
                operator.bind(x, device_weight, out=out)
                if rms
                else operator.bind(x, out=out)
            )
        # Deliberately no eager call: bind must prepare the selected entry.
        fixtures.append((source, x, out, call))
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(20):
            for _, _, _, call in fixtures:
                call()
    pointers = [(x.data_ptr(), out.data_ptr()) for _, x, out, _ in fixtures]
    for generation in range(2):
        with torch.npu.stream(stream):
            for index, (source, x, out, call) in enumerate(fixtures):
                source = torch.randn_like(source) * (1e-8 if generation else 1)
                x.copy_(source)
                out.fill_(float("nan"))
                fixtures[index] = source, x, out, call
            graph.replay()
        stream.synchronize()
        for index, (source, x, out, _) in enumerate(fixtures):
            if rms:
                value = source.double()
                expected = (
                    value
                    * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20)
                    * weight.double()
                ).to(dtype)
            else:
                gate, up = source.double().chunk(2, -1)
                gate = gate.clamp_max(10)
                expected = (gate * torch.sigmoid(gate) * up.clamp(-10, 10)).to(dtype)
            torch.testing.assert_close(
                out.cpu().float(),
                expected.float(),
                rtol=0.001 if dtype == torch.float16 else 0.008,
                atol=2e-6,
            )
            assert pointers[index] == (x.data_ptr(), out.data_ptr())
    print(
        f"PASS {kind} {'streaming then single' if reverse else 'single then streaming'} direct Graph",
        flush=True,
    )


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", choices=("rms", "bf16", "fp16"))
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if args.one:
        one(args.one, args.reverse)
        return
    # A fresh process prevents a previous direction/family from preparing an
    # entry that this case must prepare through its own bind call.
    for kind in ("rms", "bf16", "fp16"):
        for reverse in (False, True):
            command = [sys.executable, str(Path(__file__).resolve()), "--one", kind]
            if reverse:
                command.append("--reverse")
            subprocess.run(command, check=True)
    print("PASS 6 fresh-process bind-order cases, 24 changed-output checks", flush=True)


if __name__ == "__main__":
    run()
