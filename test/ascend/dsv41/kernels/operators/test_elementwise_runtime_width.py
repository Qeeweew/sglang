"""Non-model Vector widths, independent FP64 references and changed Graph inputs."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
    PackedClippedSwiGLU,
    RMSNorm,
)


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(414175)
    cases = 0
    for width in (64, 320, 1024, 6144, 8192):
        for rows in (6, 33):
            source = torch.randn((rows, width)).bfloat16()
            weight = torch.randn((width,)).bfloat16()
            x, w = source.npu(), weight.npu()
            out = torch.empty_like(x)
            norm = RMSNorm(dimension=width)
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                call = norm.bind(x, w, out=out)
                call()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                call()
            pointers = x.data_ptr(), w.data_ptr(), out.data_ptr()
            for generation in range(2):
                source = torch.randn((rows, width)).bfloat16() * (
                    1e-8 if generation else 1
                )
                x.copy_(source)
                out.fill_(float("nan"))
                torch.npu.synchronize()
                graph.replay()
                stream.synchronize()
                reference = (
                    source.double()
                    * torch.rsqrt(
                        source.double().square().mean(-1, keepdim=True) + 1e-20
                    )
                    * weight.double()
                ).bfloat16()
                torch.testing.assert_close(
                    out.cpu().float(), reference.float(), rtol=0.008, atol=2e-6
                )
                assert pointers == (x.data_ptr(), w.data_ptr(), out.data_ptr())
                cases += 1
    for width in (64, 288, 1024, 6144):
        for dtype in (torch.bfloat16, torch.float16):
            source = torch.randn((33, width * 2)).to(dtype) * 15
            x = source.npu()
            out = torch.empty((33, width), dtype=dtype, device="npu")
            op = PackedClippedSwiGLU(
                dimension=width, dtype="bf16" if dtype == torch.bfloat16 else "f16"
            )
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                call = op.bind(x, out=out)
                call()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                call()
            pointers = x.data_ptr(), out.data_ptr()
            for generation in range(2):
                source = (torch.randn((33, width * 2)) * 15).to(dtype)
                x.copy_(source)
                out.fill_(float("nan"))
                torch.npu.synchronize()
                graph.replay()
                stream.synchronize()
                gate, up = source.double().chunk(2, -1)
                gate = gate.clamp_max(10)
                reference = (gate * torch.sigmoid(gate) * up.clamp(-10, 10)).to(dtype)
                torch.testing.assert_close(
                    out.cpu().float(),
                    reference.float(),
                    rtol=0.008 if dtype == torch.bfloat16 else 0.001,
                    atol=2e-6,
                )
                assert pointers == (x.data_ptr(), out.data_ptr())
                cases += 1
    print({"status": "passed", "changed_graph_cases": cases}, flush=True)


if __name__ == "__main__":
    run()
