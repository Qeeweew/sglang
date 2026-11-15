"""RoPE tile boundaries, non-power-of-two heads and large eager/Graph storage."""

import argparse
import json
import os
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.position_quant import (
    PositionQuant,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.position_quant import rope_reference


def guarded(shape):
    n = shape[0] * shape[1] * shape[2]
    storage = torch.full((n + 256,), -321, dtype=torch.bfloat16, device="npu")
    return storage[:n].view(shape), storage[n:]


def check_prefix(x, y):
    for start in range(0, x.shape[0], 1024):
        assert torch.equal(
            x[start : start + 1024, :, :-64], y[start : start + 1024, :, :-64]
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-dir", type=Path, default=ROOT / "build/native")
    p.add_argument("--physical", type=int, required=True)
    a = p.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(a.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(104728)
    op = PositionQuant(library_dir=a.library_dir)
    cases = 0
    for m, h, d in (
        (33, 3, 128),
        (129, 5, 512),
        (129, 65, 128),
        (2049, 1, 512),
        (4097, 1, 128),
        (8192, 64, 512),
        (32768, 64, 512),
        (32769, 64, 512),
    ):
        x, xguard = guarded((m, h, d))
        x.normal_()
        out, guard = guarded((m, h, d))
        indices = torch.unique(
            torch.cat(
                (
                    torch.linspace(0, m - 1, min(m, 129)).long(),
                    torch.tensor([0, 1, 31, 32, m - 1]),
                )
            )
        )
        selected = indices.npu()
        phase = torch.randn(m, 32)
        cos, sin = phase.cos() * 1.17, phase.sin() * 1.17
        c, s = cos.npu(), sin.npu()
        for inverse in (False, True):
            before = x.index_select(0, selected).cpu()
            expected = rope_reference(before, cos[indices], sin[indices], inverse)
            op.rope(x, c, s, inverse=inverse, out=out)
            assert torch.equal(out.index_select(0, selected).cpu(), expected)
            check_prefix(x, out)
            op.rope(x, c, s, inverse=inverse, out=x)
            assert torch.equal(x.index_select(0, selected).cpu(), expected)
            check_prefix(x, out)
            assert bool((guard.cpu() == -321).all())
            assert bool((xguard.cpu() == -321).all())
            cases += 2
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        c.fill_(1)
        s.zero_()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            for _ in range(2):
                op.rope(x, c, s, out=x)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            op.rope(x, c, s, out=x)
        for _ in range(3):
            phase.normal_()
            cos, sin = phase.cos() * 1.17, phase.sin() * 1.17
            before = x.index_select(0, selected).cpu()
            c.copy_(cos)
            s.copy_(sin)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            expected = rope_reference(before, cos[indices], sin[indices])
            assert torch.equal(x.index_select(0, selected).cpu(), expected)
            check_prefix(x, out)
            assert bool((xguard.cpu() == -321).all())
            cases += 1
        del graph
        print(
            json.dumps({"M": m, "H": h, "D": d, "cases": cases, "status": "passed"}),
            flush=True,
        )
    print(
        json.dumps({"status": "passed", "cases": cases, "physical": a.physical}),
        flush=True,
    )


if __name__ == "__main__":
    main()
