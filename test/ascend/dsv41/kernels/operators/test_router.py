"""Finite-logit V4.1 routing, padding/Graph and output canaries."""

import argparse
import gc
import json
import weakref
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router
from dsv41.kernels.references.router import compare, reference


def guarded(m, k, dtype):
    storage = torch.full((m * k + 128,), -1234, dtype=dtype, device="npu")
    return storage[: m * k].view(m, k), storage


def numeric(op):
    cases = 0
    for e in (384, 128):
        k = 6 if e == 384 else 3
        for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257, 2049):
            for biased in (False, True):
                x = torch.randn(m, e) * 3
                bias = torch.randn(e) * 0.2 if biased else None
                ig, is_ = guarded(m, k, torch.int32)
                wg, ws = guarded(m, k, torch.float32)
                ids, w = op(
                    x.npu(),
                    bias.npu() if bias is not None else None,
                    ids_out=ig,
                    weights_out=wg,
                )
                compare(ids, w, reference(x, bias))
                assert not op._last, "eager router retained logits"
                assert bool((is_[m * k :].cpu() == -1234).all())
                assert bool((ws[m * k :].cpu() == -1234).all())
                cases += 1
        # Finite logits, including small/subnormal scores; ranks are separated.
        for value in (-20.0, -90.0, -100.0, 20.0, 100.0):
            x = torch.full((6, e), value)
            bias = torch.linspace(-0.1, 0.1, e)
            ids, w = op(x.npu(), bias.npu())
            compare(ids, w, reference(x, bias))
            cases += 1
        # Correction changes membership; gathered weights exclude that bias.
        x = torch.arange(e, dtype=torch.float32)[None, :].repeat(6, 1) / e
        bias = torch.full((e,), -10.0)
        bias[:k] = 100.0
        ids, w = op(x.npu(), bias.npu(), factor=1.0)
        compare(ids, w, reference(x, bias, 1.0))
        cases += 1
        print(json.dumps({"experts": e, "cases": cases, "status": "pass"}), flush=True)
    return cases


def graph(op, e):
    m = 6
    k = 6 if e == 384 else 3
    x = torch.randn(m, e, device="npu")
    bias = torch.randn(e, device="npu") * 0.1
    live = torch.tensor([m], dtype=torch.int32, device="npu")
    ids = torch.empty(m, k, dtype=torch.int32, device="npu")
    w = torch.empty(m, k, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            op(x, bias, valid_rows=live, ids_out=ids, weights_out=w)
    stream.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        op(x, bias, valid_rows=live, ids_out=ids, weights_out=w)
    for valid in (6, 1, 0, 5):
        x.copy_(torch.randn(m, e))
        bias.copy_(torch.randn(e) * 0.1)
        live.fill_(valid)
        temporary = torch.randn(257, e, device="npu")
        borrowed = weakref.ref(temporary)
        op(temporary)
        del temporary
        gc.collect()
        assert borrowed() is None, "router held an eager activation after return"
        torch.npu.synchronize()
        g.replay()
        torch.npu.synchronize()
        compare(ids, w, reference(x.cpu(), bias.cpu(), live=valid))
    print(json.dumps({"graph_experts": e, "status": "pass"}), flush=True)


def vision(op, e):
    """Mixed text/image correction affects selection, never output weights."""
    m, k = 6, 6 if e == 384 else 3
    x = torch.randn(m, e, device="npu") * 0.1
    bias = torch.zeros(e, device="npu")
    alt = torch.zeros_like(bias)
    bias[:k] = 10.0
    alt[-k:] = 10.0
    mask = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.int32, device="npu")
    live = torch.tensor([m], dtype=torch.int32, device="npu")
    ids = torch.empty(m, k, dtype=torch.int32, device="npu")
    w = torch.empty(m, k, device="npu")

    def run():
        return op(
            x,
            bias,
            alternate_bias=alt,
            image_mask=mask,
            valid_rows=live,
            ids_out=ids,
            weights_out=w,
        )

    def check():
        expected = [
            reference(x[row : row + 1].cpu(), alt.cpu() if flag else bias.cpu())
            for row, flag in enumerate(mask.cpu().tolist())
        ]
        ei = torch.cat([v[0] for v in expected])
        ew = torch.cat([v[1] for v in expected])
        count = int(live.cpu()[0])
        ei[count:] = -1
        ew[count:] = 0
        compare(ids, w, (ei, ew))

    run()
    check()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        run()
    stream.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        run()
    for count in (6, 5, 1, 0):
        mask.copy_(1 - mask)
        x.normal_(0, 0.1)
        live.fill_(count)
        torch.npu.synchronize()
        g.replay()
        torch.npu.synchronize()
        check()
    print(json.dumps({"vision_experts": e, "status": "pass"}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library-dir", type=Path, default=Path("build/dsv41"))
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(9811)
    op = Router(library_dir=a.library_dir.resolve())
    count = numeric(op)
    for e in (384, 128):
        graph(op, e)
        vision(op, e)
    print(json.dumps({"cases": count, "status": "pass"}), flush=True)


if __name__ == "__main__":
    main()
