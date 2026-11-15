"""Finite-logit router: batched reductions, counts and stable expert buckets."""

import argparse
import json
import os
from pathlib import Path

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import (
    Router,
    RoutingWorkspace,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.router import compare, reference


def guarded(shape, dtype):
    count = 1
    for n in shape:
        count *= n
    storage = torch.full((count + 64,), -1234, dtype=dtype, device="npu")
    return storage[:count].view(shape), storage[count:]


def buffers(op, m, e):
    k = 6 if e == 384 else 3
    shapes = [
        (m, k),
        (m, k),
        (e,),
        (e + 1,),
        (m * k,),
        (4 * (e + 1),),
        (m, k),
        (
            min(op.cores, e),
            (e + 15) // 16 * 16,
        ),
    ]
    dtypes = [
        torch.int32,
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.int32,
    ]
    pairs = [guarded(s, d) for s, d in zip(shapes, dtypes)]
    values = [v[0] for v in pairs]
    return {
        "ids_out": values[0],
        "weights_out": values[1],
        "counts_out": values[2],
        "offsets_out": values[3],
        "order_out": values[4],
        "metadata_out": values[5],
        "workspace": RoutingWorkspace(*values[6:]),
    }, [v[1] for v in pairs]


def check(plan, logits, bias, valid, guards, alternate=None, mask=None):
    m, e = logits.shape
    if alternate is None:
        expected = reference(
            logits.cpu(), None if bias is None else bias.cpu(), live=valid
        )
    else:
        expected_rows = [
            reference(logits[r : r + 1].cpu(), (alternate if mask[r] else bias).cpu())
            for r in range(m)
        ]
        expected = (
            torch.cat([v[0] for v in expected_rows]),
            torch.cat([v[1] for v in expected_rows]),
        )
        expected[0][valid:] = -1
        expected[1][valid:] = 0
    compare(plan.ids, plan.weights, expected)
    flat = plan.ids.cpu().flatten()
    counts = torch.bincount(flat[flat >= 0].long(), minlength=e).int()
    assert torch.equal(plan.counts.cpu(), counts)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32), counts.cumsum(0, dtype=torch.int32))
    )
    torch.testing.assert_close(plan.offsets.cpu(), offsets, rtol=0, atol=0)
    perm = torch.argsort(flat, stable=True)
    perm = perm[flat[perm] >= 0]
    order = plan.order.cpu()
    at = 0
    for expert, n in enumerate(counts.tolist()):
        begin = int(offsets[expert])
        actual = order[begin : begin + n]
        assert torch.equal(actual, perm[at : at + n]), (
            "bucket",
            expert,
            actual,
            perm[at : at + n],
        )
        at += n
    assert at == valid * (6 if e == 384 else 3)
    assert all(bool((g.cpu() == -1234).all()) for g in guards)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library-dir", type=Path, default=ROOT / "build/native")
    p.add_argument("--physical", type=int, required=True)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(a.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    torch.manual_seed(12571)
    op = Router(library_dir=a.library_dir)
    cases = 0
    for e in (384, 128):
        for m in (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            33,
            127,
            128,
            129,
            255,
            256,
            257,
            8192,
            32768,
        ):
            x = torch.randn(m, e, device="npu") * 3
            b = torch.randn(e, device="npu") * 0.2
            outputs, guards = buffers(op, m, e)
            for live in sorted({0, m, max(0, m - 1)}):
                valid = torch.tensor([live], dtype=torch.int32, device="npu")
                # Unlaunched per-worker histogram slots are deliberately dirty.
                # Consumers must not include them, including zero-row plans.
                outputs["workspace"].histogram.fill_(12345)
                result = op.route(x, b, valid_rows=valid, **outputs)
                check(result, x, b, live, guards)
                cases += 1
            print(
                json.dumps({"E": e, "M": m, "cases": cases, "status": "passed"}),
                flush=True,
            )
        m = 33
        x = torch.randn(m, e, device="npu")
        b = torch.randn(e, device="npu") * 0.2
        alt = torch.randn_like(b) * 0.2
        mask = torch.randint(0, 2, (m,), dtype=torch.int32, device="npu")
        valid = torch.tensor([m], dtype=torch.int32, device="npu")
        outputs, guards = buffers(op, m, e)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                op.route(
                    x,
                    b,
                    alternate_bias=alt,
                    image_mask=mask,
                    valid_rows=valid,
                    **outputs,
                )
        stream.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, stream=stream):
            result = op.route(
                x, b, alternate_bias=alt, image_mask=mask, valid_rows=valid, **outputs
            )
        for live in (33, 6, 1, 0, 32, 33):
            x.normal_()
            b.normal_(0, 0.2)
            alt.normal_(0, 0.2)
            mask.copy_(1 - mask)
            valid.fill_(live)
            outputs["workspace"].histogram.fill_(12345)
            torch.npu.synchronize()
            g.replay()
            torch.npu.synchronize()
            check(result, x, b, live, guards, alt, mask.cpu().tolist())
            cases += 1
        # Finite negative logits retain the stable softplus underflow behavior.
        for value in (-20.0, -90.0, -100.0, 20.0, 100.0):
            x = torch.full((6, e), value, device="npu")
            b = torch.linspace(-0.2, 0.2, e, device="npu")
            outputs, guards = buffers(op, 6, e)
            result = op.route(x, b, **outputs)
            check(result, x, b, 6, guards)
            cases += 1
    report = {
        "status": "passed",
        "cases": cases,
        "physical": a.physical,
        "library": str((a.library_dir / "libdsv41_kernels.so").resolve()),
    }
    if a.report:
        a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
