"""Runtime mHC dimensions, panel/row tails and changed-input Graph replay."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import MHC
from dsv41.kernels.references.mhc import (
    check_bf16,
    check_coeff,
    check_combine,
    post_reference,
    stats_reference,
)


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(44171)
    op = MHC()
    # The mixed Fn ND2NZ instruction has a uint16 source-stride field.
    # Reject the excluded geometry at host instead of a row-copy fallback.
    rejected = torch.empty((9, 8, 8192), dtype=torch.bfloat16, device="npu")
    fn_rejected = torch.empty((80, 65536), dtype=torch.float32, device="npu")
    scale_rejected = torch.empty(3, dtype=torch.float32, device="npu")
    base_rejected = torch.empty(80, dtype=torch.float32, device="npu")
    try:
        op.stats(rejected, fn_rejected, scale_rejected, base_rejected)
    except ValueError as error:
        assert "ND2NZ source stride exceeds uint16" in str(error)
    else:
        raise AssertionError(
            "accepted mixed projection with overflowing ND2NZ source stride"
        )
    del rejected, fn_rejected, scale_rejected, base_rejected
    print("PASS mixed ND2NZ host rejection", flush=True)
    cases = 0
    for m, b, h in (
        (1, 1, 128),
        (6, 3, 1024),
        (9, 2, 6144),
        (9, 8, 8064),
        (33, 3, 1024),
        (129, 8, 5120),
        (257, 5, 768),
        (513, 2, 1024),
    ):
        print("START", m, b, h, flush=True)
        shapes = (
            (m, b, h),
            (b * (b + 2), b * h),
            (3,),
            (b * (b + 2),),
            (m, b),
            (h,),
            (m, h),
        )
        cpu = [torch.randn(shape) for shape in shapes]
        cpu[0] = cpu[0].bfloat16()
        cpu[1] *= 0.007
        cpu[4] = cpu[4].sigmoid()
        cpu[6] = cpu[6].bfloat16()
        dev = [v.npu() for v in cpu]
        r, fn, scale, base, pre, weight, x = dev
        ws = op.workspace(r)
        bm = op.geometry(r)[0]
        computed_rows = (m + bm - 1) // bm * bm
        ws.projection[:, computed_rows:].fill_(float("nan"))
        ws.square_sum[:, computed_rows:].fill_(float("nan"))
        out = torch.empty_like(x)
        updated = torch.empty_like(r)
        fused_updated = torch.empty_like(r)
        packed = torch.full((m, 5 * h), 17, dtype=torch.bfloat16, device="npu")
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())

        def invoke():
            op.stats(r, fn, scale, base, workspace=ws)
            op.post_combine(
                x,
                r,
                pre,
                ws.triplet.post,
                ws.triplet.comb,
                weight,
                updated=updated,
                out=out,
            )
            op.mean_packed(r, packed, 3)
            op.stats_post(x, r, fn, scale, base, workspace=ws, out=fused_updated)

        with torch.npu.stream(stream):
            invoke()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            for _ in range(20):
                invoke()
        pointers = [
            v.data_ptr()
            for v in (
                *dev,
                ws.projection,
                ws.square_sum,
                *ws.triplet,
                out,
                updated,
                fused_updated,
                packed,
            )
        ]
        for generation in range(2):
            if generation:
                cpu[0] = torch.randn((m, b, h)).bfloat16()
                cpu[6] = torch.randn((m, h)).bfloat16()
                cpu[1].normal_(std=0.007)
                cpu[3].normal_()
                cpu[4].uniform_()
                for dst, src in zip(dev, cpu):
                    dst.copy_(src)
            for target in (
                ws.projection,
                ws.square_sum,
                *ws.triplet,
                out,
                updated,
                fused_updated,
            ):
                target.fill_(float("nan"))
            packed.fill_(17)
            torch.npu.synchronize()
            for _ in range(20):
                graph.replay()
            stream.synchronize()
            cr, cf, cs, cb, cp, cw, cx = cpu
            check_coeff(ws.triplet, stats_reference(cr, cf, cs, cb))
            check_bf16(
                updated,
                post_reference(cx, cr, ws.triplet.post.cpu(), ws.triplet.comb.cpu()),
            )
            check_combine(out, updated.cpu(), cp, cw)
            check_bf16(
                fused_updated,
                post_reference(cx, cr, ws.triplet.post.cpu(), ws.triplet.comb.cpu()),
            )
            check_bf16(packed[:, 3 * h : 4 * h], cr.double().mean(1).bfloat16())
            assert bool((packed[:, : 3 * h].cpu() == 17).all()) and bool(
                (packed[:, 4 * h :].cpu() == 17).all()
            )
            assert pointers == [
                v.data_ptr()
                for v in (
                    *dev,
                    ws.projection,
                    ws.square_sum,
                    *ws.triplet,
                    out,
                    updated,
                    fused_updated,
                    packed,
                )
            ]
            assert (
                torch.isfinite(ws.projection[:, :computed_rows]).all()
                and torch.isfinite(ws.square_sum[:, :computed_rows]).all()
            )
            assert (
                torch.count_nonzero(ws.projection[:, :computed_rows, b * (b + 2) :])
                == 0
            )
            assert (
                torch.count_nonzero(ws.projection[:, m:computed_rows]) == 0
                and torch.count_nonzero(ws.square_sum[:, m:computed_rows]) == 0
            )
            assert torch.isnan(ws.projection[:, computed_rows:]).all()
            assert torch.isnan(ws.square_sum[:, computed_rows:]).all()
            cases += 1
        print("PASS", m, b, h, flush=True)
    print(
        {
            "status": "passed",
            "changed_graph_cases": cases,
            "dispatches": 20,
            "replays": 20,
        },
        flush=True,
    )


if __name__ == "__main__":
    run()
