"""Mean/post tile coverage, independent references and changed-input Graph replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import MHC
from dsv41.kernels.references.mhc import check_bf16, post_reference


def guarded(shape):
    size = 1
    for dimension in shape:
        size *= dimension
    storage = torch.full((size + 64,), 123, dtype=torch.bfloat16, device="npu")
    return storage[32:-32].view(shape), storage


def run(args):
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(414170817)
    op, records = MHC(), []
    for case, m in enumerate((1, 6, 33, 128, 257, 2048, 8192, 32768)):
        print(f"state tiles M={m}", flush=True)
        column = case % 3
        base_r = torch.randn((m, 4, 5120), generator=rng).bfloat16()
        base_x = torch.randn((m, 5120), generator=rng).bfloat16()
        posts = torch.rand((m, 4), generator=rng) * 2
        comb = torch.rand((m, 4, 4), generator=rng)
        comb /= comb.sum(-2, keepdim=True)
        residual, x, post, matrix = [v.npu() for v in (base_r, base_x, posts, comb)]
        packed, packed_storage = guarded((m, 15360))
        updated, updated_storage = guarded((m, 4, 5120))
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())

        def invoke():
            op.mean_packed(residual, packed, column)
            op.post(x, residual, post, matrix, out=updated)

        indices = torch.arange(m) if m <= 257 else torch.linspace(0, m - 1, 129).long()

        def validate(source_r, source_x, amplitude, mode):
            scale = max(1.0, amplitude)
            r, xx = source_r[indices], source_x[indices]
            mean = packed.cpu()[indices, column * 5120 : (column + 1) * 5120]
            reference = r.double().mean(1).bfloat16()
            serial = r[:, 0].float()
            for source in range(1, 4):
                serial = serial + r[:, source].float()
            error = check_bf16(mean, reference, scale, fp32=(serial * 0.25).bfloat16())
            records.append(
                dict(op="mean", m=m, column=column, mode=mode, relative_l2=error)
            )
            error = check_bf16(
                updated.cpu()[indices],
                post_reference(xx, r, posts[indices], comb[indices]),
                scale,
                fp32=post_reference(
                    xx, r, posts[indices], comb[indices], torch.float32
                ),
            )
            records.append(dict(op="post", m=m, mode=mode, relative_l2=error))
            assert torch.isfinite(updated).all()
            live = packed[:, column * 5120 : (column + 1) * 5120]
            assert torch.isfinite(live).all()
            if amplitude == 0:
                # Full output coverage, including rows outside the sampled oracle.
                assert torch.all(updated == 0)
                assert torch.all(live == 0)
            for other in range(3):
                if other != column:
                    assert torch.all(
                        packed[:, other * 5120 : (other + 1) * 5120] == 123
                    )
            for storage in (packed_storage, updated_storage):
                assert torch.all(storage[:32] == 123)
                assert torch.all(storage[-32:] == 123)

        with torch.npu.stream(stream):
            invoke()
        stream.synchronize()
        validate(base_r, base_x, 1.0, "eager")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            invoke()
        tensors = (residual, x, post, matrix, packed, updated)
        pointers = [v.data_ptr() for v in tensors]
        for amplitude in (0.0, 1e-8, 1e4):
            source_r, source_x = base_r * amplitude, base_x * amplitude
            with torch.npu.stream(stream):
                residual.copy_(source_r)
                x.copy_(source_x)
                packed.fill_(123)
                updated.fill_(123)
                graph.replay()
            stream.synchronize()
            validate(source_r, source_x, amplitude, f"graph/{amplitude:g}")
            assert pointers == [v.data_ptr() for v in tensors]
        del graph
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(dict(physical=args.physical, cases=records), indent=2) + "\n"
    )
    print(f"PASS {len(records)} mean/post tile cases; {args.report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args())
