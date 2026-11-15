"""Runtime row tiles, stable workspaces and interleaved Graph replay."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import MHC, Workspace
from dsv41.kernels.references.mhc import (
    check_coeff,
    data,
    stats_reference,
)


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(414180)
    op = MHC()
    # The framework head supplies a flat residual, including on idle DP ranks.
    # Infer width from its physical shape: view(0, branches, -1) is ambiguous.
    for m in (0, 1, 6):
        residual_cpu = torch.randn((m, 4, 5120), generator=rng).bfloat16()
        pre_cpu = torch.rand((m, 4), generator=rng)
        residual = residual_cpu.view(m, 4 * 5120).npu()
        pre = pre_cpu.npu()
        storage = torch.full((m * 5120 + 64,), -127, dtype=torch.bfloat16, device="npu")
        out = storage[32:-32].view(m, 5120)
        result = op.collapse(residual, pre, out=out)
        assert result is out
        expected = torch.einsum(
            "mb,mbh->mh", pre_cpu.double(), residual_cpu.double()
        ).bfloat16()
        torch.testing.assert_close(out.cpu(), expected, rtol=0.015, atol=0.01)
        assert torch.all(storage[:32].cpu() == -127)
        assert torch.all(storage[-32:].cpu() == -127)
    print("PASS mHC flat head M0/M1/M6, caller output and canaries", flush=True)
    cases = {}
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for m in (6, 9, 48, 64, 65, 96, 128, 257):
            cpu = data(m, rng)[:4]
            dev = [v.npu() for v in cpu]
            workspace = op.workspace(dev[0])
            bm, rows, splits = op.project_geometry(m)
            assert workspace.projection.shape == (splits, rows, 32)
            assert workspace.square_sum.shape == (splits, rows)
            op.stats(*dev, workspace=workspace)
            stream.synchronize()
            for dtype in (torch.float64, torch.float32):
                check_coeff(workspace.triplet, stats_reference(*cpu, dtype))
            cases[m] = dict(cpu=cpu, dev=dev, workspace=workspace, bm=bm)

        # Caller-owned buffers must match the same geometry as allocation.
        for m, wrong_rows in ((48, 128), (65, 64)):
            c = cases[m]
            w = c["workspace"]
            bad_projection = torch.empty(
                (10, wrong_rows, 32), dtype=torch.float32, device="npu"
            )
            bad_squares = torch.empty(
                (10, wrong_rows), dtype=torch.float32, device="npu"
            )
            for bad, name in (
                (Workspace(bad_projection, w.triplet, w.square_sum), "projection"),
                (Workspace(w.projection, w.triplet, bad_squares), "square_sum"),
            ):
                try:
                    op.stats(*c["dev"], workspace=bad)
                except ValueError as error:
                    assert name in str(error), str(error)
                else:
                    raise AssertionError(f"accepted wrong {name} geometry for M{m}")

        graphs = {}
        for m in (48, 96):
            c = cases[m]
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                op.stats(*c["dev"], workspace=c["workspace"])
            graphs[m] = graph
        stream.synchronize()
        ring_addresses = {
            key: tuple(t.data_ptr() for t in buffers)
            for key, buffers in op._rings.items()
        }
        assert {key[2] for key in ring_addresses} == {
            c["bm"] for m, c in cases.items() if m > 8
        }
        addresses = {
            m: [
                t.data_ptr()
                for t in (
                    *c["dev"],
                    *c["workspace"].triplet,
                    c["workspace"].projection,
                    c["workspace"].square_sum,
                )
            ]
            for m, c in cases.items()
        }

        # Mutate every stats input, interleave eager shapes, then alternate
        # graphs that own different runtime ring geometries on the same stream.
        for generation in range(2):
            for m in (48, 96, 9, 65, 64, 48, 96):
                c = cases[m]
                c["cpu"] = data(m, rng)[:4]
                for target, source in zip(c["dev"], c["cpu"]):
                    target.copy_(source)
                if m in graphs:
                    graphs[m].replay()
                else:
                    op.stats(*c["dev"], workspace=c["workspace"])
                stream.synchronize()
                for dtype in (torch.float64, torch.float32):
                    check_coeff(
                        c["workspace"].triplet, stats_reference(*c["cpu"], dtype)
                    )
                assert addresses[m] == [
                    t.data_ptr()
                    for t in (
                        *c["dev"],
                        *c["workspace"].triplet,
                        c["workspace"].projection,
                        c["workspace"].square_sum,
                    )
                ]
                assert ring_addresses == {
                    key: tuple(t.data_ptr() for t in buffers)
                    for key, buffers in op._rings.items()
                }
    stream.synchronize()
    print(
        "PASS mHC compact geometry: 8 eager shapes, caller workspaces, "
        "2 mixed Graphs, changed inputs and stable rings"
    )


if __name__ == "__main__":
    run()
