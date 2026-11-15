"""Dynamic projection extents, long ND stride, state isolation and Graph."""

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.projection import (
    DenseProjection,
    WoAProjection,
    repack_linear,
)


def main():
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "6"
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(1641)
    cases = [
        (False, m, k, n, 1, fp32)
        for m in (0, 1, 6, 33, 128, 257)
        for k, n, fp32 in ((16, 16, True), (272, 144, False), (5120, 512, True))
    ]
    cases += [
        (True, m, k, n, g, False)
        for m in (0, 1, 6, 33, 128, 257)
        for k, n, g in ((16, 16, 3), (272, 144, 3), (256, 256, 8))
    ]
    cases += [(True, 6, 1280, 144, 65, False), (False, 1025, 272, 512, 1, True)]
    pending = []
    for grouped, m, k, n, groups, fp32 in cases:
        x = (torch.randn((m, groups, k) if grouped else (m, k)) * 0.1).bfloat16()
        w = (torch.randn(groups, n, k) * 0.1).bfloat16()
        reference = (
            torch.einsum("mgk,gnk->mgn", x.double(), w.double())
            if grouped
            else x.double() @ w[0].double().T
        )
        dtype = torch.float32 if fp32 else torch.bfloat16
        reference = reference.to(dtype)
        device_x = x.npu()
        device_w = w.npu() if grouped else repack_linear(w[0]).npu()
        shape = (m, groups, n) if grouped else (m, n)
        storage = torch.full((m * groups * n + 64,), 7, dtype=dtype, device="npu")
        output = storage[32:-32].view(shape)
        op = (
            WoAProjection(k=k, groups=groups, rank=n)
            if grouped
            else DenseProjection(k=k, columns=n, out_dtype="f32" if fp32 else "bf16")
        )
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            run = op.bind(device_x, device_w, out=output)
            output.fill_(float("nan"))
            run()
        stream.synchronize()
        torch.testing.assert_close(
            output.cpu().float(),
            reference.float(),
            rtol=1e-4 if fp32 else 0.01,
            atol=1e-4 if fp32 else 0.005,
        )
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            run()
        stream.synchronize()
        pending.append(
            (
                device_x,
                output,
                storage,
                graph,
                reference,
                fp32,
                (grouped, m, k, n, groups),
                device_w,
                run,
                stream,
            )
        )
        print("PASS eager dynamic projection", grouped, m, k, n, groups, flush=True)
    # Every replay follows different matrix-state dimensions/weight layouts.
    for iteration in range(2):
        for (
            x,
            output,
            storage,
            graph,
            reference,
            fp32,
            case,
            weight,
            run,
            stream,
        ) in reversed(pending):
            x.neg_()
            output.fill_(float("nan"))
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            expected = -reference if iteration == 0 else reference
            torch.testing.assert_close(
                output.cpu().float(),
                expected.float(),
                rtol=1e-4 if fp32 else 0.01,
                atol=1e-4 if fp32 else 0.005,
            )
            assert (storage[:32].cpu() == 7).all() and (storage[-32:].cpu() == 7).all()
        print(
            "PASS interleaved changed Graph and canaries",
            iteration,
            len(pending),
            flush=True,
        )
    print("PASS projection DeepJIT", len(cases), flush=True)


if __name__ == "__main__":
    main()
