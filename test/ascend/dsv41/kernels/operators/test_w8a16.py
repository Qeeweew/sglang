"""W8A16 full numerical/Graph/canary tests. Does not measure performance."""

import argparse
import json
import os
import time
from pathlib import Path


def workspace_elements(workers, bn, bk):
    return workers * 2 * bn * bk


def exhaustive_decode(torch, args):
    """All permitted positive-half scale bit patterns × all 256 INT8 codes.

    Identity activation exposes decoded weights without reproducing the device
    cast loop. Dot accumulation may normalize signed zero; all nonzero BF16
    results are compared by their bits, and zeros must remain numerically zero.
    """
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
        W8A16,
        repack_weight,
    )
    from dsv41.kernels.references.w8a16 import dequantize

    # 0x5fff is FP16 511.75 = 65504/128, the largest admitted scale.
    scale_patterns = 0x6000
    n, groups = 1024, scale_patterns // 1024
    k = groups * 256
    wc = torch.arange(k).remainder(256).sub(128).to(torch.int8).repeat(n, 1)
    sc = (
        torch.arange(scale_patterns, dtype=torch.int16)
        .view(torch.float16)
        .reshape(groups, n)
    )
    x = torch.eye(k, dtype=torch.bfloat16).npu()
    p, s = (t.npu() for t in repack_weight(wc, sc))
    out_backing = torch.full((k * n + 64,), -77, device="npu", dtype=torch.bfloat16)
    out = out_backing[32:-32].view(k, n)
    workers = torch.npu.get_device_properties(0).cube_core_num
    workspace_backing = torch.full(
        (workspace_elements(workers, 128, 256) + 64,),
        -79,
        device="npu",
        dtype=torch.bfloat16,
    )
    workspace = workspace_backing[32:-32]
    call = W8A16().bind(x, p, s, out=out, workspace=workspace, block_m=256, block_n=128)
    pointers = [t.data_ptr() for t in (x, p, s, out, workspace)]

    def check():
        expected = dequantize(wc, sc, torch.bfloat16).T.contiguous()
        actual = out.cpu()
        nonzero = expected != 0
        assert torch.equal(
            actual.view(torch.int16)[nonzero], expected.view(torch.int16)[nonzero]
        )
        assert bool((actual[~nonzero] == 0).all())
        assert bool((out_backing[:32] == -77).all()) and bool(
            (out_backing[-32:] == -77).all()
        )
        assert bool((workspace_backing[:32] == -79).all()) and bool(
            (workspace_backing[-32:] == -79).all()
        )
        assert pointers == [t.data_ptr() for t in (x, p, s, out, workspace)]

    call()
    torch.npu.synchronize()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call()
    for _ in range(2):
        wc.bitwise_not_()
        sc = sc.flip(1).contiguous()
        pc, spc = repack_weight(wc, sc)
        p.copy_(pc)
        s.copy_(spc)
        graph.replay()
        torch.npu.synchronize()
        check()
    result = dict(
        status="passed",
        scale_patterns=scale_patterns,
        int8_codes=256,
        combinations=scale_patterns * 256,
        nonzero_bf16_bit_exact=True,
        zero_sign="dot accumulation may normalize signed zero",
        changed_graph=True,
        prefix_suffix_canaries=True,
        m=k,
        plan=list(call.plan),
    )
    (args.output / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


def workspace_reuse(torch):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
        W8A16,
        repack_weight,
    )
    from torch_npu.npu import memory

    from dsv41.kernels.references.w8a16 import reference

    stream = torch.npu.Stream()
    rows, n, k = 11, 2304, 1024
    xc = (torch.randn(rows, k) * 0.125).bfloat16()
    wc = torch.randint(-128, 128, (n, k), dtype=torch.int8)
    sc = torch.full((k // 256, n), 2**-10, dtype=torch.float16)
    with torch.npu.stream(stream):
        x = xc.npu()
        packed, scales = (value.npu() for value in repack_weight(wc, sc))
        out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
        sort_input = torch.randn((64, 4096), device=x.device)
        sort_out = (
            torch.empty_like(sort_input),
            torch.empty_like(sort_input, dtype=torch.long),
        )
        op = W8A16()
        call = op.bind(x, packed, scales, out=out)
        call()
        torch.sort(sort_input, out=sort_out)
    stream.synchronize()
    bm, bn, bk = call.plan
    cores = torch.npu.get_device_properties(0).cube_core_num
    workers = min(cores, (rows + bm - 1) // bm * (n // bn))
    expected_bytes = workspace_elements(workers, bn, bk) * x.element_size()

    def allocations():
        return [
            (event["addr"], event["size"])
            for trace in memory._snapshot()["device_traces"]
            for event in trace
            if event["action"] == "alloc"
        ]

    memory._record_memory_history("all", context=None, max_entries=1024)
    try:
        with torch.npu.stream(stream):
            start = len(allocations())
            op.bind(x, packed, scales, out=out)
            assert len(allocations()) == start
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            call()
            torch.sort(sort_input, out=sort_out)
        scratch = [(addr, size) for addr, size in allocations()[start:] if size > 1024]
        assert len(scratch) == 2, scratch
        assert scratch[0][1] == expected_bytes, scratch
        assert scratch[0][0] == scratch[1][0], scratch
        with torch.npu.stream(stream):
            xc.neg_()
            x.copy_(xc)
            sort_input.uniform_()
            start = len(allocations())
            graph.replay()
        stream.synchronize()
        assert len(allocations()) == start
        torch.testing.assert_close(
            out.cpu().float(), reference(xc, wc, sc).float(), rtol=0.008, atol=0.002
        )
        torch.testing.assert_close(
            sort_out[0].cpu(), torch.sort(sort_input.cpu()).values
        )
    finally:
        memory._record_memory_history(None)
    return dict(
        case="workspace_reuse",
        status="passed",
        active_workers=workers,
        w8a16_bytes=expected_bytes,
        sort_bytes=scratch[1][1],
    )


def run(args):
    if (
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical)
        or os.environ.get("ASCEND_DEVICE_ID") != "0"
    ):
        raise ValueError("bind the physical NPU before importing Torch")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
        W8A16,
        repack_weight,
    )
    from dsv41.kernels.references.w8a16 import reference

    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(1616)
    if args.exhaustive:
        exhaustive_decode(torch, args)
        return
    cases = [
        (1, 1, 128, 256),
        (6, 1, 128, 256),
        (16, 1, 128, 256),
        (6, 1, 384, 768),
        (16, 1, 128, 512),
        (17, 1, 128, 256),
        (48, 2, 256, 512),
        (129, 1, 256, 768),
        (273, 1, 128, 256),
        (0, 1, 128, 256),
        (6, 8, 1024, 4096),
        # Multiple output tiles, odd K panel count and non-fractal tails
        # under changed weight/scale replay.
        (2049, 1, 3072, 768),
        (4097, 1, 256, 512),
    ]
    cases = [(*case, 256) for case in cases]
    # Cover both a half-group panel and two groups, including changed
    # weight/scale Graphs and the corresponding workspace capacities.
    cases += [(17, 1, 128, 512, 128), (17, 1, 128, 512, 512)]
    cases = [
        (*case, 16 if case[0] <= 16 else 64 if case[0] <= 64 else 256) for case in cases
    ]
    # The other finite prefill tile plan, M/N swizzle boundary windows and
    # batched activation/output pitches in the shared batch/M/N task space.
    cases += [
        (2049, 1, 1280, 768, 256, 128),
        (4097, 1, 256, 512, 256, 128),
        (273, 2, 256, 512, 256, 256),
    ]
    cases = [(*case, 128) for case in cases]
    cases += [
        (4097, 1, 256, 512, 256, 256, 64),
        (2049, 1, 1280, 768, 256, 256, 64),
        (273, 2, 256, 512, 256, 256, 64),
        (11, 1, 2304, 1024, None, None, None),
        (79, 3, 1536, 768, None, None, None),
    ]
    records = [workspace_reuse(torch)]
    stream = torch.npu.Stream()
    for dtype in (torch.bfloat16, torch.float16):
        for m, b, n, k, bk, bm, bn in cases:
            shape = (m, b, k) if b > 1 else (m, k)
            xc = (torch.randn(shape) * 0.125).to(dtype)
            wc = torch.randint(-128, 128, (b * n, k), dtype=torch.int8)
            sc = (torch.rand(k // 256, b * n) * 0.001 + 0.0001).half()
            # Include zero and minimum FP16 scales and every signed code.
            sc[:, :16] = 0
            sc[:, 16:32] = 2**-24
            wc[0] = torch.arange(k).remainder(256).sub(128).to(torch.int8)
            pc, spc = repack_weight(wc, sc)
            x, p, s = xc.npu(), pc.npu(), spc.npu()
            workers = torch.npu.get_device_properties(0).cube_core_num
            if bn is None:
                allocated = W8A16().bind(x, p, s)
                assert allocated.output.shape == (*x.shape[:-1], n)
                expected_plan = (16, 128, 512) if m == 11 else (128, 256, 256)
                if workers == 20:
                    assert allocated.plan == expected_plan
                capacity = workspace_elements(workers, *allocated.plan[1:])
                assert allocated.workspace is None
            else:
                capacity = workspace_elements(workers, bn, bk)
            out_storage = torch.full((m * b * n + 32,), -77, device="npu", dtype=dtype)
            workspace_storage = torch.full(
                (capacity + 32,),
                -79,
                device="npu",
                dtype=dtype,
            )
            out = out_storage[:-32].view((m, b, n) if b > 1 else (m, n))
            workspace = workspace_storage[:-32]
            started = time.monotonic()
            call = W8A16().bind(
                x,
                p,
                s,
                out=out,
                workspace=workspace,
                block_m=bm,
                block_n=bn,
                block_k=bk,
            )
            prepare_seconds = time.monotonic() - started
            assert call.output.data_ptr() == out.data_ptr()
            assert call.workspace.data_ptr() == workspace.data_ptr()
            pointers = [t.data_ptr() for t in (x, p, s, out, workspace)]
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                call()
            stream.synchronize()

            def check():
                expected = reference(xc, wc, sc)
                torch.testing.assert_close(
                    out.cpu().float(), expected.float(), rtol=0.008, atol=0.002
                )
                assert bool((out_storage[-32:] == -77).all())
                assert bool((workspace_storage[-32:] == -79).all())
                assert pointers == [t.data_ptr() for t in (x, p, s, out, workspace)]

            check()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                call()
            for change in range(3):
                if change == 0:
                    xc.neg_()
                elif change == 1:
                    wc.bitwise_not_()
                else:
                    sc.mul_(1.25)
                pc, spc = repack_weight(wc, sc)
                with torch.npu.stream(stream):
                    x.copy_(xc)
                    p.copy_(pc)
                    s.copy_(spc)
                    graph.replay()
                stream.synchronize()
                check()
            del graph
            record = dict(
                shape=[m, b, n, k],
                dtype=str(dtype),
                plan=list(call.plan),
                prepare_seconds=prepare_seconds,
                status="passed",
                changed_graph=True,
                canaries=True,
                independent_reference="full FP64 dot",
            )
            records.append(record)
            (args.output / "checks.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(record, flush=True)

        # A one-hot input exposes every decoded weight directly, including
        # half subnormals, the double-rounding boundary and -128*max_scale.
        wc = (
            torch.arange(128 * 256)
            .remainder(256)
            .sub(128)
            .to(torch.int8)
            .view(128, 256)
        )
        sc = torch.ones((1, 128), dtype=torch.float16)
        sc[0, :32] = 2**-24
        sc[0, 32:64] = 4.178285598754883e-05
        sc[0, 64:96] = 65504 / 128
        xc = torch.eye(256, dtype=dtype)
        p, s = (t.npu() for t in repack_weight(wc, sc))
        call = W8A16().bind(xc.npu(), p, s)
        call()
        torch.npu.synchronize()
        torch.testing.assert_close(
            call.output.cpu(), reference(xc, wc, sc), rtol=0, atol=0
        )
        record = dict(
            case="one_hot_dequantization",
            dtype=str(dtype),
            status="passed",
            bit_exact=True,
        )
        records.append(record)
        (args.output / "checks.json").write_text(json.dumps(records, indent=2) + "\n")
        print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="all admitted FP16 scales and INT8 codes, identity dot and changed Graph",
    )
    run(parser.parse_args())
