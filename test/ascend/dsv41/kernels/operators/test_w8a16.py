"""W8A16 full numerical/Graph/canary tests. Does not measure performance."""

import argparse
import json
import os
import time
from pathlib import Path


def exhaustive_decode(torch, args):
    """All permitted positive-half scale bit patterns × all 256 INT8 codes.

    Identity activation exposes decoded weights without reproducing the device
    cast loop. Dot accumulation may normalize signed zero; all nonzero BF16
    results are compared by their bits, and zeros must remain numerically zero.
    """
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import (
        W8A16,
        repack_weight,
        workspace_elements,
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
        workspace_elements,
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
    ]
    records = []
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
            out_storage = torch.full((m * b * n + 32,), -77, device="npu", dtype=dtype)
            workspace_storage = torch.full(
                (workspace_elements(workers, bn, bk) + 32,),
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
