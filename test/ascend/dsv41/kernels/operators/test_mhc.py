"""Independent FP64/FP32 references, V4.1 lagged pre and 910B Graph replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mhc import (
    MHC,
    Triplet,
    Workspace,
)
from dsv41.kernels.references.mhc import (
    check_bf16,
    check_coeff,
    check_combine,
    combine_reference,
    data,
    post_reference,
    stats_reference,
)


def guarded(shape, dtype):
    storage = torch.full((shape[0] + 2, *shape[1:]), -127.0, dtype=dtype, device="npu")
    return storage[1:-1], storage


def check_guards(storages):
    for storage in storages:
        assert torch.all(storage[[0, -1]].cpu() == -127)


def run(args):
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(41417)
    count, largest_relative = 0, 0.0
    op = MHC()
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
        16,
        32,
        33,
        128,
        129,
        255,
        256,
        257,
        385,
        513,
        641,
        769,
        1281,
    ):
        print(f"mHC M={m}", flush=True)
        cpu = data(m, generator)
        r, fn, scale, base, previous, weight, x = cpu
        nr, nf, ns, nb, np, nw, nx = [value.npu() for value in cpu]
        bm, rows, splits = op.project_geometry(m)
        computed_rows = (m + bm - 1) // bm * bm
        projection, projection_storage = guarded((splits, rows, 32), torch.float32)
        square_sum, square_storage = guarded((splits, rows), torch.float32)
        projection[:, computed_rows:].fill_(float("nan"))
        square_sum[:, computed_rows:].fill_(float("nan"))
        pre, pre_storage = guarded((m, 4), torch.float32)
        post, post_storage = guarded((m, 4), torch.float32)
        comb, comb_storage = guarded((m, 4, 4), torch.float32)
        out, out_storage = guarded((m, 5120), torch.bfloat16)
        updated, updated_storage = guarded((m, 4, 5120), torch.bfloat16)
        workspace = Workspace(projection, Triplet(pre, post, comb), square_sum)
        result = op.pre(nr, nf, ns, nb, nw, np, workspace=workspace, out=out)
        oracle = stats_reference(r, fn, scale, base)
        check_coeff(result.triplet, oracle)
        oracle32 = stats_reference(r, fn, scale, base, torch.float32)
        check_coeff(result.triplet, oracle32)
        largest_relative = max(
            largest_relative,
            check_combine(result.input, r, previous, weight),
        )
        count += 2
        initial = op.combine(nr, None, nw)
        largest_relative = max(
            largest_relative,
            check_combine(initial, r, None, weight),
        )
        count += 1
        collapsed = op.collapse(nr, np)
        check_bf16(
            collapsed,
            torch.einsum("mi,mih->mh", previous.double(), r.double()).bfloat16(),
        )
        packed = torch.full((m, 3 * 5120), 123.0, dtype=torch.bfloat16, device="npu")
        mean = op.mean_packed(nr, packed, 1)
        check_bf16(mean, r.double().mean(1).bfloat16())
        assert torch.all(packed[:, :5120] == 123)
        assert torch.all(packed[:, 10240:] == 123)
        count += 2
        op.post(nx, nr, post, comb, out=updated)
        largest_relative = max(
            largest_relative,
            check_bf16(
                updated,
                post_reference(x, r, oracle.post, oracle.comb),
                fp32=post_reference(x, r, post.cpu(), comb.cpu(), torch.float32),
            ),
        )
        count += 1
        actual_head = op.head(updated, pre, nw)
        # Feed the actual BF16 post handoff into an independent head oracle.
        largest_relative = max(
            largest_relative,
            check_combine(actual_head, updated.cpu(), pre.cpu(), weight),
        )
        count += 1
        op.post_combine(nx, nr, pre, post, comb, nw, updated=updated, out=out)
        largest_relative = max(
            largest_relative,
            check_bf16(
                updated,
                post_reference(x, r, oracle.post, oracle.comb),
                fp32=post_reference(x, r, post.cpu(), comb.cpu(), torch.float32),
            ),
            check_combine(out, updated.cpu(), pre.cpu(), weight),
        )
        count += 2
        op.stats_post(nx, nr, nf, ns, nb, workspace=workspace, out=updated)
        check_coeff(workspace.triplet, oracle)
        check_coeff(workspace.triplet, oracle32)
        largest_relative = max(
            largest_relative,
            check_bf16(
                updated,
                post_reference(x, r, oracle.post, oracle.comb),
                fp32=post_reference(x, r, post.cpu(), comb.cpu(), torch.float32),
            ),
        )
        count += 2
        check_guards(
            (
                projection_storage,
                square_storage,
                pre_storage,
                post_storage,
                comb_storage,
                out_storage,
                updated_storage,
            )
        )
        if m:
            torch.testing.assert_close(
                projection.cpu().double().sum(0)[:m, :24],
                r.double().flatten(1) @ fn.double().t(),
                rtol=2e-4,
                atol=5e-5,
            )
            torch.testing.assert_close(
                square_sum.cpu().double().sum(0)[:m],
                r.double().flatten(1).square().sum(-1),
                rtol=2e-6,
                atol=1e-5,
            )
            assert torch.count_nonzero(projection[:, :computed_rows, 24:]) == 0
            assert torch.count_nonzero(projection[:, m:computed_rows]) == 0
            assert torch.count_nonzero(square_sum[:, m:computed_rows]) == 0
            assert torch.isnan(projection[:, computed_rows:]).all()
            assert torch.isnan(square_sum[:, computed_rows:]).all()
            # Check each K split independently; a wrong mapping must not
            # hide behind a matching total or duplicate/missing panels.
            rp = r.double().reshape(m, 10, 2048)
            fp = fn.double().reshape(24, 10, 2048)
            for split in range(splits):
                rx = rp[:, split::splits, :].flatten(1)
                fx = fp[:, split::splits, :].flatten(1)
                torch.testing.assert_close(
                    projection[split, :m, :24].cpu().double(),
                    rx @ fx.t(),
                    rtol=2e-4,
                    atol=5e-5,
                )
                torch.testing.assert_close(
                    square_sum[split, :m].cpu().double(),
                    rx.square().sum(-1),
                    rtol=2e-6,
                    atol=1e-5,
                )
            count += 1
    # Directed zero/tiny/large residuals and saturated sigmoid/softmax.
    for m, amplitude, saturated in (
        (6, 0.0, False),
        (6, 1e-8, False),
        (6, 1e4, False),
        (6, 1.0, True),
        (129, 0.0, False),
        (129, 1e-8, False),
        (129, 1.0, True),
    ):
        r, fn, scale, base, previous, weight, x = data(m, generator, amplitude)
        if saturated:
            base[:8] = torch.tensor(
                [-1000.0, 1000.0, -80.0, 80.0, -1000.0, 1000.0, -80.0, 80.0]
            )
            base[8:] = torch.tensor([1000.0, -1000.0, -1000.0, -1000.0] * 4)
        nr, nf, ns, nb, np, nw, nx = [
            v.npu() for v in (r, fn, scale, base, previous, weight, x)
        ]
        result = op.pre(nr, nf, ns, nb, nw, np)
        oracle = stats_reference(r, fn, scale, base)
        oracle32 = stats_reference(r, fn, scale, base, torch.float32)
        check_coeff(result.triplet, oracle)
        check_combine(result.input, r, previous, weight)
        updated = op.post(nx, nr, result.triplet.post, result.triplet.comb)
        check_bf16(
            updated,
            post_reference(x, r, oracle.post, oracle.comb),
            max(1.0, amplitude),
            fp32=post_reference(
                x,
                r,
                result.triplet.post.cpu(),
                result.triplet.comb.cpu(),
                torch.float32,
            ),
        )
        fused, triplet = op.stats_post(nx, nr, nf, ns, nb)
        check_coeff(triplet, oracle)
        check_coeff(triplet, oracle32)
        check_bf16(
            fused,
            post_reference(x, r, oracle.post, oracle.comb),
            max(1.0, amplitude),
            fp32=post_reference(
                x, r, triplet.post.cpu(), triplet.comb.cpu(), torch.float32
            ),
        )
        count += 1
    # V4.1 lag: current pre must not replace the preceding connection's pre.
    r, fn, scale, base, previous, weight, x = data(1, generator)
    previous.zero_()
    previous[:, 0] = 1
    result = op.pre(
        r.npu(),
        torch.zeros_like(fn).npu(),
        torch.zeros_like(scale).npu(),
        torch.zeros_like(base).npu(),
        weight.npu(),
        previous.npu(),
    )
    check_bf16(result.input, combine_reference(r, previous, weight))
    assert not torch.equal(
        result.input.cpu(),
        combine_reference(r, torch.full_like(previous, 0.500001), weight),
    )
    count += 1
    # The small-M selection is a performance policy, not a kernel batch
    # limit. Exercise its row-tile loop beyond BM and poison all outputs.
    for vm in (9, 129, 257):
        r, fn, *_ = data(vm, generator)
        vr, vf = r.npu(), fn.npu()
        padded = op.project_geometry(vm)[1]
        vp, vp_storage = guarded((10, padded, 32), torch.float32)
        vs, vs_storage = guarded((10, padded), torch.float32)
        op.vector_project_kernel(
            Residual=vr.flatten(1), Fn=vf, Projection=vp, SquareSum=vs
        )
        torch.testing.assert_close(
            vp.cpu().double().sum(0)[:vm, :24],
            r.double().flatten(1) @ fn.double().t(),
            rtol=2e-4,
            atol=5e-5,
        )
        torch.testing.assert_close(
            vs.cpu().double().sum(0)[:vm],
            r.double().flatten(1).square().sum(-1),
            rtol=2e-6,
            atol=1e-5,
        )
        assert torch.count_nonzero(vp[:, :, 24:]) == 0
        assert torch.count_nonzero(vp[:, vm:]) == 0
        assert torch.count_nonzero(vs[:, vm:]) == 0
        check_guards((vp_storage, vs_storage))
        count += 1
    # Nondefault stream; Graph updates fn, scale/base, previous_pre, x/R.
    cpu = data(6, generator)
    nr, nf, ns, nb, np, nw, nx = inputs = [value.npu() for value in cpu]
    workspace = op.workspace(nr)
    output = torch.empty((6, 5120), dtype=torch.bfloat16, device="npu")
    updated = torch.empty_like(nr)
    head = torch.empty_like(output)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())

    def chain():
        result = op.pre(nr, nf, ns, nb, nw, np, workspace=workspace, out=output)
        op.post_combine(
            nx,
            nr,
            result.triplet.pre,
            result.triplet.post,
            result.triplet.comb,
            nw,
            updated=updated,
            out=head,
        )

    with torch.npu.stream(stream):
        for _ in range(3):
            chain()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        chain()
    addresses = [
        v.data_ptr()
        for v in (
            *inputs,
            workspace.projection,
            workspace.square_sum,
            *workspace.triplet,
            output,
            updated,
            head,
        )
    ]
    for _ in range(3):
        cpu = data(6, generator)
        with torch.npu.stream(stream):
            for target, source in zip(inputs, cpu):
                target.copy_(source)
        graph.replay()
        stream.synchronize()
        r, fn, scale, base, previous, weight, x = cpu
        oracle = stats_reference(r, fn, scale, base)
        oracle32 = stats_reference(r, fn, scale, base, torch.float32)
        check_coeff(workspace.triplet, oracle)
        check_combine(output, r, previous, weight)
        check_bf16(
            updated,
            post_reference(x, r, oracle.post, oracle.comb),
            fp32=post_reference(
                x,
                r,
                workspace.triplet.post.cpu(),
                workspace.triplet.comb.cpu(),
                torch.float32,
            ),
        )
        check_combine(head, updated.cpu(), workspace.triplet.pre.cpu(), weight)
        assert addresses == [
            v.data_ptr()
            for v in (
                *inputs,
                workspace.projection,
                workspace.square_sum,
                *workspace.triplet,
                output,
                updated,
                head,
            )
        ]
        count += 1
    del graph

    # Both vector/Cube policies, a partial tile and ring reuse.
    # Update every input after capture; addresses and the internal ring
    # remain stable across replay on a nondefault stream.
    for gm in (1, 6, 8, 9, 33, 257, 641, 2561):
        values = data(gm, generator)
        gr, gf, gs, gb = [v.npu() for v in values[:4]]
        gx = values[-1].npu()
        gu = torch.empty_like(gr)
        gw = op.workspace(gr)
        tensors = (
            gr,
            gf,
            gs,
            gb,
            gx,
            gu,
            gw.projection,
            gw.square_sum,
            *gw.triplet,
        )
        pointers = [v.data_ptr() for v in tensors]
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            op.stats_post(gx, gr, gf, gs, gb, workspace=gw, out=gu)
        stream.synchronize()
        mixed_graph = torch.npu.NPUGraph()
        with torch.npu.graph(mixed_graph, stream=stream):
            op.stats_post(gx, gr, gf, gs, gb, workspace=gw, out=gu)
        for amplitude in (0.0, 1e-8, 1.0, 1e4):
            values = data(gm, generator, amplitude)
            if amplitude == 1.0:
                values[3][:8] = torch.tensor([-1000.0, 1000.0] * 4)
                values[3][8:] = torch.tensor([1000.0, -1000.0, -1000.0, -1000.0] * 4)
            with torch.npu.stream(stream):
                for target, source in zip(
                    (gr, gf, gs, gb, gx), (*values[:4], values[-1])
                ):
                    target.copy_(source)
            mixed_graph.replay()
            stream.synchronize()
            check_coeff(gw.triplet, stats_reference(*values[:4]))
            check_coeff(gw.triplet, stats_reference(*values[:4], torch.float32))
            oracle = stats_reference(*values[:4])
            check_bf16(
                gu,
                post_reference(values[-1], values[0], oracle.post, oracle.comb),
                max(1.0, amplitude),
                fp32=post_reference(
                    values[-1],
                    values[0],
                    gw.triplet.post.cpu(),
                    gw.triplet.comb.cpu(),
                    torch.float32,
                ),
            )
            assert pointers == [v.data_ptr() for v in tensors]
            count += 1
        del mixed_graph

    # Real prefill sizes: full-output finiteness and an independent sample
    # spanning every group, repeated row tile and the end of the request.
    for lm in (8192, 32768):
        print(f"mHC large prefill M={lm}", flush=True)
        r = torch.randn((lm, 4, 5120), generator=generator).bfloat16()
        _, fn, scale, base, *_ = data(1, generator)
        nr_large, nf_large, ns_large, nb_large = [v.npu() for v in (r, fn, scale, base)]
        large_workspace = op.workspace(nr_large)
        actual = op.stats(
            nr_large, nf_large, ns_large, nb_large, workspace=large_workspace
        )
        indices = torch.linspace(0, lm - 1, 129).long()
        sampled = Triplet(*(v.cpu()[indices] for v in actual))
        check_coeff(sampled, stats_reference(r[indices], fn, scale, base))
        check_coeff(
            sampled, stats_reference(r[indices], fn, scale, base, torch.float32)
        )
        assert all(torch.isfinite(v).all() for v in actual)
        torch.testing.assert_close(
            large_workspace.projection.cpu().double().sum(0)[indices, :24],
            r[indices].double().flatten(1) @ fn.double().t(),
            rtol=2e-4,
            atol=5e-5,
        )
        torch.testing.assert_close(
            large_workspace.square_sum.cpu().double().sum(0)[indices],
            r[indices].double().flatten(1).square().sum(-1),
            rtol=2e-6,
            atol=1e-5,
        )
        count += 1
        del nr_large, nf_large, ns_large, nb_large, large_workspace, actual, r

    if args.model:
        from safetensors import safe_open

        mapping = json.loads((args.model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer in (0, 20, 39):
            for connection in ("attn", "ffn"):
                prefix = f"layers.{layer}.hc_{connection}_"
                tensors = []
                for suffix in ("fn", "scale", "base"):
                    name = prefix + suffix
                    with safe_open(
                        args.model / mapping[name], framework="pt", device="cpu"
                    ) as handle:
                        tensors.append(handle.get_tensor(name).float())
                fn, scale, base = tensors
                for cm in (1, 6, 8, 33, 257):
                    r = data(cm, generator)[0]
                    result = op.stats(*(v.npu() for v in (r, fn, scale, base)))
                    check_coeff(result, stats_reference(r, fn, scale, base))
                    check_coeff(
                        result, stats_reference(r, fn, scale, base, torch.float32)
                    )
                    count += 1

    # Explicitly reject forbidden output/workspace aliases and dtype errors.
    def rejects(call):
        try:
            call()
        except (ValueError, RuntimeError):
            return
        raise AssertionError("invalid alias/dtype was accepted")

    rejects(
        lambda: op.post(nx, nr, workspace.triplet.post, workspace.triplet.comb, out=nr)
    )
    nr1 = nr[:1].contiguous()
    rejects(lambda: op.combine(nr1, np[:1].contiguous(), nw, out=nr1[:, 0]))
    bad = Workspace(
        workspace.projection,
        Triplet(
            nb[:4].view(1, 4),
            workspace.triplet.post[:1],
            workspace.triplet.comb[:1],
        ),
        workspace.square_sum,
    )
    rejects(lambda: op.stats(nr1, nf, ns, nb, workspace=bad))
    rejects(lambda: op.stats(nr, nf.bfloat16(), ns, nb))
    rejects(lambda: op.head(nr, None, nw))
    rejects(
        lambda: op.post_combine(
            nx,
            nr,
            workspace.triplet.pre,
            workspace.triplet.post,
            workspace.triplet.comb,
            nw,
            updated=nr,
        )
    )
    rejects(
        lambda: op.post_combine(
            nx,
            nr,
            workspace.triplet.pre,
            workspace.triplet.post,
            workspace.triplet.comb,
            nw,
            out=nx,
        )
    )
    rejects(lambda: op.stats_post(nx, nr, nf, ns, nb, out=nr))
    rejects(lambda: op.stats_post(nx.float(), nr, nf, ns, nb))
    count += 9
    count += state_graph()
    print(f"PASS {count} mHC cases; largest BF16 relative L2={largest_relative:.3g}")


def state_graph():
    """Direct collapse and packed mean: changed inputs, canaries and all columns."""
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    op = MHC()
    rng = torch.Generator().manual_seed(4141709)
    count = 0
    for m in (1, 6, 33, 128, 257, 2048, 8192, 32768):
        print(f"mHC state Graph M={m}", flush=True)
        cpu = data(m, rng)
        residual, pre = cpu[0].npu(), cpu[4].npu()
        indices = torch.arange(m) if m <= 2048 else torch.linspace(0, m - 1, 129).long()
        collapse_storage = torch.full(
            (m * 5120 + 64,), 123, dtype=torch.bfloat16, device="npu"
        )
        packed_storage = torch.full(
            (m * 15360 + 64,), 123, dtype=torch.bfloat16, device="npu"
        )
        collapsed = collapse_storage[32:-32].view(m, 5120)
        packed = packed_storage[32:-32].view(m, 15360)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        for column in (0, 1, 2):

            def invoke():
                op.collapse(residual, pre, out=collapsed)
                op.mean_packed(residual, packed, column)

            with torch.npu.stream(stream):
                invoke()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                invoke()
            pointers = [t.data_ptr() for t in (residual, pre, collapsed, packed)]
            for amplitude in (0.0, 1e-8, 1e4):
                if m <= 2048:
                    source = data(m, rng, amplitude)
                    r, p = source[0], source[4]
                else:
                    # The added large cases reuse the seeded residual while
                    # changing its amplitude and the coefficients each replay.
                    r = cpu[0] * amplitude
                    p = torch.rand((m, 4), generator=rng)
                with torch.npu.stream(stream):
                    residual.copy_(r)
                    pre.copy_(p)
                    packed.fill_(123)
                    graph.replay()
                stream.synchronize()
                # Keep full independent references for the original shapes;
                # sample129 rows in the added large cases, with full-output
                # zero/finiteness coverage in addition to pointer/canary checks.
                rr, pp = r[indices], p[indices]
                reference = torch.einsum(
                    "mi,mih->mh", pp.double(), rr.double()
                ).bfloat16()
                serial = rr[:, 0].float() * pp[:, :1]
                for i in range(1, 4):
                    serial = serial + rr[:, i].float() * pp[:, i : i + 1]
                check_bf16(
                    collapsed.cpu()[indices],
                    reference,
                    max(1.0, amplitude),
                    fp32=serial.bfloat16(),
                )
                live = packed[:, column * 5120 : (column + 1) * 5120]
                check_bf16(
                    live.cpu()[indices],
                    rr.double().mean(1).bfloat16(),
                    max(1.0, amplitude),
                )
                assert torch.isfinite(collapsed).all()
                assert torch.isfinite(live).all()
                if amplitude == 0:
                    assert torch.all(collapsed == 0)
                    assert torch.all(live == 0)
                for other in range(3):
                    if other != column:
                        assert torch.all(
                            packed[:, other * 5120 : (other + 1) * 5120] == 123
                        )
                assert torch.all(collapse_storage[:32] == 123)
                assert torch.all(collapse_storage[-32:] == 123)
                assert torch.all(packed_storage[:32] == 123)
                assert torch.all(packed_storage[-32:] == 123)
                assert pointers == [
                    t.data_ptr() for t in (residual, pre, collapsed, packed)
                ]
                count += 2
            del graph
    print(f"PASS {count} direct mHC state Graph cases", flush=True)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, help="also check real mHC weights at layers 0/20/39"
    )
    run(parser.parse_args())
