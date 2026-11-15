"""Finite extreme ranks emit unique experts in every routing entry and Graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def run(args):
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(args.physical)
    assert os.environ["ASCEND_DEVICE_ID"] == "0"
    args.output.mkdir(parents=True, exist_ok=False)
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.health import (
        health_snapshot,
    )

    health = health_snapshot(args.physical, args.output)
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router

    torch.set_num_threads(1)
    torch.npu.set_device(0)
    router = Router(library_dir=args.library_dir)
    largest = torch.finfo(torch.float32).max
    records = []

    def check_topk(result, source, valid, experts):
        ids, weights = result[0].cpu(), result[1].cpu()
        assert bool(((ids[:valid] >= 0) & (ids[:valid] < experts)).all())
        assert bool((ids[valid:] == -1).all())
        assert bool((weights[valid:] == 0).all())
        sorted_ids = ids[:valid].sort(dim=1).values
        assert bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()), ids
        score = torch.nn.functional.softplus(source.double()).sqrt()
        selected = score[:valid].gather(1, ids[:valid].long())
        expected = selected / (selected.sum(1, keepdim=True) + 1e-20) * 1.5
        torch.testing.assert_close(
            weights[:valid], expected.float(), rtol=2e-5, atol=2e-6
        )
        assert bool(torch.isfinite(weights).all())
        return ids

    def check_buckets(result, ids, valid, experts, topk):
        flat = ids.flatten()
        counts = torch.bincount(flat[flat >= 0].long(), minlength=experts).int()
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int32), counts.cumsum(0, dtype=torch.int32)]
        )
        torch.testing.assert_close(result.counts.cpu(), counts)
        torch.testing.assert_close(result.offsets.cpu(), offsets)
        order = flat.argsort(stable=True)
        order = order[flat[order] >= 0].int()
        torch.testing.assert_close(result.order.cpu()[: order.numel()], order)
        active = torch.nonzero(counts, as_tuple=True)[0].int()
        table = result.metadata.cpu().view(-1, 4)
        torch.testing.assert_close(
            table[0],
            torch.tensor([active.numel(), valid * topk, 0, 1], dtype=torch.int32),
        )
        if active.numel():
            expected = torch.stack(
                [active, offsets[active], counts[active], torch.zeros_like(active)],
                dim=1,
            )
            torch.testing.assert_close(table[1 : active.numel() + 1], expected)

    for experts, topk in ((64, 2), (192, 5), (384, 6), (768, 11), (1536, 11)):
        for rows in (1, 6, 33):
            x = torch.zeros(rows, experts, device="npu")
            bias = torch.full((experts,), -largest, device="npu")
            alternate = torch.full_like(bias, -largest)
            images = (torch.arange(rows, dtype=torch.int32) % 2).npu()
            live = torch.tensor([rows], dtype=torch.int32, device="npu")
            common = dict(
                topk=topk, valid_rows=live, alternate_bias=alternate, image_mask=images
            )
            workspace = router.workspace(x, topk=topk)
            grouped = router.route(x, bias, workspace=workspace, **common)
            plain = router(x, bias, **common)
            grouped_outputs = dict(
                ids_out=grouped.ids,
                weights_out=grouped.weights,
                counts_out=grouped.counts,
                offsets_out=grouped.offsets,
                order_out=grouped.order,
                metadata_out=grouped.metadata,
            )
            plain_outputs = dict(ids_out=plain[0], weights_out=plain[1])

            def invoke_grouped():
                return router.route(
                    x, bias, workspace=workspace, **common, **grouped_outputs
                )

            def invoke_plain():
                return router(x, bias, **common, **plain_outputs)

            torch.npu.synchronize()
            graphs = [torch.npu.NPUGraph(), torch.npu.NPUGraph()]
            with torch.npu.graph(graphs[0]):
                invoke_grouped()
            with torch.npu.graph(graphs[1]):
                invoke_plain()
            pointers = [v.data_ptr() for v in (*grouped, *plain)]
            for generation, valid in enumerate((rows, rows // 2, 0, rows)):
                source = torch.zeros(rows, experts)
                if generation == 1:
                    source[:, ::2], source[:, 1::2] = largest, -largest
                elif generation == 2:
                    source.fill_(-largest)
                elif generation == 3:
                    source[:] = torch.linspace(-20, 20, experts)
                x.copy_(source)
                # All-minimum correction exposes retirement by -FLT_MAX;
                # updates exercise score extremes and the alternate bias path.
                bias.fill_(-largest if generation != 2 else largest)
                alternate.copy_(bias)
                if generation == 3:
                    alternate[::2] = largest
                live.fill_(valid)
                for api, result, eager, graph in (
                    ("grouped", grouped, invoke_grouped, graphs[0]),
                    ("plain", plain, invoke_plain, graphs[1]),
                ):
                    for mode, invoke in (("eager", eager), ("graph", graph.replay)):
                        invoke()
                        torch.npu.synchronize()
                        ids = check_topk(result, source, valid, experts)
                        if api == "grouped":
                            check_buckets(result, ids, valid, experts, topk)
                        assert pointers == [v.data_ptr() for v in (*grouped, *plain)]
                        records.append(
                            dict(
                                experts=experts,
                                topk=topk,
                                rows=rows,
                                generation=generation,
                                valid=valid,
                                api=api,
                                mode=mode,
                            )
                        )
            del graphs
        print(f"PASS finite extremes E={experts} K={topk}", flush=True)
    report = dict(status="passed", health=health, checks=len(records), records=records)
    (args.output / "device.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="passed", checks=len(records))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
