"""Top-k emits grouped GEMM metadata: independent routing and replay checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run(args):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.health import (
        health_snapshot,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    health = health_snapshot(args.physical, args.output)
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router
    from dsv41.kernels.references.router import compare, reference

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(434)
    router = Router(library_dir=args.library_dir)
    records = []

    def allocate(x):
        m, e = x.shape
        k = 6 if e == 384 else 3
        kwargs, guards = {}, []
        for name, shape, dtype in (
            ("ids_out", (m, k), torch.int32),
            ("weights_out", (m, k), torch.float32),
            ("counts_out", (e,), torch.int32),
            ("offsets_out", (e + 1,), torch.int32),
            ("order_out", (m * k,), torch.int32),
            ("metadata_out", (4 * (e + 1),), torch.int32),
        ):
            n = int(np.prod(shape))
            storage = torch.full((n + 64,), -1234, dtype=dtype, device="npu:0")
            kwargs[name] = storage[32:-32].view(shape)
            guards.extend((storage[:32], storage[-32:]))
        kwargs["workspace"] = router.workspace(x)
        return kwargs, guards

    def check(result, x, bias, live, guards, alternate=None, mask=None):
        m, e = x.shape
        if alternate is None:
            expected = reference(
                x.cpu(), None if bias is None else bias.cpu(), live=live
            )
        else:
            values = [
                reference(
                    x[r : r + 1].cpu(), (alternate if int(mask[r]) else bias).cpu()
                )
                for r in range(m)
            ]
            expected = tuple(torch.cat([v[i] for v in values]) for i in range(2))
            expected[0][live:] = -1
            expected[1][live:] = 0
        compare(result.ids, result.weights, expected)
        ids = result.ids.cpu().flatten().numpy()
        counts = np.bincount(ids[ids >= 0], minlength=e)
        offsets = np.r_[0, np.cumsum(counts)]
        np.testing.assert_array_equal(result.counts.cpu(), counts)
        np.testing.assert_array_equal(result.offsets.cpu(), offsets)
        order = np.argsort(ids, kind="stable")
        order = order[ids[order] >= 0]
        np.testing.assert_array_equal(result.order.cpu()[: len(order)], order)
        active = np.flatnonzero(counts)
        metadata = result.metadata.cpu().numpy().reshape(-1, 4)
        np.testing.assert_array_equal(metadata[0], [len(active), offsets[-1], 0, 1])
        np.testing.assert_array_equal(
            metadata[1 : len(active) + 1],
            np.stack(
                (active, offsets[active], counts[active], np.zeros_like(active)), axis=1
            ),
        )
        assert all(bool((guard == -1234).all()) for guard in guards)
        records.append({"rows": m, "experts": e, "live": live, "active": len(active)})

    for e in (128, 384):
        for m in (0, 1, 6, 8, 9, 33, 129, 8192, 32768):
            x = torch.randn((m, e), device="npu:0") * 3
            bias = torch.randn(e, device="npu:0") * 0.2
            kwargs, guards = allocate(x)
            for live in sorted({0, m, max(m - 1, 0)}):
                valid = torch.tensor([live], dtype=torch.int32, device="npu:0")
                result = router.route(x, bias, valid_rows=valid, **kwargs)
                check(result, x, bias, live, guards)
        for m in (6, 33):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                x = torch.randn((m, e), device="npu:0")
                bias = torch.randn(e, device="npu:0") * 0.2
                alt = torch.randn_like(bias)
                mask = torch.randint(0, 2, (m,), dtype=torch.int32, device="npu:0")
                live = torch.tensor([m], dtype=torch.int32, device="npu:0")
                kwargs, guards = allocate(x)

                def invoke():
                    return router.route(
                        x,
                        bias,
                        alternate_bias=alt,
                        image_mask=mask,
                        valid_rows=live,
                        **kwargs,
                    )

                invoke()
                stream.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    result = invoke()
                addresses = [v.data_ptr() for v in result]
                for count in (m, 6, 1, 0, m - 1):
                    x.normal_()
                    bias.normal_(0, 0.2)
                    alt.normal_(0, 0.2)
                    mask.copy_(1 - mask)
                    live.fill_(count)
                    graph.replay()
                    stream.synchronize()
                    check(result, x, bias, count, guards, alt, mask.cpu())
                    assert [v.data_ptr() for v in result] == addresses
    record = {"status": "passed", "health": health, "cases": records}
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"PASS: {len(records)} fused router metadata cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", required=True, type=int)
    parser.add_argument("--library-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
