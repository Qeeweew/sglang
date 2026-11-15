"""All exact FP16 scale powers, mixed sign tags and in-place Graph updates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


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
    from dsv41.kernels.references.quant_packing import pack_experts
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import NativeGroupedGemm

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    k, n, experts, capacity = 1280, 64, 3, 127
    codes = np.full((n, k // 2), 0x11, dtype=np.uint8)
    exponents = np.broadcast_to(np.arange(104, 144, dtype=np.uint8), (n, 40)).copy()
    source = np.zeros((capacity, k), dtype=np.float16)
    source[np.arange(120), np.arange(120) % 40 * 32] = 1
    x = torch.from_numpy(source).npu()
    records = []
    for bm in (16, 128, 256):
        op = NativeGroupedGemm(k, n, experts=experts, block_m=bm)

        def pack(exponent):
            return pack_experts(
                [(codes, exponent)] * experts,
                k=k,
                n=n,
                count=experts,
                compact_scales=True,
            )

        ph, sh = pack(exponents)
        packed, scales = (torch.from_numpy(v).npu() for v in (ph, sh))
        entries = [3, 120, 0, 1, 0, 0, 40, 0, 1, 40, 40, 0, 2, 80, 40, 0]
        routing = torch.tensor(entries, dtype=torch.int32, device="npu:0")
        storage = torch.full(
            (capacity * n + 512,), 123, dtype=torch.float16, device="npu:0"
        )
        out = storage[256:-256].view(capacity, n)
        bound = op.bind(x, packed, scales, routing, out=out)
        bound()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            bound()
        for update in range(3):
            exponent = exponents if update % 2 == 0 else exponents[:, ::-1].copy()
            _, updated_scales = pack(exponent)
            scales.copy_(torch.from_numpy(updated_scales))
            sign = -1 if update == 2 else 1
            x.copy_(torch.from_numpy(source * sign))
            expected = np.full((capacity, n), 123, dtype=np.float16)
            values = np.ldexp(np.ones(40), exponent[0].astype(np.int16) - 128)
            expected[:120] = np.tile(values * sign, experts)[:, None]
            for invoke in (bound, graph.replay):
                out.fill_(123)
                invoke()
                torch.npu.synchronize()
                np.testing.assert_array_equal(
                    out.cpu().numpy().view(np.uint16), expected.view(np.uint16)
                )
                assert bool((storage[:256] == 123).all())
                assert bool((storage[-256:] == 123).all())
            records.append(dict(bm=bm, update=update))
        del graph, bound, op
    report = dict(
        status="passed", health=health, records=records, checks=len(records) * 2
    )
    (args.output / "device.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="passed", checks=report["checks"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
