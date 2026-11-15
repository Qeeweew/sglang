"""Global TopK -> fused local buckets -> actual resident MXFP4 MoE, changed Graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical):
        raise ValueError("explicit physical binding required before Torch import")
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.quant_grouped_gemm import (
        MXFP4LocalExperts,
        NativeGroupedGemm,
    )
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router
    from dsv41.kernels.references.mxfp4_grouped import fixture, weight
    from dsv41.kernels.references.quant_packing import pack_experts
    from dsv41.kernels.references.router import reference
    from sglang.srt.hardware_backend.npu.dsv41.moe import ResidentMXFP4Method

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(6182)
    hidden, intermediate = 128, 128
    router = Router()
    records = []
    stream = torch.npu.Stream()
    for experts, local_experts, topk, padded in ((384, 48, 6, 48), (128, 16, 3, 40)):
        host_weights = []
        packed_weights = []
        for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
            codes, scales = fixture(local_experts, n, k)
            packed, packed_scales = pack_experts(
                zip(codes, scales), n=n, k=k, count=local_experts, compact_scales=True
            )
            packed_weights.append(
                (torch.from_numpy(packed).npu(), torch.from_numpy(packed_scales).npu())
            )
            host_weights.append([weight(c, s, k) for c, s in zip(codes, scales)])
        method = ResidentMXFP4Method()
        method.ep, method.first_expert, method.e = (
            8,
            experts - local_experts,
            local_experts,
        )
        method.h, method.i = hidden, intermediate
        method.local_experts = MXFP4LocalExperts(
            NativeGroupedGemm(
                hidden, 2 * intermediate, experts=local_experts, block_m=16
            ),
            NativeGroupedGemm(intermediate, hidden, experts=local_experts, block_m=16),
            *packed_weights,
        )
        for rows in (6, padded, padded * 8, 1024):
            logits = torch.randn(rows, experts, device="npu")
            bias = torch.zeros(experts, device="npu")
            source = torch.randn(rows, hidden, device="npu").bfloat16() * 0.02
            stride = rows // 8 if rows % 8 == 0 else rows
            ranks = rows // stride
            counts = torch.full((ranks,), stride, dtype=torch.int64, device="npu")

            def invoke():
                top = router.route(
                    logits,
                    bias,
                    dp_counts=counts,
                    dp_stride=stride,
                    first=method.first_expert,
                    experts=method.e,
                )
                return method.routed(source, top), top

            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                invoke()
                invoke()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    output, top = invoke()
                addresses = [tensor.data_ptr() for tensor in (output, *top)]
                for mode in ("uniform", "owned", "holes", "empty"):
                    host_logits = torch.randn(rows, experts)
                    host_bias = torch.zeros(experts)
                    if mode == "owned":
                        host_bias.fill_(-100)
                        # Keep one nonlocal selected slot to catch local renormalization.
                        host_bias[
                            method.first_expert : method.first_expert + topk - 1
                        ] = 100
                        host_bias[0] = 100
                    host_counts = torch.full((ranks,), stride, dtype=torch.int64)
                    if mode == "holes":
                        host_counts[::2] = 0
                        host_counts[-1] = stride - 1
                    elif mode == "empty":
                        host_counts.zero_()
                    host_source = (torch.randn(rows, hidden) * 0.02).bfloat16()
                    live = torch.arange(rows) % stride < host_counts.repeat_interleave(
                        stride
                    )
                    host_source[~live] = float("nan")
                    logits.copy_(host_logits)
                    bias.copy_(host_bias)
                    counts.copy_(host_counts)
                    source.copy_(host_source)
                    graph.replay()
                    stream.synchronize()
                    ids, coefficients = reference(host_logits, host_bias)
                    ids[~live] = -1
                    coefficients[~live] = 0
                    local_ids = torch.where(
                        (ids >= method.first_expert)
                        & (ids < method.first_expert + local_experts),
                        ids - method.first_expert,
                        -1,
                    )
                    torch.testing.assert_close(top.ids.cpu(), local_ids, rtol=0, atol=0)
                    torch.testing.assert_close(
                        top[1].cpu(), coefficients, rtol=3e-6, atol=2e-7
                    )
                    slot_values = torch.zeros(rows, topk, hidden)
                    x = host_source.float().half().numpy()
                    for expert in range(local_experts):
                        tokens, slots = torch.where(ids == method.first_expert + expert)
                        if not len(tokens):
                            continue
                        gu = (
                            x[tokens].astype(np.float64)
                            @ host_weights[0][expert].astype(np.float64).T
                        ).astype(np.float16)
                        gate, up = torch.from_numpy(gu).float().chunk(2, -1)
                        activation = (
                            (
                                torch.nn.functional.silu(gate.clamp(max=10))
                                * up.clamp(-10, 10)
                            )
                            .half()
                            .numpy()
                        )
                        down = (
                            activation.astype(np.float64)
                            @ host_weights[1][expert].astype(np.float64).T
                        ).astype(np.float16)
                        slot_values[tokens, slots] = torch.from_numpy(down).float()
                    expected = torch.zeros(rows, hidden)
                    for slot in range(topk):
                        expected += slot_values[:, slot] * coefficients[:, slot, None]
                    torch.testing.assert_close(
                        output.cpu(), expected.bfloat16(), rtol=0.025, atol=0.01
                    )
                    assert torch.isfinite(output).all()
                    assert addresses == [tensor.data_ptr() for tensor in (output, *top)]
                    records.append(dict(experts=experts, rows=rows, mode=mode))
                del graph
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(dict(status="passed", cases=records), output, indent=2)
        output.write("\n")
    print(
        f"PASS {len(records)} actual resident MoE changed-route Graph cases", flush=True
    )


if __name__ == "__main__":
    main()
