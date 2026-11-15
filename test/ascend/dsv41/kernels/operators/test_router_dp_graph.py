"""Direct INT32/INT64 DP counts, stable grouped outputs and changed-count Graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def run(args):
    if (
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical)
        or os.environ.get("ASCEND_DEVICE_ID") != "0"
    ):
        raise ValueError("bind the physical device before importing Torch")
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router

    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(7361)
    records = []
    router = Router()
    for rows in (6, 130):
        stride, experts, topk = rows // 2, 384, 6
        for dtype in (torch.int32, torch.int64):
            logits_cpu = torch.randn(rows, experts)
            bias_cpu = torch.randn(experts) * 0.1
            logits, bias = logits_cpu.npu(), bias_cpu.npu()
            counts = torch.tensor([stride, stride], dtype=dtype, device="npu")
            invoke = router.bind(
                logits, bias, plan=True, dp_counts=counts, dp_stride=stride
            )
            result = invoke()
            torch.npu.synchronize()
            addresses = [value.data_ptr() for value in (logits, bias, counts, *result)]
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                invoke()
            for valid_counts in ((stride, stride), (0, stride // 2), (stride // 2, 0)):
                logits_cpu.normal_()
                logits.copy_(logits_cpu)
                counts.copy_(torch.tensor(valid_counts, dtype=dtype))
                graph.replay()
                torch.npu.synchronize()
                score = torch.nn.functional.softplus(logits_cpu.double()).sqrt()
                expected_ids = (score + bias_cpu.double()).argsort(
                    descending=True, stable=True
                )[:, :topk]
                weights = score.gather(1, expected_ids)
                weights = (
                    weights / (weights.sum(1, keepdim=True) + 1e-20) * 1.5
                ).float()
                valid = torch.cat(
                    [torch.arange(stride) < count for count in valid_counts]
                )
                expected_ids[~valid] = -1
                weights[~valid] = 0
                torch.testing.assert_close(
                    result.ids.cpu(), expected_ids.int(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    result.weights.cpu(), weights, rtol=2e-5, atol=2e-6
                )
                flat = expected_ids.flatten()
                expected_counts = torch.bincount(
                    flat[flat >= 0], minlength=experts
                ).int()
                offsets = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32),
                        expected_counts.cumsum(0, dtype=torch.int32),
                    )
                )
                order = flat.argsort(stable=True)
                order = order[flat[order] >= 0].int()
                torch.testing.assert_close(
                    result.counts.cpu(), expected_counts, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    result.offsets.cpu(), offsets, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    result.order.cpu()[: order.numel()], order, rtol=0, atol=0
                )
                active = torch.nonzero(expected_counts, as_tuple=True)[0].int()
                metadata = result.metadata.cpu().view(-1, 4)
                torch.testing.assert_close(
                    metadata[0],
                    torch.tensor(
                        [active.numel(), order.numel(), 0, 1], dtype=torch.int32
                    ),
                    rtol=0,
                    atol=0,
                )
                expected_metadata = torch.stack(
                    (
                        active,
                        offsets[active],
                        expected_counts[active],
                        torch.zeros_like(active),
                    ),
                    1,
                )
                torch.testing.assert_close(
                    metadata[1 : 1 + active.numel()], expected_metadata, rtol=0, atol=0
                )
                assert addresses == [
                    value.data_ptr() for value in (logits, bias, counts, *result)
                ]
            records.append(dict(rows=rows, count_dtype=str(dtype), status="passed"))
            (args.output / "checks.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            print(f"PASS direct DP counts rows={rows} dtype={dtype}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
