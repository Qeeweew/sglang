"""Runtime expert/top-k geometry, stable buckets and changed Graph metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(61423)
    router = Router(library_dir=args.library_dir)
    cases = 0
    for experts, topk in (
        (64, 2),
        (128, 64),
        (192, 5),
        (320, 7),
        (384, 64),
        (512, 9),
        (768, 5),
        (1536, 11),
    ):
        for rows in (0, 1, 6, 33, 257):
            x = torch.randn(rows, experts).npu()
            bias = (torch.randn(experts) * 0.1).npu()
            # K=64 exercises the full 256B output with separated ranks;
            # random near-ties can exchange FP32 order under the API contract.
            rank_gap = torch.arange(experts, device="npu", dtype=torch.float32) * 4
            if topk == 64:
                bias.add_(rank_gap)
            live = torch.tensor([rows], dtype=torch.int32, device="npu")
            workspace = router.workspace(x, topk=topk)
            result = router.route(
                x, bias, topk=topk, valid_rows=live, workspace=workspace
            )

            def check(valid):
                score = torch.nn.functional.softplus(x.cpu().double()).sqrt()
                expected_ids = (
                    (score + bias.cpu().double()).topk(topk, dim=1).indices.int()
                )
                expected_weights = score.gather(1, expected_ids.long())
                expected_weights = (
                    expected_weights / expected_weights.sum(1, keepdim=True) * 1.5
                )
                expected_ids[valid:] = -1
                expected_weights[valid:] = 0
                torch.testing.assert_close(
                    result.ids.cpu(), expected_ids, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    result.weights.cpu(), expected_weights.float(), rtol=2e-5, atol=2e-6
                )
                flat = expected_ids.flatten()
                counts = torch.bincount(flat[flat >= 0].long(), minlength=experts).int()
                offsets = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32),
                        counts.cumsum(0, dtype=torch.int32),
                    ]
                )
                order = flat.argsort(stable=True)
                order = order[flat[order] >= 0].int()
                torch.testing.assert_close(result.counts.cpu(), counts)
                torch.testing.assert_close(result.offsets.cpu(), offsets)
                torch.testing.assert_close(result.order.cpu()[: order.numel()], order)
                active = torch.nonzero(counts, as_tuple=True)[0].int()
                table = result.metadata.cpu().view(-1, 4)
                torch.testing.assert_close(
                    table[0],
                    torch.tensor(
                        [active.numel(), valid * topk, 0, 1], dtype=torch.int32
                    ),
                )
                if active.numel():
                    expected = torch.stack(
                        [
                            active,
                            offsets[active],
                            counts[active],
                            torch.zeros_like(active),
                        ],
                        dim=1,
                    )
                    torch.testing.assert_close(table[1 : active.numel() + 1], expected)

            check(rows)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = router.route(
                    x,
                    bias,
                    topk=topk,
                    valid_rows=live,
                    workspace=workspace,
                    ids_out=result.ids,
                    weights_out=result.weights,
                    counts_out=result.counts,
                    offsets_out=result.offsets,
                    order_out=result.order,
                    metadata_out=result.metadata,
                )
            addresses = [v.data_ptr() for v in result]
            for valid in sorted({0, rows // 2, rows}):
                x.normal_()
                bias.normal_(0, 0.1)
                if topk == 64:
                    bias.add_(rank_gap)
                live.fill_(valid)
                graph.replay()
                check(valid)
                assert addresses == [v.data_ptr() for v in result]
                cases += 1
            print(f"PASS router rows={rows} experts={experts} topk={topk}", flush=True)
    print(f"PASS {cases} runtime router replay cases", flush=True)


if __name__ == "__main__":
    main()
