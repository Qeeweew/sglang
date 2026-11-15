"""Independent BF16/FP16 route conversion at arbitrary vector-width tails."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.moe_routes import MoERoutes
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import RoutingPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical):
        raise ValueError("explicit physical NPU binding required")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(6231)
    op = MoERoutes()
    checks = 0
    for rows in (1, 6, 33):
        for width in (1, 17, 63, 64, 65, 127, 129, 513, 2049):
            for topk in (2, 7):
                slots = rows * topk
                # All slots route to the same expert; duplicates remain separate
                # weighted contributions, with a deterministic expert-major order.
                weights_cpu = torch.rand(rows, topk)
                routing = RoutingPlan(
                    torch.zeros((rows, topk), dtype=torch.int32, device="npu"),
                    weights_cpu.npu(),
                    torch.tensor([slots], dtype=torch.int32, device="npu"),
                    torch.tensor([0, slots], dtype=torch.int32, device="npu"),
                    torch.arange(slots, dtype=torch.int32, device="npu"),
                    torch.tensor(
                        [1, slots, 0, 1, 0, 0, slots, 0],
                        dtype=torch.int32,
                        device="npu",
                    ),
                )
                x_cpu = (torch.randn(rows, width) / 8).bfloat16()
                if (rows, width, topk) == (6, 65, 2):
                    # BF16 -> FP16 must overflow to Inf, and preserve IEEE
                    # nonfinite values through FP32 combine and changed replay.
                    x_cpu[0, :7] = torch.tensor(
                        [
                            0.0,
                            -0.0,
                            float("inf"),
                            -float("inf"),
                            float("nan"),
                            2**20,
                            -(2**20),
                        ]
                    )
                x = x_cpu.npu()
                values, inverse = op.gather(x, routing)
                storage = torch.full(
                    (rows * width + 64,), -77, dtype=torch.bfloat16, device="npu"
                )
                output = storage[32:-32].view(rows, width)

                def call():
                    op.gather(x, routing, out=values, inverse=inverse)
                    op.combine(values, inverse, routing, out=output)

                call()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    call()
                for generation in range(3):
                    source = x_cpu if generation % 2 == 0 else -x_cpu * 2
                    x.copy_(source)
                    output.fill_(float("nan"))
                    (call if generation == 0 else graph.replay)()
                    torch.npu.synchronize()
                    expected_values = source.repeat_interleave(topk, 0).half()
                    accumulator = torch.zeros(rows, width)
                    for slot in range(topk):
                        accumulator += (
                            expected_values[slot::topk].float()
                            * weights_cpu[:, slot, None]
                        )
                    torch.testing.assert_close(
                        values.cpu(), expected_values, rtol=0, atol=0, equal_nan=True
                    )
                    torch.testing.assert_close(
                        output.cpu(),
                        accumulator.bfloat16(),
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    assert bool((storage[:32] == -77).all()) and bool(
                        (storage[-32:] == -77).all()
                    )
                    checks += 1
                del graph
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="passed", checks=checks)) + "\n")
    print(json.dumps(dict(status="passed", checks=checks)), flush=True)


if __name__ == "__main__":
    main()
