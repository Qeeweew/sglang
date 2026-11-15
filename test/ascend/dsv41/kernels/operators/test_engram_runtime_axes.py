"""Non-model hash/history axes, independent integer oracle and changed Graph inputs."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram_ops import (
    EngramOps,
    HashParameters,
)
from dsv41.kernels.references.engram_hash import (
    oracle_commit,
    oracle_hash,
)


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(94113)
    op = EngramOps()
    cases = 0
    for layers, history_tokens, heads in (
        (1, 1, 3),
        (3, 2, 5),
        (2, 5, 7),
        (4, 7, 1),
        (2, 3, 9),
        (1, 2, 16),
    ):
        p = dict(
            token_map=torch.arange(63, dtype=torch.int32) % 29,
            multipliers=torch.randint(
                1, 997, (layers, history_tokens + 1), dtype=torch.int64
            ),
            primes=torch.randint(
                1009, 4093, (layers, history_tokens, heads), dtype=torch.int64
            ),
            offsets=torch.arange(layers * history_tokens * heads).view(layers, -1)
            * 4096,
            pad_id=29,
            image_id=62,
        )
        parameters = HashParameters(**p)
        cpu = [
            torch.randint(0, 63, (17,), dtype=torch.int32),
            torch.arange(7, 24),
            torch.tensor([0, 0, 6, 17], dtype=torch.int32),
            torch.tensor([-1, 2, 0], dtype=torch.int32),
            torch.randint(0, 63, (4, history_tokens), dtype=torch.int32),
            torch.ones(17, dtype=torch.int32),
        ]
        dev = [v.npu() for v in cpu]
        accepted = torch.tensor([0, 3, 11], dtype=torch.int32)
        accepted_npu = accepted.npu()
        out = torch.empty(
            (layers, 17, history_tokens * heads), dtype=torch.int64, device="npu"
        )
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            op.hash(*dev, parameters, out=out)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            op.hash(*dev, parameters, out=out)
            op.commit(dev[0], dev[2], dev[3], accepted_npu, dev[4])
        pointers = [v.data_ptr() for v in (*dev, out)]
        for generation in range(3):
            cpu[0].random_(0, 63)
            cpu[4].random_(0, 63)
            cpu[5].fill_(1)
            cpu[5][generation] = 0
            cpu[0][3] = 1_000_003
            for dst, src in zip(dev, cpu):
                dst.copy_(src)
            out.fill_(-987654321)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu(), oracle_hash(*cpu, p), rtol=0, atol=0)
            torch.testing.assert_close(
                dev[4].cpu(),
                oracle_commit(cpu[0], cpu[2], cpu[3], accepted, cpu[4]),
                rtol=0,
                atol=0,
            )
            assert pointers == [v.data_ptr() for v in (*dev, out)]
            cases += 1
    print({"status": "passed", "graph_cases": cases}, flush=True)


if __name__ == "__main__":
    run()
