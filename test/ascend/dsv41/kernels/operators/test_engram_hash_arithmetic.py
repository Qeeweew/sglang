"""Exact full-width hash arithmetic and changed-input replay for both cache schedules."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram_ops import (
    EngramOps,
    HashParameters,
)
from dsv41.kernels.references.engram_hash import oracle_hash


def run():
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("bind one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(910141)
    op = EngramOps()
    cases = 0
    for rows, layers in ((1, 2), (6, 2), (257, 2), (257, 3), (2048, 1)):
        history_tokens, heads, vocab = 5, 9, 63
        prime_edges = torch.tensor(
            [1, 2, 3, 97, 2**31 - 1, 2**32 - 1, 2**32 + 15, 2**62 + 57, 2**63 - 1],
            dtype=torch.int64,
        )
        p = dict(
            token_map=torch.arange(vocab, dtype=torch.int32),
            multipliers=torch.randint(
                1, (2**63 - 1) // (vocab - 1), (layers, history_tokens + 1)
            ),
            primes=prime_edges.repeat(layers, history_tokens, 1),
            offsets=torch.zeros((layers, history_tokens * heads), dtype=torch.int64),
            pad_id=vocab - 1,
            image_id=-1,
        )
        p["multipliers"][0, 0] = (2**63 - 1) // (vocab - 1)
        parameters = HashParameters(**p)
        cpu = [
            torch.randint(0, vocab, (rows,), dtype=torch.int32),
            torch.arange(history_tokens, rows + history_tokens),
            torch.tensor([0, rows], dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.randint(0, vocab, (1, history_tokens), dtype=torch.int32),
            torch.ones(rows, dtype=torch.int32),
        ]
        cpu[0][0] = vocab - 1
        dev = [x.npu() for x in cpu]
        out = torch.empty(
            (layers, rows, history_tokens * heads), dtype=torch.int64, device="npu"
        )
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            op.hash(*dev, parameters, out=out)
        stream.synchronize()
        torch.testing.assert_close(out.cpu(), oracle_hash(*cpu, p), rtol=0, atol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            op.hash(*dev, parameters, out=out)
        pointers = [v.data_ptr() for v in (*dev, out)]
        for generation in range(2):
            cpu[0].random_(0, vocab)
            cpu[4].random_(0, vocab)
            cpu[0][0] = vocab - 1
            cpu[5][generation % rows] = 0
            for dst, src in zip(dev, cpu):
                dst.copy_(src)
            out.fill_(-1)
            torch.npu.synchronize()
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(out.cpu(), oracle_hash(*cpu, p), rtol=0, atol=0)
            assert pointers == [v.data_ptr() for v in (*dev, out)]
            cases += 1
    print({"status": "passed", "full_width_graph_cases": cases}, flush=True)


if __name__ == "__main__":
    run()
