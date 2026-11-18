"""Native TP1/2/4 head widths: independent oracle, canaries and live Graph inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
    SparseAttention,
    WindowAttention,
)
from dsv41.kernels.references.attention import (
    attention_data,
    attention_oracle,
)


def stream_isolation(records):
    """Two graphs on different streams may share one attention owner."""
    for mode in ("swa", "c2"):
        op = (
            WindowAttention(heads=16)
            if mode == "swa"
            else SparseAttention(heads=16, ratio=2)
        )
        streams = [torch.npu.Stream(), torch.npu.Stream()]
        graphs, tensors, outputs, references = [], [], [], []
        for stream, case in zip(streams, ("short", "full")):
            cpu = attention_data(mode, 1, 6, 4096, case, [6], heads=16)
            values = [x.npu() for x in cpu]
            output = torch.empty_like(values[0])
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                op(*values, out=output)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                op(*values, out=output)
            graphs.append(graph)
            tensors.append(values)
            outputs.append(output)
            references.append(attention_oracle(mode, cpu))
        for _ in range(3):
            for stream, graph, output in zip(streams, graphs, outputs):
                with torch.npu.stream(stream):
                    output.fill_(float("nan"))
                    graph.replay()
        torch.npu.synchronize()
        for output, expected in zip(outputs, references):
            torch.testing.assert_close(output.cpu(), expected, atol=0.005, rtol=0.02)
        records.append(f"{mode}/two-stream-graph-owner-isolation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(4182)
    records = []
    for heads in (16, 32, 64):
        for mode in ("swa", "c1", "c2"):
            for batch in (1, 4):
                op = (
                    WindowAttention(heads=heads)
                    if mode == "swa"
                    else SparseAttention(
                        ratio=int(mode[1]),
                        heads=heads,
                        window_reuse=batch == 1,
                    )
                )
                print("QUALIFY", heads, mode, batch, flush=True)
                for tokens in (1, 6, 33, 257):
                    selected = (
                        list(range(tokens))
                        if tokens < 257
                        else list(range(0, tokens, 16))
                    )

                    def check(cpu, output, storage):
                        torch.testing.assert_close(
                            output[selected].cpu(),
                            attention_oracle(mode, cpu, token_indices=selected),
                            atol=0.005,
                            rtol=0.02,
                        )
                        assert bool((storage[[0, -1]] == 83).all())

                    qlens = (
                        [tokens]
                        if batch == 1
                        else [0, tokens // 3, 0, tokens - tokens // 3]
                    )
                    storage = torch.full(
                        (tokens + 2, heads, 512),
                        83,
                        dtype=torch.bfloat16,
                        device="npu",
                    )
                    output = storage[1:-1]
                    for case in (
                        "empty",
                        "short",
                        "full",
                        "duplicate",
                        "padding",
                        "sinks",
                    ):
                        cpu = attention_data(
                            mode, batch, tokens, 4096, case, qlens, heads=heads
                        )
                        device = [x.npu() for x in cpu]
                        op(*device, out=output)
                        check(cpu, output, storage)
                        records.append(
                            f"H{heads}/{mode}/B{batch}/T{tokens}/{case}/eager"
                        )
                    stream = torch.npu.Stream()
                    stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.stream(stream):
                        op(*device, out=output)
                    stream.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        op(*device, out=output)
                    for case in ("short", "full", "empty", "padding"):
                        cpu = attention_data(
                            mode,
                            batch,
                            tokens,
                            4096,
                            case,
                            list(reversed(qlens)),
                            heads=heads,
                        )
                        for dst, src in zip(device, cpu):
                            dst.copy_(src)
                        torch.npu.synchronize()
                        graph.replay()
                        torch.npu.synchronize()
                        check(cpu, output, storage)
                        records.append(
                            f"H{heads}/{mode}/B{batch}/T{tokens}/{case}/graph"
                        )
                    del graph, device, output, storage
    stream_isolation(records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"status": "passed", "physical": args.physical, "cases": records}, indent=2
        )
    )
    print("PASSED", len(records), flush=True)


if __name__ == "__main__":
    main()
