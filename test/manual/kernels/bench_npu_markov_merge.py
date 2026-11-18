"""Compare the Torch and fused 910B Markov paths without starting a process group."""

import json
import os
from statistics import median

import torch
import torch_npu  # noqa: F401

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.srt.hardware_backend.npu.dsv41.dspark.markov import AscendMarkovStep
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


def measure(calls, depth, rounds=15):
    graphs = []
    stream = torch.npu.Stream()
    reference = calls[0]().cpu()
    for call in calls:
        torch.testing.assert_close(call().cpu(), reference)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            for _ in range(depth):
                tokens = call()
        graph.replay()
        torch.testing.assert_close(tokens.cpu(), reference)
        graphs.append((graph, tokens))

    samples = [[], []]
    start = torch.npu.Event(enable_timing=True)
    stop = torch.npu.Event(enable_timing=True)
    for iteration in range(rounds):
        for index in (iteration % 2, 1 - iteration % 2):
            start.record()
            graphs[index][0].replay()
            stop.record()
            stop.synchronize()
            samples[index].append(start.elapsed_time(stop) * 1000 / depth)
    return {
        "depth": depth,
        "torch_us": median(samples[0]),
        "asc_us": median(samples[1]),
        "samples_us": samples,
    }


def inputs(ranks, rows):
    width = 64640
    scores = torch.randn(ranks, rows, device="npu")
    ids = torch.randint(width, (ranks, rows), dtype=torch.int32, device="npu")
    ids += torch.arange(ranks, dtype=torch.int32, device="npu")[:, None] * width
    packet = torch.stack((scores, ids.view(torch.float32)), dim=-1)
    peers = packet[1:].reshape(-1, 2)
    base = torch.randn(rows, 5, width, dtype=torch.bfloat16, device="npu")[:, 2]
    args = dict(
        base_logits=base,
        bias=torch.randn_like(base).contiguous(),
        temperatures=torch.zeros(rows, device="npu"),
        key=42,
        offset=torch.zeros(1, dtype=torch.int64, device="npu"),
        step=2,
        step_dtype=torch.float32,
        has_sampled_rows=False,
        gather=lambda local: torch.cat((local, peers)),
    )
    return packet, args


def run(ranks, rows):
    packet, args = inputs(ranks, rows)
    native = load_library().markov_merge_candidates
    return {
        "ranks": ranks,
        "rows": rows,
        "merge": measure(
            (
                lambda: MarkovSampleStep.finish_candidates(packet, rows),
                lambda: native(packet),
            ),
            128,
        ),
        "step": measure(
            (
                lambda: MarkovSampleStep.torch(**args),
                lambda: MarkovSampleStep.execute(**args),
            ),
            32,
        ),
        "local_gain": measure(
            (lambda: AscendMarkovStep.torch(**args), lambda: MarkovSampleStep.execute(**args)), 32
        ),
    }


if __name__ == "__main__":
    if torch.npu.device_count() != 1 or os.environ.get("SGLANG_DSV41_910B") != "1":
        raise SystemExit("Select one NPU and source the DeepSeek V4.1 910B common.sh")
    torch.manual_seed(17)
    for shape in ((2, 1), (2, 2), (3, 7), (4, 16), (8, 17)):
        print(json.dumps(run(*shape)), flush=True)
