"""Single-910B graph timings for packed split + two weighted RMSNorms."""

import argparse
import json
from statistics import median

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


def benchmark(rows, depth=64, rounds=15):
    dtype = torch.bfloat16
    packed = torch.randn(rows, 1792, device="npu", dtype=dtype)
    weights = [torch.randn(width, device="npu", dtype=dtype) for width in (1280, 512)]
    contiguous = [part.contiguous() for part in packed.split((1280, 512), dim=1)]
    eps = 1e-20
    fused = load_library().split_rms_norm

    def vendor(parts):
        return tuple(
            torch_npu.npu_rms_norm(x, w, eps)[0] for x, w in zip(parts, weights)
        )

    calls = {
        "baseline": lambda: vendor((packed[..., :1280], packed[..., 1280:])),
        "pre_split_vendor": lambda: vendor(contiguous),
        "fused": lambda: fused(packed, *weights, eps, eps),
    }
    stream = torch.npu.Stream()
    graphs, samples = {}, {name: [] for name in calls}
    reference = calls["baseline"]()
    for name, call in calls.items():
        torch.testing.assert_close(call(), reference, rtol=1 / 128, atol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            for _ in range(depth):
                outputs = call()
        graph.replay()
        torch.testing.assert_close(outputs, reference, rtol=1 / 128, atol=0)
        graphs[name] = graph, outputs
    begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    names = list(calls)
    for iteration in range(rounds):
        for index in range(len(names)):
            name = names[(index + iteration) % len(names)]
            begin.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / depth)
    return {
        "rows": rows,
        "median_us": {k: median(v) for k, v in samples.items()},
        "depth": depth,
        "samples_us": samples,
    }


if __name__ == "__main__":
    if torch.npu.device_count() != 1:
        raise SystemExit("Select one idle NPU with ASCEND_RT_VISIBLE_DEVICES")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 6, 12, 24, 48, 96, 192, 512, 2048]
    )
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    torch.manual_seed(31)
    for rows in args.rows:
        print(json.dumps(benchmark(rows, args.depth, args.rounds)), flush=True)
