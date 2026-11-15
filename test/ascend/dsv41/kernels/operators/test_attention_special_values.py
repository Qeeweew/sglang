"""NaN/overflow propagation and recovery on three attention device entries.

Preserve the existing online normalization order: all -Inf key logits produce
NaN before the sink is combined. This is not a torch.softmax equivalence test
for arbitrary nonfinite input; normal finite inputs have independent references.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.attention import (
    DraftAttention,
    SparseAttention,
    WindowAttention,
)
from dsv41.kernels.references.attention import attention_data


def run(args):
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    checks = []
    for mode in ("swa", "c1", "c2", "draft"):
        cpu = attention_data(
            "swa" if mode == "draft" else mode,
            1,
            6,
            128,
            "short",
            [6],
            heads=16,
            physical_blocks=(1, 1),
        )
        if mode == "draft":
            cpu = cpu[:6]
            op = DraftAttention(heads=16)
        elif mode == "swa":
            op = WindowAttention(heads=16)
        else:
            op = SparseAttention(heads=16, ratio=1 if mode == "c1" else 2)
        data = [tensor.npu() for tensor in cpu]
        q, kv = data[:2]
        keys = [kv, data[2]] if mode in {"c1", "c2"} else [kv]
        # Equal positive values make the IEEE oracle independent of pagination
        # and duplicate keys. Every query sees at least one key and a finite sink.
        sinks = data[-2] if mode == "swa" else data[-1]
        sinks.zero_()
        out = torch.empty_like(q)
        bound = op.bind(*data, out=out)
        bound()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            bound()
        patterns = (
            ("nan", float("nan"), 1.0, "nan"),
            ("positive_inf", float("inf"), 1.0, "nan"),
            ("negative_inf_online_normalization", -float("inf"), 1.0, "nan"),
            (
                "finite_fp32_overflow",
                torch.finfo(torch.bfloat16).max,
                torch.finfo(torch.bfloat16).max,
                "nan",
            ),
            ("finite_recovery", 1.0, 1.0, "one"),
        )
        for name, query_value, key_value, expected in patterns:
            q.fill_(query_value)
            for key in keys:
                key.fill_(key_value)
            for execution, invoke in (("eager", bound), ("graph", graph.replay)):
                out.fill_(1234)
                torch.npu.synchronize()
                invoke()
                torch.npu.synchronize()
                actual = out.cpu()
                if expected == "nan":
                    assert bool(torch.isnan(actual).all()), (mode, name, execution)
                else:
                    target = (
                        torch.zeros_like(actual)
                        if expected == "zero"
                        else torch.ones_like(actual)
                    )
                    torch.testing.assert_close(
                        actual, target, rtol=0, atol=0, msg=f"{mode}/{name}/{execution}"
                    )
                checks.append(dict(mode=mode, pattern=name, execution=execution))
    report = dict(status="passed", checks=checks, physical=args.physical)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(checks)} attention IEEE/recovery checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, required=True)
    parser.add_argument("--report", type=Path)
    run(parser.parse_args())
