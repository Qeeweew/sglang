"""Independent numerical references and fixtures for dspark."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import msgspec
import torch
import sglang
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT as ROOT

SGLANG = Path(sglang.__file__).resolve().parent

SOURCE = SGLANG / "kernels/ops/speculative/dspark/dspark_schedule.py"

DISPATCH = SOURCE.with_name("dispatch.py")

CONFIG = SGLANG / "srt/speculative/dspark_components/dspark_planner.py"


def load_official():
    scope = {"torch": torch, "msgspec": msgspec}
    selections = {
        DISPATCH: {"inputs_on_cuda"},
        CONFIG: {"DSparkScheduleConfig"},
        SOURCE: {
            "ScheduleVerifyLensTopk",
            "compute_sort_survival",
            "schedule_verify_lens_topk",
            "schedule_verify_lens_topk_from_survival",
            "_value_independent_descending_order",
        },
    }
    for path, names in selections.items():
        tree = ast.parse(path.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        ]
        assert {n.name for n in nodes} == names
        for node in nodes:
            node.decorator_list = []
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef):
                    child.decorator_list = [
                        decorator
                        for decorator in child.decorator_list
                        if not ast.unparse(decorator).startswith("_dsv41_")
                    ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        selected = ast.fix_missing_locations(
            ast.Module(body=[future, *nodes], type_ignores=[])
        )
        exec(compile(selected, str(path), "exec"), scope)
    return scope


def ranking_oracle(survival, budget, cfg):
    """Rank FP32 probabilities exactly, position first then request on ties."""
    b, gamma = survival.shape
    max_len = cfg.resolved_max_verify_len()
    columns = min(max_len, gamma)
    candidates = []
    for request in range(b):
        for position in range(columns):
            value = float(survival[request, position])
            valid = value >= cfg.survival_eps
            candidates.append(
                (-(value if valid else -math.inf), position, request, valid)
            )
    candidates.sort(key=lambda x: x[:3])
    selected = [0] * b
    for _, _, request, valid in candidates[: max(0, int(budget))]:
        selected[request] += int(valid)
    return torch.tensor(
        [
            min(max(cfg.min_verify_len + n, max(cfg.min_verify_len, 1)), max_len)
            for n in selected
        ],
        dtype=torch.int32,
    )


def test_data(requests, gamma, pattern):
    if pattern == "random":
        return torch.rand(requests, gamma)
    if pattern == "ones":
        return torch.ones(requests, gamma)
    if pattern == "zeros":
        return torch.zeros(requests, gamma)
    if pattern == "nextafter":
        values = torch.ones(requests, gamma)
        if requests:
            values[::3, 0] = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
            values[1::3, 0] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
        return values
    base = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -1.0, -0.0, 0.0, 0.5, 1.0, 2.0]
    )
    return base[torch.arange(requests * gamma).reshape(requests, gamma) % base.numel()]


def check_output(output, survival, budget, cfg, label, records):
    expected = ranking_oracle(survival, budget, cfg)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0, msg=label)
    record = {
        "case": label,
        "requests": survival.shape[0],
        "gamma": survival.shape[1],
        "budget": budget,
        "min": cfg.min_verify_len,
        "max": cfg.resolved_max_verify_len(),
        "eps": cfg.survival_eps,
        "status": "passed",
    }
    records.append(record)
    return record
