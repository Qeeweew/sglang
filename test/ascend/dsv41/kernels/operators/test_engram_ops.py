"""Independent integer/FP32 oracles, history commit and updated Graph replay."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
import sglang
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.engram_ops import (
    EngramOps,
    HashParameters,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.engram_hash import (
    oracle_commit,
    oracle_gate,
    oracle_hash,
)


def case_hash(op, p, params, counts, histories, records):
    total, batch = sum(counts), len(counts)
    cu = torch.tensor([0] + list(np.cumsum(counts)), dtype=torch.int32)
    slots = torch.arange(batch, dtype=torch.int32)
    history = torch.zeros((batch + 1, 3), dtype=torch.int32)
    if batch:
        history[:, :3] = torch.randint(
            0, len(p["token_map"]), (batch + 1, 3), dtype=torch.int32
        )
    ids = torch.randint(0, len(p["token_map"]), (total,), dtype=torch.int32)
    pos = (
        torch.cat(
            [torch.arange(old, old + n) for old, n in zip(histories, counts)]
        ).long()
        if batch
        else torch.empty(0, dtype=torch.int64)
    )
    active = (torch.rand(total) > 0.15).int()
    if total > 6:
        ids[2] = p["image_id"]
        ids[4] = 1_000_003
    accepted = torch.tensor(
        [min(n, (b % 7)) for b, n in enumerate(counts)], dtype=torch.int32
    )
    cpu = [ids, pos, cu, slots, history, active]
    device = [x.npu() for x in cpu]
    storage = torch.full(
        (2 * total * 24 + 16,), -987654321, dtype=torch.int64, device="npu"
    )
    out = storage[: 2 * total * 24].view(2, total, 24)
    op.hash(*device, params, out=out)
    torch.testing.assert_close(out.cpu(), oracle_hash(*cpu, p), rtol=0, atol=0)
    assert torch.equal(device[4].cpu(), history), "hash modified accepted history"
    assert bool((storage[2 * total * 24 :].cpu() == -987654321).all())
    expected_history = oracle_commit(ids, cu, slots, accepted, history)
    op.commit(device[0], device[2], device[3], accepted.npu(), device[4])
    torch.testing.assert_close(device[4].cpu(), expected_history, rtol=0, atol=0)
    records.append({"kind": "hash_commit", "counts": counts, "status": "passed"})


def graph_hash(op, p, params, records):
    counts, old = [0, 6, 2, 0, 1], [0, 10, 1, 0, 0]
    cu = torch.tensor([0] + list(np.cumsum(counts)), dtype=torch.int32)
    slots = torch.tensor([0, 3, 1, -1, 2], dtype=torch.int32)
    ids = torch.randint(0, len(p["token_map"]), (9,), dtype=torch.int32)
    positions = torch.cat(
        [torch.arange(prefix, prefix + n) for prefix, n in zip(old, counts)]
    ).long()
    history = torch.randint(0, len(p["token_map"]), (4, 3), dtype=torch.int32)
    active = torch.ones(9, dtype=torch.int32)
    cpu = [ids, positions, cu, slots, history, active]
    dev = [x.npu() for x in cpu]
    accepted = torch.tensor([0, 6, 0, 0, 1], dtype=torch.int32).npu()
    out = torch.empty((2, 9, 24), dtype=torch.int64, device="npu")
    hash_bound = op.bind_hash(*dev, params, out=out)
    commit_bound = op.bind_commit(dev[0], dev[2], dev[3], accepted, dev[4])
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        hash_bound()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        hash_bound()
        commit_bound()
    for generation in range(7):
        cpu[0] = torch.randint(0, len(p["token_map"]), (9,), dtype=torch.int32)
        if generation % 2 == 0:
            cpu[0][3] = p["image_id"]
        if generation == 3:
            cpu[2] = torch.tensor([0, 1, 1, 7, 9, 9], dtype=torch.int32)
            cpu[3] = torch.tensor([2, -1, 0, 1, 3], dtype=torch.int32)
            cpu[1] = torch.cat(
                [
                    torch.arange(prefix, prefix + int(end - begin))
                    for prefix, begin, end in zip(
                        (3, 0, 20, 5, 0), cpu[2][:-1], cpu[2][1:]
                    )
                ]
            ).long()
        n = cpu[2][1:] - cpu[2][:-1]
        commit = torch.minimum(n, torch.full_like(n, generation))
        accepted.copy_(commit)
        expected = oracle_hash(*cpu, p)
        history_expected = oracle_commit(cpu[0], cpu[2], cpu[3], commit, cpu[4])
        for dest, src in zip(dev, cpu):
            dest.copy_(src)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(dev[4].cpu(), history_expected, rtol=0, atol=0)
        cpu[4] = history_expected
    records.append(
        {"kind": "hash_graph", "accepted_prefixes": "0..6", "status": "passed"}
    )


def gates(op, records, hidden=5120):
    for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257):
        x = torch.randn(m, 4, hidden).bfloat16()
        kv = torch.randn(m, 5 * hidden).bfloat16()
        qw, kw = [torch.randn(4, hidden).bfloat16() for _ in range(2)]
        active = (torch.arange(m) % 3 != 1).int()
        if m:
            x[0, 0].zero_()
            kv[0, :hidden].zero_()
            x[0, 1].fill_(1e-30)
            kv[0, hidden : 2 * hidden].fill_(1e-10)
        cpu = [x, kv, qw, kw, active]
        dev = [a.npu() for a in cpu]
        canary = torch.full(
            (m * 4 * hidden + 32,), 17.0, dtype=torch.bfloat16, device="npu"
        )
        out = canary[: m * 4 * hidden].view(m, 4, hidden)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for _ in range(2):
                op.gate(*dev, out=out)
        stream.synchronize()
        expected = oracle_gate(*cpu)
        torch.testing.assert_close(
            out.cpu().float(), expected.float(), atol=0.001, rtol=0.008
        )
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            op.gate(*dev, out=out)
        for _ in range(2):
            cpu = [
                torch.randn_like(a) if a.dtype != torch.int32 else 1 - a for a in cpu
            ]
            for dest, src in zip(dev, cpu):
                dest.copy_(src)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            expected = oracle_gate(*cpu)
            torch.testing.assert_close(
                out.cpu().float(), expected.float(), atol=0.001, rtol=0.008
            )
        assert bool((canary[m * 4 * hidden :].cpu() == 17).all())
        records.append(
            {
                "kind": "gate",
                "rows": m,
                "status": "passed",
                "max_abs": (out.cpu().float() - expected.float()).abs().max().item()
                if m
                else 0,
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical",
        type=int,
        default=int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "-1")),
    )
    parser.add_argument("--library", type=Path, default=None)
    parser.add_argument("--model-parameters", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(903)
    op = EngramOps(library=args.library)
    primes = np.arange(97, 145).reshape(2, 3, 8)
    p = {
        "token_map": np.arange(257) % 113,
        "multipliers": np.array(
            [[13, 101, 717, 1397], [99, 32769, 65537, 131071]], dtype=np.int64
        ),
        "primes": primes,
        "offsets": np.cumsum(
            np.c_[np.zeros(2, dtype=np.int64), primes.reshape(2, 24)[:, :-1]], axis=1
        ),
        "pad_id": 2,
        "image_id": 256,
    }
    if args.model_parameters:
        p = HashParameters.model_arrays()
        # Extract pure reference functions by AST, without importing its module
        # and hardware dependencies. Their source is read-only.
        source = Path(sglang.__file__).resolve().parent / "srt/layers/engram.py"
        tree = ast.parse(source.read_text())
        nodes = [
            x
            for x in tree.body
            if isinstance(x, ast.FunctionDef)
            and x.name in ("compute_hash_multipliers", "build_compressed_token_map")
        ]
        scope = {"torch": torch, "np": np}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope
        )
        expected = scope["compute_hash_multipliers"]((1, 14), 4, 99092)
        torch.testing.assert_close(
            torch.from_numpy(p["multipliers"]), expected, atol=0, rtol=0
        )
        from tokenizers import Tokenizer

        class TokenizerView:
            backend_tokenizer = Tokenizer.from_file(
                str(Path(os.environ["DSV41_MODEL_PATH"]) / "tokenizer.json")
            )

            def __len__(self):
                return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

        expected_map, size = scope["build_compressed_token_map"](TokenizerView())
        assert size == 99092 and p["token_map"] == expected_map
    params = HashParameters(**p)
    records = []
    for counts in (
        [],
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [33],
        [257],
        [0, 3, 6, 2, 0, 1],
    ):
        case_hash(op, p, params, counts, [i % 4 for i in range(len(counts))], records)
    graph_hash(op, p, params, records)
    gates(op, records)
    gates(op, records, hidden=256)
    directory = ROOT / "runs/engram-ops"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "physical_device": args.physical,
                "model_parameters": args.model_parameters,
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS {len(records)} Engram hash/history/gate cases", flush=True)


if __name__ == "__main__":
    main()
