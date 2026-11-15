"""Independent projected C1/C2 formulas, transactional state and NPU Graph replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.builders.native import build_native
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.compressor import (
    CompressorState,
    ProjectedCompressor,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import ROOT
from dsv41.kernels.references.compressor import (
    oracle_commit,
    oracle_prepare,
)
from dsv41.kernels.support.compressor import validate_prepare


def validate_commit(
    engine,
    provisional,
    accepted,
    expected,
    label,
    records,
    *,
    positions=None,
    status=None,
):
    got_positions, got_rope, got_status = engine.commit(
        provisional, accepted.npu(), positions=positions, status=status
    )
    pending, lengths, positions, rope_positions, status = expected
    assert torch.equal(got_positions.cpu(), positions), label + "/accepted positions"
    assert torch.equal(got_rope.cpu(), rope_positions), (
        label + "/accepted RoPE positions"
    )
    assert torch.equal(got_status.cpu(), status), label + "/status"
    assert torch.equal(engine.state.pending.cpu(), pending), (
        label + "/pending exact FP32"
    )
    assert torch.equal(engine.state.committed_lengths.cpu(), lengths), (
        label + "/lengths"
    )
    records.append({"case": label})


def seed_state(slots=8):
    pending = torch.randn(slots, 2, 512)
    lengths = torch.tensor([0, 1, 126, 127, 128, 129, 1023, 1024], dtype=torch.int64)[
        :slots
    ]
    return pending, lengths


def eager(library, records):
    for ratio in (1, 2):
        for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257):
            for start in (0, 1, 127):
                pending, lengths = seed_state()
                lengths[3] = start
                state = CompressorState(pending.npu(), lengths.npu())
                gamma = torch.randn(512) * 0.1 + 1
                engine = ProjectedCompressor(ratio, gamma.npu(), state, library=library)
                kv = torch.randn(m, 512).to(
                    torch.bfloat16 if ratio == 1 else torch.float32
                )
                scores = torch.randn(m, 512) * 20 if ratio == 2 else None
                if ratio == 2 and m:
                    # Distinguish preserved FP32 values from premature BF16 rounding.
                    kv[0] = torch.arange(512).float() * 0.000013 + 1.00001
                    scores[0, ::2] = 1000
                    scores[0, 1::2] = -1000
                if ratio == 1 and m:
                    kv[0].zero_()
                cu = torch.tensor([0, m], dtype=torch.int32)
                ids = torch.tensor([3], dtype=torch.int32)
                args = (
                    kv.npu(),
                    scores.npu() if scores is not None else None,
                    cu.npu(),
                    ids.npu(),
                )
                provisional = engine.prepare(*args)
                expected = oracle_prepare(
                    ratio, kv, scores, gamma, pending, lengths, cu, ids
                )
                label = f"C{ratio}/M{m}/prefix{start}"
                validate_prepare(
                    provisional,
                    expected,
                    state,
                    pending,
                    lengths,
                    label + "/prepare",
                    records,
                )
                for keep in sorted({0, m // 2, m}):
                    state.pending.copy_(pending)
                    state.committed_lengths.copy_(lengths)
                    accepted = torch.tensor([keep], dtype=torch.int32)
                    expected_commit = oracle_commit(
                        ratio,
                        kv,
                        scores,
                        pending,
                        lengths,
                        cu,
                        ids,
                        expected[4],
                        accepted,
                    )
                    validate_commit(
                        engine,
                        provisional,
                        accepted,
                        expected_commit,
                        label + f"/accept{keep}",
                        records,
                    )


def split_invariance(library, records):
    for ratio in (1, 2):
        for m in (33, 257):
            for prefix in (0, 1, 127):
                pending, lengths = seed_state()
                lengths[2] = prefix
                gamma = torch.randn(512) * 0.1 + 1
                kv = torch.randn(m, 512).to(
                    torch.bfloat16 if ratio == 1 else torch.float32
                )
                scores = torch.randn(m, 512) if ratio == 2 else None
                ids = torch.tensor([2], dtype=torch.int32)
                full = oracle_prepare(
                    ratio,
                    kv,
                    scores,
                    gamma,
                    pending,
                    lengths,
                    torch.tensor([0, m]),
                    ids,
                )
                final = oracle_commit(
                    ratio,
                    kv,
                    scores,
                    pending,
                    lengths,
                    torch.tensor([0, m]),
                    ids,
                    full[4],
                    torch.tensor([m]),
                )
                state = CompressorState(pending.npu(), lengths.npu())
                engine = ProjectedCompressor(ratio, gamma.npu(), state, library=library)
                all_latent, all_pool, all_positions, all_rope = [], [], [], []
                start = 0
                chunks = (1, 2, 3, 5, 6, 7, 8)
                step = 0
                while start < m:
                    count = min(chunks[step % len(chunks)], m - start)
                    x = kv[start : start + count].npu()
                    s = (
                        scores[start : start + count].npu()
                        if scores is not None
                        else None
                    )
                    provisional = engine.prepare(
                        x,
                        s,
                        torch.tensor([0, count], dtype=torch.int32).npu(),
                        ids.npu(),
                    )
                    all_latent.append(provisional.latent.cpu())
                    all_pool.append(provisional.pooled.cpu())
                    all_positions.append(provisional.positions.cpu())
                    all_rope.append(provisional.rope_positions.cpu())
                    accepted = torch.tensor([count], dtype=torch.int32).npu()
                    _, _, status = engine.commit(provisional, accepted)
                    assert torch.equal(status.cpu(), torch.zeros(1, dtype=torch.int32))
                    start += count
                    step += 1
                torch.testing.assert_close(
                    torch.cat(all_latent).float(),
                    full[0].float(),
                    atol=0.0001,
                    rtol=0.008,
                )
                torch.testing.assert_close(
                    torch.cat(all_pool), full[1], atol=2e-6, rtol=2e-6
                )
                assert torch.equal(torch.cat(all_positions), full[2])
                assert torch.equal(torch.cat(all_rope), full[3])
                # At even length pending is inactive. Different chunking may
                # leave a different old row there; only live state is semantic.
                actual_pending = state.pending.cpu()
                for slot in range(len(lengths)):
                    if slot != 2 or int(final[1][slot]) % 2:
                        assert torch.equal(actual_pending[slot], final[0][slot])
                assert torch.equal(state.committed_lengths.cpu(), final[1])
                records.append(
                    {"case": f"C{ratio}/M{m}/prefix{prefix}/split-invariance"}
                )


def graph_and_isolation(library, records):
    for ratio in (1, 2):
        pending, lengths = seed_state()
        state = CompressorState(pending.npu(), lengths.npu())
        gamma = torch.ones(512)
        engine = ProjectedCompressor(ratio, gamma.npu(), state, library=library)
        kv = (
            torch.randn(6, 512)
            .to(torch.bfloat16 if ratio == 1 else torch.float32)
            .npu()
        )
        scores = torch.randn(6, 512).npu() if ratio == 2 else None
        cu = torch.tensor([0, 2, 2, 6], dtype=torch.int32).npu()
        ids = torch.tensor([3, 1, 6], dtype=torch.int32).npu()
        accepted = torch.tensor([1, 0, 3], dtype=torch.int32).npu()
        latent = torch.empty(6, 512, dtype=torch.bfloat16, device="npu")
        pooled = torch.empty(6, 512, dtype=torch.float32, device="npu")
        positions = torch.empty(6, dtype=torch.int64, device="npu")
        rope_positions = torch.empty_like(positions)
        starts = torch.empty(3, dtype=torch.int64, device="npu")
        accepted_positions = torch.empty_like(positions)
        accepted_rope = torch.empty_like(positions)
        status = torch.empty(3, dtype=torch.int32, device="npu")

        def queue(
            engine=engine,
            kv=kv,
            scores=scores,
            cu=cu,
            ids=ids,
            latent=latent,
            pooled=pooled,
            positions=positions,
            rope_positions=rope_positions,
            starts=starts,
            accepted=accepted,
            accepted_positions=accepted_positions,
            accepted_rope=accepted_rope,
            status=status,
        ):
            provisional = engine.prepare(
                kv,
                scores,
                cu,
                ids,
                out=latent,
                pooled=pooled,
                positions=positions,
                rope_positions=rope_positions,
                starts=starts,
            )
            engine.commit(
                provisional,
                accepted,
                positions=accepted_positions,
                rope_positions=accepted_rope,
                status=status,
            )

        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for _ in range(2):
                queue()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            queue()
        for iteration, (qlens, request_slots, keeps) in enumerate(
            (
                ((2, 0, 4), (3, 1, 6), (1, 0, 3)),
                ((0, 6, 0), (7, 2, 4), (0, 5, 0)),
                ((1, 2, 3), (1, 3, 5), (0, 2, 1)),
                ((3, 3, 0), (6, 0, 2), (3, 0, 0)),
            )
        ):
            p0, l0 = seed_state()
            x0 = torch.randn(6, 512).to(torch.bfloat16 if ratio == 1 else torch.float32)
            s0 = torch.randn(6, 512) * 10 if ratio == 2 else None
            g0 = torch.randn(512) * 0.1 + 1
            cu0 = torch.tensor(
                [0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32
            )
            id0 = torch.tensor(request_slots, dtype=torch.int32)
            a0 = torch.tensor(keeps, dtype=torch.int32)
            for dest, src in (
                (state.pending, p0),
                (state.committed_lengths, l0),
                (engine.gamma, g0),
                (kv, x0),
                (cu, cu0),
                (ids, id0),
                (accepted, a0),
            ):
                dest.copy_(src)
            if scores is not None:
                scores.copy_(s0)
            expected = oracle_prepare(ratio, x0, s0, g0, p0, l0, cu0, id0)
            committed = oracle_commit(ratio, x0, s0, p0, l0, cu0, id0, expected[4], a0)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                latent.cpu().float(), expected[0].float(), atol=0.0001, rtol=0.008
            )
            torch.testing.assert_close(pooled.cpu(), expected[1], atol=2e-6, rtol=2e-6)
            assert torch.equal(positions.cpu(), expected[2])
            assert torch.equal(rope_positions.cpu(), expected[3])
            assert torch.equal(starts.cpu(), expected[4])
            assert torch.equal(state.pending.cpu(), committed[0])
            assert torch.equal(state.committed_lengths.cpu(), committed[1])
            assert torch.equal(accepted_positions.cpu(), committed[2])
            assert torch.equal(accepted_rope.cpu(), committed[3])
            assert torch.equal(status.cpu(), committed[4])
            records.append({"case": f"C{ratio}/M6/multi-request-graph{iteration}"})


def rejection_and_guards(library, records):
    for ratio in (1, 2):
        pending, lengths = seed_state()
        state = CompressorState(pending.npu(), lengths.npu())
        gamma = torch.ones(512)
        engine = ProjectedCompressor(ratio, gamma.npu(), state, library=library)
        kv = torch.randn(6, 512).to(torch.bfloat16 if ratio == 1 else torch.float32)
        scores = torch.randn(6, 512) if ratio == 2 else None
        cu = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
        ids = torch.tensor([0, 3, -1], dtype=torch.int32)
        provisional = engine.prepare(
            kv.npu(), scores.npu() if scores is not None else None, cu.npu(), ids.npu()
        )
        expected = oracle_prepare(ratio, kv, scores, gamma, pending, lengths, cu, ids)
        validate_prepare(
            provisional,
            expected,
            state,
            pending,
            lengths,
            f"C{ratio}/invalid-slot-prepare",
            records,
        )
        accepted = torch.tensor([2, -1, 0], dtype=torch.int32)
        commit = oracle_commit(
            ratio, kv, scores, pending, lengths, cu, ids, expected[4], accepted
        )
        validate_commit(
            engine,
            provisional,
            accepted,
            commit,
            f"C{ratio}/invalid-accept-rejected",
            records,
        )
        changed = lengths.clone()
        changed[0] += 1
        changed[3] += 2
        state.committed_lengths.copy_(changed)
        accepted = torch.tensor([1, 1, 0], dtype=torch.int32)
        commit = oracle_commit(
            ratio, kv, scores, pending, changed, cu, ids, expected[4], accepted
        )
        validate_commit(
            engine,
            provisional,
            accepted,
            commit,
            f"C{ratio}/stale-snapshot-rejected",
            records,
        )
        # Dtype guard forbids premature BF16 C2 projections.
        if ratio == 2:
            try:
                engine.prepare(kv.bfloat16().npu(), scores.npu(), cu.npu(), ids.npu())
            except ValueError:
                pass
            else:
                raise AssertionError("C2 accepted BF16 projection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=int, default=4)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/native/libdsv41_kernels.so"
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "runs/compressor-agent/validation.json"
    )
    args = parser.parse_args()
    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    if args.build:
        build_native(args.library.parent, components=["compressor"])
    health = subprocess.check_output(["npu-smi", "info"], text=True)
    assert health.count("OK") >= 8
    assert f"No running processes found in NPU {args.physical}" in health
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(87463)
    records = []
    eager(args.library, records)
    print("PASS eager", len(records), flush=True)
    split_invariance(args.library, records)
    print("PASS split-invariance", len(records), flush=True)
    graph_and_isolation(args.library, records)
    rejection_and_guards(args.library, records)
    torch.npu.synchronize()
    report = {
        "status": "passed",
        "physical": args.physical,
        "cases": records,
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            (ROOT / "native/compressor/compressor.asc").read_bytes()
        ).hexdigest(),
        "health": health,
    }
    target = args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(records)} compressor checks: {target}", flush=True)


if __name__ == "__main__":
    main()
