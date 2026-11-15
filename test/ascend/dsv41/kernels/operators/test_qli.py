"""Independent CPU oracle and 910B eager/Graph checks for the QLI port.

Run from repository root with ASCEND_RT_VISIBLE_DEVICES=0.
"""

from __future__ import annotations

import argparse
import gc
import json
import weakref
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.qli import QLIPlan
from dsv41.kernels.references.qli import compare, oracle
from dsv41.kernels.support.qli_fixtures import make_case


def graph_case(library_dir, mode):
    plan, inputs, candidate = make_case(
        [6], [4097], ratio=2, residual=[1], mode=mode, library_dir=library_dir, seed=37
    )
    # Warm custom launch registration before capture.
    for _ in range(2):
        plan.run(*inputs, candidates=candidate)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, _ = plan.run(*inputs, candidates=candidate)
    before = output.cpu().clone()
    metadata_names = ("cu_q", "lengths", "residual", "pages")
    addresses = [getattr(plan, name).data_ptr() for name in metadata_names]
    for generation in (1, 2):
        inputs[0].copy_(
            torch.randint(-127, 128, inputs[0].shape, dtype=torch.int8).npu()
        )
        # Schedule/length update outside capture, keeping all device addresses.
        length = 4097 - generation * 123
        pages = plan._pages_cpu.roll(1, dims=1) if generation == 1 else None
        plan.update([0, 6], [length], residual=[generation % 2], pages=pages)
        assert addresses == [getattr(plan, name).data_ptr() for name in metadata_names]
        if candidate is not None:
            candidate.fill_(-1)
            candidate[:, :17].copy_(
                torch.arange(17, dtype=torch.int32, device="npu")[None, :].expand(6, -1)
            )
        graph.replay()
        score, valid, expected_candidates = oracle(
            *(x.cpu() for x in inputs),
            plan._pages_cpu,
            [0, 6],
            [length],
            [generation % 2],
            2,
            mode,
            candidate.cpu() if candidate is not None else None,
        )
        compare(output, score, valid)
        if mode == "source":
            for row in range(plan.rows):
                assert set(plan.candidate_out[row].cpu().tolist()) == set(
                    expected_candidates[row].tolist()
                )
    assert not torch.equal(before, output.cpu()), "Graph did not observe updated inputs"


def device_metadata_graph(library_dir):
    """Change device Cu, pages and lengths without any host plan update."""
    for mode in ("off", "source", "consumer"):
        plan, inputs, candidate = make_case(
            [3, 0, 6, 2],
            [105, 0, 4097, 1],
            ratio=2,
            residual=[1, 0, 1, 0],
            mode=mode,
            library_dir=library_dir,
            seed=103,
        )
        cu, lengths, residual, pages = (
            value.clone()
            for value in (plan.cu_q, plan.lengths, plan.residual, plan.pages)
        )
        plan.bind_metadata(
            cu, lengths, pages, residual=residual, page_count=inputs[1].shape[0]
        )
        plan.run(*inputs, candidates=candidate)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output, candidate_output = plan.run(*inputs, candidates=candidate)
        for counts, kl, rs in (
            ([0, 6, 3, 2], [0, 257, 4097, 1], [0, 1, 0, 0]),
            ([6, 1, 2, 2], [129, 0, 1025, 1], [0, 1, 1, 0]),
        ):
            partition = [0]
            for count in counts:
                partition.append(partition[-1] + count)
            host_pages = pages.cpu().roll(1, dims=1)
            cu.copy_(torch.tensor(partition, dtype=torch.int32))
            lengths.copy_(torch.tensor(kl, dtype=torch.int32))
            residual.copy_(torch.tensor(rs, dtype=torch.int32))
            pages.copy_(host_pages)
            graph.replay()
            score, valid, expected_candidates = oracle(
                *(x.cpu() for x in inputs),
                host_pages,
                partition,
                kl,
                rs,
                2,
                mode,
                candidate.cpu() if candidate is not None else None,
            )
            compare(output, score, valid)
            if mode == "source":
                torch.testing.assert_close(candidate_output.cpu(), expected_candidates)
        print(
            json.dumps({"graph": "device-metadata/" + mode, "status": "pass"}),
            flush=True,
        )


def chained_graph(library_dir):
    """Source/consumer share candidate GM addresses inside one captured graph."""
    source, inputs, _ = make_case(
        [6], [20003], mode="source", library_dir=library_dir, seed=95
    )
    consumer = QLIPlan(
        [0, 6],
        [20003],
        source._pages_cpu,
        candidate_mode="consumer",
        library_dir=library_dir,
    )
    for _ in range(2):
        source.run(*inputs)
        consumer.run(*inputs, candidates=source.candidate_out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        source.run(*inputs)
        output, _ = consumer.run(*inputs, candidates=source.candidate_out)
    for length in (20003, 19013):
        torch.npu.synchronize()
        inputs[0].copy_(
            torch.randint(-127, 128, inputs[0].shape, dtype=torch.int8).npu()
        )
        source.update([0, 6], [length])
        consumer.update([0, 6], [length])
        graph.replay()
        score, valid, _ = oracle(
            *(x.cpu() for x in inputs),
            source._pages_cpu,
            [0, 6],
            [length],
            [0],
            1,
            "consumer",
            source.candidate_out.cpu(),
        )
        compare(output, score, valid)
    print(json.dumps({"graph": "source->consumer", "status": "pass"}), flush=True)


def empty_transition_graph(library_dir):
    """All-empty K: TND rows differ from B*maxQ; candidate output is poisoned.

    Previously the upstream empty-work fastpath used B*maxQ and left candidate
    output unchanged. Capture with nonempty K, then exercise both directions.
    """
    plan, inputs, _ = make_case(
        [1, 0, 0],
        [129, 0, 0],
        ratio=2,
        residual=[1, 0, 0],
        mode="source",
        library_dir=library_dir,
        seed=99,
    )
    out_storage = torch.full((512 + 1024,), -1234567, dtype=torch.int32, device="npu")
    candidate_storage = torch.full(
        (2048 + 1024,), -1234567, dtype=torch.int32, device="npu"
    )
    output = out_storage[:512].view(1, 512)
    candidate_output = candidate_storage[:2048].view(1, 2048)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        plan.run(*inputs, out=output, candidate_out=candidate_output)
    for length in (0, 129, 0):
        torch.npu.synchronize()
        output.fill_(7654321)
        candidate_output.fill_(7654321)
        plan.update([0, 1, 1, 1], [length, 0, 0], residual=[1, 0, 0])
        graph.replay()
        score, valid, expected = oracle(
            *(x.cpu() for x in inputs),
            plan._pages_cpu,
            [0, 1, 1, 1],
            [length, 0, 0],
            [1, 0, 0],
            2,
            "source",
        )
        compare(output, score, valid)
        assert set(candidate_output.cpu()[0].tolist()) == set(expected[0].tolist())
        assert bool((out_storage[512:].cpu() == -1234567).all())
        assert bool((candidate_storage[2048:].cpu() == -1234567).all())
    print(
        json.dumps({"graph": "emptyK<->nonemptyK TND canary", "status": "pass"}),
        flush=True,
    )


def edge_contracts(library_dir):
    for size in (16, 64, 256):
        for mode in ("off", "source", "consumer"):
            make_case(
                [6],
                [1573],
                ratio=2,
                residual=[1],
                mode=mode,
                library_dir=library_dir,
                page_size=size,
                padded=True,
                signed=True,
            )
    plan = QLIPlan(
        [0], [], torch.empty((0, 1), dtype=torch.int32), library_dir=library_dir
    )
    result, _ = plan.run(
        torch.empty((0, 32, 128), dtype=torch.int8, device="npu"),
        torch.empty((1, 128, 128), dtype=torch.int8, device="npu"),
        torch.empty((0, 32), dtype=torch.float16, device="npu"),
        torch.empty((0, 32), dtype=torch.float16, device="npu"),
        torch.empty((1, 128), dtype=torch.float16, device="npu"),
    )
    assert result.shape == (0, 512)
    for kwargs in (
        {"cu_q": [0, 1], "lengths": [1], "pages": [[0]], "ratio": 2, "residual": [2]},
        {"cu_q": [0, 1], "lengths": [129], "pages": [[0]]},
        {"cu_q": [0, 1], "lengths": [1], "pages": [[-1]]},
    ):
        try:
            QLIPlan(**kwargs, library_dir=library_dir)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid metadata accepted")
    plan, inputs, _ = make_case([6], [1573], library_dir=library_dir)
    alias = inputs[1].view(torch.int32).flatten()[: 6 * 512].view(6, 512)
    try:
        plan.run(*inputs, out=alias)
    except ValueError as exc:
        assert "alias" in str(exc)
    else:
        raise AssertionError("overlapping output accepted")
    query_ref = weakref.ref(inputs[0])
    del inputs
    gc.collect()
    assert query_ref() is None, "QLI retained an eager query after launch"
    print(
        json.dumps(
            {
                "edge_contracts": "signed heads, page16/64/256, empty B, invalid metadata, alias",
                "status": "pass",
            }
        ),
        flush=True,
    )


def runtime_geometry(library_dir):
    # Same single device artifact for all legal slot extents and launch grids.
    for heads, depth, topk, cores in (
        (16, 32, 128, 1),
        (32, 64, 256, 3),
        (48, 96, 512, None),
        (64, 128, 1024, None),
    ):
        for mode in ("off", "source", "consumer"):
            make_case(
                [6],
                [1573],
                library_dir=library_dir,
                mode=mode,
                heads=heads,
                depth=depth,
                topk=topk,
                cores=cores,
                padded=True,
            )
    print(
        json.dumps(
            {
                "runtime_geometry": "H16/32/48/64 D32/64/96/128 topk128..1024 cores1/3/device",
                "status": "pass",
            }
        ),
        flush=True,
    )


def prefill_cases(library_dir, rows):
    """Execute every prefill query; check boundary rows with the CPU oracle."""
    for counts in ([rows], [1, 127, 128, 129, 1, 127, 128, rows - 641]):
        cu = [0]
        for count in counts:
            cu.append(cu[-1] + count)
        assert cu[-1] == rows
        samples = sorted(
            {
                0,
                1,
                127,
                128,
                rows // 2,
                rows - 2,
                rows - 1,
                *cu[:-1],
                *(v - 1 for v in cu[1:]),
            }
        )
        for ratio in (1, 2):
            original_lengths = [n + 257 for n in counts]
            lengths = [n // ratio for n in original_lengths]
            residual = [n % ratio for n in original_lengths]
            for mode in ("off", "source", "consumer"):
                plan, inputs, candidate = make_case(
                    counts,
                    lengths,
                    ratio=ratio,
                    residual=residual,
                    mode=mode,
                    library_dir=library_dir,
                    reference_rows=samples,
                )
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output, _ = plan.run(*inputs, candidates=candidate)
                inputs[0].copy_(
                    torch.randint(-127, 128, inputs[0].shape, dtype=torch.int8).npu()
                )
                original_lengths = [n - 1 for n in original_lengths]
                plan.update(
                    cu,
                    [n // ratio for n in original_lengths],
                    residual=[n % ratio for n in original_lengths],
                )
                graph.replay()
                torch.npu.synchronize()
                score, valid, expected = oracle(
                    *(x.cpu() for x in inputs),
                    plan._pages_cpu,
                    cu,
                    [n // ratio for n in original_lengths],
                    [n % ratio for n in original_lengths],
                    ratio,
                    mode,
                    candidate.cpu() if candidate is not None else None,
                    row_indices=samples,
                )
                compare(output[samples], score, valid)
                if mode == "source":
                    for i, row in enumerate(samples):
                        assert set(plan.candidate_out[row].cpu().tolist()) == set(
                            expected[i].tolist()
                        )
                print(
                    f"PASS QLI B{len(counts)}T{rows} C{ratio} {mode}: eager + updated-input Graph",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-dir", type=Path, default=Path("build/dsv41"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prefill-rows", type=int, default=0)
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="qualify six-row schedules at 32K/128K/1M boundaries",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = args.library_dir.resolve()
    if args.prefill_rows:
        if args.prefill_rows <= 641:
            parser.error("--prefill-rows must exceed 641 for the eight-request fixture")
        prefill_cases(root, args.prefill_rows)
        return
    if args.long_context:
        for ratio in (1, 2):
            for original in (32769, 131073, 1048575, 1048576):
                plan, inputs, _ = make_case(
                    [6],
                    [original // ratio],
                    ratio=ratio,
                    residual=[original % ratio],
                    mode="source",
                    library_dir=root,
                    seed=original + ratio,
                )
                consumer = QLIPlan(
                    [0, 6],
                    [original // ratio],
                    plan._pages_cpu,
                    ratio=ratio,
                    residual=[original % ratio],
                    page_size=128,
                    candidate_mode="consumer",
                    library_dir=root,
                )
                for _ in range(2):
                    plan.run(*inputs)
                    consumer.run(*inputs, candidates=plan.candidate_out)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    plan.run(*inputs)
                    output, _ = consumer.run(*inputs, candidates=plan.candidate_out)
                for original_length in (original - 1, original):
                    for active in (plan, consumer):
                        active.update(
                            [0, 6],
                            [original_length // ratio],
                            residual=[original_length % ratio],
                        )
                    inputs[0].random_(-127, 128)
                    torch.npu.synchronize()
                    graph.replay()
                    score, valid, _ = oracle(
                        *(x.cpu() for x in inputs),
                        plan._pages_cpu,
                        [0, 6],
                        [original_length // ratio],
                        [original_length % ratio],
                        ratio,
                        "consumer",
                        plan.candidate_out.cpu(),
                    )
                    compare(output, score, valid)
                print(
                    json.dumps(
                        {
                            "long_original": original,
                            "ratio": ratio,
                            "graph": "source->consumer",
                            "status": "pass",
                        }
                    ),
                    flush=True,
                )
        return
    cases = 0
    for ratio in (1, 2):
        for m in (1,) if args.smoke else range(1, 9):
            make_case(
                [m],
                [513 + m],
                ratio=ratio,
                residual=[ratio - 1],
                library_dir=root,
                seed=10 + m,
            )
            cases += 1
            print(
                json.dumps(
                    {"case": cases, "rows": m, "ratio": ratio, "status": "pass"}
                ),
                flush=True,
            )
    if args.smoke:
        return
    for mode in ("off", "source", "consumer"):
        for rows, lens, ratio, residual in (
            ([3, 0, 6, 2], [105, 0, 4097, 1], 2, [1, 0, 1, 0]),
            ([33], [1573], 1, [0]),
            ([129], [2113], 2, [1]),
            ([6], [20003], 1, [0]),
            ([1], [0], 2, [1]),
        ):
            make_case(
                rows,
                lens,
                ratio=ratio,
                residual=residual,
                mode=mode,
                library_dir=root,
                seed=70 + cases,
                padded=True,
            )
            cases += 1
            print(
                json.dumps(
                    {
                        "case": cases,
                        "rows": rows,
                        "lengths": lens,
                        "mode": mode,
                        "status": "pass",
                    }
                ),
                flush=True,
            )
        make_case([0], [0], mode=mode, library_dir=root)
        graph_case(root, mode)
        print(json.dumps({"graph": mode, "status": "pass"}), flush=True)
    chained_graph(root)
    device_metadata_graph(root)
    empty_transition_graph(root)
    edge_contracts(root)
    runtime_geometry(root)
    print(json.dumps({"cases": cases, "graph_modes": 3, "status": "pass"}), flush=True)


if __name__ == "__main__":
    main()
