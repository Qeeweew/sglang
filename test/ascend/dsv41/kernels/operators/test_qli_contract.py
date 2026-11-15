"""Independent QLI precision witnesses, ties and live-metadata Graph checks.

Run with one explicitly bound physical NPU and --library-dir pointing at an
immutable native publication. --reference-only needs no NPU initialization.
The oracle uses integer-exact FP64 dots and explicit storage roundings; neither
the archived implementation nor its vendor sources supplies expected results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


def score_reference(inputs, pages, cu, lengths, residual, ratio, candidates=None):
    q, k, weights, qscale, kscale = inputs
    result = []
    for batch, length in enumerate(lengths):
        # Unused cells may contain arbitrary allocator sentinels, including
        # positive out-of-range IDs. Do not gather them before slicing length.
        needed = (length + k.shape[1] - 1) // k.shape[1]
        physical = pages[batch, :needed].long()
        keys = k[physical].reshape(-1, q.shape[-1])[:length].double()
        scales = kscale[physical].reshape(-1)[:length].float()
        count = cu[batch + 1] - cu[batch]
        for local in range(count):
            row = cu[batch] + local
            visible = (length * ratio + residual[batch] - count + local + 1) // ratio
            valid = max(0, min(length, visible))
            dot = q[row].double() @ keys.T
            positive = (dot.clamp_min(0) / 1024).half().double()
            head = (weights[row].double() * qscale[row].double()).half().double()
            score = (head[:, None] * positive).sum(0).float() * scales
            score[valid:] = -torch.inf
            if candidates is not None and valid:
                allowed = candidates[row].long()
                allowed = allowed[(allowed >= 0) & (allowed < (length + 7) // 8)]
                mask = torch.isin(torch.arange(valid) // 8, allowed)
                score[:valid].masked_fill_(~mask, -1e30)
            result.append((score, valid))
    return result


def check_indices(output, expected):
    output = output.cpu()
    assert output.shape[0] == len(expected)
    for row, (score, valid) in enumerate(expected):
        count = min(output.shape[1], valid)
        indices = output[row, :count].long()
        assert bool(((indices >= 0) & (indices < valid)).all()), ("unreachable", row)
        assert indices.unique().numel() == count, ("duplicate index", row)
        assert bool((output[row, count:] == -1).all()), ("padding", row)
        if count:
            # Ties do not prescribe an index permutation. Compare each selected
            # rank, retaining the established FP32 head-reduction budget.
            target = score[:valid].sort(descending=True).values[:count]
            torch.testing.assert_close(score[indices], target, rtol=2e-5, atol=2e-7)


def check_candidates(output, expected):
    for row, (score, valid) in enumerate(expected):
        values = output[row].cpu().long()
        blocks = (valid + 7) // 8
        count = min(2048, blocks)
        selected = values[:count]
        assert bool((values[count:] == -1).all()), ("candidate padding", row)
        assert bool(((selected >= 0) & (selected < blocks)).all())
        assert selected.unique().numel() == count, ("duplicate candidate", row)
        if not count:
            continue
        assert int(selected[0]) == blocks - 1, ("newest block must be first", row)
        block_score = (
            torch.nn.functional.pad(score[:valid], (0, -valid % 8), value=-torch.inf)
            .reshape(-1, 8)
            .amax(1)
        )
        # Exclude the mandatory pinned block before checking the score cutoff.
        # This permits any equally scored block at a truncated top-2048 tie.
        target = block_score[:-1].sort(descending=True).values[: count - 1]
        chosen = block_score[selected[1:]].sort(descending=True).values
        torch.testing.assert_close(chosen, target, rtol=2e-5, atol=2e-7)


def precision_witnesses():
    """Find finite rank reversals for four independently removable roundings."""
    generator = torch.Generator().manual_seed(910241)
    heads, depth, count = 16, 32, 32768
    q = torch.zeros((heads, depth), dtype=torch.int8)
    for head in range(heads):
        q[head, 2 * head] = 127
        q[head, 2 * head + 1] = 1
    keys = torch.randint(
        -128, 128, (count, depth), dtype=torch.int8, generator=generator
    )
    weights = (torch.rand(heads, generator=generator) * 1.5 + 0.1).half()
    qscale = (torch.rand(heads, generator=generator) * 1.5 + 0.01).half()
    kscale = (torch.rand(count, generator=generator) * 1.5 + 0.1).half()
    dot = q.double() @ keys.double().T
    head = (weights.double() * qscale.double()).half().double()
    positive = (dot.clamp_min(0) / 1024).half().double()
    products = head[:, None] * positive
    correct = products.sum(0).float() * kscale.float()
    mutants = {
        "weight_qscale_fp16": ((weights.double() * qscale.double())[:, None] * positive)
        .sum(0)
        .float()
        * kscale.float(),
        "relu_dot_fp16": (head[:, None] * (dot.clamp_min(0) / 1024)).sum(0).float()
        * kscale.float(),
        "head_product_fp32": products.half().double().sum(0).float() * kscale.float(),
        "kscale_after_heads": (
            head[:, None]
            * ((dot.clamp_min(0) / 1024) * kscale.double()[None, :]).half().double()
        )
        .sum(0)
        .float(),
    }
    order = correct.argsort(descending=True, stable=True)
    first, second = order[:-1], order[1:]
    margin = 5e-5 * correct[first].abs() + 2e-7
    witnesses = []
    for name, mutant in mutants.items():
        choices = (
            (
                (correct[first] - correct[second] > margin)
                & (mutant[second] - mutant[first] > margin)
            )
            .nonzero()
            .flatten()
        )
        assert choices.numel(), ("missing robust rank witness", name)
        high, low = int(first[choices[0]]), int(second[choices[0]])
        pair = keys[[high, low]]
        fixture_k = torch.zeros((1, 16, depth), dtype=torch.int8)
        fixture_k[0, :7] = pair[0]
        fixture_k[0, 7:9] = pair
        fixture_scale = torch.ones((1, 16), dtype=torch.float16)
        fixture_scale[0, :7] = kscale[high] * 4
        fixture_scale[0, 7:9] = kscale[[high, low]]
        inputs = (q[None], fixture_k, weights[None], qscale[None], fixture_scale)
        expected = score_reference(inputs, torch.tensor([[0]]), [0, 1], [9], [0], 1)[0][
            0
        ]
        assert set(expected.argsort(descending=True)[:8].tolist()) == set(range(8))
        witnesses.append(
            (
                name,
                inputs,
                {
                    "correct_pair": correct[[high, low]].tolist(),
                    "mutant_pair": mutant[[high, low]].tolist(),
                    "expected_top8_set": list(range(8)),
                },
            )
        )
    return witnesses


def relu_witness():
    q = torch.zeros((1, 16, 32), dtype=torch.int8)
    q[0, 0, :2] = torch.tensor([127, 1], dtype=torch.int8)
    q[0, 1, 2:4] = torch.tensor([127, 1], dtype=torch.int8)
    k = torch.zeros((1, 16, 32), dtype=torch.int8)

    def put(row, head, dot):
        value = math.trunc(dot / 127)
        k[0, row, 2 * head] = value
        k[0, row, 2 * head + 1] = dot - value * 127

    for row in range(7):
        put(row, 1, 4096)
    put(7, 0, -8192)
    put(7, 1, 1024)
    put(8, 1, 2048)
    w = torch.zeros((1, 16), dtype=torch.float16)
    w[0, :2] = torch.tensor([-1, 1], dtype=torch.float16)
    ks = torch.ones((1, 16), dtype=torch.float16)
    ks[0, :7] = 4
    return q, k, w, torch.ones_like(w), ks


def metadata(counts, lengths, page_size, generator):
    cu = [0]
    for count in counts:
        cu.append(cu[-1] + count)
    capacity = (
        max(1, max(((n + page_size - 1) // page_size for n in lengths), default=0)) + 2
    )
    physical = max(1, len(counts) * capacity)
    pages = (
        torch.randperm(physical, generator=generator)[: len(counts) * capacity]
        .reshape(len(counts), capacity)
        .int()
    )
    for batch, length in enumerate(lengths):
        needed = (length + page_size - 1) // page_size
        pages[batch, needed::2] = -1
        pages[batch, needed + 1 :: 2] = physical + 37
    return cu, pages, physical


def random_inputs(rows, physical, page_size, heads, depth, generator):
    return (
        torch.randint(
            -128, 128, (rows, heads, depth), dtype=torch.int8, generator=generator
        ),
        torch.randint(
            -128,
            128,
            (physical, page_size, depth),
            dtype=torch.int8,
            generator=generator,
        ),
        (torch.rand((rows, heads), generator=generator) - 0.35).half(),
        (torch.rand((rows, heads), generator=generator) * 0.02 + 0.001).half(),
        (torch.rand((physical, page_size), generator=generator) * 0.02 + 0.001).half(),
    )


def sparse_candidates(rows, length):
    values = torch.full((rows, 2048), -1, dtype=torch.int32)
    blocks = (length + 7) // 8
    pattern = [blocks + 17, 2, -7, 0, 2, blocks - 1, 1, 0, 2147483647]
    if rows:
        values[:, : len(pattern)] = torch.tensor(pattern, dtype=torch.int32)
    return values


def run(args, witnesses):
    visibility = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visibility or "," in visibility:
        raise ValueError("bind exactly one physical NPU with ASCEND_RT_VISIBLE_DEVICES")
    import torch_npu  # noqa: F401
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.qli import QLIPlan

    torch.npu.set_device(0)
    records = []
    generator = torch.Generator().manual_seed(741910)

    def exercise(
        name,
        inputs,
        cu,
        lengths,
        residual,
        pages,
        *,
        mode="off",
        ratio=1,
        topk=8,
        cores=None,
        candidates=None,
        graph_states=(),
    ):
        heads, depth = inputs[0].shape[1:]
        plan = QLIPlan(
            cu,
            lengths,
            pages,
            ratio=ratio,
            residual=residual,
            page_size=inputs[1].shape[1],
            candidate_mode=mode,
            heads=heads,
            depth=depth,
            topk=topk,
            cores=cores,
            library_dir=args.library_dir,
        )
        device = [value.npu() for value in inputs]
        # Distinct padded physical strides for K and Kscale. Padding is poison.
        kpad = torch.full(
            (inputs[1].shape[0], inputs[1].shape[1] + 16, depth),
            -99,
            dtype=torch.int8,
            device="npu",
        )
        spad = torch.full(
            (inputs[4].shape[0], inputs[4].shape[1] + 32),
            17,
            dtype=torch.float16,
            device="npu",
        )
        device[1], device[4] = (
            kpad[:, : inputs[1].shape[1]],
            spad[:, : inputs[4].shape[1]],
        )
        device[1].copy_(inputs[1])
        device[4].copy_(inputs[4])
        guards = []

        def guarded(shape, dtype, sentinel):
            prefix = 512 // torch.empty((), dtype=dtype).element_size()
            storage = torch.full(
                (math.prod(shape) + 2 * prefix,), sentinel, dtype=dtype, device="npu"
            )
            guards.append((storage, prefix, sentinel))
            return storage[prefix:-prefix].view(shape)

        out = guarded((plan.rows, topk), torch.int32, -172913)
        candidate_out = (
            guarded((plan.rows, 2048), torch.int32, -172913)
            if mode == "source"
            else None
        )
        plan.workspace = guarded((plan.workspace_bytes,), torch.uint8, 173)
        candidate = None if candidates is None else candidates.npu()
        # Borrowed metadata must be supplied before the first captured call.
        live = tuple(
            value.clone()
            for value in (plan.cu_q, plan.lengths, plan.pages, plan.residual)
        )
        plan.bind_metadata(
            live[0], live[1], live[2], residual=live[3], page_count=inputs[1].shape[0]
        )

        def invoke():
            return plan.run(
                *device, candidates=candidate, out=out, candidate_out=candidate_out
            )

        def check(cpu_inputs, host_pages, host_cu, kl, rs):
            expected = score_reference(
                cpu_inputs, host_pages, host_cu, kl, rs, ratio, candidates
            )
            check_indices(out, expected)
            if mode == "source":
                check_candidates(candidate_out, expected)
            for storage, prefix, sentinel in guards:
                assert bool((storage[:prefix].cpu() == sentinel).all())
                assert bool((storage[-prefix:].cpu() == sentinel).all())
            assert bool((kpad[:, inputs[1].shape[1] :].cpu() == -99).all())
            assert bool((spad[:, inputs[4].shape[1] :].cpu() == 17).all())

        invoke()
        torch.npu.synchronize()
        check(inputs, pages, cu, lengths, residual)
        records.append({"case": name, "mode": mode, "phase": "eager", "status": "pass"})
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(3):
                invoke()
        captured = (*device, *live, out, plan.workspace, candidate, candidate_out)
        pointers = [value.data_ptr() for value in captured if value is not None]
        states = graph_states or ((cu, lengths, residual),)
        for generation, (host_cu, kl, rs) in enumerate(states):
            if graph_states:
                inputs = random_inputs(
                    plan.rows,
                    inputs[1].shape[0],
                    inputs[1].shape[1],
                    heads,
                    depth,
                    generator,
                )
                for target, source in zip(device, inputs):
                    target.copy_(source)
            # Remap only currently used cells, leaving invalid unused sentinels.
            host_pages = pages.clone()
            for batch, length in enumerate(kl):
                needed = (length + plan.page_size - 1) // plan.page_size
                host_pages[batch, :needed] = (
                    torch.arange(needed) + generation * 7 + batch
                ) % inputs[1].shape[0]
            if not graph_states:
                host_pages = pages
            for target, source in zip(
                live,
                (
                    torch.tensor(host_cu, dtype=torch.int32),
                    torch.tensor(kl, dtype=torch.int32),
                    host_pages,
                    torch.tensor(rs, dtype=torch.int32),
                ),
            ):
                target.copy_(source)
            if candidate is not None:
                candidates = sparse_candidates(plan.rows, max(kl, default=0)).roll(
                    generation, dims=1
                )
                candidate.copy_(candidates)
            out.fill_(7654321)
            if candidate_out is not None:
                candidate_out.fill_(7654321)
            torch.npu.synchronize()
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
            check(inputs, host_pages, host_cu, kl, rs)
            captured = (*device, *live, out, plan.workspace, candidate, candidate_out)
            assert pointers == [
                value.data_ptr() for value in captured if value is not None
            ]
            records.append(
                {
                    "case": name,
                    "mode": mode,
                    "phase": "graph",
                    "generation": generation,
                    "status": "pass",
                }
            )
        print("PASS", name, mode, flush=True)

    for name, inputs, _ in witnesses:
        exercise(
            name,
            inputs,
            [0, 1],
            [9],
            [0],
            torch.tensor([[0, -1, 99]], dtype=torch.int32),
            cores=1,
        )
    inputs = relu_witness()
    exercise(
        "relu_before_signed_heads",
        inputs,
        [0, 1],
        [9],
        [0],
        torch.tensor([[0]], dtype=torch.int32),
        cores=1,
    )
    for mode in ("off", "source", "consumer"):
        counts, lengths, rs = (
            [3, 0, 1, 0, 2, 0],
            [257, 0, 0, 0, 17, 0],
            [1, 0, 1, 0, 0, 0],
        )
        cu, pages, physical = metadata(counts, lengths, 48, generator)
        inputs = random_inputs(6, physical, 48, 48, 96, generator)
        states = (
            ([0, 0, 3, 3, 5, 6, 6], [0, 129, 0, 257, 1, 0], [0, 0, 0, 1, 0, 0]),
            (list(range(7)), [0] * 6, [1] * 6),
            ([0, 1, 1, 3, 3, 3, 6], [9, 0, 17, 0, 0, 257], [0, 0, 1, 0, 0, 1]),
        )
        exercise(
            "live_Cu_length_residual_pages_empty_transition",
            inputs,
            cu,
            lengths,
            rs,
            pages,
            mode=mode,
            ratio=2,
            topk=520,
            cores=1,
            graph_states=states,
            candidates=sparse_candidates(6, 257) if mode == "consumer" else None,
        )
        for topk, counts, lengths, heads, depth, page_size, cores in (
            (520, [33], [513], 32, 64, 16, 3),
            (2048, [6], [2113], 64, 128, 80, None),
        ):
            cu, pages, physical = metadata(counts, lengths, page_size, generator)
            inputs = random_inputs(
                sum(counts), physical, page_size, heads, depth, generator
            )
            exercise(
                "runtime_geometry_topk" + str(topk),
                inputs,
                cu,
                lengths,
                [0],
                pages,
                mode=mode,
                topk=topk,
                cores=cores,
                candidates=(
                    sparse_candidates(sum(counts), lengths[0])
                    if mode == "consumer"
                    else None
                ),
            )
    for all_zero in (True, False):
        length = (
            16393  # 2050 reachable blocks, exceeding the candidate output capacity.
        )
        cu, pages, physical = metadata([1], [length], 80, generator)
        q = torch.full((1, 16, 32), -128, dtype=torch.int8)
        k = torch.full((physical, 80, 32), -128, dtype=torch.int8)
        w = torch.zeros((1, 16), dtype=torch.float16)
        if not all_zero:
            w[0, 0] = 1
        ks = torch.ones((physical, 80), dtype=torch.float16)
        if not all_zero:
            logical = (torch.arange(length) // 8 + 1).float().div(2048).half()
            for position in range(length):
                ks[pages[0, position // 80], position % 80] = logical[position]
        exercise(
            "candidate_truncated_" + ("ties" if all_zero else "ordered"),
            (q, k, w, torch.ones_like(w), ks),
            cu,
            [length],
            [0],
            pages,
            mode="source",
            topk=8,
            cores=1,
        )
    # INT8 -128 and D128 produce a 2**21 dot: safe INT32, but a premature
    # INT32->FP16 conversion before division by1024 would overflow.
    q = torch.full((1, 16, 128), -128, dtype=torch.int8)
    k = torch.full((1, 16, 128), -128, dtype=torch.int8)
    w = torch.zeros((1, 16), dtype=torch.float16)
    w[0, 0] = 1
    ks = torch.arange(1, 17).half()[None] / 16
    exercise(
        "int8_extreme_dot_before_fp16",
        (q, k, w, torch.ones_like(w), ks),
        [0, 1],
        [9],
        [0],
        torch.tensor([[0]], dtype=torch.int32),
        cores=1,
    )
    empty = random_inputs(0, 1, 16, 16, 32, generator)
    exercise(
        "empty_request_batch",
        empty,
        [0],
        [],
        [],
        torch.empty((0, 2), dtype=torch.int32),
        topk=8,
    )
    # Invalid *used* CPU page IDs must fail before a kernel is launched. Device
    # metadata owners remain responsible for their documented validity bounds.
    for bad in (-1, 999):
        try:
            plan = QLIPlan(
                [0, 1],
                [1],
                [[bad]],
                heads=16,
                depth=32,
                topk=8,
                page_size=16,
                library_dir=args.library_dir,
            )
            plan.run(
                *(
                    value[:1].npu()
                    for value in random_inputs(1, 1, 16, 16, 32, generator)
                )
            )
        except (ValueError, RuntimeError):
            records.append(
                {"case": "invalid_used_page", "page": bad, "status": "rejected"}
            )
        else:
            raise AssertionError("invalid used physical page accepted")
    report = {
        "library": str(Path(args.library_dir).resolve()),
        "records": records,
        "precision_witnesses": {name: info for name, _, info in witnesses},
        "reference": (
            "Exact integer dots in FP64; explicit FP16 weight/activation storage; "
            "FP32 head-reduction boundary and Kscale; tie-aware ranks, no vendor oracle."
        ),
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", len(records), "independent QLI contract checks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    witnesses = precision_witnesses()
    if args.reference_only:
        print(json.dumps({name: info for name, _, info in witnesses}, indent=2))
    else:
        if args.library_dir is None:
            parser.error("--library-dir is required for device checks")
        run(args, witnesses)
