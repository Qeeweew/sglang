"""Single-910B checks for native Markov sampling and its Torch fallback."""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep


def step_inputs(rows, width, padded):
    return dict(
        base_logits=torch.zeros((rows, padded), dtype=torch.bfloat16, device="npu"),
        bias=torch.zeros((rows, width), dtype=torch.bfloat16, device="npu"),
        temperatures=torch.zeros(rows, device="npu"),
        key=42,
        offset=torch.zeros(1, dtype=torch.int64, device="npu"),
        step=0,
        step_dtype=torch.float32,
        has_sampled_rows=False,
    )


def pack(scores, ids):
    return torch.stack((scores, ids.contiguous().view(torch.float32)), dim=-1).reshape(
        -1, 2
    )


@pytest.mark.parametrize("ranks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_rank_major_candidates_replay(ranks, strided):
    inf, nan = float("inf"), float("nan")
    invalid = torch.iinfo(torch.int32).max
    scores = torch.tensor(
        [
            [rank, -rank, 0, -inf, nan if rank == 0 else inf, nan, nan, inf]
            for rank in range(ranks)
        ]
    )
    ids = torch.tensor(
        [
            [
                rank,
                rank,
                ranks - rank,
                ranks - rank,
                10 + rank,
                ranks - rank,
                invalid if rank else 0,
                invalid,
            ]
            for rank in range(ranks)
        ],
        dtype=torch.int32,
    )
    expected = torch.tensor([ranks - 1, 0, 1, 1, 10, 1, 0, invalid])
    packet = pack(scores, ids)
    candidates = torch.empty(packet.shape[0], 4 if strided else 2, device="npu")
    if strided:
        candidates = candidates[:, ::2]
    candidates.copy_(packet)
    inputs = step_inputs(scores.shape[1], 8, 8)
    inputs["gather"] = lambda _: candidates
    torch.testing.assert_close(MarkovSampleStep.execute(**inputs).cpu(), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream(), auto_dispatch_capture=True):
        result = MarkovSampleStep.execute(**inputs)
    graph.replay()
    torch.testing.assert_close(result.cpu(), expected)
    candidates.copy_(pack(scores.flip(1), ids.flip(1)))
    graph.replay()
    torch.testing.assert_close(result.cpu(), expected.flip(0))


@pytest.mark.parametrize(
    "ranks,rows", [(1, 0), (1, 1), (3, 9), (8, 17), (32, 4), (33, 5), (65, 333)]
)
def test_native_merge_bit_patterns(ranks, rows):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    ops = load_library()
    ops.precompile_dspark_markov()
    stream = torch.npu.Stream()
    candidates = torch.empty(ranks, rows, 2, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        actual = ops.markov_merge_candidates(candidates)
    specials = torch.tensor(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0xFF810001,
            0x7F7FFFFF,
            0xFF7FFFFF,
        ]
    )
    invalid = (1 << 31) - 1
    for seed in (17, 91, 103):
        rng = torch.Generator().manual_seed(seed)
        bits = specials[torch.randint(len(specials), (ranks, rows), generator=rng)]
        ids = torch.randint(0, invalid, (ranks, rows), generator=rng)
        ids[:, ::7] = invalid
        ids[1::2] |= 1 << 31
        if rows >= 5:
            bits[:, 1:5] = torch.tensor([0x80000000, 0xFF800000, 0x7F800000, 0])
            bits[-1, 1:5] = torch.tensor([1, 0xFF7FFFFF, 0xFF810001, 0x80000000])
            ids[-1, 4] = 0
        signed_ids = ids & invalid
        ordered = (
            bits.int()
            .view(torch.float32)
            .double()
            .nan_to_num(nan=2e300, posinf=1e300, neginf=-1e300)
        )
        ordered[signed_ids == invalid] = -float("inf")
        winners = ordered == ordered.amax(dim=0)
        expected = torch.where(winners, signed_ids, invalid).amin(dim=0)
        host = torch.stack((bits, ids), -1).int().view(torch.float32)
        with torch.npu.stream(stream):
            candidates.copy_(host)
            graph.replay()
            eager = ops.markov_merge_candidates(candidates)
            for output in (actual, eager):
                torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "strided,available,want_native",
    [(False, True, True), (True, True, False), (False, False, False)],
)
def test_markov_merge_dispatch(monkeypatch, strided, available, want_native):
    from sglang.srt.hardware_backend.npu.dsv41.dspark import markov

    calls = []
    _, native = markov._operators()
    assert native is not None, "rebuild with the dspark_markov component"

    def tracked(packet):
        calls.append(tuple(packet.shape))
        return native(packet)

    monkeypatch.setattr(
        markov, "_operators", lambda: (None, tracked if available else None)
    )
    inputs = step_inputs(3, 7, 8)
    inputs["has_sampled_rows"] = True
    inputs["temperatures"][1] = 0.5
    inputs["logits_out"] = torch.empty_like(inputs["base_logits"], dtype=torch.float32)

    def gather(partial):
        packet = partial.repeat(5, 1)
        if strided:
            packet = packet.repeat_interleave(2, dim=1)[:, ::2]
        return packet

    inputs["gather"] = gather
    expected = MarkovSampleStep.torch(**inputs)
    torch.testing.assert_close(MarkovSampleStep.execute(**inputs), expected)
    assert calls == ([(5, 3, 2)] if want_native else [])


def test_empty_rows_preserve_gather():
    calls = []

    def gather(packet):
        calls.append(tuple(packet.shape))
        return packet

    inputs = step_inputs(0, 7, 8)
    result = MarkovSampleStep.execute(**inputs, gather=gather)
    assert calls == [(0, 2)]
    torch.testing.assert_close(result.cpu(), torch.empty(0, dtype=torch.int64))


@pytest.mark.parametrize("with_peer", [False, True])
@pytest.mark.parametrize("width", [0, 7, 8])
def test_shard_padding(width, with_peer):
    rows = 2
    inputs = step_inputs(rows, width, 8)
    inputs["base_logits"][:, -1] = 100
    peer = pack(
        torch.full((rows,), -1.0), torch.arange(rows, dtype=torch.int32) + 8
    ).to("npu")
    if with_peer:
        inputs["gather"] = lambda partial: torch.cat((partial, peer))
    expected = (
        torch.arange(rows) + 8
        if width == 0
        else torch.full((rows,), 7 if width == 8 else 0)
    )
    if width == 0 and not with_peer:
        expected = torch.full((rows,), torch.iinfo(torch.int32).max)
    torch.testing.assert_close(MarkovSampleStep.execute(**inputs).cpu(), expected)


@pytest.mark.parametrize("width", [0, 1, 63, 64, 65, 2047, 2048, 2049, 64640])
@pytest.mark.parametrize("base_dtype,bias_dtype", [
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16), (torch.float32, torch.float32),
])
def test_local_packet(width, base_dtype, bias_dtype):
    from sglang.srt.hardware_backend.npu.dsv41.dspark.markov import _operators

    local, _ = _operators()
    rows, start = 7, 64640
    base = torch.empty(rows, 3, width + 5, device="npu", dtype=base_dtype)[:, 1, 1:]
    bias = torch.empty(rows, width + 7, device="npu", dtype=bias_dtype)[:, 2:width+2]
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        local(base, bias, start)
        with torch.npu.graph(graph := torch.npu.NPUGraph(), stream=stream, auto_dispatch_capture=True):
            output = local(base, bias, start)
        for seed in (17, 101):
            rng = torch.Generator().manual_seed(seed)
            raw = torch.randn((rows, width + 4), generator=rng).to(base_dtype)
            delta = torch.randn((rows, width), generator=rng).to(bias_dtype)
            raw[:, width:] = float("inf")
            if width:
                raw[1, :width] = -float("inf")
                raw[2, width-1] = float("nan")
                raw[3, min(64, width-1)] = raw[3, width-1] = float("inf")
                raw[4, :width] = -0.0
                delta[4] = -0.0
                raw[5, min(65, width-1)] = raw[5, width-1] = float("nan")
                raw[6, width-1], delta[6, width-1] = float("inf"), -float("inf")
            base.copy_(raw)
            bias.copy_(delta)
            graph.replay()
            if width:
                scores = raw[:, :width].float() + delta.float()
                ids = scores.argmax(dim=1)
                expected = scores.gather(1, ids[:, None])[:, 0]
                ids = (ids + start).int()
            else:
                expected = torch.full((rows,), -float("inf"))
                ids = torch.full((rows,), 2**31-1, dtype=torch.int32)
            host = output.cpu()
            torch.testing.assert_close(host[:, 0], expected, atol=0, rtol=0, equal_nan=True)
            torch.testing.assert_close(host[:, 1].contiguous().view(torch.int32), ids)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
