"""Independent RoPE/INT8 oracle, native K comparison and Graph/cache checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.position_quant import (
    PositionQuant,
    RopeTables,
)
from dsv41.kernels.references.position_quant import (
    checked_k,
    k_reference,
    q_reference,
    rope_reference,
    scatter_reference,
)


def checked_equal(actual, expected, name):
    actual = actual.cpu()
    if not torch.equal(actual, expected):
        mismatch = actual != expected
        raise AssertionError(
            (name, int(mismatch.sum()), actual[mismatch][:12], expected[mismatch][:12])
        )


def guarded(shape, dtype, fill):
    n = 1
    for v in shape:
        n *= v
    storage = torch.full((n + 256,), fill, dtype=dtype, device="npu")
    return storage[:n].view(shape), storage, n, fill


def check_guard(item):
    _, storage, n, fill = item
    assert bool((storage[n:].cpu() == fill).all()), "output canary modified"


def numeric_cases(op):
    count = 0
    for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257):
        phase = torch.randn((m, 32))
        c = phase.cos() * 1.17
        s = phase.sin() * 1.17
        cd, sd = c.npu(), s.npu()
        for h, d in ((64, 512), (32, 128), (1, 512), (1, 128)):
            x = torch.randn((m, h, d)).bfloat16()
            xd = x.npu()
            for inverse in (False, True):
                guard = guarded(x.shape, torch.bfloat16, -321)
                y = op.rope(xd, cd, sd, inverse=inverse, out=guard[0])
                checked_equal(y, rope_reference(x, c, s, inverse), "rope")
                check_guard(guard)
                # Explicit in-place path and exact prefix copy.
                xx = xd.clone()
                op.rope(xx, cd, sd, inverse=inverse, out=xx)
                checked_equal(xx, rope_reference(x, c, s, inverse), "rope inplace")
                count += 1
        q = torch.randn((m, 32, 128)).bfloat16()
        k = torch.randn((m, 128)).bfloat16()
        for fused in (False, True):
            kwargs = {"cos": cd, "sin": sd} if fused else {}
            for query, x, ref in ((True, q, q_reference), (False, k, k_reference)):
                code_guard = guarded(x.shape, torch.int8, -111)
                scale_guard = guarded(
                    x.shape[:-1], torch.float16 if query else torch.float32, -321
                )
                quant = op.quantize_query if query else op.quantize_key
                actual, scale = quant(
                    x.npu(), out=code_guard[0], scale_out=scale_guard[0], **kwargs
                )
                xr = rope_reference(x, c, s) if fused else x
                expected, expected_scale = ref(xr)
                checked_equal(scale, expected_scale, "scales")
                if not query and m:
                    nq, ns = torch_npu.npu_dynamic_quant(xr.npu(), dst_type=torch.int8)
                    checked_equal(actual, nq.cpu(), "native K codes")
                    checked_equal(scale, ns.cpu(), "native K scale")
                if query:
                    checked_equal(actual, expected, "Q codes")
                else:
                    checked_k(actual, expected, xr)
                check_guard(code_guard)
                check_guard(scale_guard)
                count += 1
        print(
            json.dumps({"numeric_M": m, "cases": count, "status": "pass"}), flush=True
        )
    return count


def edge_quantization(op):
    x = torch.zeros((2, 32, 128), dtype=torch.bfloat16)
    x[0, 1, 0] = 127
    x[0, 1, 1:10] = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 126.5])
    for h, value in enumerate(
        (2**-133, 2**-126, 1e-20, 2**-24, 2**-10, 1.0, 127.0, 1e20)
    ):
        x[1, h].fill_(value)
    q, s = op.quantize_query(x.npu())
    rq, rs = q_reference(x)
    checked_equal(q, rq, "Q zero/ties/extremes")
    checked_equal(s, rs, "Q scales zero/ties/extremes")
    k = x.reshape(-1, 128).contiguous()
    q, s = op.quantize_key(k.npu())
    rq, rs = k_reference(k)
    checked_k(q, rq, k)
    checked_equal(s, rs, "K scales zero/ties/extremes")
    nq, ns = torch_npu.npu_dynamic_quant(k.npu(), dst_type=torch.int8)
    checked_equal(q, nq.cpu(), "K native edge codes")
    checked_equal(s, ns.cpu(), "K native edge scales")
    print(
        json.dumps(
            {"quant_edges": "zero/RNE ties/BF16 subnormal/large", "status": "pass"}
        ),
        flush=True,
    )


def scatter_cases(op):
    for m in (0, 1, 6, 8, 33, 257):
        for d in (128, 512):
            values = torch.randn((m, d)).bfloat16()
            base = torch.randn((7, 16, d)).bfloat16()
            slots = torch.randint(-8, 120, (m,), dtype=torch.int64)
            if m > 2:
                slots[:3] = torch.tensor([0, 1, 1])
            buf = guarded(base.shape, torch.bfloat16, -321)
            buf[0].copy_(base)
            op.scatter_bf16(values.npu(), slots.npu(), buf[0])
            checked_equal(
                buf[0], scatter_reference(base, values, slots), "BF16 scatter"
            )
            check_guard(buf)
        values = torch.randint(-127, 128, (m, 128), dtype=torch.int8)
        scales = torch.rand((m,), dtype=torch.float32) * 0.02
        base = torch.randint(-127, 128, (7, 16, 128), dtype=torch.int8)
        sb = torch.rand((7, 16)).half()
        cb = guarded(base.shape, torch.int8, -111)
        cs = guarded(sb.shape, torch.float16, -321)
        cb[0].copy_(base)
        cs[0].copy_(sb)
        op.scatter_index(values.npu(), scales.npu(), slots.npu(), cb[0], cs[0])
        checked_equal(cb[0], scatter_reference(base, values, slots), "INT8 scatter")
        expected = scatter_reference(
            sb[..., None], scales.half()[:, None], slots
        ).squeeze(-1)
        checked_equal(cs[0], expected, "FP16 scale scatter")
        check_guard(cb)
        check_guard(cs)
    print(
        json.dumps(
            {
                "scatter": "M0/1/6/8/33/257; negative/OOB/duplicates/neighbors; canaries",
                "status": "pass",
            }
        ),
        flush=True,
    )


def cache_gather_cases(op):
    for length in (0, 17):
        cc = torch.randn(length, 32)
        sc = torch.randn(length, 32)
        for m in (0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 257):
            pos = torch.randint(-3, length + 4, (m,), dtype=torch.int64)
            slots = torch.arange(m, dtype=torch.int64) - 1
            cg = guarded((m, 32), torch.float32, -321)
            sg = guarded((m, 32), torch.float32, -321)
            vg = guarded((m,), torch.int32, -321)
            pg = guarded((m,), torch.int64, -321)
            out = RopeTables(cg[0], sg[0], vg[0], pg[0])
            result = op.gather_cos_sin(
                cc.npu(), sc.npu(), pos.npu(), slots=slots.npu(), out=out
            )
            valid = (pos >= 0) & (pos < length)
            c = torch.ones(m, 32)
            s = torch.zeros(m, 32)
            if bool(valid.any()):
                c[valid] = cc[pos[valid]]
                s[valid] = sc[pos[valid]]
            checked_equal(result.cos, c, "cache cos")
            checked_equal(result.sin, s, "cache sin")
            checked_equal(result.valid, valid.int(), "cache validity")
            checked_equal(
                result.slots, torch.where(valid, slots, -1), "cache safe slots"
            )
            for item in (cg, sg, vg, pg):
                check_guard(item)
    # Explicitly reject logically-contiguous FRACTAL_NZ storage.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    try:
        torch.npu.config.allow_internal_format = True
        nz = torch_npu.empty_with_format(
            (6, 128), dtype=torch.bfloat16, device="npu", acl_format=29
        )
    finally:
        if isinstance(previous, bytes):
            previous = previous.decode()
        torch_npu._C._npu_setOption({"ALLOW_INTERNAL_FORMAT": previous or "disable"})
    assert torch_npu.get_npu_format(nz) == 29
    try:
        op.quantize_key(nz)
    except ValueError as exc:
        assert "format" in str(exc)
    else:
        raise AssertionError("format29 input accepted")
    print(
        json.dumps(
            {
                "cos_sin_cache": "T0/1..8/33/257, L0/17, invalid identity/mask/slots, canaries, format29 reject",
                "status": "pass",
            }
        ),
        flush=True,
    )


def cache_gather_graph(op):
    cc = torch.randn(17, 32, device="npu")
    sc = torch.randn(17, 32, device="npu")
    positions = torch.tensor([0, 1, -1, 16, 17, 4], dtype=torch.int64, device="npu")
    slots = torch.arange(6, dtype=torch.int64, device="npu")
    tables = RopeTables(
        torch.empty(6, 32, device="npu"),
        torch.empty(6, 32, device="npu"),
        torch.empty(6, dtype=torch.int32, device="npu"),
        torch.empty(6, dtype=torch.int64, device="npu"),
    )
    key = torch.randn(6, 128).bfloat16().npu()
    code = torch.empty(6, 128, dtype=torch.int8, device="npu")
    scale = torch.empty(6, device="npu")
    kc = torch.full((1, 16, 128), -31, dtype=torch.int8, device="npu")
    ks = torch.full((1, 16), -31.0, dtype=torch.float16, device="npu")

    def enqueue():
        op.gather_cos_sin(cc, sc, positions, slots=slots, out=tables)
        op.quantize_key(key, cos=tables.cos, sin=tables.sin, out=code, scale_out=scale)
        op.scatter_index(code, scale, tables.slots, kc, ks)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            enqueue()
    stream.synchronize()
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g, stream=stream):
        enqueue()
    for p in ([0, 1, -1, 16, 17, 4], [-1, -1, -1, -1, -1, -1], [16, 2, 1, 0, -1, 18]):
        positions.copy_(torch.tensor(p, dtype=torch.int64))
        key.copy_(torch.randn(6, 128).bfloat16())
        cc.copy_(torch.randn(17, 32))
        sc.copy_(torch.randn(17, 32))
        kc.fill_(-31)
        ks.fill_(-31)
        torch.npu.synchronize()
        g.replay()
        torch.npu.synchronize()
        valid = (positions.cpu() >= 0) & (positions.cpu() < 17)
        checked_equal(tables.valid, valid.int(), "Graph cache validity")
        expected_c = torch.ones(6, 32)
        expected_s = torch.zeros(6, 32)
        expected_c[valid] = cc.cpu()[positions.cpu()[valid]]
        expected_s[valid] = sc.cpu()[positions.cpu()[valid]]
        checked_equal(tables.cos, expected_c, "Graph cached cos")
        checked_equal(tables.sin, expected_s, "Graph cached sin")
        safe = torch.where(valid, slots.cpu(), -1)
        checked_equal(tables.slots, safe, "Graph masked slots")
        expected = scatter_reference(torch.full_like(kc.cpu(), -31), code.cpu(), safe)
        checked_equal(kc, expected, "Graph invalid positions cannot mutate K cache")
        expected_scales = scatter_reference(
            torch.full((*ks.shape, 1), -31.0, dtype=torch.float16),
            scale.cpu().half()[:, None],
            safe,
        ).squeeze(-1)
        checked_equal(
            ks, expected_scales, "Graph invalid positions cannot mutate scale cache"
        )
    print(
        json.dumps(
            {
                "graph": "device RoPE positions/cache -> fused K -> safe-slot scatter; invalid rows leave cache unchanged",
                "status": "pass",
            }
        ),
        flush=True,
    )


def graph_stream(op):
    m = 6
    main = torch.randn(m, 512).bfloat16().npu()
    q = torch.randn(m, 32, 128).bfloat16().npu()
    k = torch.randn(m, 128).bfloat16().npu()
    c = torch.randn(m, 32, device="npu")
    s = torch.randn(m, 32, device="npu")
    slots = torch.tensor([0, 1, -1, 95, 96, 1], dtype=torch.int64, device="npu")
    mr = torch.empty_like(main)
    qi = torch.empty_like(q, dtype=torch.int8)
    qs = torch.empty((m, 32), dtype=torch.float16, device="npu")
    ki = torch.empty_like(k, dtype=torch.int8)
    ks = torch.empty((m,), dtype=torch.float32, device="npu")
    main_cache = torch.full((6, 16, 512), -31.0, dtype=torch.bfloat16, device="npu")
    key_cache = torch.full((6, 16, 128), -31, dtype=torch.int8, device="npu")
    scale_cache = torch.full((6, 16), -31.0, dtype=torch.float16, device="npu")

    def enqueue():
        op.rope(main, c, s, out=mr)
        op.quantize_query(q, cos=c, sin=s, out=qi, scale_out=qs)
        op.quantize_key(k, cos=c, sin=s, out=ki, scale_out=ks)
        op.scatter_bf16(mr, slots, main_cache)
        op.scatter_index(ki, ks, slots, key_cache, scale_cache)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            enqueue()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        enqueue()
    for generation in range(3):
        for x in (main, q, k):
            x.copy_(torch.randn(x.shape).bfloat16())
        c.copy_(torch.randn(m, 32))
        s.copy_(torch.randn(m, 32))
        slots.copy_(torch.tensor([generation, 1, 1, -1, 95, 96], dtype=torch.int64))
        main_cache.fill_(-31)
        key_cache.fill_(-31)
        scale_cache.fill_(-31)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        cm, sm = c.cpu(), s.cpu()
        mm = rope_reference(main.cpu(), cm, sm)
        qr = q_reference(rope_reference(q.cpu(), cm, sm))
        kr = k_reference(rope_reference(k.cpu(), cm, sm))
        checked_equal(mr, mm, "Graph main RoPE")
        checked_equal(qi, qr[0], "Graph Q")
        checked_equal(qs, qr[1], "Graph Q scale")
        checked_k(ki, kr[0], rope_reference(k.cpu(), cm, sm))
        checked_equal(ks, kr[1], "Graph K scale")
        native_codes, native_scales = torch_npu.npu_dynamic_quant(
            rope_reference(k.cpu(), cm, sm).npu(), dst_type=torch.int8
        )
        checked_equal(ki, native_codes.cpu(), "Graph native K")
        checked_equal(ks, native_scales.cpu(), "Graph native K scale")
        checked_equal(
            main_cache,
            scatter_reference(torch.full_like(main_cache.cpu(), -31), mm, slots.cpu()),
            "Graph main cache",
        )
        checked_equal(
            key_cache,
            scatter_reference(
                torch.full_like(key_cache.cpu(), -31), ki.cpu(), slots.cpu()
            ),
            "Graph K cache",
        )
        expected_scale = scatter_reference(
            torch.full((*scale_cache.shape, 1), -31.0, dtype=torch.float16),
            ks.cpu().half()[:, None],
            slots.cpu(),
        ).squeeze(-1)
        checked_equal(scale_cache, expected_scale, "Graph scale cache")
    print(
        json.dumps(
            {
                "graph": "nondefault stream; RoPE -> Q/K quant -> both cache writes; input/trig/slot replay",
                "status": "pass",
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physical", type=int, required=True)
    p.add_argument("--library-dir", type=Path, default=Path("build/dsv41"))
    args = p.parse_args()
    import os

    assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.physical)
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    torch.manual_seed(1103)
    op = PositionQuant(library_dir=args.library_dir.resolve())
    count = numeric_cases(op)
    edge_quantization(op)
    scatter_cases(op)
    graph_stream(op)
    cache_gather_cases(op)
    cache_gather_graph(op)
    import gc
    import weakref

    temporary = torch.randn(33, 64, 512, dtype=torch.bfloat16, device="npu")
    borrowed = weakref.ref(temporary)
    cos = torch.ones(33, 32, device="npu")
    sin = torch.zeros_like(cos)
    result = op.rope(temporary, cos, sin)
    del temporary
    gc.collect()
    assert borrowed() is None, "position operator retained an eager activation"
    torch.npu.synchronize()
    assert bool(torch.isfinite(result).all().cpu())
    print(json.dumps({"numeric_cases": count, "status": "pass"}), flush=True)


if __name__ == "__main__":
    main()
