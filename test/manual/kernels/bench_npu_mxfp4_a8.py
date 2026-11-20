"""Single-910B paired Graph timings; include test/ascend in PYTHONPATH."""

import argparse
import json
from statistics import median

import numpy as np
import torch
import torch_npu  # noqa: F401
from dsv41.kernels.references.mxfp4_grouped import (
    a8_embedding,
    checkpoint,
    fixture,
    weight,
)
from dsv41.kernels.references.quant_packing import pack_experts
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import mxfp4_a8
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.quant_grouped_gemm import (
    NativeGroupedGemm,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


def audit(codes, exps, depth, first_expert, projection, layer):
    total = changed = 0
    worst = (0.0, None)
    for expert, (raw, exponents) in enumerate(zip(codes, exps), first_expert):
        total += raw.size * 2
        wide = np.flatnonzero(np.ptp(exponents, axis=1) > 3)
        if not wide.size:
            continue
        error = 0.0
        for start in range(0, wide.size, 32):
            rows = wide[start : start + 32]
            decoded = weight(raw[rows], exponents[rows], depth)
            difference = a8_embedding(decoded, exponents[rows]) - decoded
            changed += np.count_nonzero(difference)
            error += np.square(difference).sum()
        if error:
            norm = sum(
                np.square(
                    weight(
                        raw[start : start + 32], exponents[start : start + 32], depth
                    ),
                    dtype=np.float64,
                ).sum()
                for start in range(0, len(raw), 32)
            )
            relative = float(np.sqrt(error / norm))
            if relative > worst[0]:
                worst = (relative, expert)
    print(
        json.dumps(
            dict(
                layer=layer,
                projection=projection,
                weights=int(total),
                changed=int(changed),
                worst_rel_l2=worst[0],
                worst_expert=worst[1],
            )
        ),
        flush=True,
    )


def prepare_layer(args, projection, layer, depth, width):
    if args.model:
        codes, exps = checkpoint(
            args.model,
            range(args.first_expert, args.first_expert + args.experts),
            projection,
            layer,
        )
    else:
        codes, exps = fixture(args.experts, width, depth, seed=17 + layer)
    if args.audit_weights:
        audit(codes, exps, depth, args.first_expert, projection, layer)
        return
    original = pack_experts(
        zip(codes, exps), k=depth, n=width, count=args.experts, compact_scales=True
    )
    fp16_weights = [torch.from_numpy(value).npu() for value in original]
    prepared = [
        mxfp4_a8.pack_expert(
            torch.from_numpy(raw.copy()),
            torch.from_numpy(scale.copy()),
            allow_rounding=args.allow_rounding,
        )
        for raw, scale in zip(codes, exps)
    ]
    weights = tuple(torch.stack(items).npu() for items in zip(*prepared))
    reference = a8_embedding(
        weight(codes[0][::32], exps[0][::32], depth), exps[0][::32]
    )
    return fp16_weights, weights, reference


def benchmark(args, projection):
    depth, width = (5120, 4608) if projection == "gate_up" else (2304, 5120)
    layers = [
        prepare_layer(args, projection, layer, depth, width) for layer in args.layer
    ]
    if args.audit_weights:
        return
    g32_weights = []
    if args.g32_model:
        for layer in args.layer:
            raw, meta = checkpoint(
                args.g32_model,
                range(args.first_expert, args.first_expert + args.experts),
                projection,
                layer,
            )
            packed = [
                mxfp4_a8.pack_g32(
                    torch.from_numpy(c.copy()), torch.from_numpy(s.copy())
                )
                for c, s in zip(raw, meta, strict=True)
            ]
            g32_weights.append(
                tuple(torch.stack(parts).npu() for parts in zip(*packed, strict=True))
            )
    op = NativeGroupedGemm(depth, width, experts=args.experts, block_m=16)
    lib = load_library()
    scratch = torch.empty(
        lib.mxfp4_a8_scratch(depth, width, 0), dtype=torch.uint8, device="npu"
    )
    for active in args.active:
        for rows in args.rows:
            count = active * rows
            x = torch.randn(count, depth, dtype=torch.float16, device="npu")
            quant, scale = mxfp4_a8.quantize_tokens(x)
            out = x.new_empty((count, width))
            for rotate in args.rotate:
                routes = []
                for start in range(0, args.experts if rotate else 1, active):
                    route = [active, count, 0, 1] + [0] * (4 * args.experts)
                    for idx in range(active):
                        route[4 + 4 * idx : 8 + 4 * idx] = [
                            (start + idx) % args.experts,
                            idx * rows,
                            rows,
                            0,
                        ]
                    routes.append(torch.tensor(route, dtype=torch.int32, device="npu"))
                sequence = [(route, layer) for route in routes for layer in layers]
                baseline = [
                    op.bind(x, *fp16, route, out=out)
                    for route, (fp16, _, _) in sequence
                ]
                gemm = [
                    mxfp4_a8.bind(
                        quant, weights, scale, route, out=out, workspace=scratch
                    )
                    for route, (_, weights, _) in sequence
                ]
                for idx, (_, _, reference) in enumerate(layers):
                    gemm[idx]()
                    ref = (
                        quant[0].cpu().numpy().astype(np.float64) @ reference.T
                    ).astype(np.float32) * scale[0].item()
                    actual = out[0, ::32].cpu().numpy()
                    np.testing.assert_array_equal(actual, ref.astype(np.float16))

                def quant_gemm(index):
                    route, (_, weights, _) = sequence[index]
                    tokens, scales = mxfp4_a8.quantize_tokens(x)
                    lib.mxfp4_a8(tokens, *weights, scales, route, out, scratch)

                calls = {
                    "fp16": lambda index: baseline[index](),
                    "a8": lambda index: gemm[index](),
                    "quant_a8": quant_gemm,
                }
                if g32_weights:
                    transposed = projection == "down"
                    source = (
                        x.reshape(-1, 8, 32).transpose(1, 2).reshape_as(x)
                        if transposed
                        else x
                    )
                    tokens, scales = mxfp4_a8.h32_quant(source, transposed)
                    g32 = [
                        mxfp4_a8.bind(
                            tokens,
                            g32_weights[idx % len(layers)],
                            scales,
                            route,
                            out=out,
                            workspace=scratch,
                            g32=True,
                        )
                        for idx, (route, _) in enumerate(sequence)
                    ]

                    def rotated_gemm(index):
                        rotated, units = mxfp4_a8.h32_quant(source, transposed)
                        lib.int4_g32_a8(
                            rotated,
                            *g32_weights[index % len(layers)],
                            units,
                            sequence[index][0],
                            out,
                            scratch,
                        )

                    calls.update(
                        int4_g32=lambda index: g32[index](),
                        h32_a8_int4_g32=rotated_gemm,
                    )
                graphs, samples = {}, {name: [] for name in calls}
                launches = len(sequence) * max(1, args.repeats // len(sequence))
                for name, call in calls.items():
                    call(0)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(
                        graph, stream=torch.npu.Stream(), auto_dispatch_capture=True
                    ):
                        for idx in range(launches):
                            call(idx % len(sequence))
                    graphs[name] = graph
                begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                for iteration in range(args.rounds):
                    for offset in range(len(graphs)):
                        name = list(graphs)[(iteration + offset) % len(graphs)]
                        graphs[name].replay()
                        begin.record()
                        graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(begin.elapsed_time(end) * 1000 / launches)
                print(
                    json.dumps(
                        dict(
                            projection=projection,
                            layers=args.layer,
                            active=active,
                            rows=rows,
                            rotate=rotate,
                            ring=len(sequence),
                            launches=launches,
                            samples_us=samples,
                            median_us={
                                name: median(data) for name, data in samples.items()
                            },
                            bytes_per_expert=sum(
                                value[0].nbytes for value in layers[0][1]
                            ),
                            fp16_bytes_per_expert=sum(
                                value[0].nbytes for value in layers[0][0]
                            ),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument(
        "--g32-model", help="add paired timings using the exported G32 checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[0],
        help="visit these layers in order before advancing the expert route",
    )
    parser.add_argument("--first-expert", type=int, default=0)
    parser.add_argument(
        "--audit-weights",
        action="store_true",
        help="CPU only: count changed weights and worst per-matrix relative L2",
    )
    parser.add_argument(
        "--allow-rounding", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--projection",
        nargs="+",
        choices=["gate_up", "down"],
        default=["gate_up", "down"],
    )
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--active", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--rotate", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if not args.audit_weights and (
        torch.npu.device_count() != 1
        or not all(0 < active <= args.experts for active in args.active)
    ):
        parser.error("select one idle NPU and 0 < active <= experts")
    torch.manual_seed(31)
    with torch.inference_mode():
        for projection in args.projection:
            benchmark(args, projection)
