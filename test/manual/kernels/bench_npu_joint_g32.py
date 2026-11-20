"""Single-rank G32 or legacy MXFP4 expert stage; fixed top-6, excludes router and EP."""

import argparse
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from dsv41.kernels.references.mxfp4_grouped import checkpoint
from dsv41.kernels.references.quant_packing import pack_experts
from safetensors import safe_open
from sglang.srt.hardware_backend.npu.dsv41 import moe
from sglang.srt.hardware_backend.npu.dsv41.joint_experts import run_joint_experts
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import (
    FP8Weight,
    Router,
)
from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.int4_g32_a8 import pack_g32
from sglang.srt.hardware_backend.npu.dsv41.linear import (
    SharedW8A8Method,
    W8A16LinearMethod,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.model_loader.weight_utils import default_weight_loader


class SharedLinear(torch.nn.Module):
    def forward(self, x):
        return self.quant_method.apply(self, x), None


def load_layer(model, layer, experts, legacy=False):
    method = moe.ResidentMXFP4Method(g32=not legacy)
    method.h, method.i, method.e, method.ep, method.limit = 5120, 2304, experts, 8, 10.0
    weights = []
    for projection in ("gate_up", "down"):
        raw, metadata = checkpoint(model, range(experts), projection, layer)
        if legacy:
            width, depth = (
                (2 * method.i, method.h)
                if projection == "gate_up"
                else (method.h, method.i)
            )
            packed = pack_experts(
                zip(raw, metadata), n=width, k=depth, count=experts, compact_scales=True
            )
            weights.append(tuple(torch.from_numpy(value).npu() for value in packed))
            continue
        packed = [
            pack_g32(torch.from_numpy(code.copy()), torch.from_numpy(scale.copy()))
            for code, scale in zip(raw, metadata, strict=True)
        ]
        weights.append(
            tuple(torch.stack(parts).npu() for parts in zip(*packed, strict=True))
        )
    method.local_experts = method.bind_local_experts(*weights)
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(names, field):
        pieces = []
        for name in names:
            key = f"layers.{layer}.ffn.shared_experts.{name}.{field}"
            with safe_open(model / index[key], framework="pt") as shard:
                pieces.append(shard.get_tensor(key))
        return torch.cat(pieces)

    def shared_linear(names):
        if legacy:
            decoder = FP8Weight()
            matrices = [
                decoder.load_cpu(read([name], "weight"), read([name], "scale"))
                for name in names
            ]
            dense = torch.cat(matrices)
            width, depth = dense.shape
            layer = SharedLinear()
            layer.quant_method = W8A16LinearMethod()
            with torch.device(dense.device):
                layer.quant_method.create_weights(
                    layer,
                    depth,
                    [len(matrix) for matrix in matrices],
                    depth,
                    width,
                    torch.bfloat16,
                    weight_loader=default_weight_loader,
                )
            layer.weight.weight_loader(layer.weight, dense)
            layer.quant_method.process_weights_after_loading(layer)
            return layer
        return SimpleNamespace(
            weight=npu_format_cast(read(names, "weight").npu().T.contiguous()),
            weight_scale_inv=read(names, "scale").npu().flatten(),
            quant_method=SharedW8A8Method(interleaved=names == ["w2"]),
        )

    shared = SimpleNamespace(
        gate_up_proj=shared_linear(["w1", "w3"]), down_proj=shared_linear(["w2"])
    )
    return method, shared


def check_captures(directory, model, layer, method, shared):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = f"layers.{layer}.ffn.gate.bias"
    with safe_open(model / index[key], framework="pt") as shard:
        bias = shard.get_tensor(key).float().npu()
    config = json.loads((model / "config.json").read_text())
    factor = config.get("text_config", config)["routed_scaling_factor"]
    router = Router()
    for doc in json.loads((directory / "corpus.json").read_text())["docs"][:4]:
        source, logits = torch.load(
            directory / f"ffn-{doc['kind']}-{layer}.pt",
            weights_only=True,
            map_location="cpu",
        )
        valid = len(doc["ids"])
        assert (
            0 < valid <= len(source) == len(logits) and source.dtype == torch.bfloat16
        )
        picks = torch.linspace(0, valid - 1, min(valid, 64)).long()
        hidden = source[picks].npu()
        plan = router.route(
            logits[picks].float().npu(), bias, first=0, experts=method.e, factor=factor
        )
        old_routed = method.routed(hidden, plan).clone()
        gate = shared.gate_up_proj(hidden)[0]
        old_shared = shared.down_proj(
            torch.ops.npu.npu_clipped_swiglu(
                gate, alpha=1, limit=method.limit, bias=0, interleaved=False
            )
        )[0]
        routed, output = run_joint_experts(
            method, shared, hidden, plan, slice(None), False
        )
        torch.testing.assert_close(routed, old_routed, atol=0, rtol=0)
        assert torch.isfinite(output).all()
        delta = (output.float() - old_shared.float()).norm() / old_shared.float().norm()
        print(
            json.dumps(
                dict(
                    capture=doc["kind"],
                    tokens=len(picks),
                    routed_equal=True,
                    shared_relative_l2=delta.item(),
                )
            ),
            flush=True,
        )


def measure(method, shared, tokens, owned_rows, rounds, repeats):
    hidden = torch.randn(tokens, method.h).bfloat16().npu()
    owned = slice(tokens - owned_rows, tokens)
    generator = torch.Generator().manual_seed(24)
    ids = torch.stack(
        [torch.randperm(384, generator=generator)[:6] for _ in range(tokens)]
    )
    ids[0] = torch.tensor([0, 1, 64, 80, 120, 170])
    weights = torch.rand(tokens, 6, generator=generator).softmax(1).npu()
    plan = Router().bucket(ids.int().npu(), weights, 0, method.e)

    def split():
        up, down = shared.gate_up_proj, shared.down_proj
        gates = up.quant_method.apply(up, hidden[owned])
        mid = torch.ops.npu.npu_clipped_swiglu(
            gates, alpha=1, limit=10, bias=0, interleaved=False
        )
        dense = down.quant_method.apply(down, mid)
        return method.routed(hidden, plan), dense

    def joint():
        return run_joint_experts(method, shared, hidden, plan, owned, False)

    before, after = split(), joint()
    if not method.g32:
        from sglang.srt.hardware_backend.npu.dsv41.joint_experts import fp16_shared_gate

        gate = fp16_shared_gate(shared, hidden[owned].half())
        mid = moe.activation(method.i, prefix=False)(gate)
        torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
        torch.testing.assert_close(after[1], shared.down_proj(mid)[0], rtol=0, atol=0)
    differences = [
        (new.float() - old.float()).norm().item()
        / max(old.float().norm().item(), 1e-30)
        for old, new in zip(before, after, strict=True)
    ]
    graphs = {}
    for name, call in (("split", split), ("joint", joint)):
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            call()
        stream.synchronize()
        with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
            for _ in range(repeats):
                call()
        graphs[name] = graph
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    samples = {name: [] for name in graphs}
    for iteration in range(rounds):
        for name in list(graphs) if iteration % 2 else reversed(graphs):
            graph = graphs[name]
            graph.replay()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repeats)
    print(
        json.dumps(
            dict(
                tokens=tokens,
                shared_rows=owned_rows,
                routed_rows=plan.metadata[1].item(),
                relative_l2=differences,
                median_us={k: median(v) for k, v in samples.items()},
                samples_us=samples,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument(
        "--captures",
        type=Path,
        help="optional legacy BF16 router-input captures from capture_target_moe",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 8, 32])
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    if args.captures and not args.legacy:
        parser.error(
            "--captures compares the legacy target against its previous shared BF16 path"
        )
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(32)
    resident, dense = load_layer(args.model, args.layer, 48, args.legacy)
    if args.captures:
        check_captures(args.captures, args.model, args.layer, resident, dense)
    for tokens in args.tokens:
        measure(resident, dense, tokens, (tokens + 7) // 8, args.rounds, args.repeats)
