"""Single-910B ablation on captured target gate inputs and logits, with fixed routing."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def gate_hook(config):
    from safetensors import safe_open

    directory = Path(config["directory"])
    marker = directory / "collect"
    model = Path(config["model"])
    known = {}
    cache = {}
    if torch.distributed.get_rank() == 0:
        index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer in config["layers"]:
            key = f"layers.{layer}.ffn.gate.bias"
            with safe_open(model / index[key], framework="pt", device="cpu") as shard:
                known[shard.get_tensor(key).float().numpy().tobytes()] = layer
        if len(known) != len(config["layers"]):
            raise ValueError("ambiguous gate biases")

    def capture(module, inputs, logits):
        if torch.distributed.get_rank() != 0 or not marker.exists() or len(logits) < 16:
            return
        if module not in cache:
            bias = module.e_score_correction_bias
            cache[module] = (
                known.get(bias.detach().float().cpu().numpy().tobytes())
                if bias is not None
                else None
            )
        layer = cache[module]
        if layer is None:
            return
        label = marker.read_text().strip()
        dest = directory / f"ffn-{label}-{layer}.pt"
        if not dest.exists():
            torch.save((inputs[0].detach().cpu(), logits.detach().cpu()), dest)

    return capture


def load_methods(read, layer_id, experts, hidden, width, limit):
    from sglang.srt.hardware_backend.npu.dsv41.moe import ResidentMXFP4Method

    methods = []
    for a8 in (False, True):
        method = ResidentMXFP4Method(a8=a8)
        layer = torch.nn.Module()
        layer.layer_id = layer_id
        layer._expert_storage_rank = 0
        layer.moe_tp_rank = 0
        layer.num_fused_shared_experts = 0
        layer.moe_ep_size = layer.moe_tp_size = 1
        layer._num_global_routed = len(experts)
        method.create_weights(
            layer,
            len(experts),
            hidden,
            width,
            torch.float16,
            moe_intermediate_size=width,
        )
        method.create_moe_runner(
            layer,
            SimpleNamespace(
                is_gated=True,
                activation="silu",
                apply_router_weight_on_input=False,
                swiglu_limit=limit,
            ),
        )
        methods.append(method)
    for local, global_id in enumerate(experts):
        for projection in ("w1", "w3", "w2"):
            for kind in ("weight", "scale"):
                key = f"layers.{layer_id}.ffn.experts.{global_id}.{projection}.{kind}"
                raw = read(key)
                for method in methods:
                    if not method.weight_loader(None, raw, key, projection, local):
                        raise ValueError(f"missing expert {key}")
    for method in methods:
        method.process_weights_after_loading(method.layer)
    return [method.local_experts for method in methods]


def relative(ref, value):
    ref, value = ref.flatten(1).double(), value.flatten(1).double()
    norm = ref.norm(dim=1)
    error = (ref - value).norm(dim=1)
    if not torch.isfinite(error).all() or (norm == 0).any():
        raise ValueError("invalid reference or observed output")
    per = (error / norm).numpy()
    return dict(
        pooled_pct=float(100 * error.norm() / norm.norm()),
        p50_p99_max_pct=(100 * np.quantile(per, [0.5, 0.99, 1])).tolist(),
    )


def token_quant_error(value, indices):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.mxfp4_a8 import (
        quantize_tokens,
    )

    codes, scales = quantize_tokens(value)
    restored = codes.cpu().float() * scales.cpu()[:, None]
    return relative(value.cpu()[indices], restored[indices])


def replay(read, layer_id, args, config, documents):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.moe_routes import MoERoutes
    from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.router import Router

    inputs, logits, labels = [], [], []
    for doc in documents:
        features, scores = torch.load(
            args.directory / f"ffn-{doc['kind']}-{layer_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        valid = len(doc["ids"])
        if len(features) < valid or len(scores) != len(features):
            raise ValueError(
                f"capture does not cover DP0 request: {features.shape}, {valid}"
            )
        picks = np.linspace(0, valid - 1, min(valid, args.tokens_per_doc), dtype=int)
        inputs.append(features[picks])
        logits.append(scores[picks])
        labels.extend((doc["kind"], int(pos)) for pos in picks)
    features = torch.cat(inputs).npu()
    logits = torch.cat(logits).float().npu()
    router = Router()
    routes = MoERoutes()
    bias = read(f"layers.{layer_id}.ffn.gate.bias").float().npu()
    ids, coefficients = router(logits, bias, factor=config["routed_scaling_factor"])
    experts, mapped = torch.unique(ids.cpu().long(), sorted=True, return_inverse=True)
    route = router.bucket(mapped.int().npu(), coefficients, 0, len(experts))
    source, inverse = routes.gather(features, route)
    hidden = config["hidden_size"]
    width = config["moe_intermediate_size"]
    limit = config["swiglu_limit"]
    projections = load_methods(read, layer_id, experts.tolist(), hidden, width, limit)
    positions = inverse.cpu().long().reshape(len(labels), -1)
    torch.testing.assert_close(
        torch.sort(positions.flatten()).values, torch.arange(ids.numel())
    )
    after, combined = {}, {}
    input_error = {"gate_up": token_quant_error(source, positions[:, 0])}
    for gate_id, gmm in enumerate(projections):
        gate_up = source.new_empty((len(source), 2 * width))
        mid = source.new_empty((len(source), width))
        gmm.gate_up.bind(source, *gmm.gate_up_weight, route.metadata, out=gate_up)()
        gmm.activation_op.bind(gate_up, out=mid, rows=route.metadata[1:2])()
        after[gate_id] = mid.cpu()[positions]
        if gate_id == 0:
            input_error["down"] = token_quant_error(mid, positions)
        for down_id, down in enumerate(projections):
            values = torch.empty_like(source)
            down.down.bind(mid, *down.down_weight, route.metadata, out=values)()
            combined[(gate_id, down_id)] = routes.combine(values, inverse, route).cpu()
    result = dict(
        layer=layer_id,
        tokens=len(features),
        routes=ids.numel(),
        distinct_experts=len(experts),
        activation_quantization=input_error,
        swiglu=relative(after[0], after[1]),
    )
    for key, mode in (((1, 0), "gate_only"), ((0, 1), "down_only"), ((1, 1), "both")):
        result[mode] = relative(combined[0, 0], combined[key])
        worst = (
            (combined[0, 0].float() - combined[key].float()).norm(dim=1)
            / combined[0, 0].float().norm(dim=1)
        ).argmax()
        result[mode]["worst_doc_position"] = labels[worst]
    return result


def main():
    import gc
    import torch_npu  # noqa: F401
    from safetensors import safe_open
    from sglang.srt.hardware_backend.npu.dsv41.checkpoint import ExpertCheckpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 19, 39])
    parser.add_argument("--tokens-per-doc", type=int, default=64)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    documents = json.loads((args.directory / "corpus.json").read_text())["docs"][:4]
    if len({doc["kind"] for doc in documents}) != 4:
        raise ValueError("expected one capture per domain")
    with torch.inference_mode(), ExitStack() as stack:
        shards = {}
        checkpoint = ExpertCheckpoint(args.model)

        def read(name):
            if ".experts." in name:
                return checkpoint._read_expert_tensor(name)
            path = checkpoint.index[name]
            if path not in shards:
                shards[path] = stack.enter_context(
                    safe_open(args.model / path, framework="pt", device="cpu")
                )
            return shards[path].get_tensor(name)

        for layer_id in args.layers:
            print(
                json.dumps(replay(read, layer_id, args, config, documents)), flush=True
            )
            gc.collect()


if __name__ == "__main__":
    main()
