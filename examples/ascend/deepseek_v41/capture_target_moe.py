"""Capture target gate inputs and logits for single-rank MoE benchmarks."""

import json
from pathlib import Path

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
