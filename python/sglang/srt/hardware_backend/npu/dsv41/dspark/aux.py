"""DSpark mean/packing fusion and prompt-residual lifetime boundaries."""

from __future__ import annotations

from functools import wraps

import torch
from sglang.srt.hardware_backend.npu.dsv41.mhc import (
    prefill_fusion_state,
    prefill_fusions_enabled,
)

from ..policy import enabled


def capture_aux(model, residual, layer_id):
    state = prefill_fusion_state()
    if state is None:
        return residual.mean(dim=1)
    captures = model.dspark_layers_to_capture
    index, width = captures.index(layer_id), residual.shape[-1]
    if "aux" not in state:
        # Mean + packing write directly to this forward's independent output.
        # No storage from attention, MoE or an earlier forward is borrowed.
        state["aux"] = torch.empty(
            (residual.shape[0], len(captures) * width),
            dtype=residual.dtype,
            device=residual.device,
        )
    output = state["aux"]
    from sglang.srt.hardware_backend.npu.dsv41.mhc import operator

    return operator().mean_packed(residual, output, index)


def adapt_prune(original):
    if not enabled():
        return original

    @wraps(original)
    def prune(
        processor,
        hidden_states,
        hidden_states_before_norm,
        aux_hidden_states,
        logits_metadata,
    ):
        if prefill_fusions_enabled() and logits_metadata.capture_hidden_mode.is_full():
            aux_hidden_states = None
        return original(
            processor,
            hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            logits_metadata,
        )

    return prune
