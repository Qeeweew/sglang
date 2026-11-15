"""Restore prepared V4.1 inputs for the merged encoder SWA replay span."""

from functools import wraps

import torch

from ..policy import enabled


def restore_replay_embeddings(view, runner):
    """Use absolute request coordinates, including rows before the KV hit."""
    if getattr(view, "input_embeds", None) is not None:
        pieces = []
        for req, start, count in zip(view.reqs, view.prefix_lens, view.extend_lens):
            source = req.input_embeds
            if source is None:
                raise ValueError("dense input embeddings must cover every request")
            piece = torch.as_tensor(
                source[start : start + count],
                dtype=view.input_embeds.dtype,
                device=runner.device,
            )
            if piece.shape[0] != count:
                raise ValueError("dense input embeddings do not cover the replay span")
            pieces.append(piece)
        view.input_embeds = torch.cat(pieces)

    values, positions = [], []
    offset = 0
    for req, start, count in zip(view.reqs, view.prefix_lens, view.extend_lens):
        overrides = req.positional_embed_overrides
        if overrides is not None:
            selected = [
                (i, offset + pos - start)
                for i, pos in enumerate(overrides.positions)
                if start <= pos < start + count
            ]
            if selected:
                indices, rows = zip(*selected)
                values.append(overrides.embeds[list(indices)].to(runner.device))
                positions.extend(rows)
        offset += count
    view.replace_embeds = torch.cat(values) if values else None
    view.replace_positions = (
        torch.tensor(positions, dtype=torch.int64, device=runner.device)
        if positions
        else None
    )
    # Image offsets remain absolute. ForwardBatch and the shared MM embedder
    # select visual rows using the view's expanded prefix/extend coordinates.


def adapt_validate(original):
    if not enabled():
        return original

    @wraps(original)
    def validate(manager, obj, input_ids):
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.runtime_context import get_exec

        if not (
            manager.server_args.device == "npu"
            and manager.model_config.hf_config.model_type == "deepseek_v41"
            and get_exec().features.enable_encoder_swa_bounded_replay
        ):
            return original(manager, obj, input_ids)
        if (
            isinstance(obj, GenerateReqInput)
            and obj.return_logprob
            and (obj.logprob_start_len not in (None, -1, len(input_ids)))
        ):
            raise ValueError("encoder SWA replay cannot return cached prompt logprobs")
        with get_exec().features.override(enable_encoder_swa_bounded_replay=False):
            return original(manager, obj, input_ids)

    return validate
