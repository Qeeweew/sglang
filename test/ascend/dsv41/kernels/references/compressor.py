"""Independent numerical references and fixtures for compressor."""

from __future__ import annotations

import torch


def oracle_prepare(ratio, kv, scores, gamma, pending, lengths, cu, request_ids):
    """Independent CPU implementation; no custom kernel or ring workspace."""
    latent = torch.zeros(kv.shape, dtype=torch.bfloat16)
    pooled = torch.zeros(kv.shape, dtype=torch.float32)
    positions = torch.full((kv.shape[0],), -1, dtype=torch.int64)
    rope_positions = torch.full_like(positions, -1)
    starts = torch.full((len(request_ids),), -1, dtype=torch.int64)
    for b, slot in enumerate(request_ids.tolist()):
        if not 0 <= slot < len(lengths):
            continue
        start = int(lengths[slot])
        starts[b] = start
        for token in range(int(cu[b]), int(cu[b + 1])):
            local = token - int(cu[b])
            position = start + local
            if ratio == 2 and position % 2 == 0:
                continue
            if ratio == 1:
                value = kv[token].float()
            else:
                previous = kv[token - 1] if local else pending[slot, 0]
                previous_score = scores[token - 1] if local else pending[slot, 1]
                weights = torch.stack((previous_score, scores[token])).softmax(0)
                value = (torch.stack((previous, kv[token])) * weights).sum(0)
            pooled[token] = value
            rounded = value.bfloat16().float()
            latent[token] = (
                rounded * torch.rsqrt(rounded.square().mean() + 1e-20) * gamma.float()
            ).bfloat16()
            positions[token] = position
            rope_positions[token] = position - (ratio - 1)
    return latent, pooled, positions, rope_positions, starts


def oracle_commit(
    ratio, kv, scores, pending, lengths, cu, request_ids, starts, accepted
):
    pending, lengths = pending.clone(), lengths.clone()
    positions = torch.full((kv.shape[0],), -1, dtype=torch.int64)
    rope_positions = torch.full_like(positions, -1)
    statuses = torch.zeros(len(request_ids), dtype=torch.int32)
    for b, slot in enumerate(request_ids.tolist()):
        begin, end = int(cu[b]), int(cu[b + 1])
        keep = int(accepted[b])
        start = int(starts[b])
        if not 0 <= slot < len(lengths) or start < 0:
            statuses[b] = 3
            continue
        if not 0 <= keep <= end - begin:
            statuses[b] = 1
            continue
        if int(lengths[slot]) != start:
            statuses[b] = 2
            continue
        for row in range(begin, begin + keep):
            pos = start + row - begin
            if ratio == 1 or pos % 2:
                positions[row] = pos
                rope_positions[row] = pos - (ratio - 1)
        if keep:
            if ratio == 2 and (start + keep) % 2:
                pending[slot, 0] = kv[begin + keep - 1]
                pending[slot, 1] = scores[begin + keep - 1]
            lengths[slot] = start + keep
    return pending, lengths, positions, rope_positions, statuses
