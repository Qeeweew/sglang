"""Independent numerical references and fixtures for engram_hash."""

from __future__ import annotations

import torch


def oracle_hash(ids, positions, cu, slots, history, active, p):
    layers, order = p["multipliers"].shape
    heads = p["primes"].shape[2]
    history_tokens = order - 1
    result = torch.empty((layers, len(ids), history_tokens * heads), dtype=torch.int64)
    for b, slot in enumerate(slots.tolist()):
        for t in range(int(cu[b]), int(cu[b + 1])):
            off = t - int(cu[b])
            values, blocked = (
                [],
                not bool(active[t]) or slot < 0 or slot >= len(history),
            )
            for shift in range(order):
                token = (
                    (
                        int(ids[t - shift])
                        if shift <= off
                        else int(history[slot, history_tokens - (shift - off)])
                    )
                    if 0 <= slot < len(history)
                    else 0
                )
                blocked = (
                    blocked
                    or int(positions[t]) < shift
                    or token < 0
                    or token >= len(p["token_map"])
                    or token == p["image_id"]
                    or token >= 1_000_000
                )
                values.append(
                    int(p["pad_id"]) if blocked else int(p["token_map"][token])
                )
            for layer in range(layers):
                rolling = values[0] * int(p["multipliers"][layer, 0])
                for shift in range(1, order):
                    rolling ^= values[shift] * int(p["multipliers"][layer, shift])
                    for head in range(heads):
                        col = (shift - 1) * heads + head
                        result[layer, t, col] = rolling % int(
                            p["primes"][layer, shift - 1, head]
                        ) + int(p["offsets"][layer, col])
    return result


def oracle_commit(ids, cu, slots, accepted, history):
    result = history.clone()
    for b, slot in enumerate(slots.tolist()):
        n = int(accepted[b])
        if 0 <= slot < len(history) and 0 < n <= int(cu[b + 1] - cu[b]):
            seq = torch.cat([history[slot], ids[int(cu[b]) : int(cu[b]) + n]])
            result[slot] = seq[-history.shape[1] :]
    return result


def oracle_gate(x, kv, qw, kw, active):
    h = x.float()
    rows, branches, hidden = x.shape
    key = kv[:, : branches * hidden].reshape(rows, branches, hidden).float()
    value = kv[:, branches * hidden :].float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(
        key.square().mean(-1) + 1e-20
    )
    dot = (h * (qw.float() * kw.float()) * key).sum(-1) * rstd * (hidden**-0.5)
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate = gate.masked_fill(~active.bool()[:, None], 0)
    return (h + gate[:, :, None] * value[:, None, :]).bfloat16()
