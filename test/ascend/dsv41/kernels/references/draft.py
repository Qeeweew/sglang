"""Independent numerical references and fixtures for draft."""

from __future__ import annotations

import torch


def oracle_window(kv, table, cu, lengths, capacity):
    out = torch.zeros((len(lengths), capacity, 512), dtype=torch.bfloat16)
    visible = torch.zeros((len(lengths), 8), dtype=torch.int32)
    for b, length in enumerate(lengths.tolist()):
        n = int(cu[b + 1] - cu[b])
        first = max(0, length - n - 128)
        count = length - first
        if n == 0:
            continue
        if count > capacity or length < n:
            visible[b, 0] = -max(1, count)
            continue
        visible[b, 0] = count
        for index, pos in enumerate(range(first, length)):
            page = int(table[b, pos // 128])
            if not 0 <= page < len(kv):
                visible[b, 0] = -max(1, count)
                break
            out[b, index] = kv[page, pos % 128, 0]
    return out, visible


def oracle_attention(q, window, cu, visible, sinks, *, token_indices=None):
    selected = (
        None
        if token_indices is None
        else {int(token): index for index, token in enumerate(token_indices)}
    )
    result = (
        torch.zeros_like(q)
        if selected is None
        else q.new_zeros((len(selected), *q.shape[1:]))
    )
    for b in range(len(visible)):
        count = int(visible[b, 0])
        if count <= 0:
            continue
        values = window[b, :count].float()
        for token in range(int(cu[b]), min(int(cu[b + 1]), len(q))):
            if selected is not None and token not in selected:
                continue
            score = q[token].float() @ values.T * (512**-0.5)
            p = torch.cat((score, sinks[:, None]), 1).softmax(-1)[:, :-1]
            result[token if selected is None else selected[token]] = (
                p @ values
            ).bfloat16()
    return result
