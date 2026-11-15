"""Independent numerical references and fixtures for attention."""

from __future__ import annotations

import torch


def attention_data(
    mode, batch, total, max_seq, case, qlens, *, physical_blocks=None, heads=64
):
    ratio = 1 if mode == "c1" else 2
    per_batch = max_seq // 128
    blocks = batch * per_batch
    ori_blocks, cmp_blocks = physical_blocks or (blocks, blocks)
    q = torch.randn(total, heads, 512).bfloat16()
    kv = (torch.randn(ori_blocks, 128, 1, 512) * 0.3).bfloat16()
    cmp = (torch.randn(cmp_blocks, 128 // ratio, 1, 512) * 0.3).bfloat16()
    if physical_blocks is None:
        tables = torch.randperm(blocks).int().reshape(batch, per_batch)
        cmp_tables = torch.randperm(blocks).int().reshape(batch, per_batch)
    else:
        # Small physical pools deliberately alias logical pages. The oracle
        # reads only selected rows, so high absolute positions need no 1M KV.
        tables = torch.randint(ori_blocks, (batch, per_batch), dtype=torch.int32)
        cmp_tables = torch.randint(cmp_blocks, (batch, per_batch), dtype=torch.int32)
    cu = torch.tensor(
        [0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32
    )
    lengths = torch.tensor(
        [max(n, min(max_seq, 129 + b * 31)) for b, n in enumerate(qlens)],
        dtype=torch.int32,
    )
    if case == "empty":
        lengths.zero_()
    if case == "short":
        lengths = torch.tensor(qlens, dtype=torch.int32)
    if case == "full":
        lengths.fill_(max_seq)
    cmplen = lengths // ratio
    residual = lengths % ratio
    window = torch.zeros(total, 1, dtype=torch.int32)
    if case == "narrow":
        window[:, 0] = torch.arange(total) % 16 + 1
    indices = torch.full((total, 1, 512), -1, dtype=torch.int32)
    for b in range(batch):
        for token in range(int(cu[b]), int(cu[b + 1])):
            count = min(512, int(cmplen[b]))
            values = torch.randperm(int(cmplen[b]))[:count].int()
            if case == "duplicate" and count:
                values[::2] = 0
            if case == "padding":
                values = values[: count // 2]
            indices[token, 0, : len(values)] = values
    sinks = torch.randn(heads)
    if case == "sinks":
        sinks[::2] = 100
        sinks[1::2] = -100
    args = [q, kv, tables, cu, lengths, sinks, window]
    if mode != "swa":
        args = [
            q,
            kv,
            cmp,
            indices,
            tables,
            cmp_tables,
            cu,
            lengths,
            cmplen,
            residual,
            window,
            sinks,
        ]
    return args


def attention_oracle(mode, args, *, token_indices=None):
    if mode == "swa":
        q, kv, table, cu, lengths, sinks, window = args
    else:
        (
            q,
            kv,
            cmp,
            indices,
            table,
            cmp_table,
            cu,
            lengths,
            cmplen,
            residual,
            window,
            sinks,
        ) = args
    selected = (
        None
        if token_indices is None
        else {int(t): i for i, t in enumerate(token_indices)}
    )
    result = (
        torch.empty_like(q)
        if selected is None
        else q.new_empty((len(selected), *q.shape[1:]))
    )
    ratio = 1 if mode == "c1" else 2
    for b in range(len(lengths)):
        n = int(cu[b + 1] - cu[b])
        for j in range(n):
            t = int(cu[b]) + j
            if selected is not None and t not in selected:
                continue
            output_row = t if selected is None else selected[t]
            right = max(0, int(lengths[b]) - n + j + 1)
            visible = min(128, right)
            if int(window[t, 0]) > 0:
                visible = min(visible, int(window[t, 0]))
            rows = [
                kv[int(table[b, p // 128]), p % 128, 0]
                for p in range(right - visible, right)
            ]
            if mode != "swa":
                cutoff = (
                    max(0, int(cmplen[b]) * ratio + int(residual[b]) - n + j + 1)
                    // ratio
                )
                count = min(512, int(cmplen[b]), cutoff)
                for index in indices[t, 0, :count].tolist():
                    if 0 <= index < cutoff:
                        rows.append(
                            cmp[
                                int(cmp_table[b, index // (128 // ratio)]),
                                index % (128 // ratio),
                                0,
                            ]
                        )
            if not rows:
                result[output_row].zero_()
                continue
            values = torch.stack(rows).float()
            scores = q[t].float() @ values.T * (512**-0.5)
            p = torch.cat([scores, sinks[:, None]], 1).softmax(-1)[:, :-1]
            result[output_row] = (p @ values).bfloat16()
    return result
