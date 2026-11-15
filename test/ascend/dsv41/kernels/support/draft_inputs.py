"""Deterministic paged draft inputs for gather qualification and timing."""

import torch


def data(batch, capacity):
    kv = torch.randn(batch * 8, 128, 1, 512).bfloat16()
    table = torch.randperm(batch * 8).int().view(batch, 8)
    counts = [257] if capacity == 512 else [6] * batch
    if batch == 4:
        counts = [0, 1, 8, 7]
    cu = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32
    )
    lengths = torch.tensor([127 + n for n in counts], dtype=torch.int32)
    return kv, table, cu, lengths
