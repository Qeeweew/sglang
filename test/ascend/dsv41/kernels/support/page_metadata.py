"""Deterministic paged metadata fixtures, separate from the reference oracle."""

import torch


def guarded(shape, dtype, value):
    count = 1
    for dim in shape:
        count *= dim
    storage = torch.full((count + 64,), value, dtype=dtype, device="npu")
    return storage[32 : 32 + count].view(shape), storage


def fixture(batch, block, context):
    columns = (context + 127) // 128
    # Full-token physical pages are reused independently of logical context.
    # This also models mapping changes after SWA ring/page recycling.
    page_ids = (
        (torch.arange((batch + 1) * columns).reshape(batch + 1, columns) * 17 + 11)
        % 4095
        + 1
    ).int()
    token_table = torch.zeros((batch + 1, context), dtype=torch.int32, device="npu")
    token_table[:, ::128] = (page_ids * 128).npu()
    mapping = torch.arange(4096 * 128, dtype=torch.int64)
    outputs, guards = {}, []
    for name, shape, dtype, sentinel in (
        ("cu", (batch + 1,), torch.int32, -787878),
        ("lengths", (batch,), torch.int32, -787878),
        ("table", (batch, columns), torch.int32, -787878),
        ("slots", (batch * block,), torch.int64, -787878),
        ("routed_rows", (1,), torch.int32, -787878),
    ):
        outputs[name], guard = guarded(shape, dtype, sentinel)
        guards.append(guard)
    requests = torch.zeros(batch, dtype=torch.int64, device="npu")
    prefix = torch.zeros(batch, dtype=torch.int64, device="npu")
    full_slots = torch.zeros(batch * block, dtype=torch.int64, device="npu")
    return token_table, page_ids, mapping, requests, prefix, full_slots, outputs, guards


def input_values(batch, block, context, page_ids, generation):
    requests = torch.arange(1, batch + 1, dtype=torch.int64)
    positions = (0, 1, 127, 128, 129, 16383, 16384, context - block)
    prefix = torch.tensor(
        [
            min(max(0, positions[(b + generation) % len(positions)]), context - block)
            for b in range(batch)
        ],
        dtype=torch.int64,
    )
    full_slots = torch.full((batch, block), -1, dtype=torch.int64)
    for b in range(batch):
        for k in range(block):
            pos = int(prefix[b]) + k
            full_slots[b, k] = int(page_ids[b + 1, pos // 128]) * 128 + pos % 128
    if batch > 1:
        requests[generation % batch] = 0
        # Dummy metadata must not dereference a malicious token or page index.
        prefix[generation % batch] = 2**40
        full_slots[generation % batch].fill_(2**40)
    if batch and generation % 2:
        full_slots[-1, 0] = -1
    full_index = torch.arange(4096 * 128, dtype=torch.int64)
    mapping = (
        (full_index // 128 * 3 + generation * 13) % 1023 + 1
    ) * 128 + full_index % 128
    return requests, prefix, full_slots.flatten(), mapping
