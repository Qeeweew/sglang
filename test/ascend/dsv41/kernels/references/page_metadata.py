"""CPU draft visibility and page mapping oracle independent of the native op."""

import torch


def oracle(outputs, page_ids, requests, prefix, full_slots, mapping, block, context):
    batch = len(requests)
    result = {key: value.clone() for key, value in outputs.items()}
    result["cu"] = torch.arange(batch + 1, dtype=torch.int32) * block
    result["lengths"] = torch.zeros(batch, dtype=torch.int32)
    result["slots"] = torch.full((batch * block,), -1, dtype=torch.int64)
    result["routed_rows"] = torch.tensor(
        [int((requests > 0).sum()) * block], dtype=torch.int32
    )
    for b, request in enumerate(requests.tolist()):
        if not request:
            continue
        start, end = max(0, int(prefix[b]) - 128), int(prefix[b]) + block
        assert 0 <= start <= end <= context
        result["lengths"][b] = end
        for page in range(start // 128, (end + 127) // 128):
            full = int(page_ids[request, page]) * 128
            result["table"][b, page] = int(mapping[full]) // 128
        for k in range(block):
            full = int(full_slots[b * block + k])
            if full >= 0:
                result["slots"][b * block + k] = mapping[full]
    return result
