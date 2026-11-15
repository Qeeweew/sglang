"""Independent CPU stable-sort oracle for local EP routing."""

import torch


def reference(ids, first, experts):
    local = torch.where((ids >= first) & (ids < first + experts), ids - first, -1)
    flat = local.flatten().long()
    slots = torch.where(flat >= 0)[0]
    order = slots[flat[slots].argsort(stable=True)].int()
    counts = torch.bincount(flat[slots], minlength=experts).int()
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int32), counts.cumsum(0, dtype=torch.int32))
    )
    active = torch.where(counts > 0)[0]
    metadata = [len(active), len(order), 0, 1]
    for expert in active.tolist():
        metadata.extend((expert, int(offsets[expert]), int(counts[expert]), 0))
    return local, counts, offsets, order, torch.tensor(metadata, dtype=torch.int32)


def compare(actual, expected):
    for name, tensor, oracle in zip(
        ("ids", "counts", "offsets", "order", "metadata"),
        (actual.ids, actual.counts, actual.offsets, actual.order, actual.metadata),
        expected,
    ):
        try:
            torch.testing.assert_close(
                tensor.flatten()[: oracle.numel()].cpu(),
                oracle.flatten(),
                rtol=0,
                atol=0,
            )
        except Exception as error:
            error.add_note(f"bucket field {name}")
            raise
