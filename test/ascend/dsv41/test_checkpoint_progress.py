"""Streaming progress follows consumed tensors and closes on early exit."""

import json

import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("rank,draft", [(0, False), (1, False), (0, True)])
def test_checkpoint_progress_counts_consumed_tensors(
    monkeypatch, tmp_path, rank, draft
):
    from sglang.srt.hardware_backend.npu.dsv41 import checkpoint as checkpoint

    prefix = "mtp.0" if draft else "layers.0"
    names = [f"{prefix}.ffn.experts.0.{part}" for part in checkpoint._PARTS]
    names.append(f"{prefix}.dense.weight")
    save_file(
        {name: torch.ones(1, dtype=torch.uint8) for name in names},
        tmp_path / "weights.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "weights.safetensors")})
    )
    bars = []

    class Progress:
        _get_free_pos = staticmethod(lambda: 0)

        def __init__(self, **kwargs):
            self.options, self.n, self.closed = kwargs, 0, False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self):
            self.n += 1

    monkeypatch.setattr(checkpoint, "tqdm", Progress)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    reader = checkpoint.ExpertCheckpoint(tmp_path, draft=draft)
    iterator = iter(reader)
    next(iterator)
    bar = bars[-1]
    assert bar.n == 0  # Consumer has not finished loading the yielded tensor.
    assert len(list(iterator)) == 6
    assert bar.n == bar.options["total"] == 7
    assert bar.options["disable"] == (rank != 0)
    assert ("draft" if draft else "target") in bar.options["desc"]
    assert bar.closed

    iterator = iter(reader)
    next(iterator)
    iterator.close()
    assert bars[-1].closed and bars[-1].n == 0
