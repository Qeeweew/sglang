"""Shared checks for projected compressor output and transactional state."""

import torch


def validate_prepare(
    provisional, expected, state, original_pending, original_lengths, label, records
):
    latent, pooled, positions, rope_positions, starts = expected
    got_pool = provisional.pooled.cpu()
    torch.testing.assert_close(
        got_pool, pooled, atol=2e-6, rtol=2e-6, msg=label + "/pool"
    )
    got_latent = provisional.latent.cpu()
    torch.testing.assert_close(
        got_latent.float(), latent.float(), atol=0.0001, rtol=0.008, msg=label + "/norm"
    )
    assert torch.equal(provisional.positions.cpu(), positions), label + "/positions"
    assert torch.equal(provisional.rope_positions.cpu(), rope_positions), (
        label + "/RoPE positions"
    )
    assert torch.equal(provisional.starts.cpu(), starts), label + "/snapshots"
    assert torch.equal(state.pending.cpu(), original_pending), (
        label + "/read-only pending"
    )
    assert torch.equal(state.committed_lengths.cpu(), original_lengths), (
        label + "/read-only lengths"
    )
    records.append(
        {
            "case": label,
            "pool_max_abs": (got_pool - pooled).abs().max().item()
            if got_pool.numel()
            else 0.0,
            "norm_bf16_mismatches": int((got_latent != latent).sum()),
        }
    )
