"""Tests for canonical temporal-alignment helpers."""

from __future__ import annotations

import pytest
import torch

from world_model.data.temporal import (
    align_time_sequence,
    build_frame_deltas,
    expand_to_latent_steps,
    latent_split_from_frame_ratio,
)


def test_build_frame_deltas_basic_spacing() -> None:
    deltas = build_frame_deltas(context_len=3, horizon_len=2, dt=0.1)
    assert deltas == pytest.approx([-0.4, -0.3, -0.2, -0.1, 0.0])


def test_latent_split_preserves_total_and_nonzero_future() -> None:
    context_steps, horizon_steps = latent_split_from_frame_ratio(
        total_latent_steps=9,
        context_frames=10,
        horizon_frames=8,
    )
    assert context_steps + horizon_steps == 9
    assert context_steps >= 1
    assert horizon_steps >= 1


def test_align_time_sequence_resamples_with_nearest_indexing() -> None:
    seq = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    out = align_time_sequence(seq, target_steps=2)
    assert out.shape == (2, 2, 1)
    assert torch.equal(out[:, 0], seq[:, 0])
    assert torch.equal(out[:, 1], seq[:, 3])


def test_expand_to_latent_steps_handles_rank2_and_rank3() -> None:
    flat = torch.tensor([[1.0, 2.0]])
    flat_out = expand_to_latent_steps(flat, target_steps=3)
    assert flat_out.shape == (1, 3, 2)
    assert torch.equal(flat_out[:, 0], flat)

    seq = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    seq_out = expand_to_latent_steps(seq, target_steps=4)
    assert torch.equal(seq_out, seq)


def test_temporal_helpers_validate_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        align_time_sequence(torch.randn(2, 3), target_steps=2)
    with pytest.raises(ValueError):
        expand_to_latent_steps(torch.randn(2, 3, 4, 5), target_steps=2)
