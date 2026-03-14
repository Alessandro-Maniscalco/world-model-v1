"""Tests for forward-pass smoke utility helpers."""

import torch
import pytest

from world_model.data.temporal import (
    build_frame_deltas,
    expand_to_latent_steps,
    latent_split_from_frame_ratio,
)


def test_build_frame_deltas_matches_window_length_and_spacing():
    deltas = build_frame_deltas(context_len=9, horizon_len=8, dt=0.1)

    assert len(deltas) == 17
    assert deltas[0] == pytest.approx(-1.6)
    assert deltas[-1] == pytest.approx(0.0)


def test_latent_split_from_frame_ratio_matches_exact_wan_groups():
    context_steps, horizon_steps = latent_split_from_frame_ratio(
        total_latent_steps=5,
        context_frames=9,
        horizon_frames=8,
    )

    assert context_steps == 3
    assert horizon_steps == 2


def test_expand_to_latent_steps_repeats_batched_vectors():
    seq = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = expand_to_latent_steps(seq, target_steps=3)

    assert out.shape == (2, 3, 2)
    assert torch.equal(out[:, 0], seq)
    assert torch.equal(out[:, 2], seq)


def test_expand_to_latent_steps_resamples_time_sequence():
    seq = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    out = expand_to_latent_steps(seq, target_steps=2)

    assert out.shape == (2, 2, 1)
    assert torch.equal(out[:, 0], seq[:, 0])
    assert torch.equal(out[:, 1], seq[:, 3])
