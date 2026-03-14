"""Tests for canonical temporal-alignment helpers."""

from __future__ import annotations

import pytest
import torch

from world_model.data.temporal import (
    align_time_sequence,
    build_future_action_plan,
    build_frame_deltas,
    expand_to_latent_steps,
    flatten_action_chunks,
    latent_split_for_wan_frames,
    latent_split_from_frame_ratio,
)


def test_build_frame_deltas_basic_spacing() -> None:
    deltas = build_frame_deltas(context_len=5, horizon_len=4, dt=0.1)
    assert deltas == pytest.approx([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])


def test_latent_split_uses_exact_wan_packing() -> None:
    context_steps, horizon_steps = latent_split_for_wan_frames(
        total_latent_steps=5,
        context_frames=9,
        horizon_frames=8,
    )
    assert context_steps == 3
    assert horizon_steps == 2


def test_latent_split_from_frame_ratio_keeps_legacy_name_but_exact_rule() -> None:
    context_steps, horizon_steps = latent_split_from_frame_ratio(
        total_latent_steps=5,
        context_frames=9,
        horizon_frames=8,
    )
    assert (context_steps, horizon_steps) == (3, 2)


def test_temporal_helpers_reject_invalid_wan_frame_counts() -> None:
    with pytest.raises(ValueError, match="context_len = 4n\\+1"):
        build_frame_deltas(context_len=10, horizon_len=8, dt=0.1)
    with pytest.raises(ValueError, match="horizon_len to be a positive multiple of 4"):
        latent_split_for_wan_frames(total_latent_steps=5, context_frames=9, horizon_frames=6)


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


def test_flatten_action_chunks_preserves_within_chunk_order() -> None:
    """Flatten contiguous chunk actions without dropping any step ordering."""
    seq = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)

    flattened = flatten_action_chunks(seq, num_chunks=2)

    assert flattened.shape == (1, 2, 8)
    assert torch.equal(flattened[0, 0], seq[0, :4].reshape(-1))
    assert torch.equal(flattened[0, 1], seq[0, 4:].reshape(-1))


def test_build_future_action_plan_chunks_full_frame_window_for_wan() -> None:
    """Convert raw frame-rate actions into exact future latent-block features."""
    seq = torch.arange(1 * 17 * 2, dtype=torch.float32).reshape(1, 17, 2)

    plan = build_future_action_plan(
        seq,
        context_frames=9,
        horizon_frames=8,
        horizon_latent_steps=2,
    )

    assert plan.shape == (1, 2, 8)
    assert torch.equal(plan[0, 0], seq[0, 9:13].reshape(-1))
    assert torch.equal(plan[0, 1], seq[0, 13:17].reshape(-1))


def test_build_future_action_plan_keeps_latent_aligned_sequences() -> None:
    """Leave already-latent-aligned future action plans unchanged."""
    seq = torch.arange(1 * 2 * 3, dtype=torch.float32).reshape(1, 2, 3)

    plan = build_future_action_plan(
        seq,
        context_frames=9,
        horizon_frames=8,
        horizon_latent_steps=2,
    )

    assert torch.equal(plan, seq)


def test_temporal_helpers_validate_invalid_shapes() -> None:
    with pytest.raises(ValueError):
        align_time_sequence(torch.randn(2, 3), target_steps=2)
    with pytest.raises(ValueError):
        expand_to_latent_steps(torch.randn(2, 3, 4, 5), target_steps=2)
