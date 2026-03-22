"""Unit tests for latent-time chunk scheduling helpers."""

import torch
import pytest

from world_model.chunking import (
    build_chunk_schedule,
    build_full_sequence_chunk_ids,
)


def test_exact_k_schedule_supports_k_equals_one_baseline() -> None:
    """Use one future chunk when k=1 under the exact-k convention."""
    schedule = build_chunk_schedule(future_steps=8, k=1)

    assert schedule.num_chunks == 1
    assert torch.equal(schedule.chunk_ids, torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]))
    assert schedule.boundaries == ((0, 8),)


def test_exact_k_schedule_distributes_remainder_to_early_chunks() -> None:
    """Distribute remainder steps across the early exact-k chunks."""
    schedule = build_chunk_schedule(future_steps=10, k=3)

    # 10 steps over 3 chunks -> sizes [4, 3, 3]
    assert schedule.boundaries == ((0, 4), (4, 7), (7, 10))
    assert torch.equal(schedule.chunk_ids, torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))


def test_full_sequence_chunk_ids_prefixes_past_steps() -> None:
    """Prefix past chunk ids ahead of the exact-k future schedule."""
    chunk_ids = build_full_sequence_chunk_ids(past_steps=3, future_steps=5, k=1, past_chunk_id=-1)
    assert torch.equal(chunk_ids, torch.tensor([-1, -1, -1, 0, 0, 0, 0, 0]))


def test_k_chunks_schedule_uses_exactly_k_future_chunks() -> None:
    """Split the future window over exactly k chunks when requested."""
    schedule = build_chunk_schedule(future_steps=8, k=2, chunk_schedule_mode="k_chunks")

    assert schedule.num_chunks == 2
    assert schedule.boundaries == ((0, 4), (4, 8))
    assert torch.equal(schedule.chunk_ids, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))


def test_legacy_k_plus_one_mode_normalizes_to_exact_k_chunking() -> None:
    """Interpret stale legacy metadata with the repo's exact-k semantics."""
    schedule = build_chunk_schedule(future_steps=8, k=2, chunk_schedule_mode="k_plus_one")

    assert schedule.num_chunks == 2
    assert schedule.boundaries == ((0, 4), (4, 8))


def test_full_sequence_chunk_ids_support_k_chunks_mode() -> None:
    """Allow full-sequence chunk ids to use exact-k chunking for short horizons."""
    chunk_ids = build_full_sequence_chunk_ids(
        past_steps=2,
        future_steps=2,
        k=2,
        chunk_schedule_mode="k_chunks",
        past_chunk_id=-1,
    )

    assert torch.equal(chunk_ids, torch.tensor([-1, -1, 0, 1]))


@pytest.mark.parametrize(
    ("future_steps", "k"),
    [
        (0, 1),
        (4, 0),
        (1, 2),  # future_steps < k
    ],
)
def test_exact_k_schedule_rejects_invalid_inputs(future_steps: int, k: int) -> None:
    """Reject non-positive schedules and future windows shorter than k."""
    with pytest.raises(ValueError):
        build_chunk_schedule(future_steps=future_steps, k=k)


def test_k_chunks_schedule_rejects_future_windows_shorter_than_k() -> None:
    """Reject exact-k schedules when the latent horizon cannot cover all chunks."""
    with pytest.raises(ValueError):
        build_chunk_schedule(future_steps=1, k=2, chunk_schedule_mode="k_chunks")
