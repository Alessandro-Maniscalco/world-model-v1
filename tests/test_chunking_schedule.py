"""Unit tests for latent-time K+1 chunk scheduling."""

import torch
import pytest

from world_model.chunking import build_full_sequence_chunk_ids, build_k_plus_one_schedule


def test_k_plus_one_schedule_supports_k_equals_one_baseline():
    schedule = build_k_plus_one_schedule(future_steps=8, k=1)

    assert schedule.num_chunks == 2
    assert torch.equal(schedule.chunk_ids, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert schedule.boundaries == ((0, 4), (4, 8))


def test_k_plus_one_schedule_distributes_remainder_to_early_chunks():
    schedule = build_k_plus_one_schedule(future_steps=10, k=2)

    # 10 steps over 3 chunks -> sizes [4, 3, 3]
    assert schedule.boundaries == ((0, 4), (4, 7), (7, 10))
    assert torch.equal(schedule.chunk_ids, torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))


def test_full_sequence_chunk_ids_prefixes_past_steps():
    chunk_ids = build_full_sequence_chunk_ids(past_steps=3, future_steps=5, k=1, past_chunk_id=-1)
    assert torch.equal(chunk_ids, torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1]))


@pytest.mark.parametrize(
    ("future_steps", "k"),
    [
        (0, 1),
        (4, 0),
        (2, 2),  # future_steps < k+1
    ],
)
def test_k_plus_one_schedule_rejects_invalid_inputs(future_steps: int, k: int):
    with pytest.raises(ValueError):
        build_k_plus_one_schedule(future_steps=future_steps, k=k)
