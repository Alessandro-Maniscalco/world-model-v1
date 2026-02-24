"""Latent-time chunk scheduling for K+1 autoregressive stages.

Build chunk ids and boundaries over future latent timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkSchedule:
    """Chunk metadata for a latent-time future window."""

    chunk_ids: torch.Tensor
    boundaries: tuple[tuple[int, int], ...]
    num_chunks: int


def build_k_plus_one_schedule(
    future_steps: int,
    k: int,
    *,
    device: torch.device | None = None,
) -> ChunkSchedule:
    """Create a K+1 chunk schedule across `future_steps` latent timesteps.

    Args:
        future_steps: Number of latent timesteps in the future window.
        k: Baseline chunk count parameter. Total chunks are `k + 1`.
        device: Optional torch device for returned chunk ids.
    """
    _validate_schedule_args(future_steps=future_steps, k=k)

    num_chunks = k + 1
    base, rem = divmod(future_steps, num_chunks)
    sizes = [base + (1 if i < rem else 0) for i in range(num_chunks)]

    chunk_ids = torch.empty(future_steps, dtype=torch.long, device=device)
    boundaries: list[tuple[int, int]] = []

    start = 0
    for chunk_id, size in enumerate(sizes):
        end = start + size
        chunk_ids[start:end] = chunk_id
        boundaries.append((start, end))
        start = end

    return ChunkSchedule(
        chunk_ids=chunk_ids,
        boundaries=tuple(boundaries),
        num_chunks=num_chunks,
    )


def build_full_sequence_chunk_ids(
    past_steps: int,
    future_steps: int,
    k: int,
    *,
    past_chunk_id: int = -1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build chunk ids for `[past, future]` latent sequence.

    Past steps are assigned to `past_chunk_id`. Future steps use K+1 schedule.
    """
    if past_steps < 0:
        raise ValueError(f"past_steps must be non-negative, got {past_steps}")

    schedule = build_k_plus_one_schedule(future_steps=future_steps, k=k, device=device)
    past_ids = torch.full((past_steps,), past_chunk_id, dtype=torch.long, device=device)
    return torch.cat((past_ids, schedule.chunk_ids), dim=0)


def _validate_schedule_args(future_steps: int, k: int) -> None:
    """Validate `build_k_plus_one_schedule` argument constraints."""
    if future_steps <= 0:
        raise ValueError(f"future_steps must be positive, got {future_steps}")
    if k < 1:
        raise ValueError(f"k must be >= 1 for K+1 chunking, got {k}")

    num_chunks = k + 1
    if future_steps < num_chunks:
        raise ValueError(
            f"future_steps ({future_steps}) must be >= number of chunks ({num_chunks})"
        )
