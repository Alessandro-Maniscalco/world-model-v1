"""Latent-time chunk scheduling helpers for chunkwise future rollout.

Build chunk ids and boundaries over future latent timesteps for selectable
chunk-count conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ChunkScheduleMode = Literal["k_plus_one", "k_chunks"]


@dataclass(frozen=True)
class ChunkSchedule:
    """Chunk metadata for a latent-time future window."""

    chunk_ids: torch.Tensor
    boundaries: tuple[tuple[int, int], ...]
    num_chunks: int


def resolve_num_chunks(*, k: int, chunk_schedule_mode: ChunkScheduleMode) -> int:
    """Resolve the effective number of future chunks for one schedule mode."""
    if chunk_schedule_mode == "k_plus_one":
        if k < 1:
            raise ValueError(f"k must be >= 1 for K+1 chunking, got {k}")
        return k + 1
    if chunk_schedule_mode == "k_chunks":
        if k < 1:
            raise ValueError(f"k must be >= 1 for K chunking, got {k}")
        return k
    raise ValueError(
        "chunk_schedule_mode must be 'k_plus_one' or 'k_chunks', "
        f"got {chunk_schedule_mode!r}"
    )


def build_chunk_schedule(
    future_steps: int,
    k: int,
    *,
    chunk_schedule_mode: ChunkScheduleMode = "k_plus_one",
    device: torch.device | None = None,
) -> ChunkSchedule:
    """Create a chunk schedule across `future_steps` latent timesteps."""
    _validate_schedule_args(
        future_steps=future_steps,
        k=k,
        chunk_schedule_mode=chunk_schedule_mode,
    )

    num_chunks = resolve_num_chunks(k=k, chunk_schedule_mode=chunk_schedule_mode)
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
    return build_chunk_schedule(
        future_steps=future_steps,
        k=k,
        chunk_schedule_mode="k_plus_one",
        device=device,
    )


def build_full_sequence_chunk_ids(
    past_steps: int,
    future_steps: int,
    k: int,
    *,
    chunk_schedule_mode: ChunkScheduleMode = "k_plus_one",
    past_chunk_id: int = -1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build chunk ids for `[past, future]` latent sequence.

    Past steps are assigned to `past_chunk_id`. Future steps use K+1 schedule.
    """
    if past_steps < 0:
        raise ValueError(f"past_steps must be non-negative, got {past_steps}")

    schedule = build_chunk_schedule(
        future_steps=future_steps,
        k=k,
        chunk_schedule_mode=chunk_schedule_mode,
        device=device,
    )
    past_ids = torch.full((past_steps,), past_chunk_id, dtype=torch.long, device=device)
    return torch.cat((past_ids, schedule.chunk_ids), dim=0)


def _validate_schedule_args(
    *,
    future_steps: int,
    k: int,
    chunk_schedule_mode: ChunkScheduleMode,
) -> None:
    """Validate chunk-schedule argument constraints."""
    if future_steps <= 0:
        raise ValueError(f"future_steps must be positive, got {future_steps}")
    num_chunks = resolve_num_chunks(k=k, chunk_schedule_mode=chunk_schedule_mode)
    if future_steps < num_chunks:
        raise ValueError(
            f"future_steps ({future_steps}) must be >= number of chunks ({num_chunks})"
        )
