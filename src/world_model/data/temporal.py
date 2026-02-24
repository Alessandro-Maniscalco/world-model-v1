"""Temporal alignment helpers for frame-time and latent-time windows."""

from __future__ import annotations

import torch


def build_frame_deltas(context_len: int, horizon_len: int, dt: float) -> list[float]:
    """Build contiguous frame-time deltas ending at t=0 for context+horizon."""
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if horizon_len <= 0:
        raise ValueError(f"horizon_len must be positive, got {horizon_len}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    total_len = context_len + horizon_len
    return [-(total_len - 1 - i) * dt for i in range(total_len)]


def latent_split_from_frame_ratio(
    total_latent_steps: int,
    context_frames: int,
    horizon_frames: int,
) -> tuple[int, int]:
    """Map frame-time context/horizon ratio to latent timesteps."""
    if total_latent_steps < 2:
        raise ValueError(
            f"total_latent_steps must be >= 2 for a context/future split, got {total_latent_steps}"
        )
    if context_frames <= 0:
        raise ValueError(f"context_frames must be positive, got {context_frames}")
    if horizon_frames <= 0:
        raise ValueError(f"horizon_frames must be positive, got {horizon_frames}")

    total_frames = context_frames + horizon_frames
    context_ratio = context_frames / total_frames

    context_steps = int(round(total_latent_steps * context_ratio))
    context_steps = min(max(context_steps, 1), total_latent_steps - 1)
    horizon_steps = total_latent_steps - context_steps
    return context_steps, horizon_steps


def align_time_sequence(seq: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Align a `[B,T,D]` tensor to `[B,target_steps,D]` via nearest resampling."""
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if seq.ndim != 3:
        raise ValueError(f"seq must be [B,T,D], got {tuple(seq.shape)}")

    t_src = seq.shape[1]
    if t_src <= 0:
        raise ValueError("seq must have a positive source time dimension")
    if t_src == target_steps:
        return seq

    idx = torch.linspace(0, t_src - 1, steps=target_steps, device=seq.device)
    idx = idx.round().long().clamp(0, t_src - 1)
    return seq.index_select(dim=1, index=idx)


def expand_to_latent_steps(seq: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Expand `[B,D]` or align `[B,T,D]` inputs to `[B,target_steps,D]`."""
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")

    if seq.ndim == 2:
        return seq.unsqueeze(1).repeat(1, target_steps, 1)
    if seq.ndim != 3:
        raise ValueError(f"seq must be [B,D] or [B,T,D], got {tuple(seq.shape)}")

    return align_time_sequence(seq, target_steps=target_steps)
