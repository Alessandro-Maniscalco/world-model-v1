"""Wan temporal alignment helpers for frame-time and latent-time windows."""

from __future__ import annotations

import torch

WAN_FRAME_GROUP_SIZE = 4


def validate_wan_temporal_window(context_len: int, horizon_len: int) -> None:
    """Validate a Wan-compatible raw-frame context/future split."""
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if horizon_len <= 0:
        raise ValueError(f"horizon_len must be positive, got {horizon_len}")
    if context_len % WAN_FRAME_GROUP_SIZE != 1:
        raise ValueError(
            "Wan temporal packing requires context_len = 4n+1 so the first frame stands alone "
            f"and the remaining context frames form 4-frame groups, got {context_len}."
        )
    if horizon_len % WAN_FRAME_GROUP_SIZE != 0:
        raise ValueError(
            "Wan temporal packing requires horizon_len to be a positive multiple of 4 so "
            f"future frames align to full 4-frame groups, got {horizon_len}."
        )


def wan_latent_steps_from_frame_count(frame_count: int) -> int:
    """Convert an exact Wan raw-frame count into its latent-time length."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if frame_count % WAN_FRAME_GROUP_SIZE != 1:
        raise ValueError(
            "Wan raw-frame counts must be 4n+1 to avoid dropping tail frames during VAE encoding, "
            f"got {frame_count}."
        )
    return 1 + ((frame_count - 1) // WAN_FRAME_GROUP_SIZE)


def build_frame_deltas(context_len: int, horizon_len: int, dt: float) -> list[float]:
    """Build contiguous frame-time deltas for an exact Wan window."""
    validate_wan_temporal_window(context_len=context_len, horizon_len=horizon_len)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    total_len = context_len + horizon_len
    return [-(total_len - 1 - i) * dt for i in range(total_len)]


def latent_split_for_wan_frames(
    total_latent_steps: int,
    context_frames: int,
    horizon_frames: int,
) -> tuple[int, int]:
    """Map an exact Wan frame split to latent-time context and future steps."""
    validate_wan_temporal_window(context_len=context_frames, horizon_len=horizon_frames)

    expected_total_latent_steps = wan_latent_steps_from_frame_count(context_frames + horizon_frames)
    if total_latent_steps != expected_total_latent_steps:
        raise ValueError(
            "Encoded latent length does not match the exact Wan frame packing rule: "
            f"expected {expected_total_latent_steps} latent steps for "
            f"context_frames={context_frames} and horizon_frames={horizon_frames}, "
            f"got total_latent_steps={total_latent_steps}."
        )

    context_steps = wan_latent_steps_from_frame_count(context_frames)
    horizon_steps = horizon_frames // WAN_FRAME_GROUP_SIZE
    if context_steps + horizon_steps != total_latent_steps:
        raise ValueError(
            "Wan frame split produced an inconsistent latent split: "
            f"context_steps={context_steps}, horizon_steps={horizon_steps}, "
            f"total_latent_steps={total_latent_steps}."
        )
    return context_steps, horizon_steps


def latent_split_from_frame_ratio(
    total_latent_steps: int,
    context_frames: int,
    horizon_frames: int,
) -> tuple[int, int]:
    """Preserve the legacy helper name while using exact Wan temporal packing."""
    return latent_split_for_wan_frames(
        total_latent_steps=total_latent_steps,
        context_frames=context_frames,
        horizon_frames=horizon_frames,
    )


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


def build_future_action_plan(
    seq: torch.Tensor,
    *,
    context_frames: int,
    horizon_frames: int,
    horizon_latent_steps: int,
) -> torch.Tensor:
    """Build future latent-step action features from raw or latent-aligned actions."""
    validate_wan_temporal_window(context_len=context_frames, horizon_len=horizon_frames)
    if horizon_latent_steps <= 0:
        raise ValueError(f"horizon_latent_steps must be positive, got {horizon_latent_steps}")

    if seq.ndim == 2:
        return seq.unsqueeze(1).repeat(1, horizon_latent_steps, 1)
    if seq.ndim != 3:
        raise ValueError(f"seq must be [B,D] or [B,T,D], got {tuple(seq.shape)}")

    source_steps = int(seq.shape[1])
    total_frames = context_frames + horizon_frames

    if source_steps == horizon_latent_steps:
        return seq
    if source_steps == 1:
        return seq.repeat(1, horizon_latent_steps, 1)
    if source_steps == horizon_frames:
        return flatten_action_chunks(seq, num_chunks=horizon_latent_steps)
    if source_steps == total_frames:
        return flatten_action_chunks(seq[:, context_frames:], num_chunks=horizon_latent_steps)
    if source_steps % horizon_latent_steps == 0:
        return flatten_action_chunks(seq, num_chunks=horizon_latent_steps)

    return align_time_sequence(seq, target_steps=horizon_latent_steps)


def flatten_action_chunks(seq: torch.Tensor, *, num_chunks: int) -> torch.Tensor:
    """Flatten contiguous action chunks into one feature vector per chunk."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if seq.ndim != 3:
        raise ValueError(f"seq must be [B,T,D], got {tuple(seq.shape)}")
    if seq.shape[1] % num_chunks != 0:
        raise ValueError(
            f"seq time dim {seq.shape[1]} must be divisible by num_chunks={num_chunks}"
        )

    batch_size, source_steps, action_dim = seq.shape
    chunk_size = source_steps // num_chunks
    return seq.reshape(batch_size, num_chunks, chunk_size * action_dim)
