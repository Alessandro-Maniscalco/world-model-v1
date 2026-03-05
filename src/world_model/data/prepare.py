"""Shared structured latent-video batch preparation for training and inference."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from world_model.data.schema import PreparedPackedBatch
from world_model.data.temporal import expand_to_latent_steps, latent_split_from_frame_ratio


class LatentEncoder(Protocol):
    """Protocol for latent encoders used in data preparation."""

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video batch into `[B, C_lat, T_lat, H_lat, W_lat]` latents."""


def prepare_packed_batch(
    *,
    batch: dict[str, Any],
    encoder: LatentEncoder,
    device: torch.device,
    video_key: str,
    context_len: int,
    horizon_len: int,
) -> PreparedPackedBatch:
    """Prepare one model-ready batch from decoded observations and conditioning."""
    video = batch[video_key].to(device)
    action = batch["action"].to(device)

    latents = encoder.encode(video)
    if latents.ndim != 5:
        raise ValueError(f"encoder must return [B,C,T,H,W], got {tuple(latents.shape)}")

    total_latent_steps = int(latents.shape[2])
    context_latent_steps, horizon_latent_steps = latent_split_from_frame_ratio(
        total_latent_steps=total_latent_steps,
        context_frames=context_len,
        horizon_frames=horizon_len,
    )
    target_steps = context_latent_steps + horizon_latent_steps
    action_seq = expand_to_latent_steps(action, target_steps=target_steps)
    z_window_video = latents[:, :, :target_steps]
    z_past_video = z_window_video[:, :, :context_latent_steps]
    z_future_video = z_window_video[:, :, context_latent_steps:context_latent_steps + horizon_latent_steps]
    a_plan = action_seq[:, context_latent_steps:context_latent_steps + horizon_latent_steps]
    latent_shape = (int(latents.shape[1]), int(latents.shape[3]), int(latents.shape[4]))
    return PreparedPackedBatch(
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        a_plan=a_plan,
        latent_shape=latent_shape,
        total_latent_steps=total_latent_steps,
        context_latent_steps=context_latent_steps,
        horizon_latent_steps=horizon_latent_steps,
    )
