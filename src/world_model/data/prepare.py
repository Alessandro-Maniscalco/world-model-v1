"""Shared batch-preparation pipeline for training and inference scripts."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from world_model.data.pack import ProprioMode, flatten_latents_per_timestep, pack_latent_window
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
    proprio_mode: ProprioMode = "last",
) -> PreparedPackedBatch:
    """Prepare one model-ready batch from decoded observations and conditioning."""
    video = batch[video_key].to(device)
    action = batch["action"].to(device)
    proprio = batch.get("observation.state")
    if proprio is not None:
        proprio = proprio.to(device)

    latents = encoder.encode(video)
    if latents.ndim != 5:
        raise ValueError(f"encoder must return [B,C,T,H,W], got {tuple(latents.shape)}")

    z_tokens = flatten_latents_per_timestep(latents)
    total_latent_steps = int(z_tokens.shape[1])
    context_latent_steps, horizon_latent_steps = latent_split_from_frame_ratio(
        total_latent_steps=total_latent_steps,
        context_frames=context_len,
        horizon_frames=horizon_len,
    )
    target_steps = context_latent_steps + horizon_latent_steps
    action_seq = expand_to_latent_steps(action, target_steps=target_steps)
    proprio_seq = None if proprio is None else expand_to_latent_steps(proprio, target_steps=target_steps)
    z_window_video = latents[:, :, :target_steps]
    z_past_video = z_window_video[:, :, :context_latent_steps]
    z_future_video = z_window_video[:, :, context_latent_steps:context_latent_steps + horizon_latent_steps]

    packed = pack_latent_window(
        z_tokens=z_tokens,
        actions=action_seq,
        proprio=proprio_seq,
        context_steps=context_latent_steps,
        horizon_steps=horizon_latent_steps,
        proprio_mode=proprio_mode,
    )
    q_last = packed.q_cond if proprio_mode == "last" else None
    latent_shape = (int(latents.shape[1]), int(latents.shape[3]), int(latents.shape[4]))
    return PreparedPackedBatch(
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        z_past=packed.z_past,
        z_future=packed.z_future,
        a_plan=packed.a_plan,
        q_last=q_last,
        latent_shape=latent_shape,
        total_latent_steps=total_latent_steps,
        context_latent_steps=context_latent_steps,
        horizon_latent_steps=horizon_latent_steps,
    )
