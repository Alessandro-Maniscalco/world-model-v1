"""Inference-time sampling utilities for world-model future prediction.

Includes chunkwise autoregressive latent sampling and latent/video formatting helpers.
"""

from __future__ import annotations

from typing import Protocol

import torch

from world_model.chunking import build_k_plus_one_schedule
from world_model.masking import build_block_causal_mask


class VelocityModel(Protocol):
    """Protocol for velocity predictors used during inference sampling."""

    def __call__(
        self,
        *,
        noisy_future_chunk: torch.Tensor,
        past_clean_chunks: torch.Tensor,
        action_conditioning: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        proprio_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity on a noisy future chunk."""


class VideoVelocityModel(Protocol):
    """Protocol for Wan VACE-style latent-video velocity predictors."""

    def __call__(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity on a noisy future latent-video chunk."""


@torch.no_grad()
def infer_future_tokens_chunkwise(
    model: VelocityModel,
    *,
    z_past: torch.Tensor,
    future_steps: int,
    action_conditioning: torch.Tensor,
    k: int,
    proprio_conditioning: torch.Tensor | None = None,
    integration_steps: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample clean future tokens with chunkwise autoregressive Euler integration."""
    _validate_infer_inputs(
        z_past=z_past,
        future_steps=future_steps,
        action_conditioning=action_conditioning,
        proprio_conditioning=proprio_conditioning,
        integration_steps=integration_steps,
    )

    batch_size = z_past.shape[0]
    feature_dim = z_past.shape[2]
    schedule = build_k_plus_one_schedule(future_steps=future_steps, k=k, device=z_past.device)
    pred_future = torch.zeros(batch_size, future_steps, feature_dim, device=z_past.device, dtype=z_past.dtype)

    dt = 1.0 / float(integration_steps)
    for start, end in schedule.boundaries:
        chunk_len = end - start
        chunk_state = torch.randn(
            batch_size,
            chunk_len,
            feature_dim,
            device=z_past.device,
            dtype=z_past.dtype,
            generator=generator,
        )
        teacher_forced_context = torch.cat([z_past, pred_future[:, :start, :]], dim=1)
        chunk_ids = torch.cat(
            [
                torch.full(
                    (teacher_forced_context.shape[1],),
                    fill_value=-1,
                    device=z_past.device,
                    dtype=torch.long,
                ),
                torch.zeros(chunk_len, device=z_past.device, dtype=torch.long),
            ],
            dim=0,
        )
        mask = build_block_causal_mask(chunk_ids, mask_format="additive")

        for step in range(integration_steps):
            t = torch.full(
                (batch_size,),
                fill_value=(step + 0.5) * dt,
                device=z_past.device,
                dtype=z_past.dtype,
            )
            velocity = model(
                noisy_future_chunk=chunk_state,
                past_clean_chunks=teacher_forced_context,
                action_conditioning=action_conditioning,
                timestep_t=t,
                block_causal_attention_mask=mask,
                proprio_conditioning=proprio_conditioning,
            )
            chunk_state = chunk_state + dt * velocity

        pred_future[:, start:end, :] = chunk_state

    return pred_future


@torch.no_grad()
def infer_future_videos_chunkwise(
    model: VideoVelocityModel,
    *,
    z_past_video: torch.Tensor,
    future_steps: int,
    action_tokens: torch.Tensor,
    k: int,
    integration_steps: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample future latent videos with chunkwise autoregressive Euler integration."""
    _validate_video_infer_inputs(
        z_past_video=z_past_video,
        future_steps=future_steps,
        action_tokens=action_tokens,
        integration_steps=integration_steps,
    )

    batch_size = z_past_video.shape[0]
    channels = z_past_video.shape[1]
    height = z_past_video.shape[3]
    width = z_past_video.shape[4]
    schedule = build_k_plus_one_schedule(future_steps=future_steps, k=k, device=z_past_video.device)
    pred_future = torch.zeros(
        batch_size,
        channels,
        future_steps,
        height,
        width,
        device=z_past_video.device,
        dtype=z_past_video.dtype,
    )

    dt = 1.0 / float(integration_steps)
    for start, end in schedule.boundaries:
        chunk_len = end - start
        chunk_state = torch.randn(
            batch_size,
            channels,
            chunk_len,
            height,
            width,
            device=z_past_video.device,
            dtype=z_past_video.dtype,
            generator=generator,
        )
        observed_video = torch.cat([z_past_video, pred_future[:, :, :start, :, :]], dim=2)
        observed_mask = torch.zeros(
            observed_video.shape[0],
            1,
            observed_video.shape[2],
            observed_video.shape[3],
            observed_video.shape[4],
            device=observed_video.device,
            dtype=observed_video.dtype,
        )
        full_chunk_ids = torch.cat(
            [
                torch.full(
                    (observed_video.shape[2],),
                    fill_value=-1,
                    device=z_past_video.device,
                    dtype=torch.long,
                ),
                torch.zeros(chunk_len, device=z_past_video.device, dtype=torch.long),
            ],
            dim=0,
        )
        mask = build_block_causal_mask(full_chunk_ids, mask_format="additive")

        for step in range(integration_steps):
            t = torch.full(
                (batch_size,),
                fill_value=(step + 0.5) * dt,
                device=z_past_video.device,
                dtype=z_past_video.dtype,
            )
            velocity = model(
                noisy_future_video=chunk_state,
                observed_video=observed_video,
                action_tokens=action_tokens[:, start:end],
                timestep_t=t,
                block_causal_attention_mask=mask,
                observed_mask=observed_mask,
                control_hidden_states_scale=None,
            )
            chunk_state = chunk_state + dt * velocity

        pred_future[:, :, start:end, :, :] = chunk_state

    return pred_future


def tokens_to_latents(
    tokens: torch.Tensor,
    *,
    latent_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Convert tokenized latent vectors `[B,T,Z]` into `[B,C,T,H,W]` latents."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B,T,Z], got {tuple(tokens.shape)}")
    c_lat, h_lat, w_lat = latent_shape
    expected_z = c_lat * h_lat * w_lat
    if tokens.shape[2] != expected_z:
        raise ValueError(
            f"Token feature dim {tokens.shape[2]} does not match latent shape product {expected_z}"
        )
    return tokens.reshape(tokens.shape[0], tokens.shape[1], c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4).contiguous()


def _validate_infer_inputs(
    *,
    z_past: torch.Tensor,
    future_steps: int,
    action_conditioning: torch.Tensor,
    proprio_conditioning: torch.Tensor | None,
    integration_steps: int,
) -> None:
    """Validate tensor shapes and scalar params for inference sampling."""
    if z_past.ndim != 3:
        raise ValueError(f"z_past must be [B,T,D], got {tuple(z_past.shape)}")
    if z_past.shape[1] <= 0:
        raise ValueError("z_past must have positive context length")
    if z_past.shape[2] <= 0:
        raise ValueError("z_past feature dimension must be positive")
    if future_steps <= 0:
        raise ValueError(f"future_steps must be positive, got {future_steps}")
    if integration_steps <= 0:
        raise ValueError(f"integration_steps must be positive, got {integration_steps}")
    if action_conditioning.ndim != 2:
        raise ValueError(
            f"action_conditioning must be [B,C], got {tuple(action_conditioning.shape)}"
        )
    if action_conditioning.shape[0] != z_past.shape[0]:
        raise ValueError("action_conditioning batch size must match z_past")
    if proprio_conditioning is not None:
        if proprio_conditioning.ndim != 2:
            raise ValueError(
                f"proprio_conditioning must be [B,C], got {tuple(proprio_conditioning.shape)}"
            )
        if proprio_conditioning.shape[0] != z_past.shape[0]:
            raise ValueError("proprio_conditioning batch size must match z_past")


def _validate_video_infer_inputs(
    *,
    z_past_video: torch.Tensor,
    future_steps: int,
    action_tokens: torch.Tensor,
    integration_steps: int,
) -> None:
    """Validate structured latent-video inputs for Wan VACE inference sampling."""
    if z_past_video.ndim != 5:
        raise ValueError(f"z_past_video must be [B,C,T,H,W], got {tuple(z_past_video.shape)}")
    if z_past_video.shape[2] <= 0:
        raise ValueError("z_past_video must have positive context length")
    if future_steps <= 0:
        raise ValueError(f"future_steps must be positive, got {future_steps}")
    if integration_steps <= 0:
        raise ValueError(f"integration_steps must be positive, got {integration_steps}")
    if action_tokens.ndim != 3:
        raise ValueError(f"action_tokens must be [B,T,D], got {tuple(action_tokens.shape)}")
    if action_tokens.shape[0] != z_past_video.shape[0]:
        raise ValueError("action_tokens batch size must match z_past_video")
    if action_tokens.shape[1] != future_steps:
        raise ValueError("action_tokens time length must match requested future_steps")
