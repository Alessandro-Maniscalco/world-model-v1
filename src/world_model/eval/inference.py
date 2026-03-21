"""Inference-time sampling utilities for Wan VACE latent-video prediction."""

from __future__ import annotations

from typing import Protocol

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from world_model.chunking import build_chunk_schedule, resolve_num_chunks
from world_model.masking import build_block_causal_mask


class VideoVelocityModel(Protocol):
    """Protocol for Wan VACE-style latent-video velocity predictors."""

    def __call__(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        action_image_tokens: torch.Tensor | None,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        observed_mask: torch.Tensor | None = None,
        future_action_control_prior: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity on a noisy future latent-video chunk."""


@torch.no_grad()
def infer_future_videos_chunkwise(
    model: VideoVelocityModel,
    *,
    z_past_video: torch.Tensor,
    future_steps: int,
    cross_attention_tokens: torch.Tensor,
    image_attention_tokens: torch.Tensor | None = None,
    future_action_control_prior: torch.Tensor | None = None,
    k: int,
    chunk_schedule_mode: str = "k_plus_one",
    integration_steps: int = 20,
    negative_cross_attention_tokens: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    chunk_conditioning: bool = True,
    single_chunk_rollout: bool = False,
    block_causal_attention: bool = True,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample future latent videos with chunkwise autoregressive flow-matching updates."""
    _validate_video_infer_inputs(
        z_past_video=z_past_video,
        future_steps=future_steps,
        cross_attention_tokens=cross_attention_tokens,
        image_attention_tokens=image_attention_tokens,
        future_action_control_prior=future_action_control_prior,
        negative_cross_attention_tokens=negative_cross_attention_tokens,
        integration_steps=integration_steps,
        chunk_conditioning=chunk_conditioning,
        guidance_scale=guidance_scale,
        single_chunk_rollout=single_chunk_rollout,
        k=k,
        chunk_schedule_mode=chunk_schedule_mode,
    )

    batch_size = z_past_video.shape[0]
    channels = z_past_video.shape[1]
    height = z_past_video.shape[3]
    width = z_past_video.shape[4]
    schedule_boundaries = _build_rollout_boundaries(
        future_steps=future_steps,
        k=k,
        chunk_schedule_mode=chunk_schedule_mode,
        single_chunk_rollout=single_chunk_rollout,
        device=z_past_video.device,
    )
    scheduler = FlowMatchEulerDiscreteScheduler() if scheduler is None else scheduler
    pred_future = torch.zeros(
        batch_size,
        channels,
        future_steps,
        height,
        width,
        device=z_past_video.device,
        dtype=z_past_video.dtype,
    )

    for start, end in schedule_boundaries:
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
        mask = None
        if block_causal_attention:
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

        positive_tokens = _select_chunk_conditioning_tokens(
            cross_attention_tokens,
            start=start,
            end=end,
            chunk_conditioning=chunk_conditioning,
        )
        positive_image_tokens = _select_chunk_conditioning_tokens(
            image_attention_tokens,
            start=start,
            end=end,
            chunk_conditioning=chunk_conditioning,
        )
        negative_tokens = _select_chunk_conditioning_tokens(
            negative_cross_attention_tokens,
            start=start,
            end=end,
            chunk_conditioning=chunk_conditioning,
        )
        active_control_prior = _select_chunk_conditioning_control_prior(
            future_action_control_prior,
            start=start,
            end=end,
        )

        scheduler.set_timesteps(integration_steps, device=z_past_video.device)
        timesteps = scheduler.timesteps
        for t in timesteps:
            timestep_t = t.expand(batch_size).to(device=z_past_video.device, dtype=z_past_video.dtype)
            velocity = model(
                noisy_future_video=chunk_state,
                observed_video=observed_video,
                action_tokens=positive_tokens,
                action_image_tokens=positive_image_tokens,
                timestep_t=timestep_t,
                block_causal_attention_mask=mask,
                observed_mask=observed_mask,
                future_action_control_prior=active_control_prior,
                control_hidden_states_scale=None,
            )
            if negative_tokens is not None:
                velocity_uncond = model(
                    noisy_future_video=chunk_state,
                    observed_video=observed_video,
                    action_tokens=negative_tokens,
                    action_image_tokens=None,
                    timestep_t=timestep_t,
                    block_causal_attention_mask=mask,
                    observed_mask=observed_mask,
                    future_action_control_prior=active_control_prior,
                    control_hidden_states_scale=None,
                )
                velocity = velocity_uncond + guidance_scale * (velocity - velocity_uncond)
            chunk_state = scheduler.step(velocity, t, chunk_state, generator=generator, return_dict=False)[0]

        pred_future[:, :, start:end, :, :] = chunk_state

    return pred_future


def _validate_video_infer_inputs(
    *,
    z_past_video: torch.Tensor,
    future_steps: int,
    cross_attention_tokens: torch.Tensor,
    image_attention_tokens: torch.Tensor | None,
    future_action_control_prior: torch.Tensor | None,
    negative_cross_attention_tokens: torch.Tensor | None,
    integration_steps: int,
    chunk_conditioning: bool,
    guidance_scale: float,
    single_chunk_rollout: bool,
    k: int,
    chunk_schedule_mode: str,
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
    if guidance_scale < 1.0:
        raise ValueError(f"guidance_scale must be >= 1.0, got {guidance_scale}")
    if not single_chunk_rollout and k < 1:
        raise ValueError(f"k must be >= 1 for chunked inference, got {k}")
    if chunk_schedule_mode not in {"k_plus_one", "k_chunks"}:
        raise ValueError(
            "chunk_schedule_mode must be 'k_plus_one' or 'k_chunks', "
            f"got {chunk_schedule_mode!r}"
        )
    if cross_attention_tokens.ndim != 3:
        raise ValueError(f"cross_attention_tokens must be [B,S,D], got {tuple(cross_attention_tokens.shape)}")
    if cross_attention_tokens.shape[0] != z_past_video.shape[0]:
        raise ValueError("cross_attention_tokens batch size must match z_past_video")
    if chunk_conditioning and cross_attention_tokens.shape[1] != future_steps:
        raise ValueError("chunk-conditioned cross_attention_tokens length must match requested future_steps")
    if cross_attention_tokens.shape[1] <= 0:
        raise ValueError("cross_attention_tokens sequence length must be positive")
    if negative_cross_attention_tokens is not None:
        if negative_cross_attention_tokens.shape != cross_attention_tokens.shape:
            raise ValueError("negative_cross_attention_tokens must match cross_attention_tokens shape")
    if image_attention_tokens is not None:
        if image_attention_tokens.ndim != 3:
            raise ValueError(
                f"image_attention_tokens must be [B,S,D], got {tuple(image_attention_tokens.shape)}"
            )
        if image_attention_tokens.shape[0] != z_past_video.shape[0]:
            raise ValueError("image_attention_tokens batch size must match z_past_video")
        if chunk_conditioning and image_attention_tokens.shape[1] != future_steps:
            raise ValueError("chunk-conditioned image_attention_tokens length must match requested future_steps")
    if future_action_control_prior is not None:
        if future_action_control_prior.ndim != 5:
            raise ValueError(
                "future_action_control_prior must be [B,C,T,H,W], "
                f"got {tuple(future_action_control_prior.shape)}"
            )
        if future_action_control_prior.shape[0] != z_past_video.shape[0]:
            raise ValueError("future_action_control_prior batch size must match z_past_video")
        if future_action_control_prior.shape[2] != future_steps:
            raise ValueError("future_action_control_prior time length must match future_steps")
        if future_action_control_prior.shape[3:] != z_past_video.shape[3:]:
            raise ValueError("future_action_control_prior spatial shape must match z_past_video")


def _build_rollout_boundaries(
    *,
    future_steps: int,
    k: int,
    chunk_schedule_mode: str,
    single_chunk_rollout: bool,
    device: torch.device,
) -> tuple[tuple[int, int], ...]:
    """Build rollout chunk boundaries for either K+1 or single-chunk inference."""
    if single_chunk_rollout or future_steps < resolve_num_chunks(k=k, chunk_schedule_mode=chunk_schedule_mode):
        return ((0, future_steps),)
    return build_chunk_schedule(
        future_steps=future_steps,
        k=k,
        chunk_schedule_mode=chunk_schedule_mode,
        device=device,
    ).boundaries


def _select_chunk_conditioning_tokens(
    cross_attention_tokens: torch.Tensor | None,
    *,
    start: int,
    end: int,
    chunk_conditioning: bool,
) -> torch.Tensor | None:
    """Select either per-chunk or global cross-attention tokens for the active rollout chunk."""
    if cross_attention_tokens is None:
        return None
    if chunk_conditioning:
        return cross_attention_tokens[:, start:end]
    return cross_attention_tokens


def _select_chunk_conditioning_control_prior(
    future_action_control_prior: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    """Select the active future-control prior slice for the denoised rollout chunk."""
    if future_action_control_prior is None:
        return None
    return future_action_control_prior[:, :, start:end, :, :]
