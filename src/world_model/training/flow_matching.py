"""Core flow-matching utilities for latent-space training.

Provides timestep sampling, noisy state construction, and timestep loss weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from world_model.chunking import build_k_plus_one_schedule
from world_model.masking import build_block_causal_mask


WeightMode = Literal["uniform", "snr", "clipped_snr"]
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000.0


class ChunkwiseVideoVelocityModel(Protocol):
    """Protocol for chunkwise velocity predictors over latent videos."""

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
        """Predict velocity for `noisy_future_video`."""


@dataclass(frozen=True)
class ChunkwiseLossInfo:
    """Diagnostics returned by `chunkwise_teacher_forcing_loss`."""

    loss: torch.Tensor
    per_chunk_losses: tuple[float, ...]
    per_chunk_lengths: tuple[int, ...]


def sample_t(
    batch_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    t_min: float = 0.0,
    t_max: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-example timesteps uniformly from `[t_min, t_max]`.

    Args:
        batch_size: Number of timesteps to sample.
        device: Optional output device.
        dtype: Output floating dtype.
        t_min: Inclusive lower bound for sampling.
        t_max: Inclusive upper bound for sampling.
        generator: Optional torch random generator.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not (0.0 <= t_min <= t_max <= 1.0):
        raise ValueError(f"Expected 0 <= t_min <= t_max <= 1, got t_min={t_min}, t_max={t_max}")

    if t_min == t_max:
        return torch.full((batch_size,), fill_value=t_min, dtype=dtype, device=device)

    return torch.rand(batch_size, device=device, dtype=dtype, generator=generator) * (t_max - t_min) + t_min


def make_noisy_and_target(
    z_clean: torch.Tensor,
    t: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct scheduler-aligned noisy state `z_t` and target velocity.

    Uses the FlowMatch Euler scheduler parameterization where normalized
    timestep `t` is the current noise scale:

    `z_t = (1 - t) * z_clean + t * noise`
    `v_target = noise - z_clean`

    Args:
        z_clean: Clean latent target with shape `[B, ...]`.
        t: Per-sample normalized scheduler timesteps with shape `[B]`.
        noise: Optional pre-sampled noise tensor `[B, ...]`.
    """
    if z_clean.ndim < 2:
        raise ValueError(f"z_clean must be at least rank-2 [B,...], got {tuple(z_clean.shape)}")
    if t.ndim != 1:
        raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
    if t.shape[0] != z_clean.shape[0]:
        raise ValueError(f"t batch {t.shape[0]} must match z_clean batch {z_clean.shape[0]}")

    if noise is None:
        noise = torch.randn_like(z_clean)
    elif noise.shape != z_clean.shape:
        raise ValueError(
            f"noise shape {tuple(noise.shape)} must match z_clean shape {tuple(z_clean.shape)}"
        )

    t = t.to(device=z_clean.device, dtype=z_clean.dtype)
    noise = noise.to(device=z_clean.device, dtype=z_clean.dtype)

    view_shape = (z_clean.shape[0],) + (1,) * (z_clean.ndim - 1)
    t_broadcast = t.view(view_shape)

    z_t = (1.0 - t_broadcast) * z_clean + t_broadcast * noise
    v_target = noise - z_clean
    return z_t, v_target


def normalized_t_to_scheduler_timestep(
    t: torch.Tensor,
    *,
    num_train_timesteps: float = DEFAULT_NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """Convert normalized FlowMatch timesteps to the scheduler scale used by Wan."""
    if not t.is_floating_point():
        raise ValueError("t must be a floating tensor")
    if num_train_timesteps <= 0:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps}")
    return t * float(num_train_timesteps)


def w(
    t: torch.Tensor,
    *,
    mode: WeightMode = "uniform",
    snr_clip_max: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-sample timestep loss weights.

    Args:
        t: Timesteps in `[0, 1]`, shape `[B]` or any tensor shape.
        mode: Weighting mode.
            - `uniform`: constant 1.
            - `snr`: `(1 - t) / (t + eps)`.
            - `clipped_snr`: same as `snr`, clipped to `snr_clip_max`.
        snr_clip_max: Upper clip used by `clipped_snr`.
        eps: Numerical stability term.
    """
    if not t.is_floating_point():
        raise ValueError("t must be a floating tensor")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    if mode == "uniform":
        return torch.ones_like(t)

    snr = (1.0 - t) / (t + eps)
    if mode == "snr":
        return snr
    if mode == "clipped_snr":
        if snr_clip_max <= 0:
            raise ValueError(f"snr_clip_max must be > 0, got {snr_clip_max}")
        return snr.clamp(max=snr_clip_max)

    raise ValueError(f"Unsupported weight mode: {mode}")


def chunkwise_teacher_forcing_loss(
    model: ChunkwiseVideoVelocityModel,
    *,
    z_past_video: torch.Tensor,
    z_future_video: torch.Tensor,
    action_tokens: torch.Tensor,
    k: int,
    t_min: float = 0.0,
    t_max: float = 1.0,
    weight_mode: WeightMode = "uniform",
    motion_loss_alpha: float = 0.0,
    motion_loss_max_weight: float = 0.0,
    motion_loss_excess_only: bool = False,
    future_loss_early_bias: float = 0.0,
    snr_clip_max: float = 5.0,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
    return_info: bool = False,
) -> torch.Tensor | ChunkwiseLossInfo:
    """Compute chunkwise teacher-forced flow matching over structured latent videos."""
    return _chunkwise_teacher_forcing_video_loss(
        model=model,
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        action_tokens=action_tokens,
        k=k,
        t_min=t_min,
        t_max=t_max,
        weight_mode=weight_mode,
        motion_loss_alpha=motion_loss_alpha,
        motion_loss_max_weight=motion_loss_max_weight,
        motion_loss_excess_only=motion_loss_excess_only,
        future_loss_early_bias=future_loss_early_bias,
        snr_clip_max=snr_clip_max,
        eps=eps,
        generator=generator,
        return_info=return_info,
    )


def _chunkwise_teacher_forcing_video_loss(
    model: ChunkwiseVideoVelocityModel,
    *,
    z_past_video: torch.Tensor | None,
    z_future_video: torch.Tensor | None,
    action_tokens: torch.Tensor | None,
    k: int,
    t_min: float,
    t_max: float,
    weight_mode: WeightMode,
    motion_loss_alpha: float,
    motion_loss_max_weight: float,
    motion_loss_excess_only: bool,
    future_loss_early_bias: float,
    snr_clip_max: float,
    eps: float,
    generator: torch.Generator | None,
    return_info: bool,
) -> torch.Tensor | ChunkwiseLossInfo:
    """Compute chunkwise teacher-forced flow matching over structured latent videos."""
    _validate_chunkwise_video_inputs(
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        action_tokens=action_tokens,
        k=k,
    )

    assert z_past_video is not None
    assert z_future_video is not None
    assert action_tokens is not None
    if motion_loss_alpha < 0.0:
        raise ValueError(f"motion_loss_alpha must be >= 0, got {motion_loss_alpha}")
    _validate_motion_loss_max_weight(motion_loss_max_weight)
    _validate_future_loss_early_bias(future_loss_early_bias)

    batch_size = z_future_video.shape[0]
    total_future_steps = z_future_video.shape[2]
    schedule = build_k_plus_one_schedule(
        future_steps=total_future_steps,
        k=k,
        device=z_future_video.device,
    )

    weighted_sum = z_future_video.new_tensor(0.0)
    weight_mass = z_future_video.new_tensor(0.0)
    per_chunk_losses: list[float] = []
    per_chunk_lengths: list[int] = []

    for start, end in schedule.boundaries:
        chunk_len = end - start
        per_chunk_lengths.append(chunk_len)

        t = sample_t(
            batch_size=batch_size,
            device=z_future_video.device,
            dtype=z_future_video.dtype,
            t_min=t_min,
            t_max=t_max,
            generator=generator,
        )
        chunk_weight = w(
            t,
            mode=weight_mode,
            snr_clip_max=snr_clip_max,
            eps=eps,
        ).to(device=z_future_video.device, dtype=z_future_video.dtype)
        chunk_weight = chunk_weight.view(batch_size, 1, 1, 1, 1)

        clean_chunk = z_future_video[:, :, start:end, :, :]
        noisy_chunk, target_chunk = make_noisy_and_target(clean_chunk, t)

        clean_suffix = z_future_video[:, :, start:, :, :]
        noisy_suffix = clean_suffix.clone()
        noisy_suffix[:, :, :chunk_len, :, :] = noisy_chunk

        observed_video = torch.cat([z_past_video, z_future_video[:, :, :start, :, :]], dim=2)
        observed_mask = torch.zeros(
            observed_video.shape[0],
            1,
            observed_video.shape[2],
            observed_video.shape[3],
            observed_video.shape[4],
            device=observed_video.device,
            dtype=observed_video.dtype,
        )
        suffix_chunk_ids = schedule.chunk_ids[start:]
        full_chunk_ids = torch.cat(
            [
                torch.full(
                    (observed_video.shape[2],),
                    fill_value=-1,
                    device=z_future_video.device,
                    dtype=torch.long,
                ),
                suffix_chunk_ids,
            ],
            dim=0,
        )
        attn_mask = build_block_causal_mask(full_chunk_ids, mask_format="additive")

        pred_suffix = model(
            noisy_future_video=noisy_suffix,
            observed_video=observed_video,
            action_tokens=action_tokens[:, start:end],
            timestep_t=normalized_t_to_scheduler_timestep(t),
            block_causal_attention_mask=attn_mask,
            observed_mask=observed_mask,
            control_hidden_states_scale=None,
        )
        pred_chunk = pred_suffix[:, :, :chunk_len, :, :]

        sq_err = (pred_chunk - target_chunk).pow(2)
        motion_weight = _compute_motion_loss_weight(
            observed_video=observed_video,
            clean_chunk=clean_chunk,
            alpha=motion_loss_alpha,
            max_weight=motion_loss_max_weight,
            excess_only=motion_loss_excess_only,
        )
        temporal_weight = _compute_future_loss_early_weight(
            start=start,
            end=end,
            total_future_steps=total_future_steps,
            bias=future_loss_early_bias,
            device=clean_chunk.device,
            dtype=clean_chunk.dtype,
        )
        sq_err = sq_err * motion_weight * temporal_weight
        per_sample_elements = clean_chunk[0].numel()
        weighted_sum = weighted_sum + (sq_err * chunk_weight).sum()
        weight_mass = weight_mass + chunk_weight.sum() * per_sample_elements

        chunk_loss = (sq_err * chunk_weight).sum() / (chunk_weight.sum() * per_sample_elements)
        per_chunk_losses.append(float(chunk_loss.detach().cpu().item()))

    total_loss = weighted_sum / weight_mass
    if return_info:
        return ChunkwiseLossInfo(
            loss=total_loss,
            per_chunk_losses=tuple(per_chunk_losses),
            per_chunk_lengths=tuple(per_chunk_lengths),
        )
    return total_loss


def _validate_chunkwise_video_inputs(
    *,
    z_past_video: torch.Tensor | None,
    z_future_video: torch.Tensor | None,
    action_tokens: torch.Tensor | None,
    k: int,
) -> None:
    """Validate structured latent-video chunkwise training inputs."""
    if z_past_video is None or z_future_video is None or action_tokens is None:
        raise ValueError("z_past_video, z_future_video, and action_tokens must all be provided for video training")
    if z_past_video.ndim != 5:
        raise ValueError(f"z_past_video must be [B,C,T,H,W], got {tuple(z_past_video.shape)}")
    if z_future_video.ndim != 5:
        raise ValueError(f"z_future_video must be [B,C,T,H,W], got {tuple(z_future_video.shape)}")
    if action_tokens.ndim != 3:
        raise ValueError(f"action_tokens must be [B,T,D], got {tuple(action_tokens.shape)}")
    if z_past_video.shape[0] != z_future_video.shape[0]:
        raise ValueError("z_past_video and z_future_video must share batch size")
    if z_past_video.shape[1] != z_future_video.shape[1]:
        raise ValueError("z_past_video and z_future_video must share channel count")
    if z_past_video.shape[3:] != z_future_video.shape[3:]:
        raise ValueError("z_past_video and z_future_video must share spatial shape")
    if action_tokens.shape[0] != z_future_video.shape[0]:
        raise ValueError("action_tokens batch size must match latent-video batch size")
    if action_tokens.shape[1] != z_future_video.shape[2]:
        raise ValueError("action_tokens time length must match z_future_video latent horizon")
    if z_future_video.shape[2] <= 0:
        raise ValueError("z_future_video must have positive time dimension")
    if k < 1:
        raise ValueError(f"k must be >= 1 for K+1 chunking, got {k}")


def _compute_motion_loss_weight(
    *,
    observed_video: torch.Tensor,
    clean_chunk: torch.Tensor,
    alpha: float,
    max_weight: float = 0.0,
    excess_only: bool = False,
) -> torch.Tensor:
    """Build a per-pixel loss weight that upweights moving latent regions."""
    _validate_motion_loss_max_weight(max_weight)
    if alpha == 0.0:
        return torch.ones_like(clean_chunk)

    previous_frame = observed_video[:, :, -1:, :, :]
    motion_source = torch.cat((previous_frame, clean_chunk), dim=2)
    motion_delta = (motion_source[:, :, 1:, :, :] - motion_source[:, :, :-1, :, :]).abs()
    motion_energy = motion_delta.mean(dim=1, keepdim=True)
    flat = motion_energy.flatten(start_dim=1)
    mean_energy = flat.mean(dim=1, keepdim=True).clamp_min(1e-6)
    normalized = (flat / mean_energy).view_as(motion_energy)
    if excess_only:
        normalized = torch.relu(normalized - 1.0)
    weight = 1.0 + alpha * normalized
    if max_weight > 0.0:
        weight = weight.clamp(max=max_weight)
    return weight


def _validate_motion_loss_max_weight(max_weight: float) -> None:
    """Reject invalid motion-loss caps before weighting the latent error."""
    if max_weight < 0.0:
        raise ValueError(f"motion_loss_max_weight must be >= 0, got {max_weight}")
    if 0.0 < max_weight < 1.0:
        raise ValueError(
            "motion_loss_max_weight must be 0 (disabled) or >= 1, got "
            f"{max_weight}"
        )


def _compute_future_loss_early_weight(
    *,
    start: int,
    end: int,
    total_future_steps: int,
    bias: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a per-frame loss weight that emphasizes earlier future positions."""
    _validate_future_loss_early_bias(bias)
    chunk_len = end - start
    if bias == 0.0:
        return torch.ones((1, 1, chunk_len, 1, 1), device=device, dtype=dtype)
    if total_future_steps <= 1:
        return torch.full((1, 1, chunk_len, 1, 1), 1.0 + bias, device=device, dtype=dtype)

    positions = torch.arange(start, end, device=device, dtype=dtype)
    reverse_progress = 1.0 - (positions / float(total_future_steps - 1))
    weight = 1.0 + bias * reverse_progress
    return weight.view(1, 1, chunk_len, 1, 1)


def _validate_future_loss_early_bias(bias: float) -> None:
    """Reject invalid early-horizon temporal weighting before loss computation."""
    if bias < 0.0:
        raise ValueError(f"future_loss_early_bias must be >= 0, got {bias}")
