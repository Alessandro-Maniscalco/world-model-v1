"""Latent-time window packing utilities.

Take raw latent representations (from a VAE), actions, and proprioceptive data,
and "pack" them into aligned windows that the model can process. Latent time is
authoritative: splits happen on latent timestep T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ProprioMode = Literal["last", "past"]


@dataclass(frozen=True)
class PackedLatentWindow:
    """Packed tensors for one latent-time training window.

    Shapes:
    - z_past:   [B, T_ctx, ...]
    - z_future: [B, T_hor, ...]
    - a_past:   [B, T_ctx, A]
    - q_cond:   None, [B, Q], or [B, T_ctx, Q] depending on `proprio_mode`
    """

    z_past: torch.Tensor
    z_future: torch.Tensor
    a_past: torch.Tensor
    q_cond: torch.Tensor | None


def flatten_latents_per_timestep(latents: torch.Tensor) -> torch.Tensor:
    """Convert VAE latents to per-timestep token vectors.

    Input:
    - latents: [B, C_lat, T_lat, H_lat, W_lat]

    Output:
    - tokens: [B, T_lat, Z], where Z = C_lat * H_lat * W_lat
    """
    if latents.ndim != 5:
        raise ValueError(f"Expected latents [B,C,T,H,W], got {tuple(latents.shape)}")

    b, c, t, h, w = latents.shape
    return latents.permute(0, 2, 1, 3, 4).contiguous().reshape(b, t, c * h * w)


def pack_latent_window(
    z_tokens: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor | None,
    context_steps: int,
    horizon_steps: int,
    proprio_mode: ProprioMode = "last",
) -> PackedLatentWindow:
    """Split latent tokens into past/future and align conditioning in latent time.

    Args:
    - z_tokens: [B, T_lat, Z]
    - actions:  [B, T_a, A] (either already latent-time or frame-time)
    - proprio:  [B, T_q, Q] or None
    - context_steps: number of latent steps for context
    - horizon_steps: number of latent steps for future target
    - proprio_mode:
      - "last": q_cond is [B, Q] from the last context latent step
      - "past": q_cond is [B, T_ctx, Q] aligned across context latent steps
    """
    _validate_inputs(z_tokens, actions, proprio, context_steps, horizon_steps, proprio_mode)

    total_steps = context_steps + horizon_steps
    if z_tokens.shape[1] < total_steps:
        raise ValueError(
            f"z_tokens has T={z_tokens.shape[1]} but needs at least context+horizon={total_steps}"
        )

    z_window = z_tokens[:, :total_steps]
    z_past = z_window[:, :context_steps]
    z_future = z_window[:, context_steps:context_steps + horizon_steps]

    actions_aligned = _align_time_sequence(actions, target_steps=total_steps)
    a_past = actions_aligned[:, :context_steps]

    if proprio is None:
        q_cond = None
    else:
        proprio_aligned = _align_time_sequence(proprio, target_steps=total_steps)
        if proprio_mode == "last":
            q_cond = proprio_aligned[:, context_steps - 1]
        else:
            q_cond = proprio_aligned[:, :context_steps]

    return PackedLatentWindow(z_past=z_past, z_future=z_future, a_past=a_past, q_cond=q_cond)


def _align_time_sequence(seq: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Align [B, T_src, D] to [B, target_steps, D] using nearest-neighbor resampling.

    If T_src == target_steps, returns the input unchanged.
    """
    if seq.ndim != 3:
        raise ValueError(f"Expected sequence [B,T,D], got {tuple(seq.shape)}")

    t_src = seq.shape[1]
    if t_src == target_steps:
        return seq
    if t_src <= 0:
        raise ValueError("Source sequence must have positive time dimension")

    idx = torch.linspace(0, t_src - 1, steps=target_steps, device=seq.device)
    idx = idx.round().long().clamp(0, t_src - 1)
    return seq.index_select(dim=1, index=idx)


def _validate_inputs(
    z_tokens: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor | None,
    context_steps: int,
    horizon_steps: int,
    proprio_mode: ProprioMode,
) -> None:
    if z_tokens.ndim != 3:
        raise ValueError(f"z_tokens must be [B,T,Z], got {tuple(z_tokens.shape)}")
    if actions.ndim != 3:
        raise ValueError(f"actions must be [B,T,A], got {tuple(actions.shape)}")
    if proprio is not None and proprio.ndim != 3:
        raise ValueError(f"proprio must be [B,T,Q] or None, got {tuple(proprio.shape)}")

    if context_steps <= 0 or horizon_steps <= 0:
        raise ValueError("context_steps and horizon_steps must be positive")

    if proprio_mode not in ("last", "past"):
        raise ValueError(f"Unsupported proprio_mode: {proprio_mode}")
