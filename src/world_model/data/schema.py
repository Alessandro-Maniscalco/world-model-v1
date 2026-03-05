"""Typed schemas for prepared world-model training/eval batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedPackedBatch:
    """Canonical prepared batch emitted by `prepare_packed_batch`.

    Attributes:
    - z_past_video: Latent context video `[B, C, T_ctx, H, W]`.
    - z_future_video: Latent future video `[B, C, T_hor, H, W]`.
    - a_plan: Action plan aligned to future latent window `[B, T_hor, A]`.
    - latent_shape: `(C_lat, H_lat, W_lat)` for the latent-video layout.
    - total_latent_steps: Encoded total latent timesteps before split.
    - context_latent_steps: Latent context steps after split.
    - horizon_latent_steps: Latent future steps after split.
    """

    z_past_video: torch.Tensor
    z_future_video: torch.Tensor
    a_plan: torch.Tensor
    latent_shape: tuple[int, int, int]
    total_latent_steps: int
    context_latent_steps: int
    horizon_latent_steps: int
