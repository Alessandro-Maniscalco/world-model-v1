"""Typed schemas for prepared world-model training/eval batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedPackedBatch:
    """Canonical prepared batch emitted by `prepare_packed_batch`.

    Attributes:
    - z_past: Latent context tokens `[B, T_ctx, D]`.
    - z_future: Latent future tokens `[B, T_hor, D]`.
    - a_plan: Action plan aligned to future latent window `[B, T_hor, A]`.
    - q_last: Optional proprio snapshot `[B, Q]`.
    - latent_shape: `(C_lat, H_lat, W_lat)` for token->latent reshaping.
    - total_latent_steps: Encoded total latent timesteps before split.
    - context_latent_steps: Latent context steps after split.
    - horizon_latent_steps: Latent future steps after split.
    """

    z_past: torch.Tensor
    z_future: torch.Tensor
    a_plan: torch.Tensor
    q_last: torch.Tensor | None
    latent_shape: tuple[int, int, int]
    total_latent_steps: int
    context_latent_steps: int
    horizon_latent_steps: int
