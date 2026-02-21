"""Action-plan encoder for AdaLN-style conditioning.

Pools an action plan over time and projects it to the model hidden size.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


PoolType = Literal["mean", "last", "flatten"]


class ActionEncoder(nn.Module):
    """Encode an action plan `[B, T, A]` into a single conditioning vector `[B, D]`.

    This module is intentionally token-free: it returns one embedding per batch
    element so actions can condition the backbone via AdaLN (or similar) without
    becoming part of the transformer's token sequence.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        *,
        pool: PoolType = "mean",
        horizon_steps: int | None = None,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        if pool == "flatten" and (horizon_steps is None or horizon_steps <= 0):
            raise ValueError("horizon_steps must be set and positive when pool='flatten'")

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool: PoolType = pool
        self.horizon_steps = int(horizon_steps) if horizon_steps is not None else None

        in_dim = self.action_dim if pool != "flatten" else self.action_dim * self.horizon_steps  # type: ignore[operator]

        if mlp_dim is None:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, self.hidden_dim),
                nn.Dropout(dropout),
            )
        else:
            if mlp_dim <= 0:
                raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, int(mlp_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(mlp_dim), self.hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        a_plan: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode an action plan into a conditioning embedding.

        Args:
            a_plan: Action plan tensor with shape `[B, T, A]`.
            valid_mask: Optional boolean mask `[B, T]` where True marks a valid
                (non-padded) timestep.
        """
        if a_plan.ndim != 3:
            raise ValueError(f"a_plan must be [B,T,A], got {tuple(a_plan.shape)}")
        if a_plan.shape[2] != self.action_dim:
            raise ValueError(
                f"a_plan last dim A={a_plan.shape[2]} does not match action_dim={self.action_dim}"
            )
        if valid_mask is not None:
            if valid_mask.ndim != 2:
                raise ValueError(f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}")
            if valid_mask.shape[0] != a_plan.shape[0] or valid_mask.shape[1] != a_plan.shape[1]:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} must match a_plan [B,T]=({a_plan.shape[0]},{a_plan.shape[1]})"
                )
            valid_mask = valid_mask.to(device=a_plan.device, dtype=torch.bool)

        pooled = _pool_action_plan(
            a_plan=a_plan,
            pool=self.pool,
            horizon_steps=self.horizon_steps,
            valid_mask=valid_mask,
        )
        return self.net(pooled)


def _pool_action_plan(
    *,
    a_plan: torch.Tensor,
    pool: PoolType,
    horizon_steps: int | None,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if pool == "mean":
        if valid_mask is None:
            return a_plan.mean(dim=1)
        weights = valid_mask.unsqueeze(-1).to(dtype=a_plan.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (a_plan * weights).sum(dim=1) / denom

    if pool == "last":
        if valid_mask is None:
            return a_plan[:, -1]
        lengths = valid_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)  # [B]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, a_plan.shape[2])
        return a_plan.gather(dim=1, index=idx).squeeze(1)

    if pool == "flatten":
        if horizon_steps is None:
            raise ValueError("horizon_steps must be provided for pool='flatten'")
        if a_plan.shape[1] != horizon_steps:
            raise ValueError(
                f"Expected a_plan T={horizon_steps} for pool='flatten', got T={a_plan.shape[1]}"
            )
        if valid_mask is not None:
            a_plan = a_plan * valid_mask.unsqueeze(-1).to(dtype=a_plan.dtype)
        return a_plan.reshape(a_plan.shape[0], a_plan.shape[1] * a_plan.shape[2])

    raise ValueError(f"Unsupported pool type: {pool}")
