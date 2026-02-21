"""Proprio encoder for AdaLN-style conditioning.

Projects proprio state to a single conditioning embedding per batch item.
"""

from __future__ import annotations

import torch
from torch import nn


class ProprioEncoder(nn.Module):
    """Encode proprio state `[B, Q]` into a conditioning vector `[B, D]`."""

    def __init__(
        self,
        proprio_dim: int,
        hidden_dim: int,
        *,
        enabled: bool = True,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if proprio_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {proprio_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        self.proprio_dim = int(proprio_dim)
        self.hidden_dim = int(hidden_dim)
        self.enabled = bool(enabled)

        if mlp_dim is None:
            self.net = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, self.hidden_dim),
                nn.Dropout(dropout),
            )
        else:
            if mlp_dim <= 0:
                raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
            self.net = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, int(mlp_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(mlp_dim), self.hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(self, proprio: torch.Tensor | None) -> torch.Tensor:
        """Encode proprio state to a single conditioning embedding.

        Args:
            proprio: Tensor with shape `[B, Q]`.
        """
        if proprio is None:
            raise ValueError("proprio must be a tensor with shape [B,Q], got None")
        if proprio.ndim != 2:
            raise ValueError(f"proprio must be [B,Q], got {tuple(proprio.shape)}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"proprio last dim Q={proprio.shape[1]} does not match proprio_dim={self.proprio_dim}"
            )

        if not self.enabled:
            return torch.zeros(
                proprio.shape[0],
                self.hidden_dim,
                device=proprio.device,
                dtype=proprio.dtype,
            )

        return self.net(proprio)
