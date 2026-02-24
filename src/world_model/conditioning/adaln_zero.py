"""AdaLN-Zero conditioning utilities.

Applies adaptive LayerNorm modulation from a conditioning embedding.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm with zero-initialized modulation.

    This module normalizes input activations and modulates them with
    shift/scale parameters predicted from a conditioning vector.
    """

    def __init__(self, hidden_dim: int, cond_dim: int | None = None, *, eps: float = 1e-6) -> None:
        """Initialize normalization and zero-initialized modulation layers."""
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if cond_dim is None:
            cond_dim = hidden_dim
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}")

        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=eps)
        self.modulation = nn.Linear(self.cond_dim, 2 * self.hidden_dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Modulate normalized activations with conditioning parameters.

        Args:
            x: Activation tensor `[..., hidden_dim]`.
            cond: Conditioning tensor `[B, cond_dim]`.
        """
        if x.ndim < 2:
            raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}")
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} does not match hidden_dim={self.hidden_dim}"
            )
        if cond.ndim != 2:
            raise ValueError(f"cond must be [B,C], got {tuple(cond.shape)}")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"cond last dim {cond.shape[-1]} does not match cond_dim={self.cond_dim}"
            )
        if cond.shape[0] != x.shape[0]:
            raise ValueError(f"cond batch {cond.shape[0]} does not match x batch {x.shape[0]}")

        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        while shift.ndim < x.ndim:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        x_norm = self.norm(x)
        return x_norm * (1.0 + scale) + shift
