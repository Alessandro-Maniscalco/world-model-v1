"""Wan VACE-compatible conditioning helpers for action tokens and control tensors."""

from __future__ import annotations

import torch
from torch import nn


class ActionTokenEncoder(nn.Module):
    """Project action sequences into the Wan cross-attention embedding space."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        *,
        mlp_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialize an action-token projection stack."""
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        if mlp_dim is None:
            self.net = nn.Sequential(
                nn.LayerNorm(self.action_dim),
                nn.Linear(self.action_dim, self.hidden_dim),
                nn.Dropout(dropout),
            )
        else:
            if mlp_dim <= 0:
                raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
            self.net = nn.Sequential(
                nn.LayerNorm(self.action_dim),
                nn.Linear(self.action_dim, int(mlp_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(mlp_dim), self.hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(self, a_plan: torch.Tensor) -> torch.Tensor:
        """Project `[B, T, A]` actions to `[B, T, D]` Wan cross-attention tokens."""
        if a_plan.ndim != 3:
            raise ValueError(f"a_plan must be [B,T,A], got {tuple(a_plan.shape)}")
        if a_plan.shape[-1] != self.action_dim:
            raise ValueError(
                f"a_plan last dim A={a_plan.shape[-1]} does not match action_dim={self.action_dim}"
            )
        return self.net(a_plan)


def build_vace_control_tensor(
    *,
    observed_latents: torch.Tensor,
    observed_mask: torch.Tensor,
    mask_channels: int = 64,
) -> torch.Tensor:
    """Build the Wan VACE control tensor `[inactive; reactive; mask]`.

    The default `mask_channels=64` matches Wan VACE 1.3B when using 16 latent
    channels and the standard diffusers mask expansion layout.
    """
    if observed_latents.ndim != 5:
        raise ValueError(f"observed_latents must be [B,C,T,H,W], got {tuple(observed_latents.shape)}")
    if observed_mask.ndim != 5:
        raise ValueError(f"observed_mask must be [B,1,T,H,W], got {tuple(observed_mask.shape)}")
    if observed_mask.shape[0] != observed_latents.shape[0] or observed_mask.shape[2:] != observed_latents.shape[2:]:
        raise ValueError(
            f"observed_mask shape {tuple(observed_mask.shape)} must match observed_latents batch/time/space "
            f"{(observed_latents.shape[0], observed_latents.shape[2], observed_latents.shape[3], observed_latents.shape[4])}"
        )
    if observed_mask.shape[1] != 1:
        raise ValueError(f"observed_mask channel dim must be 1, got {observed_mask.shape[1]}")
    if mask_channels <= 0:
        raise ValueError(f"mask_channels must be positive, got {mask_channels}")

    mask = observed_mask.to(device=observed_latents.device, dtype=observed_latents.dtype)
    inactive = observed_latents * (1.0 - mask)
    reactive = observed_latents * mask
    mask_features = mask.expand(-1, mask_channels, -1, -1, -1)
    return torch.cat([inactive, reactive, mask_features], dim=1)
