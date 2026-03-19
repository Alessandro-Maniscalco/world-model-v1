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
        mlp_residual: bool = False,
        dropout: float = 0.0,
        input_layernorm: bool = True,
        temporal_difference_scale: float = 0.0,
    ) -> None:
        """Initialize an action-token projection stack."""
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        if temporal_difference_scale < 0:
            raise ValueError(
                "temporal_difference_scale must be non-negative, got "
                f"{temporal_difference_scale}"
            )

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_layernorm = bool(input_layernorm)
        self.mlp_residual = bool(mlp_residual)
        self.temporal_difference_scale = float(temporal_difference_scale)
        input_norm = nn.LayerNorm(self.action_dim) if self.input_layernorm else nn.Identity()

        if self.mlp_residual and mlp_dim is None:
            raise ValueError("mlp_residual requires a positive mlp_dim")

        if mlp_dim is None:
            self.net = nn.Sequential(
                input_norm,
                nn.Linear(self.action_dim, self.hidden_dim),
                nn.Dropout(dropout),
            )
            self.residual_net: nn.Sequential | None = None
        else:
            if mlp_dim <= 0:
                raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")
            if self.mlp_residual:
                self.net = nn.Sequential(
                    input_norm,
                    nn.Linear(self.action_dim, self.hidden_dim),
                    nn.Dropout(dropout),
                )
                self.residual_net = nn.Sequential(
                    nn.Linear(self.action_dim, int(mlp_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(mlp_dim), self.hidden_dim),
                    nn.Dropout(dropout),
                )
            else:
                self.net = nn.Sequential(
                    input_norm,
                    nn.Linear(self.action_dim, int(mlp_dim)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(mlp_dim), self.hidden_dim),
                    nn.Dropout(dropout),
                )
                self.residual_net = None

    def forward(self, a_plan: torch.Tensor) -> torch.Tensor:
        """Project `[B, T, A]` actions to `[B, T, D]` Wan cross-attention tokens."""
        if a_plan.ndim != 3:
            raise ValueError(f"a_plan must be [B,T,A], got {tuple(a_plan.shape)}")
        if a_plan.shape[-1] != self.action_dim:
            raise ValueError(
                f"a_plan last dim A={a_plan.shape[-1]} does not match action_dim={self.action_dim}"
            )
        tokens = self._project_tokens(a_plan)
        if self.temporal_difference_scale <= 0.0 or a_plan.shape[1] <= 1:
            return tokens
        delta_source = _build_temporal_differences(a_plan)
        delta_tokens = self._project_tokens(delta_source) - self._project_tokens(torch.zeros_like(delta_source))
        return tokens + (self.temporal_difference_scale * delta_tokens)

    def _project_tokens(self, a_plan: torch.Tensor) -> torch.Tensor:
        """Project one action tensor through the configured base and residual paths."""
        if self.residual_net is not None:
            normalized_actions = self.net[0](a_plan)
            base_tokens = self.net[1:](normalized_actions)
            residual_tokens = self.residual_net(normalized_actions)
            return base_tokens + residual_tokens
        return self.net(a_plan)


class NullConditioningEncoder(nn.Module):
    """Emit constant zero cross-attention tokens while ignoring conditioning inputs."""

    def __init__(self, hidden_dim: int) -> None:
        """Store the output token width used by the Wan backbone."""
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)

    def forward(self, token_source: torch.Tensor) -> torch.Tensor:
        """Return `[B,T,D]` zero tokens using only batch/time from `token_source`."""
        if token_source.ndim != 3:
            raise ValueError(f"token_source must be [B,T,F], got {tuple(token_source.shape)}")
        batch_size, steps = token_source.shape[:2]
        return token_source.new_zeros(batch_size, steps, self.hidden_dim)


def build_vace_control_tensor(
    *,
    observed_latents: torch.Tensor,
    observed_mask: torch.Tensor,
    inactive_fill_latents: torch.Tensor | None = None,
    reactive_fill_latents: torch.Tensor | None = None,
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
    inactive_fill_latents = _resolve_control_fill_latents(
        fill_latents=inactive_fill_latents,
        reference_latents=observed_latents,
        name="inactive_fill_latents",
    )
    reactive_fill_latents = _resolve_control_fill_latents(
        fill_latents=reactive_fill_latents,
        reference_latents=observed_latents,
        name="reactive_fill_latents",
    )
    mask_bool = mask.to(dtype=torch.bool).expand_as(observed_latents)
    inactive = torch.where(mask_bool, inactive_fill_latents, observed_latents)
    reactive = torch.where(mask_bool, observed_latents, reactive_fill_latents)
    mask_features = mask.expand(-1, mask_channels, -1, -1, -1)
    return torch.cat([inactive, reactive, mask_features], dim=1)


def _build_temporal_differences(a_plan: torch.Tensor) -> torch.Tensor:
    """Encode step-to-step action deltas with a zero first frame for alignment."""
    deltas = torch.zeros_like(a_plan)
    deltas[:, 1:] = a_plan[:, 1:] - a_plan[:, :-1]
    return deltas


def _resolve_control_fill_latents(
    *,
    fill_latents: torch.Tensor | None,
    reference_latents: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate optional control fill latents against the active latent tensor."""
    if fill_latents is None:
        return torch.zeros_like(reference_latents)
    if fill_latents.shape != reference_latents.shape:
        raise ValueError(
            f"{name} must match observed_latents shape {tuple(reference_latents.shape)}, "
            f"got {tuple(fill_latents.shape)}"
        )
    return fill_latents.to(device=reference_latents.device, dtype=reference_latents.dtype)
