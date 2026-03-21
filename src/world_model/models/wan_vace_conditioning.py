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
        latent_summary_channels: int = 0,
        mlp_dim: int | None = None,
        mlp_residual: bool = False,
        dropout: float = 0.0,
        input_layernorm: bool = True,
        order_conditioning: bool = False,
        temporal_difference_scale: float = 0.0,
        temporal_mixer_kernel_size: int = 0,
        temporal_mixer_scale: float = 0.0,
        token_scale: float = 1.0,
    ) -> None:
        """Initialize an action-token projection stack."""
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if latent_summary_channels < 0:
            raise ValueError(
                "latent_summary_channels must be non-negative, got "
                f"{latent_summary_channels}"
            )
        if dropout < 0:
            raise ValueError(f"dropout must be non-negative, got {dropout}")
        if temporal_difference_scale < 0:
            raise ValueError(
                "temporal_difference_scale must be non-negative, got "
                f"{temporal_difference_scale}"
            )
        if temporal_mixer_kernel_size < 0:
            raise ValueError(
                "temporal_mixer_kernel_size must be non-negative, got "
                f"{temporal_mixer_kernel_size}"
            )
        if temporal_mixer_kernel_size not in (0, 1) and temporal_mixer_kernel_size % 2 == 0:
            raise ValueError(
                "temporal_mixer_kernel_size must be odd so temporal mixing preserves sequence length, "
                f"got {temporal_mixer_kernel_size}"
            )
        if temporal_mixer_scale < 0:
            raise ValueError(
                "temporal_mixer_scale must be non-negative, got "
                f"{temporal_mixer_scale}"
            )
        if temporal_mixer_scale > 0.0 and temporal_mixer_kernel_size <= 1:
            raise ValueError(
                "temporal_mixer_scale requires temporal_mixer_kernel_size >= 3, got "
                f"kernel_size={temporal_mixer_kernel_size}"
            )
        if token_scale < 0:
            raise ValueError(f"token_scale must be non-negative, got {token_scale}")

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_summary_channels = int(latent_summary_channels)
        self.input_layernorm = bool(input_layernorm)
        self.mlp_residual = bool(mlp_residual)
        self.order_conditioning = bool(order_conditioning)
        self.temporal_difference_scale = float(temporal_difference_scale)
        self.temporal_mixer_kernel_size = int(temporal_mixer_kernel_size)
        self.temporal_mixer_scale = float(temporal_mixer_scale)
        self.token_scale = float(token_scale)
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

        self.order_net: nn.Sequential | None = None
        if self.order_conditioning:
            self.order_net = nn.Sequential(
                nn.Linear(2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            nn.init.zeros_(self.order_net[-1].weight)
            if self.order_net[-1].bias is not None:
                nn.init.zeros_(self.order_net[-1].bias)

        self.temporal_mixer: nn.Conv1d | None = None
        if self.temporal_mixer_kernel_size > 1:
            padding = self.temporal_mixer_kernel_size // 2
            self.temporal_mixer = nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                kernel_size=self.temporal_mixer_kernel_size,
                padding=padding,
                groups=self.hidden_dim,
            )
            nn.init.zeros_(self.temporal_mixer.weight)
            if self.temporal_mixer.bias is not None:
                nn.init.zeros_(self.temporal_mixer.bias)
        self.latent_summary_head: nn.Linear | None = None
        if self.latent_summary_channels > 0:
            self.latent_summary_head = nn.Linear(self.hidden_dim, self.latent_summary_channels)

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
            return self._apply_output_scale(self._apply_temporal_mixer(tokens))
        delta_source = _build_temporal_differences(a_plan)
        delta_tokens = self._project_tokens(delta_source) - self._project_tokens(torch.zeros_like(delta_source))
        mixed = self._apply_temporal_mixer(tokens + (self.temporal_difference_scale * delta_tokens))
        return self._apply_output_scale(mixed)

    def _project_tokens(self, a_plan: torch.Tensor) -> torch.Tensor:
        """Project one action tensor through the configured base and residual paths."""
        if self.residual_net is not None:
            normalized_actions = self.net[0](a_plan)
            base_tokens = self.net[1:](normalized_actions)
            residual_tokens = self.residual_net(normalized_actions)
            tokens = base_tokens + residual_tokens
        else:
            tokens = self.net(a_plan)
        if self.order_net is None:
            return tokens
        return tokens + self.order_net(_build_plan_progress_features(a_plan))

    def _apply_temporal_mixer(self, tokens: torch.Tensor) -> torch.Tensor:
        """Optionally add a lightweight temporal residual over projected action tokens."""
        if self.temporal_mixer is None or self.temporal_mixer_scale <= 0.0 or tokens.shape[1] <= 1:
            return tokens
        mixed = self.temporal_mixer(tokens.transpose(1, 2)).transpose(1, 2)
        return tokens + (self.temporal_mixer_scale * mixed)

    def _apply_output_scale(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scale projected action tokens before they enter Wan cross-attention."""
        if self.token_scale == 1.0:
            return tokens
        return tokens * self.token_scale

    def predict_future_latent_summary(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-step latent summaries `[B,C,T]` from projected action tokens."""
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"tokens last dim D={tokens.shape[-1]} does not match hidden_dim={self.hidden_dim}"
            )
        if self.latent_summary_head is None:
            raise ValueError("latent_summary_head is unavailable because latent_summary_channels=0")
        return self.latent_summary_head(tokens).permute(0, 2, 1)

    def allowed_missing_state_dict_keys(self) -> set[str]:
        """List optional state-dict keys that older checkpoints may legitimately omit."""
        missing: set[str] = set()
        if self.order_net is not None:
            missing.update(
                {
                    "order_net.0.weight",
                    "order_net.0.bias",
                    "order_net.2.weight",
                    "order_net.2.bias",
                }
            )
        if self.temporal_mixer is not None:
            missing.update(
                {
                    "temporal_mixer.weight",
                    "temporal_mixer.bias",
                }
            )
        if self.latent_summary_head is not None:
            missing.update(
                {
                    "latent_summary_head.weight",
                    "latent_summary_head.bias",
                }
            )
        return missing


class ActionControlProjector(nn.Module):
    """Project action plans into a per-step latent control prior for future VACE fillers."""

    def __init__(
        self,
        action_dim: int,
        latent_channels: int,
        *,
        init_mode: str = "zero",
        observed_context_mode: str = "none",
    ) -> None:
        """Initialize the action-to-latent prior projection layer."""
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}")
        if init_mode not in {"zero", "linear_default"}:
            raise ValueError(
                "init_mode must be 'zero' or 'linear_default', got "
                f"{init_mode!r}"
            )
        if observed_context_mode not in {"none", "last_frame"}:
            raise ValueError(
                "observed_context_mode must be 'none' or 'last_frame', got "
                f"{observed_context_mode!r}"
            )
        self.action_dim = int(action_dim)
        self.latent_channels = int(latent_channels)
        self.init_mode = str(init_mode)
        self.observed_context_mode = str(observed_context_mode)
        self.projection = nn.Linear(self.action_dim + 2, self.latent_channels)
        self.context_projection: nn.Linear | None = None
        if self.observed_context_mode == "last_frame":
            self.context_projection = nn.Linear(self.latent_channels, self.latent_channels)
        if self.init_mode == "zero":
            nn.init.zeros_(self.projection.weight)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        a_plan: torch.Tensor,
        *,
        latent_height: int,
        latent_width: int,
        observed_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project `[B,T,A]` actions into `[B,C,T,H,W]` broadcast latent priors."""
        if a_plan.ndim != 3:
            raise ValueError(f"a_plan must be [B,T,A], got {tuple(a_plan.shape)}")
        if a_plan.shape[-1] != self.action_dim:
            raise ValueError(
                f"a_plan last dim A={a_plan.shape[-1]} does not match action_dim={self.action_dim}"
            )
        if latent_height <= 0:
            raise ValueError(f"latent_height must be positive, got {latent_height}")
        if latent_width <= 0:
            raise ValueError(f"latent_width must be positive, got {latent_width}")

        plan_features = torch.cat((_build_plan_progress_features(a_plan), a_plan), dim=-1)
        projected = self.projection(plan_features).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        projected = projected + self._build_observed_context_bias(
            observed_latents=observed_latents,
            steps=a_plan.shape[1],
            dtype=projected.dtype,
            device=projected.device,
        )
        return projected.expand(-1, -1, -1, latent_height, latent_width)

    def allowed_missing_state_dict_keys(self) -> set[str]:
        """List optional projector keys that older checkpoints may legitimately omit."""
        missing = {
            "projection.weight",
            "projection.bias",
        }
        if self.context_projection is not None:
            missing.update(
                {
                    "context_projection.weight",
                    "context_projection.bias",
                }
            )
        return missing

    def _build_observed_context_bias(
        self,
        *,
        observed_latents: torch.Tensor | None,
        steps: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Project pooled observed-latent state into a broadcast future latent bias."""
        if self.observed_context_mode == "none":
            if observed_latents is None:
                return torch.zeros(1, self.latent_channels, steps, 1, 1, device=device, dtype=dtype)
            return observed_latents.new_zeros(
                observed_latents.shape[0],
                self.latent_channels,
                steps,
                1,
                1,
            ).to(device=device, dtype=dtype)
        if observed_latents is None:
            raise ValueError("observed_latents are required when observed_context_mode='last_frame'")
        if observed_latents.ndim != 5:
            raise ValueError(
                f"observed_latents must be [B,C,T,H,W], got {tuple(observed_latents.shape)}"
            )
        if observed_latents.shape[1] != self.latent_channels:
            raise ValueError(
                "observed_latents channel dim must match latent_channels, got "
                f"{observed_latents.shape[1]} vs {self.latent_channels}"
            )
        if observed_latents.shape[2] <= 0:
            raise ValueError("observed_latents must include at least one timestep")
        assert self.context_projection is not None
        context_summary = observed_latents[:, :, -1].mean(dim=(2, 3))
        context_bias = self.context_projection(context_summary).to(device=device, dtype=dtype)
        return context_bias.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, steps, 1, 1)


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


class NullActionControlProjector(nn.Module):
    """Emit zero latent control priors while ignoring action-plan values."""

    def __init__(self, latent_channels: int) -> None:
        """Store the latent-channel width used by the world-model control stream."""
        super().__init__()
        if latent_channels <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}")
        self.latent_channels = int(latent_channels)

    def forward(
        self,
        a_plan: torch.Tensor,
        *,
        latent_height: int,
        latent_width: int,
        observed_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a zero `[B,C,T,H,W]` latent prior using only batch/time from `a_plan`."""
        del observed_latents
        if a_plan.ndim != 3:
            raise ValueError(f"a_plan must be [B,T,A], got {tuple(a_plan.shape)}")
        if latent_height <= 0:
            raise ValueError(f"latent_height must be positive, got {latent_height}")
        if latent_width <= 0:
            raise ValueError(f"latent_width must be positive, got {latent_width}")
        batch_size, steps = a_plan.shape[:2]
        return a_plan.new_zeros(batch_size, self.latent_channels, steps, latent_height, latent_width)

    def allowed_missing_state_dict_keys(self) -> set[str]:
        """Return the empty optional-key set because this stub has no parameters."""
        return set()


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


def _build_plan_progress_features(a_plan: torch.Tensor) -> torch.Tensor:
    """Build continuous forward/reverse progress features for each action timestep."""
    if a_plan.ndim != 3:
        raise ValueError(f"a_plan must be [B,T,A], got {tuple(a_plan.shape)}")
    steps = a_plan.shape[1]
    if steps <= 1:
        progress = torch.zeros((steps,), device=a_plan.device, dtype=a_plan.dtype)
    else:
        progress = torch.arange(steps, device=a_plan.device, dtype=a_plan.dtype) / float(steps - 1)
    reverse_progress = 1.0 - progress
    plan_features = torch.stack((progress, reverse_progress), dim=-1)
    return plan_features.unsqueeze(0).expand(a_plan.shape[0], -1, -1)


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
