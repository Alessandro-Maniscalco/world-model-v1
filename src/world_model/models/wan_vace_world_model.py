"""Local adapter around the vendored Wan VACE backbone for world-model training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from world_model.models.wan_vace_conditioning import build_vace_control_tensor
from world_model.vendor.wan import WanVACETransformer3DModel


def expand_block_causal_mask_to_patch_tokens(
    block_causal_attention_mask: torch.Tensor,
    *,
    patches_per_frame: int,
) -> torch.Tensor:
    """Expand a latent-frame additive mask into Wan patch-token space."""
    if patches_per_frame <= 0:
        raise ValueError(f"patches_per_frame must be positive, got {patches_per_frame}")
    if block_causal_attention_mask.ndim == 2:
        return block_causal_attention_mask.repeat_interleave(patches_per_frame, dim=0).repeat_interleave(
            patches_per_frame, dim=1
        )
    if block_causal_attention_mask.ndim == 3:
        return block_causal_attention_mask.repeat_interleave(patches_per_frame, dim=1).repeat_interleave(
            patches_per_frame, dim=2
        )
    raise ValueError(
        "block_causal_attention_mask must have shape [L, L] or [B, L, L], "
        f"got {tuple(block_causal_attention_mask.shape)}"
    )


class WanVACEWorldModel(nn.Module):
    """Adapt repo chunkwise tensors to the vendored Wan VACE backbone."""

    def __init__(
        self,
        *,
        backbone: nn.Module | None = None,
        control_scale: float = 1.0,
        mask_channels: int = 64,
    ) -> None:
        """Store the backbone and control-stream defaults."""
        super().__init__()
        self.backbone = backbone if backbone is not None else WanVACETransformer3DModel()
        self.control_scale = float(control_scale)
        self.mask_channels = int(mask_channels)

    def forward(
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
        """Run the Wan VACE backbone on chunkwise world-model inputs."""
        if noisy_future_video.ndim != 5:
            raise ValueError(f"noisy_future_video must be [B,C,T,H,W], got {tuple(noisy_future_video.shape)}")
        if observed_video.ndim != 5:
            raise ValueError(f"observed_video must be [B,C,T,H,W], got {tuple(observed_video.shape)}")
        if action_tokens.ndim != 3:
            raise ValueError(f"action_tokens must be [B,T,D], got {tuple(action_tokens.shape)}")
        if timestep_t.ndim != 1:
            raise ValueError(f"timestep_t must be [B], got {tuple(timestep_t.shape)}")

        if observed_mask is None:
            observed_mask = torch.zeros(
                observed_video.shape[0],
                1,
                observed_video.shape[2],
                observed_video.shape[3],
                observed_video.shape[4],
                device=observed_video.device,
                dtype=observed_video.dtype,
            )
        elif observed_mask.shape != (
            observed_video.shape[0],
            1,
            observed_video.shape[2],
            observed_video.shape[3],
            observed_video.shape[4],
        ):
            raise ValueError(
                "observed_mask must match observed_video as [B,1,T,H,W], "
                f"got {tuple(observed_mask.shape)} for observed_video {tuple(observed_video.shape)}"
            )

        full_hidden_states = torch.cat([observed_video, noisy_future_video], dim=2)
        future_control_video = torch.zeros_like(noisy_future_video)
        future_control_mask = torch.ones(
            noisy_future_video.shape[0],
            1,
            noisy_future_video.shape[2],
            noisy_future_video.shape[3],
            noisy_future_video.shape[4],
            device=noisy_future_video.device,
            dtype=noisy_future_video.dtype,
        )
        control_video = torch.cat([observed_video, future_control_video], dim=2)
        control_mask = torch.cat([observed_mask, future_control_mask], dim=2)

        control_hidden_states = build_vace_control_tensor(
            observed_latents=control_video,
            observed_mask=control_mask,
            mask_channels=self.mask_channels,
        )
        expected_frames = full_hidden_states.shape[2]
        if block_causal_attention_mask.shape[-1] != expected_frames or block_causal_attention_mask.shape[-2] != expected_frames:
            raise ValueError(
                "block_causal_attention_mask must match the full rollout length "
                f"{expected_frames}, got {tuple(block_causal_attention_mask.shape)}"
            )
        patches_per_frame = _patches_per_frame(video=full_hidden_states, backbone=self.backbone)
        attention_mask = expand_block_causal_mask_to_patch_tokens(
            block_causal_attention_mask,
            patches_per_frame=patches_per_frame,
        ).to(device=full_hidden_states.device, dtype=full_hidden_states.dtype)
        control_scale = _resolve_control_scale(
            backbone=self.backbone,
            control_hidden_states_scale=control_hidden_states_scale,
            default_scale=self.control_scale,
            device=full_hidden_states.device,
            dtype=full_hidden_states.dtype,
        )

        output = self.backbone(
            hidden_states=full_hidden_states,
            timestep=timestep_t,
            encoder_hidden_states=action_tokens,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=control_scale,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sample = getattr(output, "sample", None)
        if sample is not None:
            return sample[:, :, observed_video.shape[2] :, :, :]
        if isinstance(output, tuple):
            return output[0][:, :, observed_video.shape[2] :, :, :]
        raise ValueError(f"Unsupported backbone output type: {type(output).__name__}")


def _patches_per_frame(*, video: torch.Tensor, backbone: nn.Module) -> int:
    """Compute the number of Wan patch tokens emitted per latent frame."""
    patch_size = getattr(getattr(backbone, "config", None), "patch_size", (1, 2, 2))
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError(f"Expected temporal patch size 1 for latent-frame masks, got {patch_t}")
    height = video.shape[-2]
    width = video.shape[-1]
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Video spatial size {(height, width)} must be divisible by patch size {(patch_h, patch_w)}"
        )
    return (height // patch_h) * (width // patch_w)


def _resolve_control_scale(
    *,
    backbone: nn.Module,
    control_hidden_states_scale: torch.Tensor | None,
    default_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build a per-VACE-layer control scale tensor when the backbone exposes that config."""
    if control_hidden_states_scale is not None:
        return control_hidden_states_scale.to(device=device, dtype=dtype)
    vace_layers = getattr(getattr(backbone, "config", None), "vace_layers", None)
    if vace_layers is None:
        return None
    return torch.full((len(vace_layers),), fill_value=default_scale, device=device, dtype=dtype)
