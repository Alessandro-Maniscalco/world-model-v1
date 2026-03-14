"""Tests for Wan VACE world-model control-stream assembly."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from world_model.models.wan_vace_conditioning import build_vace_control_tensor
from world_model.models.wan_vace_world_model import WanVACEWorldModel


class _RecordingBackbone(torch.nn.Module):
    """Record the control tensor passed into the Wan VACE backbone."""

    def __init__(self) -> None:
        """Store a minimal Wan config and initialize capture state."""
        super().__init__()
        self.config = SimpleNamespace(patch_size=(1, 1, 1), vace_layers=(0,))
        self.last_control_hidden_states: torch.Tensor | None = None
        self.last_control_hidden_states_scale: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        control_hidden_states_scale: torch.Tensor | None,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        """Capture the constructed control tensor and echo the hidden states."""
        del timestep, encoder_hidden_states, attention_mask, return_dict
        self.last_control_hidden_states = control_hidden_states.detach().clone()
        self.last_control_hidden_states_scale = None
        if control_hidden_states_scale is not None:
            self.last_control_hidden_states_scale = control_hidden_states_scale.detach().clone()
        return SimpleNamespace(sample=hidden_states)


def test_wan_vace_world_model_uses_black_fill_latents_with_gray_future_template() -> None:
    """Match the public VACE control layout instead of defaulting masked regions to latent zero."""
    backbone = _RecordingBackbone()
    model = WanVACEWorldModel(
        backbone=backbone,
        control_scale=0.75,
        mask_channels=1,
        control_black_latents=torch.full((1, 2, 2, 1, 1), -1.0),
        control_gray_latents=torch.full((1, 2, 2, 1, 1), 0.5),
    )
    observed_video = torch.tensor([[[[[2.0]]], [[[3.0]]]]], dtype=torch.float32)
    noisy_future_video = torch.tensor([[[[[4.0]]], [[[5.0]]]]], dtype=torch.float32)
    action_tokens = torch.randn(1, 1, 4)
    timestep_t = torch.tensor([0.25], dtype=torch.float32)
    block_causal_attention_mask = torch.zeros(2, 2, dtype=torch.float32)

    output = model(
        noisy_future_video=noisy_future_video,
        observed_video=observed_video,
        action_tokens=action_tokens,
        timestep_t=timestep_t,
        block_causal_attention_mask=block_causal_attention_mask,
    )

    expected_control_video = torch.tensor(
        [[[[[2.0]], [[0.5]]], [[[3.0]], [[0.5]]]]],
        dtype=torch.float32,
    )
    expected_control_mask = torch.tensor([[[[[0.0]], [[1.0]]]]], dtype=torch.float32)
    expected_control_hidden_states = build_vace_control_tensor(
        observed_latents=expected_control_video,
        observed_mask=expected_control_mask,
        inactive_fill_latents=torch.full_like(expected_control_video, -1.0),
        reactive_fill_latents=torch.full_like(expected_control_video, -1.0),
        mask_channels=1,
    )

    assert torch.equal(output, noisy_future_video)
    assert backbone.last_control_hidden_states is not None
    assert torch.equal(backbone.last_control_hidden_states, expected_control_hidden_states)
    assert backbone.last_control_hidden_states_scale is not None
    assert torch.equal(backbone.last_control_hidden_states_scale, torch.tensor([0.75], dtype=torch.float32))


def test_wan_vace_world_model_allows_missing_block_causal_attention_mask() -> None:
    """Skip patch-token mask expansion when the caller wants full attention."""
    backbone = _RecordingBackbone()
    model = WanVACEWorldModel(
        backbone=backbone,
        control_scale=1.0,
        mask_channels=1,
        control_black_latents=torch.full((1, 2, 2, 1, 1), -1.0),
        control_gray_latents=torch.full((1, 2, 2, 1, 1), 0.5),
    )

    output = model(
        noisy_future_video=torch.zeros(1, 2, 1, 1, 1),
        observed_video=torch.ones(1, 2, 1, 1, 1),
        action_tokens=torch.randn(1, 1, 4),
        timestep_t=torch.tensor([0.5], dtype=torch.float32),
        block_causal_attention_mask=None,
    )

    assert torch.equal(output, torch.zeros(1, 2, 1, 1, 1))
