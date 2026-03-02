"""Tests for the Wan VACE world-model adapter."""

from __future__ import annotations

import types

import torch
from torch import nn

from world_model.models.wan_vace_world_model import WanVACEWorldModel, expand_block_causal_mask_to_patch_tokens


def test_expand_block_causal_mask_to_patch_tokens_repeats_frame_blocks() -> None:
    """Expand a latent-time mask into Wan patch-token space."""
    mask = torch.tensor(
        [
            [0.0, float("-inf")],
            [0.0, 0.0],
        ]
    )

    expanded = expand_block_causal_mask_to_patch_tokens(mask, patches_per_frame=4)

    assert expanded.shape == (8, 8)
    assert torch.all(expanded[:4, :4] == 0.0)
    assert torch.all(expanded[:4, 4:] == float("-inf"))
    assert torch.all(expanded[4:, :4] == 0.0)


def test_wan_vace_world_model_forwards_chunk_inputs() -> None:
    """Build Wan-compatible inputs and return the backbone sample tensor."""

    class _FakeBackbone(nn.Module):
        """Capture adapter kwargs and echo the hidden-state shape."""

        def __init__(self) -> None:
            """Initialize fake config and call storage."""
            super().__init__()
            self.config = types.SimpleNamespace(patch_size=(1, 2, 2), vace_layers=[0])
            self.kwargs: dict[str, torch.Tensor | None] = {}

        def forward(self, **kwargs):
            """Capture kwargs and return a diffusers-style output object."""
            self.kwargs = kwargs
            return types.SimpleNamespace(sample=kwargs["hidden_states"])

    backbone = _FakeBackbone()
    model = WanVACEWorldModel(backbone=backbone)
    noisy_future_video = torch.randn(2, 16, 4, 8, 8)
    observed_video = torch.randn(2, 16, 6, 8, 8)
    action_tokens = torch.randn(2, 4, 4096)
    timestep_t = torch.rand(2)
    frame_mask = torch.zeros(10, 10)

    out = model(
        noisy_future_video=noisy_future_video,
        observed_video=observed_video,
        action_tokens=action_tokens,
        timestep_t=timestep_t,
        block_causal_attention_mask=frame_mask,
    )

    assert out.shape == noisy_future_video.shape
    assert backbone.kwargs["hidden_states"].shape == (2, 16, 10, 8, 8)
    assert backbone.kwargs["encoder_hidden_states"].shape == action_tokens.shape
    assert backbone.kwargs["control_hidden_states"].shape == (2, 96, 10, 8, 8)
    assert backbone.kwargs["attention_mask"].shape == (160, 160)
