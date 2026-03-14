"""Tests for Wan VACE-compatible conditioning helpers."""

from __future__ import annotations

import torch

from world_model.models.wan_vace_conditioning import ActionTokenEncoder, build_vace_control_tensor


def test_action_token_encoder_projects_actions_to_wan_text_width() -> None:
    """Project action sequences to the Wan cross-attention width."""
    encoder = ActionTokenEncoder(action_dim=7, hidden_dim=4096)
    tokens = encoder(torch.randn(2, 4, 7))

    assert tokens.shape == (2, 4, 4096)


def test_build_vace_control_tensor_matches_vace_channel_contract() -> None:
    """Build `[inactive; reactive; mask]` channels matching Wan VACE defaults."""
    latents = torch.randn(2, 16, 6, 8, 8)
    mask = torch.randint(0, 2, (2, 1, 6, 8, 8), dtype=torch.float32)

    control = build_vace_control_tensor(observed_latents=latents, observed_mask=mask)

    assert control.shape == (2, 96, 6, 8, 8)
    assert torch.allclose(control[:, :16], latents * (1.0 - mask))
    assert torch.allclose(control[:, 16:32], latents * mask)
    assert control[:, 32:].shape[1] == 64


def test_build_vace_control_tensor_uses_fill_latents_for_masked_regions() -> None:
    """Use fill latents instead of mathematical zeroes in masked control regions."""
    latents = torch.tensor(
        [[[[[1.0]]], [[[2.0]]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[[[0.0]]]]], dtype=torch.float32)
    inactive_fill = torch.full_like(latents, -3.0)
    reactive_fill = torch.full_like(latents, -5.0)

    observed_control = build_vace_control_tensor(
        observed_latents=latents,
        observed_mask=mask,
        inactive_fill_latents=inactive_fill,
        reactive_fill_latents=reactive_fill,
        mask_channels=1,
    )
    generated_control = build_vace_control_tensor(
        observed_latents=latents,
        observed_mask=torch.ones_like(mask),
        inactive_fill_latents=inactive_fill,
        reactive_fill_latents=reactive_fill,
        mask_channels=1,
    )

    assert torch.equal(observed_control[:, :2], latents)
    assert torch.equal(observed_control[:, 2:4], reactive_fill)
    assert torch.equal(generated_control[:, :2], inactive_fill)
    assert torch.equal(generated_control[:, 2:4], latents)
