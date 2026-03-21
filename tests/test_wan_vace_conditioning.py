"""Tests for Wan VACE-compatible conditioning helpers."""

from __future__ import annotations

import pytest
import torch

from world_model.models.wan_vace_conditioning import (
    ActionControlProjector,
    ActionTokenEncoder,
    build_vace_control_tensor,
)


def test_action_token_encoder_projects_actions_to_wan_text_width() -> None:
    """Project action sequences to the Wan cross-attention width."""
    encoder = ActionTokenEncoder(action_dim=7, hidden_dim=4096)
    tokens = encoder(torch.randn(2, 4, 7))

    assert tokens.shape == (2, 4, 4096)


def test_action_token_encoder_supports_two_layer_mlp_projection() -> None:
    """Use the optional hidden MLP width when a deeper action encoder is requested."""
    encoder = ActionTokenEncoder(action_dim=7, hidden_dim=32, mlp_dim=11, input_layernorm=False)
    tokens = encoder(torch.randn(2, 4, 7))

    assert tokens.shape == (2, 4, 32)
    assert encoder.net[1].out_features == 11
    assert encoder.net[4].out_features == 32


def test_action_token_encoder_supports_residual_mlp_projection() -> None:
    """Allow the optional action MLP to augment the legacy linear path residually."""
    encoder = ActionTokenEncoder(action_dim=7, hidden_dim=32, mlp_dim=11, mlp_residual=True, input_layernorm=False)
    tokens = encoder(torch.randn(2, 4, 7))

    assert tokens.shape == (2, 4, 32)
    assert encoder.net[1].out_features == 32
    assert encoder.residual_net is not None
    assert encoder.residual_net[0].out_features == 11
    assert encoder.residual_net[3].out_features == 32


def test_action_token_encoder_residual_mlp_preserves_linear_path_when_zeroed() -> None:
    """Keep the legacy linear projection exactly when the residual MLP contributes zero."""
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 7)

    linear_encoder = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    residual_encoder = ActionTokenEncoder(
        action_dim=7,
        hidden_dim=32,
        mlp_dim=11,
        mlp_residual=True,
        input_layernorm=False,
    )
    residual_encoder.net.load_state_dict(linear_encoder.net.state_dict())
    for parameter in residual_encoder.residual_net.parameters():
        parameter.data.zero_()

    assert torch.allclose(linear_encoder(actions), residual_encoder(actions))


def test_action_token_encoder_scale_invariance_depends_on_input_layernorm() -> None:
    """Show how the optional input LayerNorm changes positive scale sensitivity."""
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 7)
    scaled = actions * 1.5

    normalized_encoder = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=True)
    unnormalized_encoder = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    unnormalized_encoder.load_state_dict(normalized_encoder.state_dict(), strict=False)

    normalized_tokens = normalized_encoder(actions)
    normalized_scaled_tokens = normalized_encoder(scaled)
    unnormalized_tokens = unnormalized_encoder(actions)
    unnormalized_scaled_tokens = unnormalized_encoder(scaled)

    assert torch.allclose(normalized_tokens, normalized_scaled_tokens, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(unnormalized_tokens, unnormalized_scaled_tokens)


def test_action_token_encoder_token_scale_multiplies_projected_tokens() -> None:
    """Scale final projected action tokens before they reach cross-attention."""
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 7)

    baseline = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    scaled = ActionTokenEncoder(
        action_dim=7,
        hidden_dim=32,
        input_layernorm=False,
        token_scale=2.5,
    )
    scaled.load_state_dict(baseline.state_dict(), strict=False)

    assert torch.allclose(scaled(actions), 2.5 * baseline(actions), atol=1e-5)


def test_action_token_encoder_temporal_difference_scale_is_noop_for_constant_actions() -> None:
    """Keep outputs unchanged when temporal differences are identically zero."""
    torch.manual_seed(0)
    actions = torch.ones(2, 4, 7)

    baseline = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    temporal = ActionTokenEncoder(
        action_dim=7,
        hidden_dim=32,
        input_layernorm=False,
        temporal_difference_scale=0.75,
    )
    temporal.load_state_dict(baseline.state_dict(), strict=False)

    assert torch.allclose(baseline(actions), temporal(actions))


def test_action_token_encoder_temporal_difference_scale_changes_varying_sequences() -> None:
    """Let the optional temporal-difference path change outputs for non-constant plans."""
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 7)

    baseline = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    temporal = ActionTokenEncoder(
        action_dim=7,
        hidden_dim=32,
        input_layernorm=False,
        temporal_difference_scale=0.75,
    )
    temporal.load_state_dict(baseline.state_dict(), strict=False)

    assert not torch.allclose(baseline(actions), temporal(actions))


def test_action_token_encoder_temporal_mixer_is_noop_until_trained() -> None:
    """Keep outputs unchanged when the optional temporal mixer starts from zero weights."""
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 7)

    baseline = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    temporal = ActionTokenEncoder(
        action_dim=7,
        hidden_dim=32,
        input_layernorm=False,
        temporal_mixer_kernel_size=3,
        temporal_mixer_scale=0.5,
    )
    temporal.load_state_dict(baseline.state_dict(), strict=False)

    assert torch.allclose(baseline(actions), temporal(actions))
    assert temporal.allowed_missing_state_dict_keys() == {"temporal_mixer.weight", "temporal_mixer.bias"}


def test_action_token_encoder_order_conditioning_is_noop_until_trained() -> None:
    """Keep legacy outputs unchanged when learned order features start from zero."""
    torch.manual_seed(0)
    actions = torch.randn(2, 4, 7)

    baseline = ActionTokenEncoder(action_dim=7, hidden_dim=32, input_layernorm=False)
    ordered = ActionTokenEncoder(
        action_dim=7,
        hidden_dim=32,
        input_layernorm=False,
        order_conditioning=True,
    )
    ordered.load_state_dict(baseline.state_dict(), strict=False)

    assert torch.allclose(baseline(actions), ordered(actions))
    assert {
        "order_net.0.weight",
        "order_net.0.bias",
        "order_net.2.weight",
        "order_net.2.bias",
    }.issubset(ordered.allowed_missing_state_dict_keys())


def test_action_token_encoder_order_conditioning_can_distinguish_permuted_plans() -> None:
    """Let learned order features change outputs when the same actions appear in a new order."""
    torch.manual_seed(0)
    actions = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0]]],
        dtype=torch.float32,
    )
    permuted = actions[:, [3, 2, 1, 0], :]

    encoder = ActionTokenEncoder(
        action_dim=1,
        hidden_dim=4,
        input_layernorm=False,
        order_conditioning=True,
    )
    encoder.net[1].weight.data.fill_(1.0)
    encoder.net[1].bias.data.zero_()
    assert encoder.order_net is not None
    encoder.order_net[0].weight.data.copy_(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, -0.5],
            ],
            dtype=torch.float32,
        )
    )
    encoder.order_net[0].bias.data.zero_()
    encoder.order_net[2].weight.data.copy_(torch.eye(4))
    encoder.order_net[2].bias.data.zero_()

    assert not torch.allclose(encoder(actions), encoder(permuted))


def test_action_token_encoder_rejects_residual_mlp_without_width() -> None:
    """Require a hidden width when enabling the residual action MLP path."""
    with pytest.raises(ValueError, match="mlp_residual requires a positive mlp_dim"):
        ActionTokenEncoder(action_dim=7, hidden_dim=32, mlp_residual=True)


def test_action_token_encoder_rejects_negative_temporal_difference_scale() -> None:
    """Reject negative temporal-difference residual scales."""
    with pytest.raises(ValueError, match="temporal_difference_scale must be non-negative"):
        ActionTokenEncoder(action_dim=7, hidden_dim=32, temporal_difference_scale=-0.1)


def test_action_token_encoder_rejects_negative_token_scale() -> None:
    """Reject negative post-projection token gains."""
    with pytest.raises(ValueError, match="token_scale must be non-negative"):
        ActionTokenEncoder(action_dim=7, hidden_dim=32, token_scale=-0.1)


def test_action_token_encoder_rejects_invalid_temporal_mixer_config() -> None:
    """Reject unsupported temporal mixer settings before building the module."""
    with pytest.raises(ValueError, match="temporal_mixer_kernel_size must be odd"):
        ActionTokenEncoder(action_dim=7, hidden_dim=32, temporal_mixer_kernel_size=4)

    with pytest.raises(ValueError, match="temporal_mixer_scale requires temporal_mixer_kernel_size >= 3"):
        ActionTokenEncoder(action_dim=7, hidden_dim=32, temporal_mixer_scale=0.5)


def test_action_control_projector_zero_init_starts_inert() -> None:
    """Keep the latent-prior branch exactly off when zero-init is requested."""
    projector = ActionControlProjector(action_dim=3, latent_channels=5, init_mode="zero")

    projected = projector(torch.randn(2, 4, 3), latent_height=2, latent_width=2)

    assert torch.count_nonzero(projected) == 0


def test_action_control_projector_linear_default_init_is_nonzero() -> None:
    """Allow a resumed latent-prior branch to start from a learned linear mapping instead of zeros."""
    torch.manual_seed(0)
    projector = ActionControlProjector(action_dim=3, latent_channels=5, init_mode="linear_default")

    projected = projector(torch.randn(2, 4, 3), latent_height=2, latent_width=2)

    assert torch.count_nonzero(projected) > 0


def test_action_control_projector_rejects_unknown_init_mode() -> None:
    """Reject unsupported projector init modes before building parameters."""
    with pytest.raises(ValueError, match="init_mode must be 'zero' or 'linear_default'"):
        ActionControlProjector(action_dim=3, latent_channels=5, init_mode="bad_mode")


def test_action_control_projector_rejects_unknown_observed_context_mode() -> None:
    """Reject unsupported observed-context modes before building parameters."""
    with pytest.raises(ValueError, match="observed_context_mode must be 'none' or 'last_frame'"):
        ActionControlProjector(action_dim=3, latent_channels=5, observed_context_mode="bad_mode")


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


def test_action_control_projector_broadcasts_latent_prior_over_space() -> None:
    """Project action plans into broadcast latent priors with the expected `[B,C,T,H,W]` shape."""
    projector = ActionControlProjector(action_dim=3, latent_channels=5)
    projector.projection.weight.data.fill_(1.0)
    projector.projection.bias.data.zero_()

    prior = projector(
        torch.tensor([[[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]], dtype=torch.float32),
        latent_height=4,
        latent_width=6,
    )

    assert prior.shape == (1, 5, 2, 4, 6)
    assert torch.allclose(prior[:, :, :, 0, 0], prior[:, :, :, -1, -1])


def test_action_control_projector_can_add_last_observed_latent_context() -> None:
    """Add a pooled last-frame latent summary when observed context is enabled."""
    projector = ActionControlProjector(
        action_dim=3,
        latent_channels=2,
        init_mode="zero",
        observed_context_mode="last_frame",
    )
    projector.context_projection.weight.data.copy_(torch.eye(2))
    projector.context_projection.bias.data.zero_()

    prior = projector(
        torch.zeros(1, 3, 3),
        latent_height=2,
        latent_width=2,
        observed_latents=torch.tensor(
            [[
                [[[1.0, 1.0], [1.0, 1.0]], [[3.0, 3.0], [3.0, 3.0]]],
                [[[2.0, 2.0], [2.0, 2.0]], [[4.0, 4.0], [4.0, 4.0]]],
            ]],
            dtype=torch.float32,
        ).permute(0, 1, 2, 3, 4),
    )

    assert prior.shape == (1, 2, 3, 2, 2)
    assert torch.allclose(prior[:, :, :, 0, 0], torch.tensor([[[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]]))
