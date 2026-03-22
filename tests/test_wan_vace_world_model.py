"""Tests for Wan VACE world-model control-stream assembly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from world_model.models.wan_vace_conditioning import build_vace_control_tensor
from world_model.models.wan_vace_world_model import (
    WanVACEWorldModel,
    _patches_per_frame,
    _resolve_control_scale,
    _slice_control_latent_template,
    expand_block_causal_mask_to_patch_tokens,
)


class _RecordingBackbone(torch.nn.Module):
    """Record the control tensor passed into the Wan VACE backbone."""

    def __init__(self, *, patch_size: tuple[int, int, int] = (1, 1, 1), vace_layers: tuple[int, ...] = (0,)) -> None:
        """Store a minimal Wan config and initialize capture state."""
        super().__init__()
        self.config = SimpleNamespace(patch_size=patch_size, vace_layers=vace_layers)
        self.last_control_hidden_states: torch.Tensor | None = None
        self.last_control_hidden_states_scale: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.last_encoder_hidden_states_image: torch.Tensor | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
        control_hidden_states: torch.Tensor,
        control_hidden_states_scale: torch.Tensor | None,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        """Capture the constructed control tensor and echo the hidden states."""
        del timestep, encoder_hidden_states, return_dict
        self.last_encoder_hidden_states_image = (
            None if encoder_hidden_states_image is None else encoder_hidden_states_image.detach().clone()
        )
        self.last_control_hidden_states = control_hidden_states.detach().clone()
        self.last_attention_mask = None if attention_mask is None else attention_mask.detach().clone()
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


def test_wan_vace_world_model_can_fill_future_control_with_last_context_frame() -> None:
    """Use the last observed latent frame as the future masked-control fill when requested."""
    backbone = _RecordingBackbone()
    model = WanVACEWorldModel(
        backbone=backbone,
        control_scale=1.0,
        future_control_fill_mode="last_context_frame",
        mask_channels=1,
        control_black_latents=torch.full((1, 2, 3, 1, 1), -1.0),
        control_gray_latents=torch.full((1, 2, 3, 1, 1), 0.5),
    )

    observed_video = torch.tensor(
        [[[[[2.0]], [[4.0]]], [[[3.0]], [[5.0]]]]],
        dtype=torch.float32,
    )
    model(
        noisy_future_video=torch.zeros(1, 2, 1, 1, 1),
        observed_video=observed_video,
        action_tokens=torch.randn(1, 1, 4),
        timestep_t=torch.tensor([0.5], dtype=torch.float32),
        block_causal_attention_mask=None,
    )

    expected_control_video = torch.tensor(
        [[[[[2.0]], [[4.0]], [[4.0]]], [[[3.0]], [[5.0]], [[5.0]]]]],
        dtype=torch.float32,
    )
    expected_control_mask = torch.tensor([[[[[0.0]], [[0.0]], [[1.0]]]]], dtype=torch.float32)
    expected_control_hidden_states = build_vace_control_tensor(
        observed_latents=expected_control_video,
        observed_mask=expected_control_mask,
        inactive_fill_latents=torch.tensor(
            [[[[[-1.0]], [[-1.0]], [[4.0]]], [[[-1.0]], [[-1.0]], [[5.0]]]]],
            dtype=torch.float32,
        ),
        reactive_fill_latents=torch.tensor(
            [[[[[-1.0]], [[-1.0]], [[4.0]]], [[[-1.0]], [[-1.0]], [[5.0]]]]],
            dtype=torch.float32,
        ),
        mask_channels=1,
    )

    assert backbone.last_control_hidden_states is not None
    assert torch.equal(backbone.last_control_hidden_states, expected_control_hidden_states)


def test_wan_vace_world_model_residualizes_future_control_stream_when_requested() -> None:
    """Shift future VACE control latents into the same residual coordinates as the sampled future chunk."""
    backbone = _RecordingBackbone()
    model = WanVACEWorldModel(
        backbone=backbone,
        control_scale=1.0,
        mask_channels=1,
        control_black_latents=torch.full((1, 2, 2, 1, 1), -1.0),
        control_gray_latents=torch.full((1, 2, 2, 1, 1), 0.5),
    )

    model(
        noisy_future_video=torch.zeros(1, 2, 1, 1, 1),
        observed_video=torch.ones(1, 2, 1, 1, 1),
        action_tokens=torch.randn(1, 1, 4),
        timestep_t=torch.tensor([0.5], dtype=torch.float32),
        block_causal_attention_mask=None,
        future_latent_residual_base=torch.full((1, 2, 1, 1, 1), 2.0),
    )

    expected_control_video = torch.tensor(
        [[[[[1.0]], [[-1.5]]], [[[1.0]], [[-1.5]]]]],
        dtype=torch.float32,
    )
    expected_control_mask = torch.tensor([[[[[0.0]], [[1.0]]]]], dtype=torch.float32)
    expected_control_hidden_states = build_vace_control_tensor(
        observed_latents=expected_control_video,
        observed_mask=expected_control_mask,
        inactive_fill_latents=torch.tensor(
            [[[[[-1.0]], [[-3.0]]], [[[-1.0]], [[-3.0]]]]],
            dtype=torch.float32,
        ),
        reactive_fill_latents=torch.tensor(
            [[[[[-1.0]], [[-3.0]]], [[[-1.0]], [[-3.0]]]]],
            dtype=torch.float32,
        ),
        mask_channels=1,
    )

    assert backbone.last_control_hidden_states is not None
    assert torch.equal(backbone.last_control_hidden_states, expected_control_hidden_states)


def test_wan_vace_world_model_zeroes_last_context_fill_under_residual_targets() -> None:
    """Turn last-context future fills into a zero-change prior after residualization."""
    backbone = _RecordingBackbone()
    model = WanVACEWorldModel(
        backbone=backbone,
        control_scale=1.0,
        future_control_fill_mode="last_context_frame",
        mask_channels=1,
        control_black_latents=torch.full((1, 2, 2, 1, 1), -1.0),
        control_gray_latents=torch.full((1, 2, 2, 1, 1), 0.5),
    )

    model(
        noisy_future_video=torch.zeros(1, 2, 1, 1, 1),
        observed_video=torch.tensor([[[[[2.0]]], [[[3.0]]]]], dtype=torch.float32),
        action_tokens=torch.randn(1, 1, 4),
        timestep_t=torch.tensor([0.5], dtype=torch.float32),
        block_causal_attention_mask=None,
        future_latent_residual_base=torch.tensor([[[[[2.0]]], [[[3.0]]]]], dtype=torch.float32),
    )

    expected_control_video = torch.tensor(
        [[[[[2.0]], [[0.0]]], [[[3.0]], [[0.0]]]]],
        dtype=torch.float32,
    )
    expected_control_mask = torch.tensor([[[[[0.0]], [[1.0]]]]], dtype=torch.float32)
    expected_control_hidden_states = build_vace_control_tensor(
        observed_latents=expected_control_video,
        observed_mask=expected_control_mask,
        inactive_fill_latents=torch.tensor(
            [[[[[-1.0]], [[0.0]]], [[[-1.0]], [[0.0]]]]],
            dtype=torch.float32,
        ),
        reactive_fill_latents=torch.tensor(
            [[[[[-1.0]], [[0.0]]], [[[-1.0]], [[0.0]]]]],
            dtype=torch.float32,
        ),
        mask_channels=1,
    )

    assert backbone.last_control_hidden_states is not None
    assert torch.equal(backbone.last_control_hidden_states, expected_control_hidden_states)


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


def test_wan_vace_world_model_forwards_action_image_tokens_to_backbone() -> None:
    """Pass optional action-image tokens into Wan's added-K/V image-conditioning slot."""
    backbone = _RecordingBackbone()
    model = WanVACEWorldModel(
        backbone=backbone,
        control_scale=1.0,
        mask_channels=1,
        control_black_latents=torch.full((1, 2, 2, 1, 1), -1.0),
        control_gray_latents=torch.full((1, 2, 2, 1, 1), 0.5),
    )
    action_image_tokens = torch.randn(1, 1, 4)

    model(
        noisy_future_video=torch.zeros(1, 2, 1, 1, 1),
        observed_video=torch.ones(1, 2, 1, 1, 1),
        action_tokens=torch.randn(1, 1, 4),
        action_image_tokens=action_image_tokens,
        timestep_t=torch.tensor([0.5], dtype=torch.float32),
        block_causal_attention_mask=None,
    )

    assert backbone.last_encoder_hidden_states_image is not None
    assert torch.equal(backbone.last_encoder_hidden_states_image, action_image_tokens)


def test_expand_block_causal_mask_to_patch_tokens_repeats_2d_masks() -> None:
    """Repeat each latent-frame mask entry across the patch-token grid."""
    mask = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]], dtype=torch.float32)

    expanded = expand_block_causal_mask_to_patch_tokens(mask, patches_per_frame=2)

    expected = torch.tensor(
        [
            [0.0, 0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf"), float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(expanded, expected)


def test_expand_block_causal_mask_to_patch_tokens_repeats_batched_masks() -> None:
    """Expand batched additive masks without mixing batch entries."""
    mask = torch.tensor([[[0.0, float("-inf")], [0.0, 0.0]]], dtype=torch.float32)

    expanded = expand_block_causal_mask_to_patch_tokens(mask, patches_per_frame=3)

    assert expanded.shape == (1, 6, 6)
    assert torch.isinf(expanded[0, :3, 3:]).all()
    assert (expanded[0, 3:, :3] == 0.0).all()


def test_slice_control_latent_template_broadcasts_batch_one_template() -> None:
    """Broadcast a shared control template across the active batch."""
    observed_video = torch.zeros(2, 3, 2, 4, 4)
    noisy_future_video = torch.zeros(2, 3, 1, 4, 4)
    template = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 3, 4, 4)

    sliced = _slice_control_latent_template(
        template=template,
        observed_video=observed_video,
        noisy_future_video=noisy_future_video,
    )

    assert sliced.shape == (2, 3, 3, 4, 4)
    assert torch.equal(sliced[0], template[0])
    assert torch.equal(sliced[1], template[0])


def test_slice_control_latent_template_validates_template_shape() -> None:
    """Reject control templates that cannot match the active rollout."""
    observed_video = torch.zeros(2, 3, 2, 4, 4)
    noisy_future_video = torch.zeros(2, 3, 2, 4, 4)

    with pytest.raises(ValueError, match="time dim 3 is smaller than rollout length 4"):
        _slice_control_latent_template(
            template=torch.zeros(1, 3, 3, 4, 4),
            observed_video=observed_video,
            noisy_future_video=noisy_future_video,
        )
    with pytest.raises(ValueError, match="channel dim 2 must match latent channels 3"):
        _slice_control_latent_template(
            template=torch.zeros(1, 2, 4, 4, 4),
            observed_video=observed_video,
            noisy_future_video=noisy_future_video,
        )
    with pytest.raises(ValueError, match="spatial shape \\(2, 4\\) must match"):
        _slice_control_latent_template(
            template=torch.zeros(1, 3, 4, 2, 4),
            observed_video=observed_video,
            noisy_future_video=noisy_future_video,
        )
    with pytest.raises(ValueError, match="batch dim 3 must be 1 or match batch size 2"):
        _slice_control_latent_template(
            template=torch.zeros(3, 3, 4, 4, 4),
            observed_video=observed_video,
            noisy_future_video=noisy_future_video,
        )


def test_patches_per_frame_validates_patch_geometry() -> None:
    """Require temporal patch size 1 and spatial divisibility for latent-frame masks."""
    video = torch.zeros(1, 2, 3, 4, 4)

    with pytest.raises(ValueError, match="temporal patch size 1"):
        _patches_per_frame(video=video, backbone=_RecordingBackbone(patch_size=(2, 2, 2)))
    with pytest.raises(ValueError, match="must be divisible by patch size"):
        _patches_per_frame(video=video, backbone=_RecordingBackbone(patch_size=(1, 3, 2)))


def test_resolve_control_scale_prefers_explicit_override() -> None:
    """Use the caller-provided control scale tensor instead of rebuilding defaults."""
    scale = _resolve_control_scale(
        backbone=_RecordingBackbone(vace_layers=(0, 2)),
        control_hidden_states_scale=torch.tensor([1.5, 2.5], dtype=torch.float64),
        default_scale=0.75,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(scale, torch.tensor([1.5, 2.5], dtype=torch.float32))


def test_wan_vace_world_model_expands_attention_mask_to_patch_tokens() -> None:
    """Expand latent-frame masks before forwarding them to the backbone."""
    backbone = _RecordingBackbone(patch_size=(1, 2, 2))
    model = WanVACEWorldModel(
        backbone=backbone,
        mask_channels=1,
        control_black_latents=torch.zeros(1, 2, 2, 4, 4),
        control_gray_latents=torch.zeros(1, 2, 2, 4, 4),
    )
    attention_mask = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]], dtype=torch.float32)

    model(
        noisy_future_video=torch.zeros(1, 2, 1, 4, 4),
        observed_video=torch.ones(1, 2, 1, 4, 4),
        action_tokens=torch.randn(1, 1, 4),
        timestep_t=torch.tensor([0.5], dtype=torch.float32),
        block_causal_attention_mask=attention_mask,
    )

    assert backbone.last_attention_mask is not None
    assert backbone.last_attention_mask.shape == (8, 8)
    assert torch.isinf(backbone.last_attention_mask[:4, 4:]).all()
