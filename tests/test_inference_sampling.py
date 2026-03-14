"""Tests for inference-time chunkwise sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from world_model.eval import infer_future_videos_chunkwise


class _RecordingVideoInferenceModel(nn.Module):
    """Record structured Wan VACE inference calls and denoise by negation."""

    def __init__(self):
        """Initialize call storage."""
        super().__init__()
        self.calls: list[dict[str, int | tuple[int, ...]]] = []

    def forward(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        observed_mask: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record active chunk shapes and return a simple velocity field."""
        del timestep_t, observed_mask, control_hidden_states_scale
        self.calls.append(
            {
                "future_frames": noisy_future_video.shape[2],
                "observed_frames": observed_video.shape[2],
                "action_frames": action_tokens.shape[1],
                "mask_shape": None if block_causal_attention_mask is None else tuple(block_causal_attention_mask.shape),
            }
        )
        return -noisy_future_video


def test_infer_future_videos_chunkwise_shapes_and_calls():
    """Sample future latent videos chunkwise with the Wan VACE inference contract."""
    torch.manual_seed(0)
    model = _RecordingVideoInferenceModel()
    z_past_video = torch.randn(2, 16, 3, 8, 8)
    cross_attention_tokens = torch.randn(2, 8, 16)

    out = infer_future_videos_chunkwise(
        model,
        z_past_video=z_past_video,
        future_steps=8,
        cross_attention_tokens=cross_attention_tokens,
        k=1,
        integration_steps=5,
    )

    assert out.shape == (2, 16, 8, 8, 8)
    assert len(model.calls) == 10
    assert model.calls[0]["observed_frames"] == 3
    assert model.calls[0]["future_frames"] == 4
    assert model.calls[0]["action_frames"] == 4
    assert model.calls[5]["observed_frames"] == 7
    assert model.calls[5]["future_frames"] == 4
    assert model.calls[5]["action_frames"] == 4


def test_infer_future_videos_chunkwise_supports_single_chunk_prompt_conditioning() -> None:
    """Use one chunk while reusing global prompt tokens on every denoising step."""
    torch.manual_seed(0)
    model = _RecordingVideoInferenceModel()
    out = infer_future_videos_chunkwise(
        model,
        z_past_video=torch.randn(2, 16, 3, 8, 8),
        future_steps=8,
        cross_attention_tokens=torch.randn(2, 6, 16),
        negative_cross_attention_tokens=torch.randn(2, 6, 16),
        guidance_scale=5.0,
        chunk_conditioning=False,
        single_chunk_rollout=True,
        k=1,
        integration_steps=5,
    )

    assert out.shape == (2, 16, 8, 8, 8)
    assert len(model.calls) == 10
    assert model.calls[0]["observed_frames"] == 3
    assert model.calls[0]["future_frames"] == 8
    assert model.calls[0]["action_frames"] == 6


def test_infer_future_videos_chunkwise_can_disable_block_causal_attention() -> None:
    """Allow pretrained base-mode sampling to reuse the local adapter without a causal mask."""
    torch.manual_seed(0)
    model = _RecordingVideoInferenceModel()

    out = infer_future_videos_chunkwise(
        model,
        z_past_video=torch.randn(1, 16, 3, 8, 8),
        future_steps=4,
        cross_attention_tokens=torch.randn(1, 6, 16),
        chunk_conditioning=False,
        single_chunk_rollout=True,
        block_causal_attention=False,
        k=1,
        integration_steps=2,
    )

    assert out.shape == (1, 16, 4, 8, 8)
    assert len(model.calls) == 2
    assert model.calls[0]["mask_shape"] is None


def test_infer_future_videos_chunkwise_auto_collapses_when_future_is_shorter_than_k_plus_one() -> None:
    """Fall back to one chunk when latent future length is too short for K+1 chunking."""
    torch.manual_seed(0)
    model = _RecordingVideoInferenceModel()
    out = infer_future_videos_chunkwise(
        model,
        z_past_video=torch.randn(2, 16, 3, 8, 8),
        future_steps=1,
        cross_attention_tokens=torch.randn(2, 1, 16),
        k=1,
        integration_steps=2,
    )

    assert out.shape == (2, 16, 1, 8, 8)
    assert len(model.calls) == 2
    assert model.calls[0]["future_frames"] == 1
    assert model.calls[0]["action_frames"] == 1


def test_infer_future_videos_chunkwise_validates_inputs() -> None:
    """Reject invalid integration-step settings on the active inference path."""
    model = _RecordingVideoInferenceModel()
    with pytest.raises(ValueError, match="integration_steps must be positive"):
        infer_future_videos_chunkwise(
            model,
            z_past_video=torch.randn(2, 16, 3, 8, 8),
            future_steps=8,
            cross_attention_tokens=torch.randn(2, 8, 16),
            k=1,
            integration_steps=0,
        )


def test_infer_future_videos_chunkwise_rejects_misaligned_chunk_conditioning() -> None:
    """Require future-step-aligned token length when chunk conditioning is enabled."""
    model = _RecordingVideoInferenceModel()
    with pytest.raises(ValueError, match="chunk-conditioned cross_attention_tokens length"):
        infer_future_videos_chunkwise(
            model,
            z_past_video=torch.randn(2, 16, 3, 8, 8),
            future_steps=8,
            cross_attention_tokens=torch.randn(2, 6, 16),
            k=1,
            integration_steps=5,
            chunk_conditioning=True,
        )
