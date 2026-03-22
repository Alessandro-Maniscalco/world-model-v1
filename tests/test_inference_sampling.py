"""Tests for inference-time chunkwise sampling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from world_model.eval import infer_future_videos_chunkwise
from world_model.eval.inference import _build_rollout_boundaries, _select_chunk_conditioning_tokens


class _RecordingVideoInferenceModel(nn.Module):
    """Record structured Wan VACE inference calls and denoise by negation."""

    def __init__(self):
        """Initialize call storage."""
        super().__init__()
        self.calls: list[dict[str, int | float | tuple[int, ...] | None]] = []

    def forward(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        action_image_tokens: torch.Tensor | None = None,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        observed_mask: torch.Tensor | None = None,
        future_latent_residual_base: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record active chunk shapes and return a simple velocity field."""
        residual_base_frames = None if future_latent_residual_base is None else future_latent_residual_base.shape[2]
        residual_base_value = None
        if future_latent_residual_base is not None:
            residual_base_value = float(future_latent_residual_base[0, 0, 0, 0, 0].item())
        del (
            action_image_tokens,
            timestep_t,
            observed_mask,
            future_latent_residual_base,
            control_hidden_states_scale,
        )
        self.calls.append(
            {
                "future_frames": noisy_future_video.shape[2],
                "observed_frames": observed_video.shape[2],
                "action_frames": action_tokens.shape[1],
                "residual_base_frames": residual_base_frames,
                "residual_base_value": residual_base_value,
                "mask_shape": None if block_causal_attention_mask is None else tuple(block_causal_attention_mask.shape),
            }
        )
        return -noisy_future_video


class _TokenDrivenInferenceModel(nn.Module):
    """Return a constant velocity field derived from the active token values."""

    def __init__(self) -> None:
        """Initialize residual-base call storage."""
        super().__init__()
        self.calls: list[dict[str, float | int | None]] = []

    def forward(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        action_image_tokens: torch.Tensor | None = None,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        observed_mask: torch.Tensor | None = None,
        future_latent_residual_base: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fill the latent chunk with the mean token value."""
        residual_base_frames = None if future_latent_residual_base is None else future_latent_residual_base.shape[2]
        residual_base_value = None
        if future_latent_residual_base is not None:
            residual_base_value = float(future_latent_residual_base[0, 0, 0, 0, 0].item())
        del (
            observed_video,
            action_image_tokens,
            timestep_t,
            block_causal_attention_mask,
            observed_mask,
            control_hidden_states_scale,
        )
        self.calls.append(
            {
                "residual_base_frames": residual_base_frames,
                "residual_base_value": residual_base_value,
            }
        )
        value = action_tokens.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        return value.expand_as(noisy_future_video)


class _EchoScheduler:
    """Return the model output as the next latent state."""

    def __init__(self) -> None:
        """Initialize the scheduler timestep storage."""
        self.timesteps = torch.tensor([], dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        """Expose a simple descending timestep tensor."""
        del num_inference_steps
        self.timesteps = torch.tensor([0.0], device=device, dtype=torch.float32)

    def step(self, model_output: torch.Tensor, t: torch.Tensor, sample: torch.Tensor, **kwargs):
        """Return the predicted velocity directly for deterministic testing."""
        del t, sample, kwargs
        return (model_output.clone(),)


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
        k=2,
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


def test_infer_future_videos_chunkwise_auto_collapses_when_future_is_shorter_than_k() -> None:
    """Fall back to one chunk when latent future length is too short for exact-k chunking."""
    torch.manual_seed(0)
    model = _RecordingVideoInferenceModel()
    out = infer_future_videos_chunkwise(
        model,
        z_past_video=torch.randn(2, 16, 3, 8, 8),
        future_steps=1,
        cross_attention_tokens=torch.randn(2, 1, 16),
        k=2,
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


def test_infer_future_videos_chunkwise_applies_classifier_free_guidance() -> None:
    """Combine conditional and unconditional predictions using the CFG equation."""
    model = _TokenDrivenInferenceModel()
    scheduler = _EchoScheduler()

    out = infer_future_videos_chunkwise(
        model,
        z_past_video=torch.zeros(1, 1, 1, 1, 1),
        future_steps=2,
        cross_attention_tokens=torch.full((1, 2, 3), 3.0),
        negative_cross_attention_tokens=torch.full((1, 2, 3), 1.0),
        guidance_scale=2.0,
        chunk_conditioning=True,
        single_chunk_rollout=True,
        block_causal_attention=False,
        scheduler=scheduler,
        k=1,
        integration_steps=1,
    )

    assert torch.allclose(out, torch.full_like(out, 5.0))


def test_infer_future_videos_chunkwise_restores_last_context_frame_baseline() -> None:
    """Add the last observed latent frame back after residual-space sampling."""
    scheduler = _EchoScheduler()
    model = _TokenDrivenInferenceModel()

    out = infer_future_videos_chunkwise(
        model,
        z_past_video=torch.full((1, 1, 2, 1, 1), 7.0),
        future_steps=2,
        cross_attention_tokens=torch.zeros(1, 2, 1),
        chunk_conditioning=True,
        single_chunk_rollout=True,
        block_causal_attention=False,
        scheduler=scheduler,
        k=1,
        integration_steps=1,
        future_latent_residual_mode="last_context_frame",
    )

    assert torch.allclose(out, torch.full_like(out, 7.0))
    assert model.calls[0]["residual_base_frames"] == 2
    assert model.calls[0]["residual_base_value"] == pytest.approx(7.0)


def test_select_chunk_conditioning_tokens_slices_multi_chunk_window() -> None:
    """Select only the token window belonging to the active chunk when chunking is enabled."""
    tokens = torch.arange(1 * 6 * 2, dtype=torch.float32).reshape(1, 6, 2)

    chunk_tokens = _select_chunk_conditioning_tokens(tokens, start=2, end=5, chunk_conditioning=True)

    assert torch.equal(chunk_tokens, tokens[:, 2:5])


def test_build_rollout_boundaries_matches_single_and_exact_k_modes() -> None:
    """Return the expected chunk boundaries for chunked and single-chunk rollout."""
    assert _build_rollout_boundaries(
        future_steps=8,
        k=2,
        chunk_schedule_mode="k_chunks",
        single_chunk_rollout=False,
        device=torch.device("cpu"),
    ) == ((0, 4), (4, 8))
    assert _build_rollout_boundaries(
        future_steps=8,
        k=3,
        chunk_schedule_mode="k_chunks",
        single_chunk_rollout=True,
        device=torch.device("cpu"),
    ) == ((0, 8),)


def test_infer_future_videos_chunkwise_rejects_batch_and_guidance_mismatches() -> None:
    """Validate batch-aligned tokens, negative tokens, guidance scale, and chunk count."""
    model = _RecordingVideoInferenceModel()

    with pytest.raises(ValueError, match="batch size must match"):
        infer_future_videos_chunkwise(
            model,
            z_past_video=torch.randn(1, 16, 3, 8, 8),
            future_steps=4,
            cross_attention_tokens=torch.randn(2, 4, 16),
            k=1,
        )
    with pytest.raises(ValueError, match="negative_cross_attention_tokens must match"):
        infer_future_videos_chunkwise(
            model,
            z_past_video=torch.randn(1, 16, 3, 8, 8),
            future_steps=4,
            cross_attention_tokens=torch.randn(1, 4, 16),
            negative_cross_attention_tokens=torch.randn(1, 3, 16),
            k=1,
        )
    with pytest.raises(ValueError, match="guidance_scale must be >= 1.0"):
        infer_future_videos_chunkwise(
            model,
            z_past_video=torch.randn(1, 16, 3, 8, 8),
            future_steps=4,
            cross_attention_tokens=torch.randn(1, 4, 16),
            guidance_scale=0.5,
            k=1,
        )
    with pytest.raises(ValueError, match="k must be >= 1"):
        infer_future_videos_chunkwise(
            model,
            z_past_video=torch.randn(1, 16, 3, 8, 8),
            future_steps=4,
            cross_attention_tokens=torch.randn(1, 4, 16),
            k=0,
        )
