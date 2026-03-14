"""Tests for Wan VACE-oriented prepared batch helpers."""

from __future__ import annotations

import torch

from world_model.data.prepare import prepare_packed_batch


class _FakeEncoder:
    """Return a fixed latent video tensor for batch-preparation tests."""

    def __init__(self, latents: torch.Tensor) -> None:
        """Store the fake latent tensor."""
        self._latents = latents

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Ignore the input video and return canned latents."""
        del video
        return self._latents


def test_prepare_packed_batch_preserves_structured_latent_videos() -> None:
    """Expose the structured latent-video fields used by the Wan VACE runtime."""
    batch_size = 2
    latents = torch.randn(batch_size, 4, 5, 2, 2)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(batch_size, 17, 3, 32, 32),
        "action": torch.randn(batch_size, 7),
        "observation.state": torch.randn(batch_size, 5),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=9,
        horizon_len=8,
    )

    assert prepared.z_past_video.ndim == 5
    assert prepared.z_future_video.ndim == 5
    assert prepared.z_past_video.shape[0] == batch_size
    assert prepared.z_future_video.shape[0] == batch_size
    assert prepared.context_latent_steps == 3
    assert prepared.horizon_latent_steps == 2
    assert prepared.z_past_video.shape[2] == prepared.context_latent_steps
    assert prepared.z_future_video.shape[2] == prepared.horizon_latent_steps
    assert prepared.a_plan.shape[1] == prepared.horizon_latent_steps
