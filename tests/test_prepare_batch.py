"""Tests for shared prepared-batch pipeline helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from world_model.data import prepare as prepare_mod
from world_model.data.prepare import load_local_video_clip, prepare_packed_batch, preprocess_video_for_vae


class _FakeEncoder:
    """Minimal encoder stub that records the preprocessed video shape."""

    def __init__(self, latents: torch.Tensor) -> None:
        """Store canned latents and initialize the last seen input shape."""
        self._latents = latents
        self.last_input_shape: tuple[int, ...] | None = None

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Return canned latents while recording the received input shape."""
        self.last_input_shape = tuple(video.shape)
        return self._latents


def test_prepare_packed_batch_shapes_and_metadata() -> None:
    """Prepare a structured latent-video batch with aligned action/proprio metadata."""
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

    assert prepared.total_latent_steps == 5
    assert prepared.context_latent_steps + prepared.horizon_latent_steps == 5
    assert prepared.context_latent_steps == 3
    assert prepared.horizon_latent_steps == 2
    assert prepared.z_past_video.shape == (batch_size, 4, prepared.context_latent_steps, 2, 2)
    assert prepared.z_future_video.shape == (batch_size, 4, prepared.horizon_latent_steps, 2, 2)
    assert prepared.control_black_latents is not None
    assert prepared.control_gray_latents is not None
    assert prepared.control_black_latents.shape == (batch_size, 4, 5, 2, 2)
    assert prepared.control_gray_latents.shape == (batch_size, 4, 5, 2, 2)
    assert prepared.a_plan.shape[1] == prepared.horizon_latent_steps
    assert prepared.latent_shape == (4, 2, 2)


def test_prepare_packed_batch_handles_missing_proprio() -> None:
    """Allow missing proprio while still preparing the latent-video batch."""
    latents = torch.randn(1, 2, 5, 1, 1)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(1, 17, 3, 32, 32),
        "action": torch.randn(1, 3),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=9,
        horizon_len=8,
    )
    assert prepared.a_plan.shape[0] == 1


def test_prepare_packed_batch_chunks_frame_rate_actions_by_future_latent_block() -> None:
    """Encode each future Wan latent block from the full raw action chunk it covers."""
    latents = torch.randn(1, 2, 5, 1, 1)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(1, 17, 3, 32, 32),
        "action": torch.arange(17 * 3, dtype=torch.float32).reshape(1, 17, 3),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=9,
        horizon_len=8,
    )

    assert prepared.a_plan.shape == (1, 2, 12)
    assert torch.equal(prepared.a_plan[0, 0], batch["action"][0, 9:13].reshape(-1))
    assert torch.equal(prepared.a_plan[0, 1], batch["action"][0, 13:17].reshape(-1))


def test_prepare_packed_batch_allows_missing_action_when_requested() -> None:
    """Build placeholder conditioning inputs when action-free training is enabled."""
    latents = torch.randn(1, 2, 5, 1, 1)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(1, 17, 3, 32, 32),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=9,
        horizon_len=8,
        allow_missing_action=True,
    )

    assert prepared.a_plan.shape[0] == 1
    assert prepared.a_plan.shape[-1] == 1


def test_preprocess_video_for_vae_center_crops_to_multiple_of_eight() -> None:
    """Crop non-aligned video sizes so VAE decode can roundtrip without shrink mismatch."""
    video = torch.randn(1, 3, 3, 180, 321)

    processed = preprocess_video_for_vae(video)

    assert processed.shape == (1, 3, 3, 176, 320)


def test_preprocess_video_for_vae_resizes_uint8_video() -> None:
    """Upsample local uint8 video inputs by casting before bilinear interpolation."""
    video = torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8)

    processed = preprocess_video_for_vae(video, frame_height=48, frame_width=48)

    assert processed.shape == (1, 2, 3, 48, 48)
    assert processed.dtype == torch.float32


def test_prepare_packed_batch_preprocesses_video_before_encoding() -> None:
    """Run the shared spatial preprocessing before handing video to the encoder."""
    latents = torch.randn(1, 2, 5, 22, 40)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(1, 17, 3, 180, 321),
        "action": torch.randn(1, 3),
    }

    prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=9,
        horizon_len=8,
    )

    assert encoder.last_input_shape == (1, 17, 3, 176, 320)


def test_prepare_packed_batch_rejects_invalid_wan_temporal_window() -> None:
    """Fail fast when the requested frame split would misalign Wan temporal groups."""
    latents = torch.randn(1, 2, 5, 1, 1)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(1, 18, 3, 32, 32),
        "action": torch.randn(1, 3),
    }

    with pytest.raises(ValueError, match="context_len = 4n\\+1"):
        prepare_packed_batch(
            batch=batch,
            encoder=encoder,
            device=torch.device("cpu"),
            video_key="observation.images.image",
            context_len=10,
            horizon_len=8,
        )


def test_load_local_video_clip_reads_contiguous_rgb_window(monkeypatch) -> None:
    """Load a fixed contiguous clip window from a local video array."""
    video = np.arange(6 * 4 * 5 * 3, dtype=np.uint8).reshape(6, 4, 5, 3)

    monkeypatch.setattr(prepare_mod.iio, "imread", lambda path: video)

    clip = load_local_video_clip("dummy.mp4", start_frame=1, total_frames=3)

    assert clip.shape == (1, 3, 3, 4, 5)
    assert clip.dtype == torch.uint8


def test_load_local_video_clip_rejects_windows_past_end(monkeypatch) -> None:
    """Reject local clip requests that exceed the video length."""
    video = np.zeros((4, 4, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(prepare_mod.iio, "imread", lambda path: video)

    with pytest.raises(ValueError, match="exceed video length"):
        load_local_video_clip("dummy.mp4", start_frame=2, total_frames=3)
