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


class _MeanValueEncoder:
    """Encode videos into latents filled with the input mean value."""

    def __init__(self, *, latent_steps: int) -> None:
        """Store the latent rollout length and initialize call recording."""
        self._latent_steps = latent_steps
        self.call_count = 0
        self.seen_inputs: list[torch.Tensor] = []

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Record the input and return a latent tensor filled with its mean."""
        self.call_count += 1
        self.seen_inputs.append(video.detach().clone())
        mean_value = video.float().mean(dim=(1, 2, 3, 4), keepdim=True)
        return mean_value.expand(video.shape[0], 1, self._latent_steps, 1, 1).clone()


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
    """Encode each future Wan latent block from the transition actions that lead into it."""
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
    assert torch.equal(prepared.a_plan[0, 0], batch["action"][0, 8:12].reshape(-1))
    assert torch.equal(prepared.a_plan[0, 1], batch["action"][0, 12:16].reshape(-1))


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


def test_preprocess_video_for_vae_center_crops_to_multiple_of_sixteen() -> None:
    """Crop non-aligned video sizes so Wan patchification keeps an even latent grid."""
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


@pytest.mark.parametrize(
    ("video", "zero_to_one_value", "expected_fill"),
    [
        (
            torch.tensor([[[[[0, 255]]]]], dtype=torch.uint8),
            128.0 / 255.0,
            torch.tensor([[[[[128, 128]]]]], dtype=torch.uint8),
        ),
        (
            torch.tensor([[[[[-1.0, 1.0]]]]], dtype=torch.float32),
            0.0,
            torch.tensor([[[[[-1.0, -1.0]]]]], dtype=torch.float32),
        ),
        (
            torch.tensor([[[[[0.0, 255.0]]]]], dtype=torch.float32),
            128.0 / 255.0,
            torch.tensor([[[[[128.0, 128.0]]]]], dtype=torch.float32),
        ),
        (
            torch.tensor([[[[[0.0, 1.0]]]]], dtype=torch.float32),
            128.0 / 255.0,
            torch.tensor([[[[[128.0 / 255.0, 128.0 / 255.0]]]]], dtype=torch.float32),
        ),
    ],
)
def test_make_constant_video_like_preserves_expected_numeric_ranges(
    video: torch.Tensor,
    zero_to_one_value: float,
    expected_fill: torch.Tensor,
) -> None:
    """Map normalized fill values into the active video numeric range."""
    constant_video = prepare_mod._make_constant_video_like(video=video, zero_to_one_value=zero_to_one_value)

    assert torch.equal(constant_video, expected_fill)


def test_prepare_packed_batch_encodes_neutral_and_gray_control_templates() -> None:
    """Encode upstream-style neutral and gray control templates with the expected latent semantics."""
    prepare_mod._CONTROL_LATENT_TEMPLATE_CACHE.clear()
    encoder = _MeanValueEncoder(latent_steps=5)
    batch_video = torch.linspace(-1.0, 1.0, steps=17 * 3 * 16 * 16, dtype=torch.float32).reshape(1, 17, 3, 16, 16)
    batch = {
        "observation.images.image": batch_video,
        "action": torch.randn(1, 17, 3),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=9,
        horizon_len=8,
    )

    expected_neutral = 0.0
    expected_gray = (128.0 / 255.0) * 2.0 - 1.0

    assert encoder.call_count == 3
    assert torch.equal(encoder.seen_inputs[0], torch.full_like(batch_video, expected_neutral))
    assert torch.allclose(encoder.seen_inputs[1], torch.full_like(batch_video, expected_gray))
    assert torch.equal(encoder.seen_inputs[2], batch_video)
    assert torch.allclose(prepared.control_black_latents, torch.full((1, 1, 5, 1, 1), expected_neutral))
    assert torch.allclose(prepared.control_gray_latents, torch.full((1, 1, 5, 1, 1), expected_gray))
    prepare_mod._CONTROL_LATENT_TEMPLATE_CACHE.clear()


def test_constant_control_latent_cache_reuses_matching_range_and_misses_on_range_change() -> None:
    """Reuse cached control templates only when shape, suffix, and numeric range all match."""
    prepare_mod._CONTROL_LATENT_TEMPLATE_CACHE.clear()
    encoder = _MeanValueEncoder(latent_steps=5)
    zero_to_one_video = torch.zeros(1, 17, 3, 16, 16, dtype=torch.float32)
    minus_one_to_one_video = torch.linspace(
        -1.0,
        1.0,
        steps=17 * 3 * 16 * 16,
        dtype=torch.float32,
    ).reshape(1, 17, 3, 16, 16)

    first = prepare_mod._get_constant_control_latents(
        encoder=encoder,
        video=zero_to_one_video,
        cache_key_suffix="gray",
        zero_to_one_value=128.0 / 255.0,
    )
    second = prepare_mod._get_constant_control_latents(
        encoder=encoder,
        video=zero_to_one_video.clone(),
        cache_key_suffix="gray",
        zero_to_one_value=128.0 / 255.0,
    )
    third = prepare_mod._get_constant_control_latents(
        encoder=encoder,
        video=minus_one_to_one_video,
        cache_key_suffix="gray",
        zero_to_one_value=128.0 / 255.0,
    )

    assert encoder.call_count == 2
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    prepare_mod._CONTROL_LATENT_TEMPLATE_CACHE.clear()


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
