"""Tests for shared prepared-batch pipeline helpers."""

from __future__ import annotations

import torch

from world_model.data.prepare import prepare_packed_batch


class _FakeEncoder:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        del video
        return self._latents


def test_prepare_packed_batch_shapes_and_metadata() -> None:
    batch_size = 2
    latents = torch.randn(batch_size, 4, 6, 2, 2)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(batch_size, 18, 3, 32, 32),
        "action": torch.randn(batch_size, 7),
        "observation.state": torch.randn(batch_size, 5),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=10,
        horizon_len=8,
        proprio_mode="last",
    )

    assert prepared.total_latent_steps == 6
    assert prepared.context_latent_steps + prepared.horizon_latent_steps == 6
    assert prepared.z_past.shape[0] == batch_size
    assert prepared.z_future.shape[0] == batch_size
    assert prepared.a_plan.shape[1] == prepared.horizon_latent_steps
    assert prepared.q_last is not None
    assert prepared.q_last.shape == (batch_size, 5)
    assert prepared.latent_shape == (4, 2, 2)


def test_prepare_packed_batch_handles_missing_proprio() -> None:
    latents = torch.randn(1, 2, 4, 1, 1)
    encoder = _FakeEncoder(latents)
    batch = {
        "observation.images.image": torch.randn(1, 18, 3, 32, 32),
        "action": torch.randn(1, 3),
    }

    prepared = prepare_packed_batch(
        batch=batch,
        encoder=encoder,
        device=torch.device("cpu"),
        video_key="observation.images.image",
        context_len=10,
        horizon_len=8,
        proprio_mode="last",
    )

    assert prepared.q_last is None
