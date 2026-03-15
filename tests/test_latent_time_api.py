"""Tests for the Wan latent-time wrapper API."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from world_model.latents import WanVAE, WanVAEConfig


class _FakeLatentDist:
    """Minimal latent distribution stub for deterministic and stochastic paths."""

    def __init__(self, base: torch.Tensor):
        """Store the raw latent tensor used by the fake VAE."""
        self.mean = base

    def mode(self) -> torch.Tensor:
        """Return the deterministic latent code."""
        return self.mean

    def sample(self) -> torch.Tensor:
        """Return a distinct stochastic sample for test coverage."""
        return self.mean + 1.0


class _FakeEncodeOut:
    """Container matching diffusers-style VAE encode output."""

    def __init__(self, dist: _FakeLatentDist):
        """Expose the fake posterior on `latent_dist`."""
        self.latent_dist = dist


class _FakeDecodeOut:
    """Container matching diffusers-style VAE decode output."""

    def __init__(self, sample: torch.Tensor):
        """Expose the decoded tensor on `sample`."""
        self.sample = sample


class _FakeVAE:
    """Fake VAE with configurable latent stats and traceable decode input."""

    def __init__(self):
        """Attach Wan-style latent stats and initialize decode tracing."""
        self.config = types.SimpleNamespace(
            latents_mean=[2.0, 4.0, 6.0],
            latents_std=[0.5, 2.0, 4.0],
            z_dim=3,
        )
        self.last_decode_input: torch.Tensor | None = None

    def encode(self, x: torch.Tensor):
        """Return a shifted latent tensor while preserving shape."""
        base = x + 10.0
        return _FakeEncodeOut(_FakeLatentDist(base))

    def decode(self, z: torch.Tensor):
        """Record raw decode latents and return a bounded output tensor."""
        self.last_decode_input = z.clone()
        sample = torch.tanh(z)
        return _FakeDecodeOut(sample)


class _FakeTypedVAE(nn.Module):
    """Fake module-backed VAE that exposes a parameter dtype/device."""

    def __init__(self, *, dtype: torch.dtype):
        """Store a parameter so the wrapper can infer runtime dtype."""
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1, dtype=dtype))
        self.config = types.SimpleNamespace(
            latents_mean=[0.0, 0.0, 0.0],
            latents_std=[1.0, 1.0, 1.0],
            z_dim=3,
        )
        self.last_encode_input_dtype: torch.dtype | None = None
        self.last_decode_input_dtype: torch.dtype | None = None

    def encode(self, x: torch.Tensor):
        """Record encode dtype and return a minimal posterior."""
        self.last_encode_input_dtype = x.dtype
        return _FakeEncodeOut(_FakeLatentDist(x))

    def decode(self, z: torch.Tensor):
        """Record decode dtype and return the input tensor on `sample`."""
        self.last_decode_input_dtype = z.dtype
        return _FakeDecodeOut(z)


def test_encode_deterministic_uses_wan_normalized_mode() -> None:
    """Deterministic encode should return Wan-normalized posterior mode latents."""
    model = WanVAE(_FakeVAE(), WanVAEConfig(deterministic=True, input_layout="BTCHW", input_range="minus_one_to_one"))
    video = torch.zeros(2, 4, 3, 8, 8)

    latents = model.encode(video)
    expected = torch.tensor([16.0, 3.0, 1.0], dtype=latents.dtype).view(1, 3, 1, 1, 1).expand_as(latents)
    assert latents.shape == (2, 3, 4, 8, 8)
    assert torch.allclose(latents, expected)


def test_encode_stochastic_uses_wan_normalized_sample() -> None:
    """Stochastic encode should normalize the sampled latent tensor."""
    model = WanVAE(_FakeVAE(), WanVAEConfig(deterministic=False, input_layout="BTCHW", input_range="minus_one_to_one"))
    video = torch.zeros(1, 2, 3, 4, 4)

    latents = model.encode(video)
    expected = torch.tensor([18.0, 3.5, 1.25], dtype=latents.dtype).view(1, 3, 1, 1, 1).expand_as(latents)
    assert torch.allclose(latents, expected)


def test_encode_accepts_bthwc_layout() -> None:
    """Wrapper should accept `BTHWC` input and preserve the VAE-native output shape."""
    model = WanVAE(_FakeVAE(), WanVAEConfig(deterministic=True, input_layout="BTHWC", input_range="uint8"))
    video = torch.zeros(1, 3, 6, 5, 3, dtype=torch.uint8)

    latents = model.encode(video)
    assert latents.shape == (1, 3, 3, 6, 5)


def test_decode_unscales_wan_latents_before_vae_decode() -> None:
    """Decode should invert Wan latent normalization before calling the VAE."""
    fake_vae = _FakeVAE()
    model = WanVAE(fake_vae, WanVAEConfig())
    latents = torch.tensor([16.0, 3.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1, 1).expand(1, 3, 2, 4, 4)

    decoded = model.decode(latents, output_layout="BTHWC", output_range="uint8")
    expected_raw = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32).view(1, 3, 1, 1, 1).expand_as(latents)

    assert fake_vae.last_decode_input is not None
    assert torch.allclose(fake_vae.last_decode_input, expected_raw)
    assert decoded.shape == (1, 2, 4, 4, 3)
    assert decoded.dtype == torch.uint8


def test_raw_latent_format_preserves_existing_behavior() -> None:
    """Raw latent mode should bypass Wan normalization for encode and decode."""
    fake_vae = _FakeVAE()
    model = WanVAE(
        fake_vae,
        WanVAEConfig(deterministic=True, input_layout="BTCHW", input_range="minus_one_to_one", latent_format="raw"),
    )
    video = torch.zeros(1, 2, 3, 4, 4)

    latents = model.encode(video)
    model.decode(latents, output_layout="BTCHW", output_range="zero_to_one")

    assert torch.allclose(latents, torch.full_like(latents, 10.0))
    assert fake_vae.last_decode_input is not None
    assert torch.allclose(fake_vae.last_decode_input, latents)


def test_encode_and_decode_cast_inputs_to_loaded_vae_dtype() -> None:
    """VAE wrapper should match floating inputs to the loaded VAE runtime dtype."""
    fake_vae = _FakeTypedVAE(dtype=torch.bfloat16)
    model = WanVAE(fake_vae, WanVAEConfig(deterministic=True, input_layout="BTCHW", input_range="minus_one_to_one"))

    latents = model.encode(torch.zeros(1, 2, 3, 4, 4, dtype=torch.float32))
    model.decode(latents.float(), output_layout="BTCHW", output_range="zero_to_one")

    assert fake_vae.last_encode_input_dtype == torch.bfloat16
    assert fake_vae.last_decode_input_dtype == torch.bfloat16


def test_encode_rejects_unsupported_input_layout() -> None:
    """Reject encode requests whose configured layout is not recognized."""
    model = WanVAE(_FakeVAE(), WanVAEConfig(input_layout="bad"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported layout"):
        model.encode(torch.zeros(1, 2, 3, 4, 4))


def test_encode_rejects_ambiguous_auto_input_range() -> None:
    """Require an explicit input range when auto-detection cannot classify the tensor values."""
    model = WanVAE(_FakeVAE(), WanVAEConfig(input_range="auto"))
    video = torch.tensor([[[[[-2.0, 2.0], [0.0, 3.0]]]]], dtype=torch.float32).expand(1, 1, 3, 2, 2)

    with pytest.raises(ValueError, match="Unable to infer input range"):
        model.encode(video)


def test_encode_requires_wan_latent_stats_when_normalizing() -> None:
    """Fail clearly when Wan latent normalization stats are missing."""
    fake_vae = _FakeVAE()
    fake_vae.config = types.SimpleNamespace(z_dim=3)
    model = WanVAE(fake_vae, WanVAEConfig(latent_format="wan", input_range="minus_one_to_one"))

    with pytest.raises(ValueError, match="latents_mean`/`latents_std"):
        model.encode(torch.zeros(1, 2, 3, 4, 4))


def test_decode_rejects_invalid_payload_and_output_options() -> None:
    """Validate decode payload extraction and output layout/range arguments."""
    fake_vae = _FakeVAE()
    model = WanVAE(fake_vae, WanVAEConfig())
    latents = torch.zeros(1, 3, 1, 2, 2)

    with pytest.raises(ValueError, match="Unsupported output layout"):
        model.decode(latents, output_layout="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported output range"):
        model.decode(latents, output_range="bad")  # type: ignore[arg-type]

    fake_vae.decode = lambda z: object()
    with pytest.raises(ValueError, match="missing tensor/sample"):
        model.decode(latents)
