import torch

from world_model.latents import WanVAE, WanVAEConfig


class _FakeLatentDist:
    def __init__(self, base: torch.Tensor):
        self.mean = base

    def sample(self) -> torch.Tensor:
        return self.mean + 1.0


class _FakeEncodeOut:
    def __init__(self, dist: _FakeLatentDist):
        self.latent_dist = dist


class _FakeDecodeOut:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class _FakeVAE:
    def encode(self, x: torch.Tensor):
        # Keep shape [B,C,T,H,W], but shift values to track path.
        base = x + 10.0
        return _FakeEncodeOut(_FakeLatentDist(base))

    def decode(self, z: torch.Tensor):
        # Return in VAE-native range [-1,1] for output conversion tests.
        sample = torch.tanh(z)
        return _FakeDecodeOut(sample)


def test_encode_deterministic_uses_mean():
    model = WanVAE(_FakeVAE(), WanVAEConfig(deterministic=True, input_layout="BTCHW", input_range="minus_one_to_one"))
    video = torch.zeros(2, 4, 3, 8, 8)

    latents = model.encode(video)
    assert latents.shape == (2, 3, 4, 8, 8)
    assert torch.allclose(latents, torch.full_like(latents, 10.0))


def test_encode_stochastic_uses_sample():
    model = WanVAE(_FakeVAE(), WanVAEConfig(deterministic=False, input_layout="BTCHW", input_range="minus_one_to_one"))
    video = torch.zeros(1, 2, 3, 4, 4)

    latents = model.encode(video)
    # sample() returns mean + 1.0
    assert torch.allclose(latents, torch.full_like(latents, 11.0))


def test_encode_accepts_bthwc_layout():
    model = WanVAE(_FakeVAE(), WanVAEConfig(deterministic=True, input_layout="BTHWC", input_range="uint8"))
    video = torch.zeros(1, 3, 6, 5, 3, dtype=torch.uint8)

    latents = model.encode(video)
    assert latents.shape == (1, 3, 3, 6, 5)


def test_decode_layout_and_uint8_output():
    model = WanVAE(_FakeVAE(), WanVAEConfig())
    latents = torch.zeros(1, 3, 2, 4, 4)

    decoded = model.decode(latents, output_layout="BTHWC", output_range="uint8")
    assert decoded.shape == (1, 2, 4, 4, 3)
    assert decoded.dtype == torch.uint8
