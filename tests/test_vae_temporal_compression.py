"""Test the temporal compression factor of the Wan2.1 VAE.

This module verifies how many raw RGB frames map to a given number of
latent timesteps when passed through the VAE encoder.
"""

import pytest
import torch
from diffusers import AutoencoderKLWan

def test_vae_temporal_compression():
    """Verify the temporal downsampling behavior of the Wan VAE.

    The Wan 2.1 VAE temporal convolution stride effectively maps:
    - 2 to 4 frames -> 1 latent
    - 5 to 8 frames -> 2 latents
    - 9 to 12 frames -> 3 latents
    - 13 to 16 frames -> 4 latents
    - 17 to 20 frames -> 5 latents
    """
    vae = AutoencoderKLWan.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', subfolder='vae')

    # Expected latent counts for frame counts 2 through 20
    expected_latents = {
        2: 1, 3: 1, 4: 1,
        5: 2, 6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3, 12: 3,
        13: 4, 14: 4, 15: 4, 16: 4,
        17: 5, 18: 5, 19: 5, 20: 5,
    }

    for t in range(2, 21):
        # Input tensor BCTHW -> (1, 3, t, 16, 16)
        x = torch.randn(1, 3, t, 16, 16)
        z = vae.encode(x).latent_dist.mean
        t_lat = z.shape[2]
        assert t_lat == expected_latents[t], f"Expected {expected_latents[t]} latents for {t} frames, got {t_lat}"
