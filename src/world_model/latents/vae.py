"""Latent-time VAE API for world-model training.

This module defines a thin wrapper around Wan's video VAE so the rest of the
project can use a stable interface:

- `encode(video) -> latents` where latent time is authoritative.
- `decode(latents) -> video` with explicit output layout/range control.
- deterministic encoding mode (`latent_dist.mean`) for reproducible caching and
  training, plus optional stochastic mode (`latent_dist.sample`).

Conventions:
- Input video layout is configured as `BTCHW` or `BTHWC`.
- Internally, data is converted to VAE-native `[B, C, T, H, W]`.
- Encoder input is normalized to `[-1, 1]`.
- Latent output is `[B, C_lat, T_lat, H_lat, W_lat]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


VideoLayout = Literal["BTCHW", "BTHWC"]
VideoRange = Literal["auto", "minus_one_to_one", "zero_to_one", "uint8"]
OutputRange = Literal["minus_one_to_one", "zero_to_one", "uint8"]


@dataclass(frozen=True)
class WanVAEConfig:
    deterministic: bool = True
    input_layout: VideoLayout = "BTCHW"
    input_range: VideoRange = "auto"


class WanVAE:
    """
    Thin latent-time API around Wan VAE encode/decode.

    Deterministic mode is intended for caching and training reproducibility.
    """

    def __init__(self, vae: Any, config: WanVAEConfig | None = None):
        """Store VAE handle and immutable encode/decode behavior config."""
        self.vae = vae
        self.config = config or WanVAEConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder: str = "vae",
        torch_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        deterministic: bool = True,
    ) -> "WanVAE":
        """Load and freeze pretrained Wan VAE weights."""
        try:
            from diffusers import AutoencoderKLWan
        except ImportError as exc:
            raise ImportError("diffusers is required to load Wan VAE") from exc

        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
        )
        if device is not None:
            vae = vae.to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        return cls(vae=vae, config=WanVAEConfig(deterministic=deterministic))

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode input video into latent-time tensor.

        Input (configured or inferred):
          - BTCHW: [B, T, C, H, W]
          - BTHWC: [B, T, H, W, C]

        Output:
          - [B, C_lat, T_lat, H_lat, W_lat]
        """
        bcthw = _to_bcthw(video, self.config.input_layout)
        bcthw = _normalize_video(bcthw, self.config.input_range)

        encoded = self.vae.encode(bcthw)

        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None:
            latents = getattr(encoded, "latents", None)
            if latents is None:
                raise ValueError("VAE encode output missing both `latent_dist` and `latents`")
            return latents

        if self.config.deterministic:
            return latent_dist.mean
        return latent_dist.sample()

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        output_layout: VideoLayout = "BTCHW",
        output_range: OutputRange = "zero_to_one",
    ) -> torch.Tensor:
        """
        Decode latent tensor back to video.

        Input:
          - latents [B, C_lat, T_lat, H_lat, W_lat]

        Output layout:
          - BTCHW -> [B, T, C, H, W]
          - BTHWC -> [B, T, H, W, C]
        """
        if latents.ndim != 5:
            raise ValueError(f"Expected latents [B,C,T,H,W], got shape {tuple(latents.shape)}")

        decoded = self.vae.decode(latents)
        video_bcthw = _extract_decoded_sample(decoded)
        video = _from_bcthw(video_bcthw, output_layout)
        return _format_output_range(video, output_range)


def _to_bcthw(video: torch.Tensor, layout: VideoLayout) -> torch.Tensor:
    """Convert configured input layout to VAE-native `[B,C,T,H,W]`."""
    if video.ndim != 5:
        raise ValueError(f"Expected video with 5 dims, got shape {tuple(video.shape)}")

    if layout == "BTCHW":
        if video.shape[2] != 3:
            raise ValueError(f"BTCHW expects C=3 at dim=2, got shape {tuple(video.shape)}")
        return video.permute(0, 2, 1, 3, 4)

    if layout == "BTHWC":
        if video.shape[-1] != 3:
            raise ValueError(f"BTHWC expects C=3 at dim=-1, got shape {tuple(video.shape)}")
        btchw = video.permute(0, 1, 4, 2, 3)
        return btchw.permute(0, 2, 1, 3, 4)

    raise ValueError(f"Unsupported layout: {layout}")


def _from_bcthw(video_bcthw: torch.Tensor, output_layout: VideoLayout) -> torch.Tensor:
    """Convert decoded VAE-native `[B,C,T,H,W]` to target output layout."""
    if video_bcthw.ndim != 5:
        raise ValueError(f"Expected decoded video [B,C,T,H,W], got shape {tuple(video_bcthw.shape)}")

    if output_layout == "BTCHW":
        return video_bcthw.permute(0, 2, 1, 3, 4)
    if output_layout == "BTHWC":
        return video_bcthw.permute(0, 2, 3, 4, 1)

    raise ValueError(f"Unsupported output layout: {output_layout}")


def _normalize_video(video_bcthw: torch.Tensor, input_range: VideoRange) -> torch.Tensor:
    """Normalize video tensor to `[-1,1]` for VAE encode input."""
    x = video_bcthw
    if input_range == "minus_one_to_one":
        return x.float()
    if input_range == "zero_to_one":
        return x.float() * 2.0 - 1.0
    if input_range == "uint8":
        return x.float() / 255.0 * 2.0 - 1.0

    # auto
    if x.dtype == torch.uint8:
        return x.float() / 255.0 * 2.0 - 1.0

    x = x.float()
    max_val = float(x.detach().max().cpu()) if x.numel() > 0 else 1.0
    min_val = float(x.detach().min().cpu()) if x.numel() > 0 else -1.0

    if min_val >= -0.1 and max_val <= 1.1:
        return x * 2.0 - 1.0
    if min_val >= -1.1 and max_val <= 1.1:
        return x
    if min_val >= 0.0 and max_val <= 255.0:
        return x / 255.0 * 2.0 - 1.0

    raise ValueError(
        f"Unable to infer input range from min={min_val:.3f}, max={max_val:.3f}; set config.input_range explicitly."
    )


def _extract_decoded_sample(decoded: Any) -> torch.Tensor:
    """Extract decoded tensor from diffusers-style output payload."""
    if torch.is_tensor(decoded):
        return decoded

    sample = getattr(decoded, "sample", None)
    if sample is not None:
        return sample

    raise ValueError("VAE decode output missing tensor/sample")


def _format_output_range(video: torch.Tensor, output_range: OutputRange) -> torch.Tensor:
    """Map decoded video from `[-1,1]` into the configured output range."""
    if output_range == "minus_one_to_one":
        return video

    if output_range == "zero_to_one":
        return ((video + 1.0) / 2.0).clamp(0.0, 1.0)

    if output_range == "uint8":
        return (((video + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    raise ValueError(f"Unsupported output range: {output_range}")
