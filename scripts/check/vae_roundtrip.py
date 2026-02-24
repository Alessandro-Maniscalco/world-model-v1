"""Run a Wan VAE encode/decode roundtrip and save visual artifacts."""

from __future__ import annotations

import random
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from diffusers import AutoencoderKLWan
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    """Convert common video layouts into `BCTHW` with batch size 1."""
    if video.ndim == 3:
        if video.shape[0] == 3:
            video = video.unsqueeze(0)
        elif video.shape[-1] == 3:
            video = video.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unrecognized single-frame shape: {tuple(video.shape)}")

    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got {video.ndim}D: {tuple(video.shape)}")

    if video.shape[1] == 3:
        video_tchw = video
    elif video.shape[-1] == 3:
        video_tchw = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")

    return video_tchw.permute(1, 0, 2, 3).unsqueeze(0)


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Normalize uint8/float image tensor to `[-1, 1]` VAE range."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.max().detach().cpu()) > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def denormalize_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert `BCTHW` `[-1,1]` tensor to `THWC` uint8 numpy array."""
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)
    return x[0].permute(1, 2, 3, 0).contiguous().detach().cpu().numpy()


@torch.no_grad()
def main() -> None:
    """Encode/decode one sample and persist side-by-side outputs."""
    out_dir = Path(__file__).resolve().parents[2] / "assets" / "vae_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    clip_len = 9
    dt = 0.1
    deltas = [-(clip_len - 1 - i) * dt for i in range(clip_len)]
    ds = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps={"observation.images.image": deltas},
        video_backend="pyav",
    )

    idx = random.randint(1000, len(ds) - 1)
    sample = ds[idx]
    video = normalize_to_minus1_1(to_bcthw(sample["observation.images.image"])).to(device)

    print("Input video B,C,T,H,W:", tuple(video.shape), video.dtype)
    print("Input range:", float(video.min().cpu()), float(video.max().cpu()))

    enc = vae.encode(video)
    latents = enc.latent_dist.sample()
    print("Latents shape:", tuple(latents.shape), latents.dtype)
    print("Latents range:", float(latents.min().cpu()), float(latents.max().cpu()))
    print("Latents mean/std:", float(latents.mean().cpu()), float(latents.std().cpu()))

    dec = vae.decode(latents).sample
    print("Decoded video B,C,T,H,W:", tuple(dec.shape), dec.dtype)
    print("Decoded range:", float(dec.min().cpu()), float(dec.max().cpu()))

    recon = denormalize_to_uint8(dec)
    orig = denormalize_to_uint8(video)

    side_by_side = np.concatenate([orig, recon], axis=2)
    mp4_path = out_dir / "orig_vs_recon.mp4"
    iio.imwrite(mp4_path, side_by_side, fps=int(round(1.0 / dt)))
    print("Saved:", mp4_path)

    for t in [0, clip_len // 2, clip_len - 1]:
        png_path = out_dir / f"frame_{t:02d}.png"
        iio.imwrite(png_path, side_by_side[t])
        print("Saved:", png_path)


if __name__ == "__main__":
    main()
