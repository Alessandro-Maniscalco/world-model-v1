import os
import random
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio

from diffusers import AutoencoderKLWan
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    """
    Accepts video as torch tensor in one of these common layouts:
      T,C,H,W
      T,H,W,C
      C,H,W (single frame)
      H,W,C (single frame)
    Returns B,C,T,H,W with B=1.
    """
    if video.ndim == 3:
        if video.shape[0] == 3:
            video = video.unsqueeze(0)  # T=1,C,H,W
        elif video.shape[-1] == 3:
            video = video.permute(2, 0, 1).unsqueeze(0)  # T=1,C,H,W
        else:
            raise ValueError(f"Unrecognized single-frame shape: {tuple(video.shape)}")

    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got {video.ndim}D: {tuple(video.shape)}")

    # Now either T,C,H,W or T,H,W,C
    if video.shape[1] == 3:
        video_tchw = video  # T,C,H,W
    elif video.shape[-1] == 3:
        video_tchw = video.permute(0, 3, 1, 2)  # T,C,H,W
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")

    # Convert to B,C,T,H,W
    video_bcthw = video_tchw.permute(1, 0, 2, 3).unsqueeze(0)
    return video_bcthw


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """
    If input looks like uint8 0..255 or float 0..1, map to [-1, 1].
    """
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        mx = float(x.max().detach().cpu())
        if mx > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def denormalize_to_uint8(x: torch.Tensor) -> np.ndarray:
    """
    Map [-1, 1] to uint8 0..255, return numpy video as T,H,W,C.
    """
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)

    # x is B,C,T,H,W
    x = x[0].permute(1, 2, 3, 0).contiguous()  # T,H,W,C
    return x.detach().cpu().numpy()


@torch.no_grad()
def main():
    out_dir = Path(__file__).resolve().parent.parent / "assets" / "vae_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load Wan video VAE from the Wan2.1 Diffusers repo (vae subfolder)
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    clip_len = 9
    dt = 0.1
    deltas = [-(clip_len - 1 - i) * dt for i in range(clip_len)]  # [-0.7, ..., 0.0]

    ds = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps={"observation.images.image": deltas},
        video_backend="pyav",
    )

    idx = random.randint(1000, len(ds) - 1)
    sample = ds[idx]

    video = sample["observation.images.image"]
    video = to_bcthw(video)
    video = normalize_to_minus1_1(video).to(device)

    print("Input video B,C,T,H,W:", tuple(video.shape), video.dtype)
    print("Input range:", float(video.min().cpu()), float(video.max().cpu()))

    # Encode to latents
    enc = vae.encode(video)
    latents = enc.latent_dist.sample()
    # For many Diffusers VAEs, scaling_factor is applied before feeding diffusion models.
    # For a pure roundtrip sanity check, we decode the same latents we sampled.
    print("Latents shape:", tuple(latents.shape), latents.dtype)
    print("Latents range:", float(latents.min().cpu()), float(latents.max().cpu()))
    print("Latents mean/std:", float(latents.mean().cpu()), float(latents.std().cpu()))

    # Decode back to video
    dec = vae.decode(latents).sample
    print("Decoded video B,C,T,H,W:", tuple(dec.shape), dec.dtype)
    print("Decoded range:", float(dec.min().cpu()), float(dec.max().cpu()))

    recon = denormalize_to_uint8(dec)
    orig = denormalize_to_uint8(video)

    # Save a side-by-side video: left original, right reconstruction
    side_by_side = np.concatenate([orig, recon], axis=2)  # concat width, T,H,2W,C
    mp4_path = out_dir / "orig_vs_recon.mp4"
    iio.imwrite(mp4_path, side_by_side, fps=int(round(1.0 / dt)))
    print("Saved:", mp4_path)

    # Save a few frame PNGs for quick inspection
    for t in [0, clip_len // 2, clip_len - 1]:
        frame = side_by_side[t]
        png_path = out_dir / f"frame_{t:02d}.png"
        iio.imwrite(png_path, frame)
        print("Saved:", png_path)


if __name__ == "__main__":
    main()
