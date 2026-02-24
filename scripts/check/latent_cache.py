"""Validate deterministic latent caching against direct VAE encoding."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers import AutoencoderKLWan
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    """Convert `TCHW`/`THWC` video to `BCTHW` with batch size 1."""
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video, got {video.ndim}D: {tuple(video.shape)}")
    if video.shape[1] == 3:
        tchw = video
    elif video.shape[-1] == 3:
        tchw = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")
    return tchw.permute(1, 0, 2, 3).unsqueeze(0)


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Normalize uint8/float image tensors to the VAE input range `[-1, 1]`."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.max().cpu()) > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Compute a stable SHA-256 digest for a tensor payload."""
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def main() -> None:
    """Encode a small subset, cache latents, and verify deterministic re-loads."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    clip_len = 8
    dt = 0.1
    deltas = [-(clip_len - 1 - i) * dt for i in range(clip_len)]
    ds = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps={"observation.images.image": deltas},
        video_backend="pyav",
    )

    cache_root = Path("cache/libero/observation.images.image")
    cache_root.mkdir(parents=True, exist_ok=True)

    indices = list(range(0, 20))
    metadata: dict[int, dict[str, object]] = {}
    for idx in tqdm(indices, desc="Caching latents"):
        sample = ds[idx]
        video = normalize_to_minus1_1(to_bcthw(sample["observation.images.image"])).to(device)

        enc = vae.encode(video)
        latents = enc.latent_dist.mean

        path = cache_root / f"z_{idx:08d}.pt"
        torch.save(latents.cpu(), path)
        metadata[idx] = {
            "path": str(path),
            "shape": tuple(latents.shape),
            "hash": sha256_tensor(latents),
        }

    print("Wrote", len(indices), "latent files to", cache_root)

    for idx in indices[:5]:
        path = Path(metadata[idx]["path"])  # type: ignore[index]
        loaded = torch.load(path, map_location="cpu")
        assert tuple(loaded.shape) == metadata[idx]["shape"]
        assert sha256_tensor(loaded) == metadata[idx]["hash"]
    print("Reload check passed for 5 samples")

    max_abs_diffs = []
    for idx in indices[:5]:
        sample = ds[idx]
        video = normalize_to_minus1_1(to_bcthw(sample["observation.images.image"])).to(device)
        direct = vae.encode(video).latent_dist.mean.detach().cpu()
        cached = torch.load(cache_root / f"z_{idx:08d}.pt", map_location="cpu")
        max_abs_diffs.append((direct - cached).abs().max().item())

    print("Max abs diff (direct vs cached) for 5 samples:", max_abs_diffs)
    print("Typical target: ~0 to 1e-6 range. If larger, check dtype, preprocessing, or VAE mode.")


if __name__ == "__main__":
    main()
