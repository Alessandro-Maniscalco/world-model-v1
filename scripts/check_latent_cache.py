import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers import AutoencoderKLWan
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    # Accept T,C,H,W or T,H,W,C
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video, got {video.ndim}D: {tuple(video.shape)}")
    if video.shape[1] == 3:
        tchw = video
    elif video.shape[-1] == 3:
        tchw = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")
    return tchw.permute(1, 0, 2, 3).unsqueeze(0)  # B,C,T,H,W


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.max().cpu()) > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def sha256_tensor(t: torch.Tensor) -> str:
    # Stable hash for quick equality check after reload
    b = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load Wan VAE
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    # Dataset window
    clip_len = 8
    dt = 0.1
    deltas = [-(clip_len - 1 - i) * dt for i in range(clip_len)]  # past to present
    ds = LeRobotDataset("lerobot/libero", delta_timestamps={"observation.images.image": deltas}, video_backend="pyav")

    cache_root = Path("cache/libero/observation.images.image")
    cache_root.mkdir(parents=True, exist_ok=True)

    # Tiny subset for the check
    indices = list(range(0, 20))  # first 20 samples

    # 1) Write cache
    meta = {}
    for idx in tqdm(indices, desc="Caching latents"):
        sample = ds[idx]
        video = sample["observation.images.image"]
        video = normalize_to_minus1_1(to_bcthw(video)).to(device)

        enc = vae.encode(video)
        latents = enc.latent_dist.mean  # deterministic for caching

        path = cache_root / f"z_{idx:08d}.pt"
        torch.save(latents.cpu(), path)

        meta[idx] = {
            "path": str(path),
            "shape": tuple(latents.shape),
            "hash": sha256_tensor(latents),
        }

    print("Wrote", len(indices), "latent files to", cache_root)

    # 2) Reload and confirm exact match to what was saved
    for idx in indices[:5]:
        path = Path(meta[idx]["path"])
        loaded = torch.load(path, map_location="cpu")
        assert tuple(loaded.shape) == meta[idx]["shape"]
        assert sha256_tensor(loaded) == meta[idx]["hash"]
    print("Reload check passed for 5 samples")

    # 3) Compare cached latents to direct encoding again
    # Because we use latent_dist.mean and eval mode, this should match extremely closely.
    max_abs_diffs = []
    for idx in indices[:5]:
        sample = ds[idx]
        video = normalize_to_minus1_1(to_bcthw(sample["observation.images.image"])).to(device)
        direct = vae.encode(video).latent_dist.mean.detach().cpu()

        cached = torch.load(cache_root / f"z_{idx:08d}.pt", map_location="cpu")
        diff = (direct - cached).abs().max().item()
        max_abs_diffs.append(diff)

    print("Max abs diff (direct vs cached) for 5 samples:", max_abs_diffs)
    print("Typical target: ~0 to 1e-6 range. If larger, check dtype, preprocessing, or VAE mode.")


if __name__ == "__main__":
    main()
