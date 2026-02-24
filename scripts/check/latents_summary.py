"""Summarize a small sample of cached latent tensors."""

from __future__ import annotations

from pathlib import Path

import torch


def main() -> None:
    """Print shape and basic stats for a subset of cached latents."""
    cache_dir = Path(".cache/lerobot_libero/vae_latents")
    if not cache_dir.exists():
        print("Cache dir not found.")
        return

    latents = []
    for file_path in list(cache_dir.glob("*.pt"))[:10]:
        latents.append(torch.load(file_path, map_location="cpu"))

    if not latents:
        print("No latents found")
        return

    stacked = torch.stack(latents)
    print("Latent shape:", stacked.shape)
    print("Latent mean:", stacked.mean().item())
    print("Latent std:", stacked.std().item())
    print("Latent min:", stacked.min().item(), "max:", stacked.max().item())


if __name__ == "__main__":
    main()
