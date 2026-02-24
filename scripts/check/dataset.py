"""Inspect LIBERO dataset samples and save a quick visualization image."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    """Load sample timesteps, print schema stats, and save image grid."""
    repo_id = "lerobot/libero"
    ds = LeRobotDataset(repo_id, video_backend="pyav")

    print("len(ds):", len(ds))
    num_timesteps = 3
    samples = [ds[i] for i in range(num_timesteps)]

    sample = samples[0]
    print("keys:", list(sample.keys()))
    for key, value in sample.items():
        try:
            print(key, getattr(value, "shape", None), getattr(value, "dtype", None))
        except Exception:
            print(key, type(value))

    image_keys = [key for key in sample if key.startswith("observation.images.")]
    if not image_keys:
        return

    n_cams = len(image_keys)
    fig, axes = plt.subplots(num_timesteps, n_cams, figsize=(5 * n_cams, 5 * num_timesteps))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and axes.shape[1] == num_timesteps:
        axes = axes.T

    for timestep, sample_t in enumerate(samples):
        state = sample_t["observation.state"]
        action = sample_t["action"]
        state_str = ", ".join(f"{x:.3f}" for x in state.tolist())
        action_str = ", ".join(f"{x:.3f}" for x in action.tolist())
        print(f"t={timestep}  state: [{state_str}]  action: [{action_str}]")
        row_title = f"t={timestep}  state: [{state_str}]  action: [{action_str}]"
        for col, key in enumerate(image_keys):
            axis = axes[timestep, col]
            img = sample_t[key].permute(1, 2, 0).clamp(0, 1).numpy()
            axis.imshow(img)
            axis.set_title(row_title if col == 0 else (key if timestep == 0 else ""), fontsize=8)
            axis.axis("off")

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parents[2] / "assets" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "check_dataset_sample.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print("Saved:", out_path)

    plt.show()


if __name__ == "__main__":
    main()
