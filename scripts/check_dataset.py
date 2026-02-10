
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    repo_id = "lerobot/libero"
    # Use pyav to avoid torchcodec/FFmpeg shared-library issues
    ds = LeRobotDataset(repo_id, video_backend="pyav")

    print("len(ds):", len(ds))
    num_timesteps = 3
    samples = [ds[i] for i in range(num_timesteps)]

    sample = samples[0]
    print("keys:", list(sample.keys()))
    for k, v in sample.items():
        try:
            print(k, getattr(v, "shape", None), getattr(v, "dtype", None))
        except Exception:
            print(k, type(v))

    # Image keys: observation.images.* (tensors CHW, float32)
    image_keys = [k for k in sample if k.startswith("observation.images.")]
    if not image_keys:
        return

    n_cams = len(image_keys)
    fig, axes = plt.subplots(num_timesteps, n_cams, figsize=(5 * n_cams, 5 * num_timesteps))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and axes.shape[1] == num_timesteps:
        axes = axes.T  # (num_timesteps, 1) from subplots(num_timesteps, 1)

    for t, sample_t in enumerate(samples):
        state = sample_t["observation.state"]
        action = sample_t["action"]
        state_str = ", ".join(f"{x:.3f}" for x in state.tolist())
        action_str = ", ".join(f"{x:.3f}" for x in action.tolist())
        print(f"t={t}  state: [{state_str}]  action: [{action_str}]")
        row_title = f"t={t}  state: [{state_str}]  action: [{action_str}]"
        for c, key in enumerate(image_keys):
            ax = axes[t, c]
            img = sample_t[key]  # (C, H, W)
            img = img.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img)
            if c == 0:
                ax.set_title(row_title, fontsize=8)
            else:
                ax.set_title(key if t == 0 else "")
            ax.axis("off")

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "assets" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "check_dataset_sample.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print("Saved:", out_path)

    plt.show()


if __name__ == "__main__":
    main()
