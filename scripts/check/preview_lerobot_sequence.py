"""Preview sequential frames from a LeRobot dataset sample.

This script saves ordered PNG frames and an MP4 preview for quick inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


LEROBOT_REPO_ID = "lerobot/libero"
LEROBOT_SAMPLE_INDEX = 0
LEROBOT_VIDEO_KEY = "observation.images.image"
NUM_CONDITION_FRAMES = 19
OUTPUT_DIR = Path("runs/check_lerobot_preview")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LeRobot frame preview export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=LEROBOT_REPO_ID)
    parser.add_argument("--sample-index", type=int, default=LEROBOT_SAMPLE_INDEX)
    parser.add_argument("--video-key", default=LEROBOT_VIDEO_KEY)
    parser.add_argument("--num-frames", type=int, default=NUM_CONDITION_FRAMES)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Preview FPS. Defaults to dataset FPS when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _to_pil_rgb(frame: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
    """Convert a frame tensor/array/image into a correctly ordered RGB PIL image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor/array, got shape={array.shape}.")

    # Normalize channel-first (C,H,W) and channel-last (H,W,C) to HWC.
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    elif array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Cannot infer channel dimension from shape={array.shape}.")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    # Preserve true colors for both [0,1] float and [0,255] uint8 inputs.
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array)) if array.size else 0.0
        min_value = float(np.min(array)) if array.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            array = (array * 255.0).round()
        array = np.clip(array, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)

    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def load_lerobot_images(
    *,
    repo_id: str,
    sample_index: int,
    video_key: str,
    num_frames: int,
) -> tuple[list[Image.Image], float]:
    """Load sequential LeRobot frames and return PIL images with dataset FPS."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    end_index = sample_index + num_frames
    if end_index > len(dataset):
        raise ValueError(
            f"Requested frames [{sample_index}, {end_index - 1}] exceed dataset length {len(dataset)}."
        )

    images: list[Image.Image] = []
    for idx in range(sample_index, end_index):
        sample = dataset[idx]
        if video_key not in sample:
            raise KeyError(f"video_key={video_key!r} not found in sample at index {idx}.")
        images.append(_to_pil_rgb(sample[video_key]))
    dataset_fps = float(getattr(dataset, "fps", 10.0))
    return images, dataset_fps


def _resolve_fps(requested_fps: float | None, dataset_fps: float) -> int:
    """Resolve final preview FPS, defaulting to dataset FPS when unspecified."""
    fps = dataset_fps if requested_fps is None else requested_fps
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}.")
    return int(round(fps))


def save_ordered_png_frames(*, images: list[Image.Image], output_dir: Path) -> list[Path]:
    """Save each image to disk with a stable sequential filename."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in frames_dir.glob("frame_*.png"):
        stale_path.unlink()
    paths: list[Path] = []
    for i, image in enumerate(images):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        image.save(frame_path)
        paths.append(frame_path)
    return paths


def export_preview_video(*, images: list[Image.Image], output_path: Path, fps: int) -> Path:
    """Export sequential frames to an MP4 preview."""
    import imageio.v2 as imageio

    video_frames = [np.array(image.convert("RGB"), dtype=np.uint8) for image in images]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        output_path,
        video_frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return output_path


def _detect_dataset_storage_type(*, repo_id: str) -> str:
    """Detect whether the local LeRobot cache stores camera streams as MP4 videos."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    videos_dir = dataset.root / "videos"
    if videos_dir.exists() and any(path.suffix.lower() == ".mp4" for path in videos_dir.rglob("*")):
        return f"mp4 files under {videos_dir}"
    return "no local mp4 files detected"


def main() -> None:
    """Load LeRobot frames and export ordered PNG + MP4 previews."""
    args = _parse_args()
    images, dataset_fps = load_lerobot_images(
        repo_id=args.repo_id,
        sample_index=args.sample_index,
        video_key=args.video_key,
        num_frames=args.num_frames,
    )
    final_fps = _resolve_fps(args.fps, dataset_fps)

    frame_paths = save_ordered_png_frames(images=images, output_dir=args.output_dir)
    video_path = export_preview_video(images=images, output_path=args.output_dir / "preview.mp4", fps=final_fps)
    storage_type = _detect_dataset_storage_type(repo_id=args.repo_id)

    print(f"Saved {len(frame_paths)} ordered frame images to: {args.output_dir / 'frames'}")
    print(f"Saved sequential preview video to: {video_path}")
    print(f"Dataset FPS: {dataset_fps:.3f}; preview FPS used: {final_fps}")
    print(f"Local dataset storage: {storage_type}")
    print(
        "Settings: "
        f"repo_id={args.repo_id}, sample_index={args.sample_index}, "
        f"video_key={args.video_key}, num_frames={args.num_frames}"
    )


if __name__ == "__main__":
    main()
