"""Shared helpers for exporting local DROID video clips from LeRobot datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class DroidPreviewExport:
    """Describe the exported local clip and its preview metadata."""

    frame_paths: list[Path]
    video_path: Path
    dataset_fps: float
    preview_fps: int
    storage_type: str


def to_pil_rgb(frame: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
    """Convert a frame tensor, array, or image into an RGB PIL image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor/array, got shape={array.shape}.")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    elif array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Cannot infer channel dimension from shape={array.shape}.")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        min_value = float(np.min(array)) if array.size else 0.0
        max_value = float(np.max(array)) if array.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            array = (array * 255.0).round()
        array = np.clip(array, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)

    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def load_droid_images(
    *,
    repo_id: str,
    episode_index: int,
    frame_offset: int,
    video_key: str,
    num_frames: int,
) -> tuple[list[Image.Image], float]:
    """Load sequential DROID frames and return RGB images plus dataset FPS."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}.")
    if frame_offset < 0:
        raise ValueError(f"frame_offset must be >= 0, got {frame_offset}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

    dataset = LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
    end_index = frame_offset + num_frames
    if end_index > len(dataset):
        raise ValueError(
            f"Requested frames [{frame_offset}, {end_index - 1}] exceed episode length {len(dataset)}."
        )

    images: list[Image.Image] = []
    for idx in range(frame_offset, end_index):
        sample = dataset[idx]
        if video_key not in sample:
            available_keys = [key for key in sample if key.startswith("observation.images.")]
            raise KeyError(
                f"video_key={video_key!r} not found in sample at index {idx}. "
                f"Available camera keys: {available_keys}"
            )
        images.append(to_pil_rgb(sample[video_key]))

    dataset_fps = float(getattr(dataset, "fps", 10.0))
    return images, dataset_fps


def resolve_preview_fps(requested_fps: float | None, dataset_fps: float) -> int:
    """Resolve the preview FPS, defaulting to dataset FPS when omitted."""
    fps = dataset_fps if requested_fps is None else requested_fps
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}.")
    return int(round(fps))


def save_ordered_png_frames(*, images: list[Image.Image], output_dir: Path) -> list[Path]:
    """Save preview images to deterministic sequential PNG filenames."""
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
    """Export sequential preview images to an MP4 clip."""
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_frames = [np.array(image.convert("RGB"), dtype=np.uint8) for image in images]
    imageio.mimsave(
        output_path,
        video_frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return output_path


def detect_dataset_storage_type(*, repo_id: str, episode_index: int) -> str:
    """Report whether cached DROID data includes mp4 files for the episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id,
        episodes=[episode_index],
        video_backend="pyav",
        download_videos=False,
    )
    videos_dir = dataset.root / "videos"
    if videos_dir.exists() and any(path.suffix.lower() == ".mp4" for path in videos_dir.rglob("*")):
        return f"mp4 files under {videos_dir}"
    return "no local mp4 files detected"


def export_droid_preview_clip(
    *,
    repo_id: str,
    episode_index: int,
    frame_offset: int,
    video_key: str,
    num_frames: int,
    output_dir: Path,
    fps: float | None = None,
) -> DroidPreviewExport:
    """Load DROID frames, save sequential PNGs, and export a local MP4 preview clip."""
    images, dataset_fps = load_droid_images(
        repo_id=repo_id,
        episode_index=episode_index,
        frame_offset=frame_offset,
        video_key=video_key,
        num_frames=num_frames,
    )
    preview_fps = resolve_preview_fps(fps, dataset_fps)
    frame_paths = save_ordered_png_frames(images=images, output_dir=output_dir)
    video_path = export_preview_video(images=images, output_path=output_dir / "preview.mp4", fps=preview_fps)
    storage_type = detect_dataset_storage_type(repo_id=repo_id, episode_index=episode_index)
    return DroidPreviewExport(
        frame_paths=frame_paths,
        video_path=video_path,
        dataset_fps=dataset_fps,
        preview_fps=preview_fps,
        storage_type=storage_type,
    )
