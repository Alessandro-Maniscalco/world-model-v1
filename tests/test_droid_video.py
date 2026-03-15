"""Tests for shared DROID preview clip export helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from world_model.data import droid_video


def _install_fake_lerobot_module(monkeypatch, dataset_cls: type) -> None:
    """Install a fake LeRobot dataset module for DROID helper tests."""
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = dataset_cls
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)


def test_load_droid_images_returns_rgb_frames_and_fps(monkeypatch) -> None:
    """load_droid_images should convert dataset frames to PIL RGB images."""

    class FakeDataset:
        """Minimal LeRobotDataset stub for DROID preview tests."""

        def __init__(self, repo_id: str, episodes: list[int], video_backend: str) -> None:
            self.repo_id = repo_id
            self.episodes = episodes
            self.video_backend = video_backend
            self.fps = 9.5
            self.samples = [
                {"observation.images.exterior_1_left": torch.ones(3, 4, 5, dtype=torch.uint8) * 32},
                {"observation.images.exterior_1_left": torch.ones(3, 4, 5, dtype=torch.uint8) * 64},
            ]

        def __len__(self) -> int:
            """Return the number of fake samples."""
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            """Return one fake dataset sample."""
            return self.samples[index]

    _install_fake_lerobot_module(monkeypatch, FakeDataset)

    images, fps = droid_video.load_droid_images(
        repo_id="lerobot/droid_1.0.1",
        episode_index=0,
        frame_offset=0,
        video_key="observation.images.exterior_1_left",
        num_frames=2,
    )

    assert fps == 9.5
    assert len(images) == 2
    assert all(isinstance(image, Image.Image) for image in images)
    assert all(image.mode == "RGB" for image in images)
    assert images[0].size == (5, 4)


def test_export_droid_preview_clip_writes_frames_and_metadata(tmp_path: Path, monkeypatch) -> None:
    """export_droid_preview_clip should save ordered PNGs and return metadata."""

    def fake_load_droid_images(**_: object) -> tuple[list[Image.Image], float]:
        """Return a deterministic two-frame fake preview batch."""
        return [Image.new("RGB", (8, 6), (255, 0, 0)), Image.new("RGB", (8, 6), (0, 255, 0))], 7.6

    def fake_export_preview_video(*, images: list[Image.Image], output_path: Path, fps: int) -> Path:
        """Create a placeholder mp4 file without relying on ffmpeg/imageio."""
        assert len(images) == 2
        assert fps == 8
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        return output_path

    monkeypatch.setattr(droid_video, "load_droid_images", fake_load_droid_images)
    monkeypatch.setattr(droid_video, "export_preview_video", fake_export_preview_video)
    monkeypatch.setattr(
        droid_video,
        "detect_dataset_storage_type",
        lambda **_: "mp4 files under /tmp/fake-videos",
    )

    export = droid_video.export_droid_preview_clip(
        repo_id="lerobot/droid_1.0.1",
        episode_index=0,
        frame_offset=0,
        video_key="observation.images.exterior_1_left",
        num_frames=2,
        output_dir=tmp_path,
    )

    assert export.dataset_fps == 7.6
    assert export.preview_fps == 8
    assert export.storage_type == "mp4 files under /tmp/fake-videos"
    assert export.video_path == tmp_path / "preview.mp4"
    assert export.video_path.exists()
    assert len(export.frame_paths) == 2
    assert export.frame_paths[0].name == "frame_0000.png"
    assert export.frame_paths[1].name == "frame_0001.png"
    assert all(path.exists() for path in export.frame_paths)


def test_to_pil_rgb_handles_channel_first_float_and_alpha_inputs() -> None:
    """Normalize float channel-first arrays and drop alpha channels when present."""
    gray = droid_video.to_pil_rgb(torch.full((1, 2, 5), 0.5, dtype=torch.float32))
    rgba = droid_video.to_pil_rgb(np.full((2, 3, 4), 255, dtype=np.uint8))

    assert gray.mode == "RGB"
    assert gray.size == (5, 2)
    assert gray.getpixel((0, 0)) == (128, 128, 128)
    assert rgba.mode == "RGB"
    assert rgba.getpixel((0, 0)) == (255, 255, 255)


def test_resolve_preview_fps_rounds_and_rejects_invalid_values() -> None:
    """Round valid FPS values and reject non-positive preview rates."""
    assert droid_video.resolve_preview_fps(None, 7.6) == 8
    assert droid_video.resolve_preview_fps(11.2, 7.6) == 11

    with pytest.raises(ValueError, match="fps must be >= 1"):
        droid_video.resolve_preview_fps(0.5, 7.6)


def test_load_droid_images_rejects_missing_video_key(monkeypatch) -> None:
    """Report available camera keys when the requested video key is absent."""
    class FakeDataset:
        """Expose one sample without the requested camera stream."""

        def __init__(self, repo_id: str, episodes: list[int], video_backend: str) -> None:
            del repo_id, episodes, video_backend
            self.fps = 10.0
            self.samples = [{"observation.images.other": torch.ones(3, 4, 5, dtype=torch.uint8)}]

        def __len__(self) -> int:
            """Return the fake dataset length."""
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            """Return one fake sample."""
            return self.samples[index]

    _install_fake_lerobot_module(monkeypatch, FakeDataset)

    with pytest.raises(KeyError, match="Available camera keys"):
        droid_video.load_droid_images(
            repo_id="lerobot/droid_1.0.1",
            episode_index=0,
            frame_offset=0,
            video_key="observation.images.exterior_1_left",
            num_frames=1,
        )


def test_load_droid_images_rejects_requests_past_episode_end(monkeypatch) -> None:
    """Fail fast when the requested frame range exceeds the episode length."""
    class FakeDataset:
        """Expose a short fake dataset for bounds-check coverage."""

        def __init__(self, repo_id: str, episodes: list[int], video_backend: str) -> None:
            del repo_id, episodes, video_backend
            self.fps = 10.0
            self.samples = [{"observation.images.exterior_1_left": torch.ones(3, 4, 5, dtype=torch.uint8)}]

        def __len__(self) -> int:
            """Return the fake dataset length."""
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            """Return one fake sample."""
            return self.samples[index]

    _install_fake_lerobot_module(monkeypatch, FakeDataset)

    with pytest.raises(ValueError, match="exceed episode length"):
        droid_video.load_droid_images(
            repo_id="lerobot/droid_1.0.1",
            episode_index=0,
            frame_offset=0,
            video_key="observation.images.exterior_1_left",
            num_frames=2,
        )
