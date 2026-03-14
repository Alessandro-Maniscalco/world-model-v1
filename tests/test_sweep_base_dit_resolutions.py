"""Tests for the upstream-base resolution sweep smoke-check script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import numpy as np
from PIL import Image


def _load_script_module():
    """Load the base-resolution sweep script directly from its file path."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "check" / "sweep_base_dit_resolutions.py"
    spec = importlib.util.spec_from_file_location("sweep_base_dit_resolutions", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def test_resize_condition_image_is_noop_for_matching_rgb_size() -> None:
    """Matching RGB frames should be returned unchanged to avoid unnecessary resampling."""
    image = Image.new("RGB", (320, 240), (12, 34, 56))

    resized = script._resize_condition_image(image=image, width=320, height=240)

    assert resized is image


def test_build_video_and_mask_uses_resized_condition_frame_pixels() -> None:
    """Conditioned frames should use the explicit resize helper output."""
    source = np.zeros((6, 8, 3), dtype=np.uint8)
    source[:, :4] = (255, 0, 0)
    source[:, 4:] = (0, 255, 0)
    image = Image.fromarray(source, mode="RGB")

    video, mask = script._build_video_and_mask_for_resolution(
        condition_images=[image],
        condition_indices=None,
        width=320,
        height=240,
    )

    expected = np.asarray(
        script._resize_condition_image(image=image, width=320, height=240),
        dtype=np.uint8,
    )

    assert np.array_equal(np.asarray(video[0], dtype=np.uint8), expected)
    assert np.array_equal(np.asarray(mask[0], dtype=np.uint8), np.zeros((240, 320), dtype=np.uint8))


def test_save_ordered_png_frames_replaces_stale_outputs(tmp_path: Path) -> None:
    """PNG frame export should replace stale numbered frames with the new sequence."""
    stale_path = tmp_path / "frame_9999.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(stale_path)

    saved_paths = script._save_ordered_png_frames(
        images=[
            Image.new("RGB", (4, 4), (10, 20, 30)),
            Image.new("RGB", (4, 4), (40, 50, 60)),
        ],
        output_dir=tmp_path,
    )

    assert stale_path.exists() is False
    assert saved_paths == [tmp_path / "frame_0000.png", tmp_path / "frame_0001.png"]
    assert all(path.exists() for path in saved_paths)


def test_save_ordered_png_frames_supports_numpy_frames(tmp_path: Path) -> None:
    """PNG frame export should accept the NumPy frames returned by the base pipeline."""
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[..., 1] = 255

    saved_paths = script._save_ordered_png_frames(images=[frame], output_dir=tmp_path)

    saved = Image.open(saved_paths[0]).convert("RGB")
    assert saved.size == (6, 4)
    assert np.array_equal(np.asarray(saved, dtype=np.uint8), frame)


def test_run_one_base_resolution_uses_explicit_prompt_arguments(tmp_path: Path, monkeypatch) -> None:
    """The base sweep should forward explicit prompt overrides to the pipeline call."""
    captured: dict[str, object] = {}

    class _FakePipe:
        """Minimal pipeline stub that records prompt kwargs and returns one clip."""

        def __call__(self, **kwargs):
            """Capture call kwargs and return one generated clip."""
            captured["call_kwargs"] = kwargs
            return types.SimpleNamespace(
                frames=[[Image.new("RGB", (kwargs["width"], kwargs["height"]), (1, 2, 3))]]
            )

    monkeypatch.setattr(
        script.base,
        "_export_video",
        lambda **_: str(tmp_path / "out.mp4"),
    )

    result = script._run_one_base_resolution(
        pipe=_FakePipe(),
        width=320,
        height=240,
        condition_images=[Image.new("RGB", (320, 240), (9, 9, 9))],
        condition_indices=None,
        output_dir=tmp_path,
        num_inference_steps=5,
        guidance_scale=1.5,
        fps=10,
        seed=0,
        prompt="",
        negative_prompt="low quality",
    )

    assert result["status"] == "ok"
    assert captured["call_kwargs"]["prompt"] == ""
    assert captured["call_kwargs"]["negative_prompt"] == "low quality"
