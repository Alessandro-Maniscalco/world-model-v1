"""Tests for reference-image routing in the Wan VACE diffuser smoke-check script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image


def _load_script_module():
    """Load the diffuser smoke-check script directly from its file path."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "check" / "wan_vace_diffuser_generate_video.py"
    spec = importlib.util.spec_from_file_location("wan_vace_diffuser_generate_video", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def test_load_reference_images_supports_aloha(monkeypatch) -> None:
    """load_reference_images should route the dense ALOHA mode to the ALOHA loader."""
    expected_image = Image.new("RGB", (8, 6), (12, 34, 56))
    monkeypatch.setattr(script, "REFERENCE_IMAGE_SOURCE", "aloha")
    monkeypatch.setattr(script, "load_aloha_images", lambda: [expected_image])

    images, indices = script.load_reference_images()

    assert images == [expected_image]
    assert indices is None


def test_load_reference_images_supports_aloha_first_last(monkeypatch) -> None:
    """load_reference_images should route sparse ALOHA mode with first-last indices."""
    first_image = Image.new("RGB", (8, 6), (0, 0, 0))
    last_image = Image.new("RGB", (8, 6), (255, 255, 255))
    monkeypatch.setattr(script, "REFERENCE_IMAGE_SOURCE", "aloha_first_last")
    monkeypatch.setattr(script, "load_aloha_first_last_images", lambda: [first_image, last_image])

    images, indices = script.load_reference_images()

    assert images == [first_image, last_image]
    assert indices == [0, script.NUM_TOTAL_FRAMES - 1]


def test_build_video_and_mask_expands_sparse_endpoint_conditions() -> None:
    """Sparse endpoint constraints should cover the latent-time buckets Wan VACE actually preserves."""
    first_image = Image.new("RGB", (8, 6), (10, 20, 30))
    last_image = Image.new("RGB", (8, 6), (200, 210, 220))

    video, mask = script.build_video_and_mask(
        [first_image, last_image],
        condition_indices=[0, script.NUM_TOTAL_FRAMES - 1],
    )

    first_rgb = np.asarray(first_image.resize((script.WIDTH, script.HEIGHT)).convert("RGB"))
    last_rgb = np.asarray(last_image.resize((script.WIDTH, script.HEIGHT)).convert("RGB"))
    placeholder_rgb = np.full((script.HEIGHT, script.WIDTH, 3), 128, dtype=np.uint8)
    keep_mask = np.zeros((script.HEIGHT, script.WIDTH), dtype=np.uint8)
    generate_mask = np.full((script.HEIGHT, script.WIDTH), 255, dtype=np.uint8)

    for frame_index in range(3):
        assert np.array_equal(np.asarray(video[frame_index]), first_rgb)
        assert np.array_equal(np.asarray(mask[frame_index]), keep_mask)

    for frame_index in range(3, 6):
        assert np.array_equal(np.asarray(video[frame_index]), placeholder_rgb)
        assert np.array_equal(np.asarray(mask[frame_index]), generate_mask)

    for frame_index in range(6, script.NUM_TOTAL_FRAMES):
        assert np.array_equal(np.asarray(video[frame_index]), last_rgb)
        assert np.array_equal(np.asarray(mask[frame_index]), keep_mask)


def test_build_video_and_mask_keeps_dense_prefix_conditions_per_frame() -> None:
    """Dense prefix conditioning should keep one source frame per timestep without sparse bucket expansion."""
    condition_images = [
        Image.new("RGB", (8, 6), (10 + idx, 20 + idx, 30 + idx))
        for idx in range(5)
    ]

    video, mask = script.build_video_and_mask(condition_images)

    keep_mask = np.zeros((script.HEIGHT, script.WIDTH), dtype=np.uint8)
    generate_mask = np.full((script.HEIGHT, script.WIDTH), 255, dtype=np.uint8)
    placeholder_rgb = np.full((script.HEIGHT, script.WIDTH, 3), 128, dtype=np.uint8)

    for frame_index, image in enumerate(condition_images):
        expected_rgb = np.asarray(image.resize((script.WIDTH, script.HEIGHT)).convert("RGB"))
        assert np.array_equal(np.asarray(video[frame_index]), expected_rgb)
        assert np.array_equal(np.asarray(mask[frame_index]), keep_mask)

    for frame_index in range(len(condition_images), script.NUM_TOTAL_FRAMES):
        assert np.array_equal(np.asarray(video[frame_index]), placeholder_rgb)
        assert np.array_equal(np.asarray(mask[frame_index]), generate_mask)


def test_generate_video_uses_torch_dtype_keyword(monkeypatch, tmp_path) -> None:
    """generate_video should load the diffusers pipeline with the supported torch_dtype keyword."""

    captured: dict[str, object] = {}

    class _FakePipeline:
        """Minimal pipeline stub that records loading kwargs and returns one frame."""

        def enable_sequential_cpu_offload(self) -> None:
            """Match the diffusers offload API."""

        def __call__(self, **kwargs):
            """Return a single generated frame without invoking any real models."""
            captured["call_kwargs"] = kwargs
            return types.SimpleNamespace(frames=[[Image.new("RGB", (8, 6), (1, 2, 3))]])

    class _FakeDiffusionPipeline:
        """Minimal loader stub that records the kwargs passed to from_pretrained."""

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            """Capture loader arguments and return the fake pipeline."""
            captured["model_id"] = model_id
            captured["from_pretrained_kwargs"] = kwargs
            return _FakePipeline()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.DiffusionPipeline = _FakeDiffusionPipeline
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr(script, "load_reference_images", lambda: ([Image.new("RGB", (8, 6), (9, 9, 9))], None))
    monkeypatch.setattr(script, "_export_video", lambda **kwargs: str(tmp_path / "out.mp4"))

    output_path = script.generate_video(tmp_path / "out.mp4")

    assert output_path == tmp_path / "out.mp4"
    assert captured["model_id"] == script.MODEL_ID
    assert captured["from_pretrained_kwargs"] == {"torch_dtype": script.torch.bfloat16}
