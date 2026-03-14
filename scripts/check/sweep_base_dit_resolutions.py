"""Sweep the public Wan VACE base pipeline over resolutions.

Loads ALOHA fork conditioning frames and exports one MP4 per resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from diffusers import DiffusionPipeline
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wan_vace_diffuser_generate_video as base


DEFAULT_OUTPUT_DIR = Path("runs/check_wan_vace_base_resolution_sweep")
DEFAULT_RESOLUTIONS = (
    "256x192",
)
DEFAULT_REPO_ID = "lerobot/aloha_static_fork_pick_up"
DEFAULT_VIDEO_KEY = "observation.images.cam_high"
DEFAULT_EPISODE_INDEX = 0
DEFAULT_START_FRAME = 0
DEFAULT_REFERENCE_LAYOUT = "dense"
DEFAULT_PROMPT = ""
DEFAULT_NEGATIVE_PROMPT = ""


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the upstream base sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="List of WIDTHxHEIGHT values to test, e.g. 512x384 320x240.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=base.NUM_INFERENCE_STEPS,
        help="Upstream Wan VACE denoising steps.",
    )
    parser.add_argument("--guidance-scale", type=float, default=base.GUIDANCE_SCALE)
    parser.add_argument("--fps", type=int, default=base.FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt used for the upstream base VACE generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt used for the upstream base VACE generation.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    parser.add_argument(
        "--start-frame",
        "--condition-start-frame-index",
        dest="start_frame",
        type=int,
        default=DEFAULT_START_FRAME,
        help="Episode-local first conditioned frame.",
    )
    parser.add_argument("--video-key", default=DEFAULT_VIDEO_KEY)
    parser.add_argument(
        "--reference-layout",
        choices=("dense", "first_last"),
        default=DEFAULT_REFERENCE_LAYOUT,
        help="Use 5 consecutive condition frames or only the first/last sparse pair.",
    )
    return parser.parse_args()


def _parse_resolution(spec: str) -> tuple[int, int]:
    """Parse one WIDTHxHEIGHT resolution string."""
    normalized = spec.lower().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WIDTHxHEIGHT, got {spec!r}")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {spec!r}")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(
            f"Resolution must be divisible by 16 for Wan VACE, got {width}x{height}."
        )
    return width, height


def _load_reference_images(
    *,
    repo_id: str,
    episode_index: int,
    start_frame: int,
    video_key: str,
    reference_layout: str,
) -> tuple[list[Image.Image], list[int] | None]:
    """Load ALOHA fork conditioning frames for the upstream base pipeline."""
    if reference_layout == "dense":
        images = base.load_lerobot_images(
            repo_id=repo_id,
            episode_index=episode_index,
            frame_offset=0,
            condition_start_frame_index=start_frame,
            video_key=video_key,
            num_frames=base.NUM_CONDITION_FRAMES,
        )
        return images, None

    images = base.load_lerobot_images_at_offsets(
        repo_id=repo_id,
        episode_index=episode_index,
        frame_offset=0,
        condition_start_frame_index=start_frame,
        video_key=video_key,
        frame_offsets=[0, base.NUM_TOTAL_FRAMES - 1],
    )
    return images, [0, base.NUM_TOTAL_FRAMES - 1]


def _select_effective_condition_images(
    *,
    condition_images: list[Image.Image],
    condition_indices: list[int] | None,
) -> tuple[list[Image.Image], list[int], bool]:
    """Select the actual conditioning frames used by the sweep at video resolution."""
    if not condition_images:
        raise ValueError("condition_images must contain at least one frame.")

    use_dense_prefix = condition_indices is None
    if use_dense_prefix:
        effective_indices = list(range(min(len(condition_images), base.NUM_TOTAL_FRAMES)))
        effective_images = condition_images[: len(effective_indices)]
    else:
        effective_indices = list(condition_indices)
        effective_images = condition_images
    return effective_images, effective_indices, use_dense_prefix


def _resize_condition_image(*, image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize one conditioning frame with explicit resampling and a no-op fast path."""
    target_size = (width, height)
    if image.mode == "RGB" and image.size == target_size:
        return image

    rgb_image = image.convert("RGB")
    if rgb_image.size == target_size:
        return rgb_image

    source_width, source_height = rgb_image.size
    downscaling = width < source_width or height < source_height
    resample = Image.Resampling.LANCZOS if downscaling else Image.Resampling.BICUBIC
    return rgb_image.resize(target_size, resample=resample)


def _save_ordered_png_frames(*, images: list[object], output_dir: Path) -> list[Path]:
    """Persist ordered PNG frames for direct visual inspection without video compression."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("frame_*.png"):
        stale_path.unlink()

    saved_paths: list[Path] = []
    for index, image in enumerate(images):
        frame_path = output_dir / f"frame_{index:04d}.png"
        base._to_pil_rgb(image).save(frame_path)
        saved_paths.append(frame_path)
    return saved_paths


def _build_video_and_mask_for_resolution(
    *,
    condition_images: list[Image.Image],
    condition_indices: list[int] | None,
    width: int,
    height: int,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Build a resolution-specific conditioning video and generation mask."""
    effective_images, effective_indices, use_dense_prefix = _select_effective_condition_images(
        condition_images=condition_images,
        condition_indices=condition_indices,
    )
    resized_frames = [
        _resize_condition_image(image=image, width=width, height=height)
        for image in effective_images
    ]
    placeholder = Image.new("RGB", (width, height), (128, 128, 128))
    keep_frame = Image.new("L", (width, height), 0)
    generate_frame = Image.new("L", (width, height), 255)

    video = [placeholder.copy() for _ in range(base.NUM_TOTAL_FRAMES)]
    mask = [generate_frame.copy() for _ in range(base.NUM_TOTAL_FRAMES)]
    if use_dense_prefix:
        frame_ranges = [range(index, index + 1) for index in range(len(resized_frames))]
    else:
        frame_ranges = base._compute_sparse_condition_ranges(
            condition_indices=effective_indices,
            num_frames=base.NUM_TOTAL_FRAMES,
        )

    for conditioned_frame, frame_range in zip(resized_frames, frame_ranges):
        for frame_index in frame_range:
            video[frame_index] = conditioned_frame
            mask[frame_index] = keep_frame.copy()
    return video, mask


def _load_base_pipeline() -> DiffusionPipeline:
    """Load the public Wan VACE pipeline once for the whole sweep."""
    pipe = DiffusionPipeline.from_pretrained(base.MODEL_ID, dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    return pipe


def _run_one_base_resolution(
    *,
    pipe: DiffusionPipeline,
    width: int,
    height: int,
    condition_images: list[Image.Image],
    condition_indices: list[int] | None,
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    fps: int,
    seed: int,
    prompt: str,
    negative_prompt: str,
) -> dict[str, object]:
    """Run one public Wan VACE generation at a specific resolution."""
    label = f"{width}x{height}"
    output_path = output_dir / f"{label}.mp4"
    resolution_dir = output_dir / label
    condition_frames_dir = resolution_dir / "condition_frames"
    generated_frames_dir = resolution_dir / "generated_frames"
    start_time = time.time()
    try:
        effective_images, _, _ = _select_effective_condition_images(
            condition_images=condition_images,
            condition_indices=condition_indices,
        )
        resized_condition_images = [
            _resize_condition_image(image=image, width=width, height=height)
            for image in effective_images
        ]
        video, mask = _build_video_and_mask_for_resolution(
            condition_images=condition_images,
            condition_indices=condition_indices,
            width=width,
            height=height,
        )
        torch.manual_seed(seed)
        frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video=video,
            mask=mask,
            height=height,
            width=width,
            num_frames=base.NUM_TOTAL_FRAMES,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        base._export_video(video_frames=list(frames), output_video_path=str(output_path), fps=fps)
        saved_condition_paths = _save_ordered_png_frames(
            images=resized_condition_images,
            output_dir=condition_frames_dir,
        )
        saved_generated_paths = _save_ordered_png_frames(
            images=list(frames),
            output_dir=generated_frames_dir,
        )
        return {
            "resolution": label,
            "status": "ok",
            "output_path": str(output_path),
            "condition_frames_dir": str(condition_frames_dir),
            "generated_frames_dir": str(generated_frames_dir),
            "num_condition_frames_saved": len(saved_condition_paths),
            "num_generated_frames_saved": len(saved_generated_paths),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "elapsed_s": time.time() - start_time,
        }
    except Exception as exc:  # pragma: no cover - manual smoke script
        return {
            "resolution": label,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.time() - start_time,
        }


def _save_summary(*, output_dir: Path, results: list[dict[str, object]]) -> Path:
    """Persist the sweep results as JSON for quick review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    """Run the upstream base resolution sweep."""
    args = _parse_args()
    parsed_resolutions = [_parse_resolution(spec) for spec in args.resolutions]
    condition_images, condition_indices = _load_reference_images(
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        video_key=args.video_key,
        reference_layout=args.reference_layout,
    )
    pipe = _load_base_pipeline()

    results: list[dict[str, object]] = []
    for width, height in parsed_resolutions:
        label = f"{width}x{height}"
        print(f"Running upstream base Wan VACE at {label}...")
        result = _run_one_base_resolution(
            pipe=pipe,
            width=width,
            height=height,
            condition_images=condition_images,
            condition_indices=condition_indices,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            fps=args.fps,
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
        )
        results.append(result)
        if result["status"] == "ok":
            print(f"{label}: saved {result['output_path']}")
        else:
            print(f"{label}: {result['error']}")

    summary_path = _save_summary(output_dir=args.output_dir, results=results)
    print(f"Saved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
