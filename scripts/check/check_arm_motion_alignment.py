"""Score generated-video motion against a reference clip with motion-first heuristics.

This script is meant to complement `check_generated_video_plausibility.py`.
The plausibility checker remains a coarse safety gate for obviously broken
videos, while this script focuses on whether the generated rollout moves in the
same region and with similar temporal commitment as the reference.

The checker derives a motion ROI directly from the reference clip, so it works
best for static-camera robot tasks where the moving manipulator occupies a
localized region of the frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class MotionConfig:
    """Configuration values for the motion-focused comparison."""

    roi_percentile: float
    min_motion_value: float
    roi_padding: int
    profile_floor: float
    motion_ratio_floor: float
    late_motion_ratio_floor: float
    profile_corr_floor: float
    spatial_iou_floor: float


@dataclass(frozen=True)
class MotionSummary:
    """Summary metrics describing reference and generated motion alignment."""

    roi_xyxy: tuple[int, int, int, int]
    roi_area_fraction: float
    reference_total_motion: float
    generated_total_motion: float
    total_motion_ratio: float
    reference_late_motion: float
    generated_late_motion: float
    late_motion_ratio: float
    peak_motion_ratio: float
    profile_correlation: float
    normalized_profile_l1: float
    spatial_motion_iou: float
    motion_verdict: str
    motion_flags: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for motion-focused video comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-video", type=Path, required=True, help="Reference target video path.")
    parser.add_argument("--generated-video", type=Path, required=True, help="Generated video path to evaluate.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults next to the generated video.",
    )
    parser.add_argument(
        "--resize-reference",
        action="store_true",
        help="Resize the reference clip to the generated resolution before comparison.",
    )
    parser.add_argument(
        "--roi-percentile",
        type=float,
        default=92.0,
        help="Reference-motion percentile used to derive the motion ROI.",
    )
    parser.add_argument(
        "--min-motion-value",
        type=float,
        default=6.0,
        help="Minimum per-pixel motion-map value in [0,255] before a pixel can enter the ROI.",
    )
    parser.add_argument(
        "--roi-padding",
        type=int,
        default=24,
        help="Extra pixels of padding applied around the inferred motion ROI.",
    )
    parser.add_argument(
        "--profile-floor",
        type=float,
        default=1.0,
        help="Small positive floor used when normalizing low-motion temporal profiles.",
    )
    parser.add_argument(
        "--motion-ratio-floor",
        type=float,
        default=0.70,
        help="Flag under-commitment when total generated/reference motion falls below this ratio.",
    )
    parser.add_argument(
        "--late-motion-ratio-floor",
        type=float,
        default=0.70,
        help="Flag early stopping when late generated/reference motion falls below this ratio.",
    )
    parser.add_argument(
        "--profile-corr-floor",
        type=float,
        default=0.70,
        help="Flag temporal mismatch when motion-profile correlation falls below this value.",
    )
    parser.add_argument(
        "--spatial-iou-floor",
        type=float,
        default=0.25,
        help="Flag spatial mismatch when reference/generated motion masks overlap less than this IoU.",
    )
    return parser.parse_args()


def load_video_rgb(path: Path) -> np.ndarray:
    """Load a local image or video into `THWC` uint8 RGB form."""
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    video = iio.imread(path)
    if video.ndim == 3:
        video = video[None, ...]
    if video.ndim != 4:
        raise ValueError(f"Expected THWC video data, got shape {tuple(video.shape)} from {path}")
    if video.shape[-1] == 4:
        video = video[..., :3]
    if video.shape[-1] != 3:
        raise ValueError(f"Expected RGB video with 3 channels, got shape {tuple(video.shape)} from {path}")

    if np.issubdtype(video.dtype, np.floating):
        max_value = float(video.max()) if video.size else 0.0
        if max_value <= 1.0:
            video = (video * 255.0).round()
        video = np.clip(video, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        video = np.clip(video, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(video)


def resize_video(video: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize a `THWC` RGB video to a fixed spatial size using bilinear sampling."""
    resized_frames: list[np.ndarray] = []
    for frame in video:
        image = Image.fromarray(frame, mode="RGB")
        resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        resized_frames.append(np.asarray(resized, dtype=np.uint8))
    return np.stack(resized_frames, axis=0)


def align_videos(
    *,
    reference_video: np.ndarray,
    generated_video: np.ndarray,
    resize_reference: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim videos to a shared frame count and optionally resize the reference clip."""
    if reference_video.shape[0] == 0 or generated_video.shape[0] == 0:
        raise ValueError("Both videos must contain at least one frame.")

    target_frames = min(int(reference_video.shape[0]), int(generated_video.shape[0]))
    reference_trimmed = reference_video[:target_frames]
    generated_trimmed = generated_video[:target_frames]

    ref_height, ref_width = reference_trimmed.shape[1:3]
    gen_height, gen_width = generated_trimmed.shape[1:3]
    if (ref_height, ref_width) != (gen_height, gen_width):
        if not resize_reference:
            raise ValueError(
                "Reference and generated videos have different sizes "
                f"{(ref_width, ref_height)} vs {(gen_width, gen_height)}. "
                "Pass --resize-reference to compare them."
            )
        reference_trimmed = resize_video(reference_trimmed, width=gen_width, height=gen_height)

    return reference_trimmed, generated_trimmed


def to_luma(video: np.ndarray) -> np.ndarray:
    """Convert `THWC` RGB video to a float32 luma video in [0,255]."""
    video_f = video.astype(np.float32, copy=False)
    return (
        0.2126 * video_f[..., 0]
        + 0.7152 * video_f[..., 1]
        + 0.0722 * video_f[..., 2]
    )


def compute_motion_volume(video: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame absolute luma change as a `T-1,H,W` motion volume."""
    luma = to_luma(video)
    if luma.shape[0] < 2:
        return np.zeros((0, luma.shape[1], luma.shape[2]), dtype=np.float32)
    return np.abs(luma[1:] - luma[:-1])


def infer_motion_roi(
    *,
    reference_motion: np.ndarray,
    percentile: float,
    min_motion_value: float,
    padding: int,
) -> tuple[int, int, int, int]:
    """Infer a bounding box around the most active region of the reference motion."""
    if reference_motion.size == 0:
        raise ValueError("Reference motion volume must contain at least one transition.")

    heat = reference_motion.mean(axis=0)
    mask = np.zeros_like(heat, dtype=bool)
    for current_percentile in (percentile, 97.0, 98.0, 99.0, 99.5):
        threshold = max(float(np.percentile(heat, current_percentile)), float(min_motion_value))
        candidate = heat >= threshold
        if bool(candidate.any()):
            mask = largest_connected_component(candidate)
            if bool(mask.any()) and float(mask.mean()) <= 0.35:
                break
    if not bool(mask.any()):
        height, width = heat.shape
        return (0, 0, width, height)

    ys, xs = np.nonzero(mask)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + 1 + padding, heat.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + 1 + padding, heat.shape[1])
    return (x0, y0, x1, y1)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected component from a boolean mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {tuple(mask.shape)}")
    if not bool(mask.any()):
        return mask

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if len(component) > len(best_component):
                best_component = component

    output = np.zeros_like(mask, dtype=bool)
    for y, x in best_component:
        output[y, x] = True
    return output


def crop_video(video: np.ndarray, *, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a `THWC` video to the provided ROI."""
    x0, y0, x1, y1 = roi_xyxy
    return video[:, y0:y1, x0:x1, :]


def compute_profile(motion_volume: np.ndarray) -> np.ndarray:
    """Reduce a `T-1,H,W` motion volume to a per-transition mean-motion profile."""
    if motion_volume.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return motion_volume.mean(axis=(1, 2))


def normalize_profile(profile: np.ndarray, *, floor: float) -> np.ndarray:
    """Normalize a nonnegative motion profile into a unit-sum distribution."""
    if profile.size == 0:
        return profile.astype(np.float32, copy=False)
    total = float(profile.sum())
    if total <= floor:
        return np.full_like(profile, fill_value=1.0 / max(int(profile.size), 1), dtype=np.float32)
    return (profile / total).astype(np.float32, copy=False)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a stable ratio for nonnegative quantities."""
    return float(numerator / max(denominator, 1e-6))


def profile_correlation(reference_profile: np.ndarray, generated_profile: np.ndarray) -> float:
    """Compute Pearson correlation between two temporal motion profiles."""
    if reference_profile.size == 0 or generated_profile.size == 0:
        return 1.0
    if reference_profile.size != generated_profile.size:
        raise ValueError("Profiles must have matching length.")
    ref = reference_profile.astype(np.float32, copy=False)
    gen = generated_profile.astype(np.float32, copy=False)
    ref_centered = ref - ref.mean()
    gen_centered = gen - gen.mean()
    denom = float(np.sqrt((ref_centered**2).sum()) * np.sqrt((gen_centered**2).sum()))
    if denom <= 1e-6:
        return 1.0
    return float(np.clip((ref_centered * gen_centered).sum() / denom, -1.0, 1.0))


def spatial_motion_iou(reference_motion: np.ndarray, generated_motion: np.ndarray) -> float:
    """Measure overlap between the strongest reference and generated motion pixels."""
    ref_heat = reference_motion.mean(axis=0)
    gen_heat = generated_motion.mean(axis=0)
    ref_mask = ref_heat >= max(float(np.percentile(ref_heat, 80.0)), 1.0)
    gen_mask = gen_heat >= max(float(np.percentile(gen_heat, 80.0)), 1.0)
    union = ref_mask | gen_mask
    if not bool(union.any()):
        return 1.0
    return float((ref_mask & gen_mask).sum() / max(int(union.sum()), 1))


def build_motion_summary(
    *,
    reference_video: np.ndarray,
    generated_video: np.ndarray,
    config: MotionConfig,
) -> MotionSummary:
    """Build motion-first metrics and a qualitative verdict for one video pair."""
    reference_motion = compute_motion_volume(reference_video)
    generated_motion = compute_motion_volume(generated_video)
    roi_xyxy = infer_motion_roi(
        reference_motion=reference_motion,
        percentile=config.roi_percentile,
        min_motion_value=config.min_motion_value,
        padding=config.roi_padding,
    )
    reference_roi_motion = compute_motion_volume(crop_video(reference_video, roi_xyxy=roi_xyxy))
    generated_roi_motion = compute_motion_volume(crop_video(generated_video, roi_xyxy=roi_xyxy))

    ref_profile = compute_profile(reference_roi_motion)
    gen_profile = compute_profile(generated_roi_motion)
    ref_total = float(ref_profile.sum())
    gen_total = float(gen_profile.sum())
    split = max(int(np.ceil(ref_profile.size * (2.0 / 3.0))), 0)
    ref_late = float(ref_profile[split:].sum()) if ref_profile.size else 0.0
    gen_late = float(gen_profile[split:].sum()) if gen_profile.size else 0.0
    total_ratio = safe_ratio(gen_total, ref_total)
    late_ratio = safe_ratio(gen_late, ref_late)
    peak_ratio = safe_ratio(float(gen_profile.max(initial=0.0)), float(ref_profile.max(initial=0.0)))
    ref_profile_norm = normalize_profile(ref_profile, floor=config.profile_floor)
    gen_profile_norm = normalize_profile(gen_profile, floor=config.profile_floor)
    profile_l1 = float(np.abs(ref_profile_norm - gen_profile_norm).mean()) if ref_profile.size else 0.0
    corr = profile_correlation(ref_profile, gen_profile)
    spatial_iou = spatial_motion_iou(reference_roi_motion, generated_roi_motion)

    height = int(reference_video.shape[1])
    width = int(reference_video.shape[2])
    x0, y0, x1, y1 = roi_xyxy
    roi_area_fraction = float(((x1 - x0) * (y1 - y0)) / max(height * width, 1))

    flags: list[str] = []
    if total_ratio < config.motion_ratio_floor:
        flags.append("undercommitted_motion")
    if late_ratio < config.late_motion_ratio_floor:
        flags.append("stops_early")
    if corr < config.profile_corr_floor:
        flags.append("temporal_profile_mismatch")
    if spatial_iou < config.spatial_iou_floor:
        flags.append("motion_region_mismatch")
    if peak_ratio > 1.6:
        flags.append("overactive_motion")

    if "stops_early" in flags or "undercommitted_motion" in flags:
        verdict = "undercommitted"
    elif "temporal_profile_mismatch" in flags or "motion_region_mismatch" in flags:
        verdict = "misaligned"
    elif "overactive_motion" in flags:
        verdict = "overactive"
    else:
        verdict = "good"

    return MotionSummary(
        roi_xyxy=roi_xyxy,
        roi_area_fraction=roi_area_fraction,
        reference_total_motion=ref_total,
        generated_total_motion=gen_total,
        total_motion_ratio=total_ratio,
        reference_late_motion=ref_late,
        generated_late_motion=gen_late,
        late_motion_ratio=late_ratio,
        peak_motion_ratio=peak_ratio,
        profile_correlation=corr,
        normalized_profile_l1=profile_l1,
        spatial_motion_iou=spatial_iou,
        motion_verdict=verdict,
        motion_flags=tuple(flags),
    )


def default_output_json(generated_video: Path) -> Path:
    """Choose the default JSON report path for a generated video."""
    return generated_video.with_name(f"{generated_video.stem}_arm_motion_report.json")


def motion_crop_video_path(generated_video: Path) -> Path:
    """Choose the output path for the cropped comparison video."""
    return generated_video.with_name(f"{generated_video.stem}_arm_crop_comparison.mp4")


def roi_preview_path(generated_video: Path) -> Path:
    """Choose the output path for the ROI preview PNG."""
    return generated_video.with_name(f"{generated_video.stem}_arm_roi_preview.png")


def save_report(*, output_json: Path, report: dict[str, object]) -> None:
    """Write a JSON motion report to disk."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def draw_roi_preview(
    *,
    reference_video: np.ndarray,
    generated_video: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    """Save a side-by-side first-frame preview with the inferred ROI outlined."""
    x0, y0, x1, y1 = roi_xyxy
    ref = Image.fromarray(reference_video[0], mode="RGB")
    gen = Image.fromarray(generated_video[0], mode="RGB")
    for image in (ref, gen):
        draw = ImageDraw.Draw(image)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 64, 64), width=3)
    canvas = Image.new("RGB", (ref.width + gen.width, ref.height), color=(255, 255, 255))
    canvas.paste(ref, (0, 0))
    canvas.paste(gen, (ref.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_motion_crop_comparison(
    *,
    reference_video: np.ndarray,
    generated_video: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    """Save a side-by-side cropped comparison video around the motion ROI."""
    ref_crop = crop_video(reference_video, roi_xyxy=roi_xyxy)
    gen_crop = crop_video(generated_video, roi_xyxy=roi_xyxy)
    side_by_side = np.concatenate([ref_crop, gen_crop], axis=2)
    height = int(side_by_side.shape[1])
    width = int(side_by_side.shape[2])
    if height % 2 != 0:
        side_by_side = np.pad(side_by_side, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="edge")
    if width % 2 != 0:
        side_by_side = np.pad(side_by_side, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="edge")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, side_by_side, fps=10, macro_block_size=1)


def print_summary(*, summary: MotionSummary, output_json: Path, crop_path: Path) -> None:
    """Print a concise terminal summary for the saved motion report."""
    print(f"Motion verdict: {summary.motion_verdict}")
    print(f"Motion flags: {', '.join(summary.motion_flags) if summary.motion_flags else '(none)'}")
    print(
        "Motion ratios: "
        f"total={summary.total_motion_ratio:.3f} "
        f"late={summary.late_motion_ratio:.3f} "
        f"peak={summary.peak_motion_ratio:.3f}"
    )
    print(
        "Temporal alignment: "
        f"corr={summary.profile_correlation:.3f} "
        f"profile_l1={summary.normalized_profile_l1:.3f}"
    )
    print(f"Spatial motion IoU: {summary.spatial_motion_iou:.3f}")
    print(f"ROI: {summary.roi_xyxy} area_fraction={summary.roi_area_fraction:.3f}")
    print(f"Saved report: {output_json}")
    print(f"Saved arm-crop comparison: {crop_path}")


def main() -> None:
    """Compare generated arm motion against a reference clip and save a report."""
    args = parse_args()
    config = MotionConfig(
        roi_percentile=args.roi_percentile,
        min_motion_value=args.min_motion_value,
        roi_padding=args.roi_padding,
        profile_floor=args.profile_floor,
        motion_ratio_floor=args.motion_ratio_floor,
        late_motion_ratio_floor=args.late_motion_ratio_floor,
        profile_corr_floor=args.profile_corr_floor,
        spatial_iou_floor=args.spatial_iou_floor,
    )

    reference_video = load_video_rgb(args.reference_video)
    generated_video = load_video_rgb(args.generated_video)
    reference_aligned, generated_aligned = align_videos(
        reference_video=reference_video,
        generated_video=generated_video,
        resize_reference=args.resize_reference,
    )
    summary = build_motion_summary(
        reference_video=reference_aligned,
        generated_video=generated_aligned,
        config=config,
    )

    output_json = args.output_json if args.output_json is not None else default_output_json(args.generated_video)
    crop_path = motion_crop_video_path(args.generated_video)
    preview_path = roi_preview_path(args.generated_video)
    draw_roi_preview(
        reference_video=reference_aligned,
        generated_video=generated_aligned,
        roi_xyxy=summary.roi_xyxy,
        output_path=preview_path,
    )
    save_motion_crop_comparison(
        reference_video=reference_aligned,
        generated_video=generated_aligned,
        roi_xyxy=summary.roi_xyxy,
        output_path=crop_path,
    )

    report = {
        "reference_video": str(args.reference_video),
        "generated_video": str(args.generated_video),
        "reference_shape": list(reference_aligned.shape),
        "generated_shape": list(generated_aligned.shape),
        "config": asdict(config),
        "summary": asdict(summary),
        "artifacts": {
            "arm_crop_comparison_video": str(crop_path),
            "roi_preview_image": str(preview_path),
        },
    }
    save_report(output_json=output_json, report=report)
    print_summary(summary=summary, output_json=output_json, crop_path=crop_path)


if __name__ == "__main__":
    main()
