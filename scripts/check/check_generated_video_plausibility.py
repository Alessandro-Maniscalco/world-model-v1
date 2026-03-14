"""Heuristically compare a generated video against a reference clip.

Flags frame-level failures such as near-black output, severe color drift, and
posterized low-palette artifacts, then saves a JSON report.

This is a PASS:
python scripts/check/check_generated_video_plausibility.py \
  --reference-video runs/check_aloha_fork_preview_start30/preview.mp4 \
  --generated-video runs/check_wan_vace_base_resolution_sweep/320x240.mp4 \
  --resize-reference 
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ThresholdConfig:
    """Thresholds used to classify a generated frame as plausible or suspicious."""

    max_frame_mae: float
    max_mean_mae: float
    black_luma_threshold: float
    max_black_fraction: float
    max_colorfulness_ratio: float
    max_top2_color_share: float
    min_quantized_colors: int
    min_edge_ratio: float
    max_temporal_delta_ratio: float
    quantization_bin_size: int


@dataclass(frozen=True)
class FrameMetrics:
    """Per-frame comparison metrics and anomaly flags."""

    frame_index: int
    mae_rgb_0_255: float
    black_fraction: float
    reference_black_fraction: float
    colorfulness: float
    reference_colorfulness: float
    colorfulness_ratio: float
    edge_energy: float
    reference_edge_energy: float
    edge_ratio: float
    quantized_unique_colors: int
    top1_color_share: float
    top2_color_share: float
    flags: tuple[str, ...]
    plausible: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for video plausibility checking."""
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
        "--max-frame-mae",
        type=float,
        default=45.0,
        help="Fail a frame when mean absolute RGB error exceeds this value in [0,255].",
    )
    parser.add_argument(
        "--max-mean-mae",
        type=float,
        default=30.0,
        help="Fail the full video when mean frame MAE exceeds this value in [0,255].",
    )
    parser.add_argument(
        "--black-luma-threshold",
        type=float,
        default=12.0,
        help="Luma threshold in [0,255] used to count near-black pixels.",
    )
    parser.add_argument(
        "--max-black-fraction",
        type=float,
        default=0.75,
        help="Fail a frame when more than this fraction of pixels are near-black.",
    )
    parser.add_argument(
        "--max-colorfulness-ratio",
        type=float,
        default=2.5,
        help="Fail a frame when generated/reference colorfulness exceeds this ratio.",
    )
    parser.add_argument(
        "--max-top2-color-share",
        type=float,
        default=0.85,
        help="Fail a frame when the two dominant quantized colors cover too much of the image.",
    )
    parser.add_argument(
        "--min-quantized-colors",
        type=int,
        default=8,
        help="Fail a frame when quantized palette size falls below this count.",
    )
    parser.add_argument(
        "--min-edge-ratio",
        type=float,
        default=0.20,
        help="Fail a frame when generated/reference edge energy falls below this ratio.",
    )
    parser.add_argument(
        "--max-temporal-delta-ratio",
        type=float,
        default=4.0,
        help="Fail the video when generated flicker greatly exceeds reference flicker.",
    )
    parser.add_argument(
        "--quantization-bin-size",
        type=int,
        default=32,
        help="RGB bin size used for palette statistics; smaller values are more sensitive.",
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
    """Resize a `THWC` RGB video to a fixed spatial size using PIL bilinear sampling."""
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


def compute_luma(frame: np.ndarray) -> np.ndarray:
    """Compute a simple Rec.709-style luma image in [0,255]."""
    frame_f = frame.astype(np.float32, copy=False)
    return (0.2126 * frame_f[..., 0]) + (0.7152 * frame_f[..., 1]) + (0.0722 * frame_f[..., 2])


def compute_mae(reference_frame: np.ndarray, generated_frame: np.ndarray) -> float:
    """Compute mean absolute RGB error in the native [0,255] scale."""
    diff = reference_frame.astype(np.float32) - generated_frame.astype(np.float32)
    return float(np.abs(diff).mean())


def compute_black_fraction(frame: np.ndarray, *, luma_threshold: float) -> float:
    """Measure the share of pixels whose luma falls below a near-black threshold."""
    luma = compute_luma(frame)
    return float((luma <= luma_threshold).mean())


def compute_edge_energy(frame: np.ndarray) -> float:
    """Estimate spatial detail using mean absolute horizontal and vertical luma gradients."""
    luma = compute_luma(frame)
    grad_y = np.abs(luma[1:, :] - luma[:-1, :]).mean() if luma.shape[0] > 1 else 0.0
    grad_x = np.abs(luma[:, 1:] - luma[:, :-1]).mean() if luma.shape[1] > 1 else 0.0
    return float((grad_x + grad_y) * 0.5 / 255.0)


def compute_colorfulness(frame: np.ndarray) -> float:
    """Compute the Hasler-Suesstrunk colorfulness score for one RGB frame."""
    frame_f = frame.astype(np.float32, copy=False)
    rg = frame_f[..., 0] - frame_f[..., 1]
    yb = 0.5 * (frame_f[..., 0] + frame_f[..., 1]) - frame_f[..., 2]
    std_root = np.sqrt(np.var(rg) + np.var(yb))
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return float(std_root + 0.3 * mean_root)


def compute_palette_stats(frame: np.ndarray, *, bin_size: int) -> tuple[int, float, float]:
    """Compute quantized palette size and dominant-color shares for one RGB frame."""
    if bin_size <= 0 or bin_size > 256:
        raise ValueError(f"bin_size must be in [1,256], got {bin_size}")

    quantized = (frame.astype(np.int16) // bin_size).astype(np.int16)
    flat = quantized.reshape(-1, 3)
    _, counts = np.unique(flat, axis=0, return_counts=True)
    counts_sorted = np.sort(counts)[::-1]
    total = int(flat.shape[0])
    top1_share = float(counts_sorted[0] / total)
    top2_share = float(counts_sorted[:2].sum() / total) if counts_sorted.size > 1 else top1_share
    return int(counts_sorted.size), top1_share, top2_share


def analyze_frame(
    *,
    frame_index: int,
    reference_frame: np.ndarray,
    generated_frame: np.ndarray,
    thresholds: ThresholdConfig,
) -> FrameMetrics:
    """Compute frame-level metrics and classify the frame with heuristic flags."""
    mae = compute_mae(reference_frame, generated_frame)
    black_fraction = compute_black_fraction(generated_frame, luma_threshold=thresholds.black_luma_threshold)
    ref_black_fraction = compute_black_fraction(reference_frame, luma_threshold=thresholds.black_luma_threshold)
    colorfulness = compute_colorfulness(generated_frame)
    ref_colorfulness = compute_colorfulness(reference_frame)
    colorfulness_ratio = float(colorfulness / max(ref_colorfulness, 1e-6))
    edge_energy = compute_edge_energy(generated_frame)
    ref_edge_energy = compute_edge_energy(reference_frame)
    edge_ratio = float(edge_energy / max(ref_edge_energy, 1e-6))
    unique_colors, top1_share, top2_share = compute_palette_stats(
        generated_frame,
        bin_size=thresholds.quantization_bin_size,
    )

    flags: list[str] = []
    if mae > thresholds.max_frame_mae:
        flags.append("high_reference_diff")
    if black_fraction > thresholds.max_black_fraction and black_fraction > (ref_black_fraction + 0.25):
        flags.append("mostly_black")
    if colorfulness_ratio > thresholds.max_colorfulness_ratio and mae > (0.5 * thresholds.max_frame_mae):
        flags.append("extreme_color_shift")
    if unique_colors < thresholds.min_quantized_colors and top2_share > thresholds.max_top2_color_share:
        flags.append("low_palette_posterization")
    if edge_ratio < thresholds.min_edge_ratio and mae > (0.5 * thresholds.max_frame_mae):
        flags.append("too_flat")

    return FrameMetrics(
        frame_index=frame_index,
        mae_rgb_0_255=mae,
        black_fraction=black_fraction,
        reference_black_fraction=ref_black_fraction,
        colorfulness=colorfulness,
        reference_colorfulness=ref_colorfulness,
        colorfulness_ratio=colorfulness_ratio,
        edge_energy=edge_energy,
        reference_edge_energy=ref_edge_energy,
        edge_ratio=edge_ratio,
        quantized_unique_colors=unique_colors,
        top1_color_share=top1_share,
        top2_color_share=top2_share,
        flags=tuple(flags),
        plausible=(len(flags) == 0),
    )


def compute_temporal_mae(video: np.ndarray) -> float:
    """Measure mean absolute RGB delta between consecutive frames in [0,255]."""
    if video.shape[0] < 2:
        return 0.0
    deltas = np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32))
    return float(deltas.mean())


def build_summary(
    *,
    reference_video: np.ndarray,
    generated_video: np.ndarray,
    frame_metrics: list[FrameMetrics],
    thresholds: ThresholdConfig,
) -> dict[str, object]:
    """Aggregate frame metrics into a video-level verdict and failure summary."""
    mean_frame_mae = float(np.mean([metric.mae_rgb_0_255 for metric in frame_metrics]))
    max_frame_mae = float(np.max([metric.mae_rgb_0_255 for metric in frame_metrics]))
    failing_frames = [metric for metric in frame_metrics if not metric.plausible]

    reference_temporal_mae = compute_temporal_mae(reference_video)
    generated_temporal_mae = compute_temporal_mae(generated_video)
    temporal_delta_ratio = float(generated_temporal_mae / max(reference_temporal_mae, 1e-6))

    video_flags: list[str] = []
    if mean_frame_mae > thresholds.max_mean_mae:
        video_flags.append("high_mean_reference_diff")
    if temporal_delta_ratio > thresholds.max_temporal_delta_ratio:
        video_flags.append("temporal_instability")
    if len(failing_frames) > max(1, len(frame_metrics) // 3):
        video_flags.append("too_many_bad_frames")

    return {
        "plausible": len(video_flags) == 0 and len(failing_frames) == 0,
        "num_frames_compared": len(frame_metrics),
        "num_failing_frames": len(failing_frames),
        "mean_frame_mae_rgb_0_255": mean_frame_mae,
        "max_frame_mae_rgb_0_255": max_frame_mae,
        "reference_temporal_mae_rgb_0_255": reference_temporal_mae,
        "generated_temporal_mae_rgb_0_255": generated_temporal_mae,
        "temporal_delta_ratio": temporal_delta_ratio,
        "video_flags": video_flags,
        "failing_frame_indices": [metric.frame_index for metric in failing_frames],
    }


def default_output_json(generated_video: Path) -> Path:
    """Choose the default JSON report path for a generated video."""
    return generated_video.with_name(f"{generated_video.stem}_plausibility_report.json")


def save_report(*, output_json: Path, report: dict[str, object]) -> None:
    """Write a JSON plausibility report to disk."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(*, summary: dict[str, object], output_json: Path) -> None:
    """Print a concise terminal summary for the saved plausibility report."""
    verdict = "PASS" if summary["plausible"] else "FAIL"
    print(f"Verdict: {verdict}")
    print(f"Frames compared: {summary['num_frames_compared']}")
    print(f"Failing frames: {summary['num_failing_frames']}")
    print(f"Mean frame MAE [0,255]: {summary['mean_frame_mae_rgb_0_255']:.3f}")
    print(f"Max frame MAE [0,255]: {summary['max_frame_mae_rgb_0_255']:.3f}")
    print(f"Temporal delta ratio: {summary['temporal_delta_ratio']:.3f}")
    if summary["video_flags"]:
        print(f"Video flags: {', '.join(summary['video_flags'])}")
    if summary["failing_frame_indices"]:
        print(f"Failing frame indices: {summary['failing_frame_indices']}")
    print(f"Saved report: {output_json}")


def main() -> None:
    """Load two local videos, compare them frame by frame, and save a plausibility report."""
    args = parse_args()
    thresholds = ThresholdConfig(
        max_frame_mae=args.max_frame_mae,
        max_mean_mae=args.max_mean_mae,
        black_luma_threshold=args.black_luma_threshold,
        max_black_fraction=args.max_black_fraction,
        max_colorfulness_ratio=args.max_colorfulness_ratio,
        max_top2_color_share=args.max_top2_color_share,
        min_quantized_colors=args.min_quantized_colors,
        min_edge_ratio=args.min_edge_ratio,
        max_temporal_delta_ratio=args.max_temporal_delta_ratio,
        quantization_bin_size=args.quantization_bin_size,
    )

    reference_video = load_video_rgb(args.reference_video)
    generated_video = load_video_rgb(args.generated_video)
    reference_aligned, generated_aligned = align_videos(
        reference_video=reference_video,
        generated_video=generated_video,
        resize_reference=args.resize_reference,
    )

    frame_metrics = [
        analyze_frame(
            frame_index=frame_index,
            reference_frame=reference_frame,
            generated_frame=generated_frame,
            thresholds=thresholds,
        )
        for frame_index, (reference_frame, generated_frame) in enumerate(zip(reference_aligned, generated_aligned))
    ]
    summary = build_summary(
        reference_video=reference_aligned,
        generated_video=generated_aligned,
        frame_metrics=frame_metrics,
        thresholds=thresholds,
    )

    output_json = args.output_json if args.output_json is not None else default_output_json(args.generated_video)
    report = {
        "reference_video": str(args.reference_video),
        "generated_video": str(args.generated_video),
        "reference_shape": list(reference_aligned.shape),
        "generated_shape": list(generated_aligned.shape),
        "thresholds": asdict(thresholds),
        "summary": summary,
        "frames": [asdict(metric) for metric in frame_metrics],
    }
    save_report(output_json=output_json, report=report)
    print_summary(summary=summary, output_json=output_json)


if __name__ == "__main__":
    main()
