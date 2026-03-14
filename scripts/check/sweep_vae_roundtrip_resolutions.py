"""Run a pure Wan VAE roundtrip sweep across multiple resize settings.

This smoke-check helps isolate resize/VAE blur from world-model generation
quality by saving raw-vs-roundtrip visualizations for each resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_model.latents import WanVAE
from world_model.data.prepare import load_local_video_clip, preprocess_video_for_vae
from world_model.data.temporal import latent_split_from_frame_ratio
from world_model.config import InferScriptConfig
from scripts.train.infer_world_model import (
    _build_frame_report,
    _build_sharpness_report,
    _resample_video_time,
    _save_grid,
    _save_json_report,
    _save_strip,
    _select_runtime_dtype,
    _to_zero_one,
)


DEFAULT_VIDEO_PATH = REPO_ROOT / "runs" / "check_droid_preview_start25" / "preview.mp4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "check_vae_resolution_sweep"
DEFAULT_RESOLUTIONS = ("224x128", "336x192", "448x256", "672x384", "832x480")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the VAE roundtrip resolution sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-len", type=int, default=9)
    parser.add_argument("--horizon-len", type=int, default=8)
    parser.add_argument("--num-vis-frames", type=int, default=0)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="List of WIDTHxHEIGHT values to sweep, e.g. 224x128 832x480.",
    )
    return parser.parse_args()


def _parse_resolution(spec: str) -> tuple[int, int]:
    """Parse one WIDTHxHEIGHT string into integer width/height values."""
    normalized = spec.lower().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WIDTHxHEIGHT, got {spec!r}")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {spec!r}")
    return width, height


def _run_one_resolution(
    *,
    vae: WanVAE,
    video_path: Path,
    output_dir: Path,
    context_len: int,
    horizon_len: int,
    num_vis_frames: int,
    width: int,
    height: int,
    device: torch.device,
) -> dict[str, object]:
    """Roundtrip a local clip through the VAE at one resize setting."""
    resolution_dir = output_dir / f"{width}x{height}"
    resolution_dir.mkdir(parents=True, exist_ok=True)

    total_frames = context_len + horizon_len
    source_video = load_local_video_clip(video_path, start_frame=0, total_frames=total_frames).to(device)
    source_video = preprocess_video_for_vae(source_video, frame_height=height, frame_width=width)

    latents = vae.encode(source_video)
    raw_video = _to_zero_one(source_video)
    raw_future = raw_video[:, context_len:context_len + horizon_len]
    context_latent_steps, horizon_latent_steps = latent_split_from_frame_ratio(
        total_latent_steps=int(latents.shape[2]),
        context_frames=context_len,
        horizon_frames=horizon_len,
    )
    roundtrip_future = vae.decode(
        latents[:, :, context_latent_steps:],
        output_layout="BTCHW",
        output_range="zero_to_one",
    )
    raw_future_aligned = _resample_video_time(raw_future, roundtrip_future.shape[1])

    _save_strip(
        video=raw_future,
        output_path=resolution_dir / "raw_future_grid.png",
        num_frames=num_vis_frames,
        label="Raw future",
    )
    _save_grid(
        pred_video=roundtrip_future,
        target_video=raw_future_aligned,
        output_path=resolution_dir / "vae_roundtrip_vs_raw_grid.png",
        num_frames=num_vis_frames,
        top_label="Raw future aligned",
        bottom_label="VAE roundtrip",
    )

    cfg = InferScriptConfig(
        context_len=context_len,
        horizon_len=horizon_len,
        num_vis_frames=num_vis_frames,
        frame_height=height,
        frame_width=width,
    )
    frame_report = _build_frame_report(
        cfg=cfg,
        prepared=type(
            "Prepared",
            (),
            {
                "total_latent_steps": int(latents.shape[2]),
                "context_latent_steps": int(context_latent_steps),
                "horizon_latent_steps": int(horizon_latent_steps),
            },
        )(),
        source_video=source_video,
        raw_future=raw_future,
        raw_future_aligned=raw_future_aligned,
        pred_video=roundtrip_future,
        target_video=roundtrip_future,
    )
    sharpness_report = _build_sharpness_report(
        raw_future_aligned=raw_future_aligned,
        target_video=roundtrip_future,
        pred_video=roundtrip_future,
    )
    _save_json_report(frame_report, resolution_dir / "frame_report.json")
    _save_json_report(sharpness_report, resolution_dir / "sharpness_report.json")

    return {
        "resolution": f"{width}x{height}",
        "output_dir": str(resolution_dir),
        "sharpness_report": str(resolution_dir / "sharpness_report.json"),
        "grid": str(resolution_dir / "vae_roundtrip_vs_raw_grid.png"),
    }


def main() -> None:
    """Run the VAE roundtrip sweep and save a summary JSON file."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_dtype = _select_runtime_dtype(device=device, disable_amp=False)
    vae = WanVAE.from_pretrained(device=device, deterministic=True, torch_dtype=runtime_dtype)

    summary: list[dict[str, object]] = []
    for spec in args.resolutions:
        width, height = _parse_resolution(spec)
        summary.append(
            _run_one_resolution(
                vae=vae,
                video_path=args.video_path,
                output_dir=args.output_dir,
                context_len=args.context_len,
                horizon_len=args.horizon_len,
                num_vis_frames=args.num_vis_frames,
                width=width,
                height=height,
                device=device,
            )
        )
        print(f"Saved VAE sweep artifacts for {width}x{height}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved VAE sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
