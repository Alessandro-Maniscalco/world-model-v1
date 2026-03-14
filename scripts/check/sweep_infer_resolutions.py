"""Run the same inference checkpoint across multiple spatial resolutions.

This smoke-check is meant for manual visual comparison of VAE roundtrip and
generated outputs at different resize settings.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "world_model_droid_local_overfit100" / "checkpoints" / "step_0000100.pt"
DEFAULT_VIDEO_PATH = REPO_ROOT / "runs" / "check_droid_preview_start25" / "preview.mp4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "check_infer_resolution_sweep"
DEFAULT_RESOLUTIONS = ("224x128", "336x192", "448x256", "672x384", "832x480")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the inference resolution sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--horizon-len", type=int, default=6)
    parser.add_argument("--integration-steps", type=int, default=20)
    parser.add_argument("--num-vis-frames", type=int, default=0)
    parser.add_argument("--single-chunk-rollout", action="store_true")
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


def _build_command(
    *,
    checkpoint: Path,
    video_path: Path,
    output_dir: Path,
    context_len: int,
    horizon_len: int,
    integration_steps: int,
    num_vis_frames: int,
    width: int,
    height: int,
    single_chunk_rollout: bool,
) -> list[str]:
    """Build one inference command for a single resolution setting."""
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "infer_world_model.py"),
        "--checkpoint",
        str(checkpoint),
        "--conditioning-mode",
        "none",
        "--video-path",
        str(video_path),
        "--context-len",
        str(context_len),
        "--horizon-len",
        str(horizon_len),
        "--frame-height",
        str(height),
        "--frame-width",
        str(width),
        "--integration-steps",
        str(integration_steps),
        "--num-vis-frames",
        str(num_vis_frames),
        "--output-dir",
        str(output_dir),
    ]
    if single_chunk_rollout:
        command.append("--single-chunk-rollout")
    return command


def _run_one_resolution(
    *,
    checkpoint: Path,
    video_path: Path,
    root_output_dir: Path,
    context_len: int,
    horizon_len: int,
    integration_steps: int,
    num_vis_frames: int,
    width: int,
    height: int,
    single_chunk_rollout: bool,
) -> dict[str, object]:
    """Run inference for one resolution and return a JSON-serializable result."""
    resolution_label = f"{width}x{height}"
    output_dir = root_output_dir / resolution_label
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    command = _build_command(
        checkpoint=checkpoint,
        video_path=video_path,
        output_dir=output_dir,
        context_len=context_len,
        horizon_len=horizon_len,
        integration_steps=integration_steps,
        num_vis_frames=num_vis_frames,
        width=width,
        height=height,
        single_chunk_rollout=single_chunk_rollout,
    )
    print(f"Running {resolution_label}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    result: dict[str, object] = {
        "resolution": resolution_label,
        "width": width,
        "height": height,
        "output_dir": str(output_dir),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "comparison_grid": str(output_dir / "comparison_grid.png"),
        "sharpness_report": str(output_dir / "sharpness_report.json"),
    }
    result["status"] = "ok" if completed.returncode == 0 else "error"
    return result


def _save_summary(*, summary: list[dict[str, object]], output_dir: Path) -> Path:
    """Persist the sweep result summary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    """Run inference across the requested resolutions and persist a summary."""
    args = _parse_args()
    results: list[dict[str, object]] = []
    for spec in args.resolutions:
        width, height = _parse_resolution(spec)
        results.append(
            _run_one_resolution(
                checkpoint=args.checkpoint,
                video_path=args.video_path,
                root_output_dir=args.output_dir,
                context_len=args.context_len,
                horizon_len=args.horizon_len,
                integration_steps=args.integration_steps,
                num_vis_frames=args.num_vis_frames,
                width=width,
                height=height,
                single_chunk_rollout=args.single_chunk_rollout,
            )
        )

    summary_path = _save_summary(summary=results, output_dir=args.output_dir)
    ok_count = sum(1 for result in results if result["status"] == "ok")
    print(f"Saved resolution sweep summary: {summary_path}")
    print(f"Completed {ok_count}/{len(results)} runs successfully.")
    for result in results:
        print(f"{result['resolution']}: {result['status']} -> {result['output_dir']}")


if __name__ == "__main__":
    main()
