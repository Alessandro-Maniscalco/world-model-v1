"""Run a fixed diagnostic matrix to compare null, prompt, and upstream VACE paths.

This smoke-check isolates whether replacing prompt-conditioned cross-attention
with null tokens is the main cause of poor DiT outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_SOURCE = "droid"
DEFAULT_CHECKPOINT_CANDIDATES = (
    REPO_ROOT / "runs" / "world_model_droid_local_overfit500" / "checkpoints" / "step_0000500.pt",
    REPO_ROOT / "runs" / "world_model_droid_local_overfit100" / "checkpoints" / "step_0000100.pt",
)
DEFAULT_VIDEO_PATH = REPO_ROOT / "runs" / "check_droid_preview_start25" / "preview.mp4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "check_dit_conditioning_diagnosis"
DEFAULT_PROMPT = "a laundry room with a washing machine, cleaning bottles on top, and a robot arm on the right"
DEFAULT_DROID_REPO_ID = "lerobot/droid_1.0.1"
DEFAULT_DROID_EPISODE_INDEX = 0
DEFAULT_DROID_FRAME_OFFSET = 25
DEFAULT_DROID_VIDEO_KEY = "observation.images.exterior_1_left"
PRIMARY_RESOLUTION = (832, 480)
SECONDARY_RESOLUTION = (224, 128)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the DiT conditioning diagnosis matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--video-source",
        choices=("droid", "local"),
        default=DEFAULT_VIDEO_SOURCE,
        help="Use a DROID-exported clip or a provided local mp4 clip.",
    )
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--context-len", type=int, default=10)
    parser.add_argument("--horizon-len", type=int, default=6)
    parser.add_argument("--integration-steps", type=int, default=20)
    parser.add_argument("--num-vis-frames", type=int, default=0)
    parser.add_argument("--repo-id", default=DEFAULT_DROID_REPO_ID)
    parser.add_argument("--episode-index", type=int, default=DEFAULT_DROID_EPISODE_INDEX)
    parser.add_argument("--frame-offset", type=int, default=DEFAULT_DROID_FRAME_OFFSET)
    parser.add_argument("--video-key", default=DEFAULT_DROID_VIDEO_KEY)
    parser.add_argument("--skip-secondary-resolution", action="store_true")
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional subset of case ids to run. Defaults to the full diagnostic matrix.",
    )
    return parser.parse_args()


def _python() -> str:
    """Return the active interpreter path for subprocess calls."""
    return sys.executable


def _resolve_checkpoint(path: Path | None) -> tuple[Path | None, str]:
    """Resolve a fine-tuned checkpoint or fall back to the pretrained backbone."""
    if path is not None:
        if not path.exists():
            return None, f"Checkpoint not found: {path}; falling back to pretrained backbone only."
        return path, f"Using explicit fine-tuned checkpoint: {path}"
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate, f"Using default fine-tuned checkpoint: {candidate}"
    return None, "No fine-tuned checkpoint found; using pretrained backbone only."


def _selected_case_ids(args: argparse.Namespace) -> set[str] | None:
    """Normalize optional case-id filtering into a set for membership checks."""
    return set(args.case_ids) if args.case_ids is not None else None


def _base_env() -> dict[str, str]:
    """Build the environment for subprocess-based smoke checks."""
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _run_command(*, command: list[str], output_path: Path) -> dict[str, object]:
    """Run one subprocess command and return a structured summary payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_base_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "command": command,
        "returncode": int(completed.returncode),
        "status": "ok" if completed.returncode == 0 else "error",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(output_path),
    }
    return result


def _build_error_result(*, message: str, output_path: Path) -> dict[str, object]:
    """Build a synthetic failed run result when prerequisites are missing."""
    return {
        "command": [],
        "returncode": 1,
        "status": "error",
        "stdout": "",
        "stderr": message,
        "output_path": str(output_path),
    }


def _prepare_video_path(args: argparse.Namespace) -> tuple[Path | None, str]:
    """Resolve the local clip path, exporting a DROID preview clip when requested."""
    if args.video_source == "local":
        if not args.video_path.exists():
            return None, f"Local video_path not found: {args.video_path}"
        return args.video_path, f"Using local clip: {args.video_path}"

    preview_dir = args.output_dir / "droid_source"
    command = [
        _python(),
        str(REPO_ROOT / "scripts" / "check" / "preview_droid_sequence.py"),
        "--repo-id",
        args.repo_id,
        "--episode-index",
        str(args.episode_index),
        "--frame-offset",
        str(args.frame_offset),
        "--video-key",
        args.video_key,
        "--num-frames",
        str(args.context_len + args.horizon_len),
        "--output-dir",
        str(preview_dir),
    ]
    result = _run_command(command=command, output_path=preview_dir / "preview.mp4")
    if result["status"] != "ok":
        return None, result["stderr"] or "Failed to export DROID preview clip."
    return (
        preview_dir / "preview.mp4",
        (
            f"Using DROID preview clip from repo_id={args.repo_id}, "
            f"episode_index={args.episode_index}, frame_offset={args.frame_offset}, "
            f"video_key={args.video_key}."
        ),
    )


def _infer_case_command(
    *,
    checkpoint: Path | None,
    conditioning_mode: str,
    prompt: str,
    negative_prompt: str,
    video_path: Path,
    output_dir: Path,
    context_len: int,
    horizon_len: int,
    integration_steps: int,
    num_vis_frames: int,
    width: int,
    height: int,
) -> list[str]:
    """Build one world-model inference command for the diagnostic matrix."""
    command = [
        _python(),
        str(REPO_ROOT / "scripts" / "train" / "infer_world_model.py"),
        "--conditioning-mode",
        conditioning_mode,
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
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if conditioning_mode == "prompt":
        command.extend(["--prompt", prompt, "--negative-prompt", negative_prompt])
    return command


def _upstream_case_command(
    *,
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    width: int,
    height: int,
) -> list[str]:
    """Build the upstream VACE pipeline command for the diagnostic matrix."""
    return [
        _python(),
        str(REPO_ROOT / "scripts" / "check" / "wan_vace_pipeline_generate_video.py"),
        "--prompt",
        prompt,
        "--negative-prompt",
        negative_prompt,
        "--output-path",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
    ]


def _vae_case_command(
    *,
    video_path: Path,
    output_dir: Path,
    context_len: int,
    horizon_len: int,
    num_vis_frames: int,
    width: int,
    height: int,
) -> list[str]:
    """Build the pure VAE roundtrip command for the diagnostic matrix."""
    return [
        _python(),
        str(REPO_ROOT / "scripts" / "check" / "sweep_vae_roundtrip_resolutions.py"),
        "--video-path",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--context-len",
        str(context_len),
        "--horizon-len",
        str(horizon_len),
        "--num-vis-frames",
        str(num_vis_frames),
        "--resolutions",
        f"{width}x{height}",
    ]


def _existing_output_paths(output_dir: Path) -> dict[str, str]:
    """Collect the standard artifact paths produced by inference checks."""
    return {
        "comparison_grid": str(output_dir / "comparison_grid.png"),
        "raw_future_grid": str(output_dir / "raw_future_grid.png"),
        "vae_roundtrip_grid": str(output_dir / "vae_roundtrip_future_grid.png"),
        "sharpness_report": str(output_dir / "sharpness_report.json"),
        "frame_report": str(output_dir / "frame_report.json"),
    }


def _run_diagnostic_matrix(args: argparse.Namespace) -> list[dict[str, object]]:
    """Execute the fixed diagnostic matrix and return a structured summary."""
    results: list[dict[str, object]] = []
    output_dir = args.output_dir
    selected_case_ids = _selected_case_ids(args)
    checkpoint = None
    checkpoint_note = ""
    checkpoint_cases = {
        "checkpoint_null_832x480",
        "checkpoint_prompt_832x480",
        "checkpoint_null_224x128",
        "checkpoint_prompt_224x128",
    }
    needs_checkpoint = selected_case_ids is None or bool(selected_case_ids & checkpoint_cases)
    if needs_checkpoint:
        checkpoint, checkpoint_note = _resolve_checkpoint(args.checkpoint)
    video_path, video_note = _prepare_video_path(args)
    width, height = PRIMARY_RESOLUTION

    primary_cases = [
        {
            "case_id": "checkpoint_null_832x480",
            "kind": "world_model",
            "conditioning_mode": "none",
            "prompt": "",
            "resolution": f"{width}x{height}",
            "output_dir": output_dir / "checkpoint_null_832x480",
            "command": _infer_case_command(
                checkpoint=checkpoint,
                conditioning_mode="none",
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                video_path=video_path if video_path is not None else args.video_path,
                output_dir=output_dir / "checkpoint_null_832x480",
                context_len=args.context_len,
                horizon_len=args.horizon_len,
                integration_steps=args.integration_steps,
                num_vis_frames=args.num_vis_frames,
                width=width,
                height=height,
            ),
        },
        {
            "case_id": "checkpoint_prompt_832x480",
            "kind": "world_model",
            "conditioning_mode": "prompt",
            "prompt": args.prompt,
            "resolution": f"{width}x{height}",
            "output_dir": output_dir / "checkpoint_prompt_832x480",
            "command": _infer_case_command(
                checkpoint=checkpoint,
                conditioning_mode="prompt",
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                video_path=video_path if video_path is not None else args.video_path,
                output_dir=output_dir / "checkpoint_prompt_832x480",
                context_len=args.context_len,
                horizon_len=args.horizon_len,
                integration_steps=args.integration_steps,
                num_vis_frames=args.num_vis_frames,
                width=width,
                height=height,
            ),
        },
        {
            "case_id": "pretrained_prompt_832x480",
            "kind": "world_model",
            "conditioning_mode": "prompt",
            "prompt": args.prompt,
            "resolution": f"{width}x{height}",
            "output_dir": output_dir / "pretrained_prompt_832x480",
            "command": _infer_case_command(
                checkpoint=None,
                conditioning_mode="prompt",
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                video_path=video_path if video_path is not None else args.video_path,
                output_dir=output_dir / "pretrained_prompt_832x480",
                context_len=args.context_len,
                horizon_len=args.horizon_len,
                integration_steps=args.integration_steps,
                num_vis_frames=args.num_vis_frames,
                width=width,
                height=height,
            ),
        },
        {
            "case_id": "upstream_vace_prompt_832x480",
            "kind": "upstream_vace",
            "conditioning_mode": "prompt",
            "prompt": args.prompt,
            "resolution": f"{width}x{height}",
            "output_dir": output_dir / "upstream_vace_prompt_832x480",
            "command": _upstream_case_command(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                output_path=output_dir / "upstream_vace_prompt_832x480" / "generated.mp4",
                width=width,
                height=height,
            ),
        },
        {
            "case_id": "vae_roundtrip_832x480",
            "kind": "vae_roundtrip",
            "conditioning_mode": "none",
            "prompt": "",
            "resolution": f"{width}x{height}",
            "output_dir": output_dir / "vae_roundtrip_832x480",
            "command": _vae_case_command(
                video_path=video_path if video_path is not None else args.video_path,
                output_dir=output_dir / "vae_roundtrip_832x480",
                context_len=args.context_len,
                horizon_len=args.horizon_len,
                num_vis_frames=args.num_vis_frames,
                width=width,
                height=height,
            ),
        },
    ]

    secondary_cases: list[dict[str, object]] = []
    if not args.skip_secondary_resolution:
        small_width, small_height = SECONDARY_RESOLUTION
        secondary_cases = [
            {
                "case_id": "checkpoint_null_224x128",
                "kind": "world_model",
                "conditioning_mode": "none",
                "prompt": "",
                "resolution": f"{small_width}x{small_height}",
                "output_dir": output_dir / "checkpoint_null_224x128",
                "command": _infer_case_command(
                    checkpoint=checkpoint,
                    conditioning_mode="none",
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    video_path=video_path if video_path is not None else args.video_path,
                    output_dir=output_dir / "checkpoint_null_224x128",
                    context_len=args.context_len,
                    horizon_len=args.horizon_len,
                    integration_steps=args.integration_steps,
                    num_vis_frames=args.num_vis_frames,
                    width=small_width,
                    height=small_height,
                ),
            },
            {
                "case_id": "checkpoint_prompt_224x128",
                "kind": "world_model",
                "conditioning_mode": "prompt",
                "prompt": args.prompt,
                "resolution": f"{small_width}x{small_height}",
                "output_dir": output_dir / "checkpoint_prompt_224x128",
                "command": _infer_case_command(
                    checkpoint=checkpoint,
                    conditioning_mode="prompt",
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    video_path=video_path if video_path is not None else args.video_path,
                    output_dir=output_dir / "checkpoint_prompt_224x128",
                    context_len=args.context_len,
                    horizon_len=args.horizon_len,
                    integration_steps=args.integration_steps,
                    num_vis_frames=args.num_vis_frames,
                    width=small_width,
                    height=small_height,
                ),
            },
        ]

    for case in primary_cases + secondary_cases:
        if selected_case_ids is not None and case["case_id"] not in selected_case_ids:
            continue
        if case["kind"] != "upstream_vace" and video_path is None:
            run_result = _build_error_result(message=video_note, output_path=case["output_dir"])
        else:
            run_result = _run_command(command=case["command"], output_path=case["output_dir"])
        result = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "conditioning_mode": case["conditioning_mode"],
            "prompt": case["prompt"],
            "resolution": case["resolution"],
            "output_dir": str(case["output_dir"]),
            "checkpoint_source": "fine_tuned" if checkpoint is not None else "pretrained_base",
            "checkpoint_note": checkpoint_note,
            "video_source": args.video_source,
            "video_note": video_note,
            **run_result,
        }
        if case["kind"] == "world_model":
            result["artifacts"] = _existing_output_paths(case["output_dir"])
        elif case["kind"] == "upstream_vace":
            result["artifacts"] = {"video": str(case["output_dir"] / "generated.mp4")}
        else:
            result["artifacts"] = {
                "grid": str(case["output_dir"] / f"{case['resolution']}" / "vae_roundtrip_vs_raw_grid.png"),
                "sharpness_report": str(case["output_dir"] / f"{case['resolution']}" / "sharpness_report.json"),
                "frame_report": str(case["output_dir"] / f"{case['resolution']}" / "frame_report.json"),
            }
        results.append(result)

    return results


def _save_summary(*, summary: list[dict[str, object]], output_dir: Path) -> Path:
    """Persist the top-level diagnosis summary JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    """Run the DiT conditioning diagnosis matrix and save its summary."""
    args = _parse_args()
    results = _run_diagnostic_matrix(args)
    summary_path = _save_summary(summary=results, output_dir=args.output_dir)
    ok_count = sum(1 for result in results if result["status"] == "ok")
    print(f"Saved diagnosis summary: {summary_path}")
    print(f"Completed {ok_count}/{len(results)} runs successfully.")
    for result in results:
        print(f"{result['case_id']}: {result['status']} -> {result['output_dir']}")


if __name__ == "__main__":
    main()
