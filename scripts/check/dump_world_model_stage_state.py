"""Dump latent packing and control-assembly state for one world-model window.

This manual smoke-check script is intended for fixed-anchor debugging. It loads
one checkpoint plus one dataset window, encodes the raw clip into Wan latents,
and saves a JSON report describing the latent split, chunk schedule, residual
target, and VACE control-tensor assembly before denoising.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_model.chunking import build_chunk_schedule, normalize_chunk_schedule_mode, resolve_num_chunks
from world_model.data.prepare import prepare_packed_batch, preprocess_video_for_vae
from world_model.latents import WanVAE
from world_model.models.wan_vace_conditioning import build_vace_control_tensor


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the stage-state dump."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path used to recover runtime config.")
    parser.add_argument("--repo-id", default="lerobot/aloha_static_fork_pick_up")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=60)
    parser.add_argument("--video-key", default="observation.images.cam_high")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--horizon-len", type=int, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--chunk-schedule-mode", default="k_chunks")
    parser.add_argument("--single-chunk-rollout", action="store_true")
    parser.add_argument("--frame-height", type=int, default=240)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--future-latent-residual-mode", default="")
    parser.add_argument("--future-control-fill-mode", default="")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _resolve_device(device_name: str) -> torch.device:
    """Resolve the requested torch device."""
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_runtime_dtype(*, device: torch.device) -> torch.dtype:
    """Choose an inference dtype that fits the active device."""
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_checkpoint_runtime_config(checkpoint_path: Path) -> tuple[dict[str, object], SimpleNamespace]:
    """Load a checkpoint and recover its saved runtime configuration."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dict.")
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        raise ValueError("Checkpoint missing extra_state.config metadata.")
    saved_cfg = extra_state.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("Checkpoint missing saved config metadata.")
    if "chunk_schedule_mode" in saved_cfg:
        saved_cfg = dict(saved_cfg)
        saved_cfg["chunk_schedule_mode"] = normalize_chunk_schedule_mode(saved_cfg["chunk_schedule_mode"])
    return checkpoint, SimpleNamespace(**saved_cfg)


def _load_dataset_window(
    *,
    repo_id: str,
    episode_index: int,
    start_frame: int,
    total_frames: int,
    video_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one contiguous dataset clip and its per-frame raw actions."""
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
    end_frame = start_frame + total_frames
    if end_frame > len(dataset):
        raise ValueError(
            f"Requested frames [{start_frame}:{end_frame}] exceed episode-local length {len(dataset)}."
        )

    frames: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for frame_index in range(start_frame, end_frame):
        sample = dataset[frame_index]
        frame = sample.get(video_key)
        if frame is None:
            available = [key for key in sample if key.startswith("observation.images.")]
            raise KeyError(f"video_key={video_key!r} not found. Available camera keys: {available}")
        action = sample.get("action")
        if action is None:
            raise KeyError("Dataset sample is missing 'action'.")
        frames.append(frame.to(dtype=torch.float32))
        actions.append(action.to(dtype=torch.float32))

    video = torch.stack(frames, dim=0).unsqueeze(0).to(device=device)
    action_seq = torch.stack(actions, dim=0).unsqueeze(0).to(device=device)
    return video, action_seq


def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    """Convert one tensor into a compact JSON-friendly summary."""
    tensor_cpu = tensor.detach().float().cpu()
    return {
        "shape": [int(dim) for dim in tensor_cpu.shape],
        "min": float(tensor_cpu.min().item()),
        "max": float(tensor_cpu.max().item()),
        "mean": float(tensor_cpu.mean().item()),
        "std": float(tensor_cpu.std(unbiased=False).item()),
        "abs_mean": float(tensor_cpu.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(tensor_cpu).item()),
    }


def _build_rollout_boundaries(
    *,
    future_steps: int,
    k: int,
    chunk_schedule_mode: str,
    single_chunk_rollout: bool,
) -> list[list[int]]:
    """Build rollout boundaries in latent time for the active inference contract."""
    chunk_schedule_mode = normalize_chunk_schedule_mode(chunk_schedule_mode)
    if single_chunk_rollout or future_steps < resolve_num_chunks(k=k, chunk_schedule_mode=chunk_schedule_mode):
        return [[0, int(future_steps)]]
    schedule = build_chunk_schedule(
        future_steps=future_steps,
        k=k,
        chunk_schedule_mode=chunk_schedule_mode,
    )
    return [[int(start), int(end)] for start, end in schedule.boundaries]


def _build_future_latent_residual_base(
    *,
    z_past_video: torch.Tensor,
    future_steps: int,
    future_latent_residual_mode: str,
) -> torch.Tensor:
    """Build the inference-time future residual baseline tensor."""
    if future_latent_residual_mode == "none":
        return z_past_video.new_zeros(
            z_past_video.shape[0],
            z_past_video.shape[1],
            future_steps,
            z_past_video.shape[3],
            z_past_video.shape[4],
        )
    if future_latent_residual_mode == "last_context_frame":
        return z_past_video[:, :, -1:, :, :].expand(-1, -1, future_steps, -1, -1)
    raise ValueError(
        "future_latent_residual_mode must be 'none' or 'last_context_frame', "
        f"got {future_latent_residual_mode!r}"
    )


def _build_control_diagnostics(
    *,
    z_past_video: torch.Tensor,
    z_future_video: torch.Tensor,
    control_black_latents: torch.Tensor,
    control_gray_latents: torch.Tensor,
    future_residual_base: torch.Tensor,
    future_control_fill_mode: str,
    mask_channels: int,
) -> dict[str, object]:
    """Recreate the VACE control assembly used before denoising."""
    future_start = int(z_past_video.shape[2])
    future_control_video = control_gray_latents[:, :, future_start:, :, :]
    inactive_fill_latents = control_black_latents.clone()
    reactive_fill_latents = control_black_latents.clone()
    future_fill_base = z_past_video[:, :, -1:, :, :].expand_as(z_future_video)

    if future_control_fill_mode == "last_context_frame":
        future_control_video = future_fill_base
        inactive_fill_latents[:, :, future_start:, :, :] = future_fill_base
        reactive_fill_latents[:, :, future_start:, :, :] = future_fill_base
    elif future_control_fill_mode != "gray":
        raise ValueError(
            "future_control_fill_mode must be 'gray' or 'last_context_frame', "
            f"got {future_control_fill_mode!r}"
        )

    future_control_video = future_control_video - future_residual_base
    inactive_fill_latents[:, :, future_start:, :, :] = (
        inactive_fill_latents[:, :, future_start:, :, :] - future_residual_base
    )
    reactive_fill_latents[:, :, future_start:, :, :] = (
        reactive_fill_latents[:, :, future_start:, :, :] - future_residual_base
    )

    observed_mask = torch.zeros(
        z_past_video.shape[0],
        1,
        z_past_video.shape[2],
        z_past_video.shape[3],
        z_past_video.shape[4],
        device=z_past_video.device,
        dtype=z_past_video.dtype,
    )
    future_control_mask = torch.ones(
        z_future_video.shape[0],
        1,
        z_future_video.shape[2],
        z_future_video.shape[3],
        z_future_video.shape[4],
        device=z_future_video.device,
        dtype=z_future_video.dtype,
    )
    control_video = torch.cat([z_past_video, future_control_video], dim=2)
    control_mask = torch.cat([observed_mask, future_control_mask], dim=2)
    control_hidden_states = build_vace_control_tensor(
        observed_latents=control_video,
        observed_mask=control_mask,
        inactive_fill_latents=inactive_fill_latents,
        reactive_fill_latents=reactive_fill_latents,
        mask_channels=mask_channels,
    )

    return {
        "future_control_fill_mode": future_control_fill_mode,
        "future_fill_base": _tensor_summary(future_fill_base),
        "future_control_video_after_residual_subtraction": _tensor_summary(future_control_video),
        "inactive_future_fill_after_residual_subtraction": _tensor_summary(
            inactive_fill_latents[:, :, future_start:, :, :]
        ),
        "reactive_future_fill_after_residual_subtraction": _tensor_summary(
            reactive_fill_latents[:, :, future_start:, :, :]
        ),
        "control_video": _tensor_summary(control_video),
        "control_mask": {
            "shape": [int(dim) for dim in control_mask.shape],
            "future_mean": float(control_mask[:, :, future_start:, :, :].float().mean().item()),
            "past_mean": float(control_mask[:, :, :future_start, :, :].float().mean().item()),
        },
        "control_hidden_states": _tensor_summary(control_hidden_states),
        "future_control_abs_max": float(future_control_video.detach().abs().max().item()),
        "future_control_is_exact_zero": bool(
            torch.count_nonzero(future_control_video.detach()).item() == 0
        ),
    }


def main() -> None:
    """Load one checkpoint window and write a latent/control stage report."""
    args = _parse_args()
    checkpoint, runtime_cfg = _load_checkpoint_runtime_config(args.checkpoint)
    del checkpoint

    device = _resolve_device(args.device)
    runtime_dtype = _select_runtime_dtype(device=device)
    total_frames = args.context_len + args.horizon_len
    video, action_seq = _load_dataset_window(
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        total_frames=total_frames,
        video_key=args.video_key,
        device=device,
    )
    preprocessed_video = preprocess_video_for_vae(
        video,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
    )

    vae = WanVAE.from_pretrained(
        device=device,
        deterministic=True,
        torch_dtype=runtime_dtype,
    )
    prepared = prepare_packed_batch(
        batch={
            args.video_key: video,
            "action": action_seq,
        },
        encoder=vae,
        device=device,
        video_key=args.video_key,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
    )

    future_latent_residual_mode = (
        args.future_latent_residual_mode
        or str(getattr(runtime_cfg, "future_latent_residual_mode", "none"))
    )
    future_control_fill_mode = (
        args.future_control_fill_mode
        or str(getattr(runtime_cfg, "future_control_fill_mode", "gray"))
    )
    mask_channels = int(getattr(runtime_cfg, "mask_channels", 64))
    future_residual_base = _build_future_latent_residual_base(
        z_past_video=prepared.z_past_video,
        future_steps=prepared.z_future_video.shape[2],
        future_latent_residual_mode=future_latent_residual_mode,
    )
    rollout_boundaries = _build_rollout_boundaries(
        future_steps=prepared.z_future_video.shape[2],
        k=args.k,
        chunk_schedule_mode=args.chunk_schedule_mode,
        single_chunk_rollout=args.single_chunk_rollout,
    )
    target_minus_last_context = prepared.z_future_video - prepared.z_past_video[:, :, -1:, :, :]
    future_target_minus_base = prepared.z_future_video - future_residual_base

    report = {
        "checkpoint": str(args.checkpoint),
        "repo_id": args.repo_id,
        "episode_index": int(args.episode_index),
        "start_frame": int(args.start_frame),
        "video_key": args.video_key,
        "frame_size": {
            "height": int(args.frame_height),
            "width": int(args.frame_width),
        },
        "raw_video": {
            "source_shape": [int(dim) for dim in video.shape],
            "preprocessed_shape": [int(dim) for dim in preprocessed_video.shape],
        },
        "prepared_batch": {
            "latent_shape": [int(dim) for dim in prepared.latent_shape],
            "total_latent_steps": int(prepared.total_latent_steps),
            "context_latent_steps": int(prepared.context_latent_steps),
            "horizon_latent_steps": int(prepared.horizon_latent_steps),
            "z_past_video": _tensor_summary(prepared.z_past_video),
            "z_future_video": _tensor_summary(prepared.z_future_video),
            "control_black_latents": _tensor_summary(prepared.control_black_latents),
            "control_gray_latents": _tensor_summary(prepared.control_gray_latents),
            "action_sequence": _tensor_summary(action_seq),
            "future_action_plan": _tensor_summary(prepared.a_plan),
        },
        "chunk_schedule": {
            "k": int(args.k),
            "chunk_schedule_mode": normalize_chunk_schedule_mode(args.chunk_schedule_mode),
            "single_chunk_rollout": bool(args.single_chunk_rollout),
            "boundaries": rollout_boundaries,
        },
        "residual": {
            "future_latent_residual_mode": future_latent_residual_mode,
            "future_residual_base": _tensor_summary(future_residual_base),
            "future_target_minus_base": _tensor_summary(future_target_minus_base),
            "future_target_minus_last_context": _tensor_summary(target_minus_last_context),
        },
        "control": _build_control_diagnostics(
            z_past_video=prepared.z_past_video,
            z_future_video=prepared.z_future_video,
            control_black_latents=prepared.control_black_latents,
            control_gray_latents=prepared.control_gray_latents,
            future_residual_base=future_residual_base,
            future_control_fill_mode=future_control_fill_mode,
            mask_channels=mask_channels,
        ),
        "note": (
            "When future_control_fill_mode and future_latent_residual_mode are both "
            "'last_context_frame', the future control stream should cancel to an exact "
            "zero-change prior after residual subtraction."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved stage report: {args.output_json}")


if __name__ == "__main__":
    main()
