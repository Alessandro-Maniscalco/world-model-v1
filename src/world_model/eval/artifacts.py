"""Shared visualization and report helpers for inference artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch

from world_model.config import InferScriptConfig


class SupportsLatentStepCounts(Protocol):
    """Describe the latent-step counters needed for frame reports."""

    total_latent_steps: int
    context_latent_steps: int
    horizon_latent_steps: int


def select_runtime_dtype(*, device: torch.device, disable_amp: bool) -> torch.dtype:
    """Choose an inference dtype that fits the active device and AMP setting."""
    if device.type != "cuda" or disable_amp:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def to_zero_one(video_btchw: torch.Tensor) -> torch.Tensor:
    """Convert a `BTCHW` video tensor into float `[0, 1]` space."""
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected BTCHW video with 5 dims, got {tuple(video_btchw.shape)}")
    if video_btchw.dtype == torch.uint8:
        return video_btchw.float() / 255.0

    video = video_btchw.float()
    max_val = float(video.max().detach().cpu()) if video.numel() > 0 else 1.0
    min_val = float(video.min().detach().cpu()) if video.numel() > 0 else 0.0
    if min_val >= -0.1 and max_val <= 1.1:
        return video.clamp(0.0, 1.0)
    if min_val >= -1.1 and max_val <= 1.1:
        return ((video + 1.0) / 2.0).clamp(0.0, 1.0)
    if min_val >= 0.0 and max_val <= 255.0:
        return (video / 255.0).clamp(0.0, 1.0)
    raise ValueError(
        f"Unable to infer video range for visualization from min={min_val:.3f}, max={max_val:.3f}."
    )


def resample_video_time(video_btchw: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Nearest-resample a `BTCHW` video to `target_steps` frames."""
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected BTCHW video with 5 dims, got {tuple(video_btchw.shape)}")

    source_steps = int(video_btchw.shape[1])
    if source_steps <= 0:
        raise ValueError("Cannot resample an empty video time dimension")
    if source_steps == target_steps:
        return video_btchw
    idx = torch.linspace(0, source_steps - 1, steps=target_steps, device=video_btchw.device)
    idx = idx.round().long().clamp(0, source_steps - 1)
    return video_btchw.index_select(dim=1, index=idx)


def save_grid(
    *,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    output_path: Path,
    num_frames: int,
    top_label: str = "Ground-truth",
    bottom_label: str = "Generated",
) -> None:
    """Save a two-row comparison grid to disk."""
    pred_frames = pred_video[0].detach().float().cpu()
    target_frames = target_video[0].detach().float().cpu()
    if pred_frames.shape[2:] != target_frames.shape[2:]:
        raise ValueError(
            "Predicted/target frame sizes must match for grid export; "
            f"got pred={tuple(pred_frames.shape[2:])}, target={tuple(target_frames.shape[2:])}"
        )
    vis_frames = resolve_visualized_frame_count(
        requested_frames=num_frames,
        available_frames=min(pred_frames.shape[0], target_frames.shape[0]),
    )
    if vis_frames <= 0:
        raise ValueError("No frames available for visualization")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        torch.save(
            {
                "pred_video": pred_video.detach().cpu(),
                "target_video": target_video.detach().cpu(),
            },
            output_path.with_suffix(".pt"),
        )
        return

    frame_h = int(pred_frames.shape[2])
    frame_w = int(pred_frames.shape[3])
    margin = 140
    gap = 12
    canvas_w = margin + vis_frames * frame_w
    canvas_h = frame_h * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, frame_h // 2), top_label, fill=(30, 30, 30))
    draw.text((24, frame_h + gap + frame_h // 2), bottom_label, fill=(30, 30, 30))

    for idx in range(vis_frames):
        gt = (
            target_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype("uint8")
        pred = (
            pred_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype("uint8")
        canvas.paste(Image.fromarray(gt), (margin + idx * frame_w, 0))
        canvas.paste(Image.fromarray(pred), (margin + idx * frame_w, frame_h + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_strip(
    *,
    video: torch.Tensor,
    output_path: Path,
    num_frames: int,
    label: str,
) -> None:
    """Save a one-row frame strip for a single video."""
    frames = video[0].detach().float().cpu()
    vis_frames = resolve_visualized_frame_count(
        requested_frames=num_frames,
        available_frames=frames.shape[0],
    )
    if vis_frames <= 0:
        raise ValueError("No frames available for visualization")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        torch.save({"video": video.detach().cpu()}, output_path.with_suffix(".pt"))
        return

    frame_h = int(frames.shape[2])
    frame_w = int(frames.shape[3])
    margin = 140
    canvas = Image.new("RGB", (margin + vis_frames * frame_w, frame_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, frame_h // 2), label, fill=(30, 30, 30))

    for idx in range(vis_frames):
        frame = (
            frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype("uint8")
        canvas.paste(Image.fromarray(frame), (margin + idx * frame_w, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_frame_report(
    *,
    cfg: InferScriptConfig,
    prepared: SupportsLatentStepCounts,
    source_video: torch.Tensor,
    raw_future: torch.Tensor,
    raw_future_aligned: torch.Tensor,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
) -> dict[str, object]:
    """Build a compact frame/latent accounting report for saved artifacts."""
    return {
        "requested_context_frames": int(cfg.context_len),
        "requested_horizon_frames": int(cfg.horizon_len),
        "raw_source_frames_after_preprocess": int(source_video.shape[1]),
        "raw_future_frames": int(raw_future.shape[1]),
        "latent_total_steps": int(prepared.total_latent_steps),
        "latent_context_steps": int(prepared.context_latent_steps),
        "latent_future_steps": int(prepared.horizon_latent_steps),
        "decoded_roundtrip_future_frames": int(target_video.shape[1]),
        "decoded_generated_future_frames": int(pred_video.shape[1]),
        "aligned_raw_future_frames": int(raw_future_aligned.shape[1]),
        "visualized_frames": int(
            resolve_visualized_frame_count(
                requested_frames=cfg.num_vis_frames,
                available_frames=min(
                    int(raw_future.shape[1]),
                    int(raw_future_aligned.shape[1]),
                    int(target_video.shape[1]),
                    int(pred_video.shape[1]),
                ),
            )
        ),
        "comparison_labels": {
            "comparison_grid.png": ["VAE roundtrip", "Generated"],
            "vae_roundtrip_future_grid.png": ["Raw future aligned", "VAE roundtrip"],
            "raw_future_grid.png": ["Raw future"],
        },
        "note": (
            "Wan VAE operates in compressed latent time, so raw horizon frames, latent future steps, "
            "and decoded future frames are different quantities."
        ),
    }


def build_sharpness_report(
    *,
    raw_future_aligned: torch.Tensor,
    target_video: torch.Tensor,
    pred_video: torch.Tensor,
) -> dict[str, object]:
    """Summarize relative sharpness between raw, VAE-roundtrip, and generated frames."""
    raw_energy = _mean_gradient_energy(raw_future_aligned)
    target_energy = _mean_gradient_energy(target_video)
    pred_energy = _mean_gradient_energy(pred_video)
    return {
        "mean_gradient_energy": {
            "raw_future_aligned": raw_energy,
            "vae_roundtrip": target_energy,
            "generated": pred_energy,
        },
        "relative_to_vae_roundtrip": {
            "generated": 0.0 if target_energy == 0.0 else pred_energy / target_energy,
            "raw_future_aligned": 0.0 if target_energy == 0.0 else raw_energy / target_energy,
        },
        "note": (
            "Higher mean gradient energy usually means a sharper image. "
            "Generated-to-roundtrip values well below 1.0 indicate extra blur beyond the VAE."
        ),
    }


def save_json_report(report: dict[str, object], output_path: Path) -> None:
    """Persist a JSON report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_visualized_frame_count(*, requested_frames: int, available_frames: int) -> int:
    """Resolve `0 => all` and clamp frame-visualization requests to availability."""
    if requested_frames < 0:
        raise ValueError(f"requested_frames must be >= 0, got {requested_frames}")
    if available_frames < 0:
        raise ValueError(f"available_frames must be >= 0, got {available_frames}")
    if requested_frames == 0:
        return available_frames
    return min(requested_frames, available_frames)


def _mean_gradient_energy(video_btchw: torch.Tensor) -> float:
    """Estimate perceptual sharpness from mean spatial gradient energy."""
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected BTCHW video with 5 dims, got {tuple(video_btchw.shape)}")
    video = video_btchw.detach().float().cpu()
    if video.numel() == 0:
        return 0.0

    gray = video.mean(dim=2)
    grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    return float(grad_y.pow(2).mean().item() + grad_x.pow(2).mean().item())
