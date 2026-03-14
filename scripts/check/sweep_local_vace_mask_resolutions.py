"""Sweep local Wan VACE masked conditioning over resolutions.

Uses this repo's backbone path with explicit video-plus-mask control and no cross-attention.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import imageio.v2 as iio
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wan_vace_diffuser_generate_video as base
from world_model.config import load_train_config
from world_model.data.prepare import preprocess_video_for_vae
from world_model.latents import WanVAE
from world_model.models.wan_vace_conditioning import build_vace_control_tensor
from world_model.vendor.wan import WanVACETransformer3DModel


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train" / "aloha_fork_pick_up.yaml"
DEFAULT_OUTPUT_DIR = Path("runs/check_wan_vace_local_mask_resolution_sweep")
DEFAULT_RESOLUTIONS = (
    "320x240",
    "384x288",
    "512x384",
)
DEFAULT_REPO_ID = "lerobot/aloha_static_fork_pick_up"
DEFAULT_VIDEO_KEY = "observation.images.cam_high"
DEFAULT_EPISODE_INDEX = 0
DEFAULT_START_FRAME = 0
DEFAULT_REFERENCE_LAYOUT = "dense"
DEFAULT_DEVICE = "auto"


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the local masked-conditioning sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
        help="Flow-matching integration steps for masked latent denoising.",
    )
    parser.add_argument("--fps", type=int, default=base.FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    parser.add_argument("--start-frame", type=int, default=DEFAULT_START_FRAME)
    parser.add_argument("--video-key", default=DEFAULT_VIDEO_KEY)
    parser.add_argument(
        "--reference-layout",
        choices=("dense", "first_last"),
        default=DEFAULT_REFERENCE_LAYOUT,
        help="Use 5 consecutive condition frames or only the first/last sparse pair.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=DEFAULT_DEVICE,
        help="Execution device. Use cpu if your GPU is busy.",
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


def _resolve_device(*, device_name: str) -> torch.device:
    """Resolve the requested execution device."""
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_runtime_dtype(*, device: torch.device) -> torch.dtype:
    """Choose the mixed-precision dtype for inference."""
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(*, device: torch.device, dtype: torch.dtype):
    """Build a lightweight autocast context for CUDA inference."""
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _offline_mode_enabled() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    import os

    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _load_runtime_config(config_path: Path):
    """Load a base runtime config with pretrained weights and no cross-attention conditioning."""
    train_cfg = load_train_config(config_path)
    return replace(
        train_cfg,
        trainable_backbone="full",
        conditioning_mode="none",
        load_pretrained_backbone=True,
    )


def _condition_indices(*, reference_layout: str) -> list[int]:
    """Return raw-frame condition indices for the chosen layout."""
    if reference_layout == "dense":
        return list(range(base.NUM_CONDITION_FRAMES))
    return [0, base.NUM_TOTAL_FRAMES - 1]


def _load_target_clip(
    *,
    repo_id: str,
    episode_index: int,
    start_frame: int,
    total_frames: int,
    video_key: str,
    device: torch.device,
) -> torch.Tensor:
    """Load a contiguous episode-local clip as BTCHW float video."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
    end_frame = start_frame + total_frames
    if end_frame > len(dataset):
        raise ValueError(
            f"Requested frames [{start_frame}:{end_frame}] exceed episode-local length {len(dataset)}."
        )

    frames: list[torch.Tensor] = []
    for frame_index in range(start_frame, end_frame):
        sample = dataset[frame_index]
        if video_key not in sample:
            available = [key for key in sample if key.startswith("observation.images.")]
            raise KeyError(
                f"video_key={video_key!r} not found in sample. Available camera keys: {available}"
            )
        frames.append(sample[video_key].to(dtype=torch.float32))
    return torch.stack(frames, dim=0).unsqueeze(0).to(device=device)


def _build_masked_control_inputs(
    *,
    target_video: torch.Tensor,
    condition_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a gray-filled raw RGB control video and binary generation mask."""
    batch_size, total_frames, _, height, width = target_video.shape
    known_bt1hw = torch.zeros(
        (batch_size, total_frames, 1, height, width),
        device=target_video.device,
        dtype=target_video.dtype,
    )
    known_bt1hw[:, condition_indices] = 1.0
    gray_video = _make_constant_video_like(video=target_video, zero_to_one_value=128.0 / 255.0)
    control_video = (target_video * known_bt1hw) + (gray_video * (1.0 - known_bt1hw))
    control_mask = (1.0 - known_bt1hw).permute(0, 2, 1, 3, 4)
    return control_video, control_mask


def _make_constant_video_like(*, video: torch.Tensor, zero_to_one_value: float) -> torch.Tensor:
    """Build a constant video that matches the active numeric range of `video`."""
    if not (0.0 <= zero_to_one_value <= 1.0):
        raise ValueError(f"zero_to_one_value must be in [0,1], got {zero_to_one_value}")

    video_float = video.float()
    if video_float.numel() == 0:
        fill_value = zero_to_one_value
    else:
        min_value = float(video_float.min().detach().cpu().item())
        max_value = float(video_float.max().detach().cpu().item())
        if min_value >= -1.1 and max_value <= 1.1 and min_value < -0.1:
            fill_value = zero_to_one_value * 2.0 - 1.0
        elif min_value >= 0.0 and max_value <= 255.0 and max_value > 1.1:
            fill_value = zero_to_one_value * 255.0
        else:
            fill_value = zero_to_one_value
    return torch.full_like(video_float, fill_value=fill_value)


def _resize_mask_to_latent(mask_bt1hw: torch.Tensor, *, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Resize a raw-frame mask to latent time and space with nearest sampling."""
    target_frames, target_height, target_width = target_shape
    return F.interpolate(
        mask_bt1hw,
        size=(target_frames, target_height, target_width),
        mode="nearest-exact",
    )


def _normalize_video_for_export(video: torch.Tensor) -> torch.Tensor:
    """Normalize BTCHW video tensors into zero-to-one range for MP4 export."""
    normalized = video.detach().float()
    if normalized.numel() == 0:
        return normalized
    if float(normalized.max().item()) > 1.0:
        normalized = normalized / 255.0
    return normalized.clamp(0.0, 1.0)


def _tensor_video_to_frames(video_btchw: torch.Tensor) -> list[np.ndarray]:
    """Convert BTCHW zero-to-one tensors into contiguous HWC uint8 frames."""
    frames: list[np.ndarray] = []
    for frame in video_btchw[0].detach().float().cpu().clamp(0.0, 1.0):
        frame_hwc = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8, copy=False)
        frames.append(np.ascontiguousarray(frame_hwc))
    return frames


def _export_video(*, video_frames: list[np.ndarray], output_video_path: str, fps: int) -> str:
    """Export generated frames to an mp4 with an explicit RGB-safe writer."""
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in video_frames:
            writer.append_data(np.ascontiguousarray(frame))
    return str(output_path)


def _build_side_by_side_video(*, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate two BTCHW videos horizontally after aligning frame counts."""
    target_steps = min(int(left.shape[1]), int(right.shape[1]))
    return torch.cat([left[:, :target_steps], right[:, :target_steps]], dim=4)


def _sample_masked_video(
    *,
    backbone: WanVACETransformer3DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    z_target: torch.Tensor,
    control_hidden_states: torch.Tensor,
    latent_mask: torch.Tensor,
    integration_steps: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Denoise a full latent clip while clamping known latent regions after each step."""
    latent_state = torch.randn(
        z_target.shape,
        device=z_target.device,
        dtype=z_target.dtype,
        generator=generator,
    )
    latent_state = latent_mask * latent_state + (1.0 - latent_mask) * z_target
    zero_tokens = torch.zeros(
        z_target.shape[0],
        1,
        int(backbone.config.text_dim),
        device=z_target.device,
        dtype=z_target.dtype,
    )

    scheduler.set_timesteps(integration_steps, device=z_target.device)
    for timestep in scheduler.timesteps:
        timestep_t = timestep.expand(z_target.shape[0]).to(device=z_target.device, dtype=z_target.dtype)
        velocity = backbone(
            hidden_states=latent_state,
            timestep=timestep_t,
            encoder_hidden_states=zero_tokens,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=None,
            attention_mask=None,
            return_dict=True,
        ).sample
        latent_state = scheduler.step(velocity, timestep, latent_state, generator=generator, return_dict=False)[0]
        latent_state = latent_mask * latent_state + (1.0 - latent_mask) * z_target

    return latent_state


def _run_one_resolution(
    *,
    cfg,
    width: int,
    height: int,
    output_dir: Path,
    repo_id: str,
    episode_index: int,
    start_frame: int,
    video_key: str,
    reference_layout: str,
    device: torch.device,
    runtime_dtype: torch.dtype,
    integration_steps: int,
    fps: int,
    seed: int,
) -> dict[str, object]:
    """Run one local masked-conditioning generation at a specific resolution."""
    label = f"{width}x{height}"
    output_path = output_dir / f"{label}.mp4"
    comparison_path = output_dir / f"{label}_comparison.mp4"
    start_time = time.time()
    try:
        total_frames = base.NUM_TOTAL_FRAMES
        generator = torch.Generator(device=device.type) if device.type == "cuda" else torch.Generator()
        generator.manual_seed(seed)
        torch.manual_seed(seed)

        target_video_raw = _load_target_clip(
            repo_id=repo_id,
            episode_index=episode_index,
            start_frame=start_frame,
            total_frames=total_frames,
            video_key=video_key,
            device=device,
        )
        target_video = preprocess_video_for_vae(target_video_raw, frame_height=height, frame_width=width)
        condition_indices = _condition_indices(reference_layout=reference_layout)
        control_video_raw, control_mask_raw = _build_masked_control_inputs(
            target_video=target_video,
            condition_indices=condition_indices,
        )

        vae = WanVAE.from_pretrained(device=device, deterministic=True, torch_dtype=runtime_dtype)
        z_target = vae.encode(target_video)
        z_control = vae.encode(control_video_raw)
        z_black = vae.encode(_make_constant_video_like(video=target_video, zero_to_one_value=0.0))
        latent_mask = _resize_mask_to_latent(
            control_mask_raw,
            target_shape=(int(z_target.shape[2]), int(z_target.shape[3]), int(z_target.shape[4])),
        ).to(device=device, dtype=runtime_dtype)

        backbone = WanVACETransformer3DModel.from_pretrained(
            cfg.wan_vace_model_id,
            subfolder=cfg.wan_vace_subfolder or None,
            local_files_only=_offline_mode_enabled(),
        ).to(device=device, dtype=runtime_dtype)
        backbone.eval()
        control_hidden_states = build_vace_control_tensor(
            observed_latents=z_control.to(device=device, dtype=runtime_dtype),
            observed_mask=latent_mask,
            inactive_fill_latents=z_black.to(device=device, dtype=runtime_dtype),
            reactive_fill_latents=z_black.to(device=device, dtype=runtime_dtype),
            mask_channels=int(cfg.mask_channels),
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            cfg.wan_vace_model_id,
            subfolder="scheduler",
            local_files_only=_offline_mode_enabled(),
        )

        with _autocast_context(device=device, dtype=runtime_dtype):
            pred_latents = _sample_masked_video(
                backbone=backbone,
                scheduler=scheduler,
                z_target=z_target.to(device=device, dtype=runtime_dtype),
                control_hidden_states=control_hidden_states,
                latent_mask=latent_mask,
                integration_steps=integration_steps,
                generator=generator,
            )
            pred_video = vae.decode(pred_latents, output_layout="BTCHW", output_range="zero_to_one")

        target_export = _normalize_video_for_export(target_video)
        pred_export = _normalize_video_for_export(pred_video)
        comparison_video = _build_side_by_side_video(left=target_export, right=pred_export)
        output_dir.mkdir(parents=True, exist_ok=True)
        _export_video(video_frames=_tensor_video_to_frames(pred_export), output_video_path=str(output_path), fps=fps)
        _export_video(
            video_frames=_tensor_video_to_frames(comparison_video),
            output_video_path=str(comparison_path),
            fps=fps,
        )
        return {
            "resolution": label,
            "status": "ok",
            "output_path": str(output_path),
            "comparison_output_path": str(comparison_path),
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
    """Run the local VACE-mask resolution sweep."""
    args = _parse_args()
    cfg = _load_runtime_config(args.config)
    device = _resolve_device(device_name=args.device)
    runtime_dtype = _select_runtime_dtype(device=device)
    results: list[dict[str, object]] = []

    for width, height in (_parse_resolution(spec) for spec in args.resolutions):
        label = f"{width}x{height}"
        print(f"Running local VACE-mask evaluator at {label}...")
        result = _run_one_resolution(
            cfg=cfg,
            width=width,
            height=height,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            episode_index=args.episode_index,
            start_frame=args.start_frame,
            video_key=args.video_key,
            reference_layout=args.reference_layout,
            device=device,
            runtime_dtype=runtime_dtype,
            integration_steps=args.num_inference_steps,
            fps=args.fps,
            seed=args.seed,
        )
        results.append(result)
        if result["status"] == "ok":
            print(f"{label}: saved {result['output_path']}")
            print(f"{label}: saved {result['comparison_output_path']}")
        else:
            print(f"{label}: {result['error']}")

    summary_path = _save_summary(output_dir=args.output_dir, results=results)
    print(f"Saved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
