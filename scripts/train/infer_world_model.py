"""Run Wan VACE world-model inference and export GT-vs-generated grids.

This entrypoint uses typed YAML-backed config plus CLI overrides and shared
batch preparation utilities.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

# Ensure local `src/` package imports work when run as `python scripts/train/infer_world_model.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    # Prevent sibling `world_model.py` from shadowing the `world_model` package.
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
loaded_world_model = sys.modules.get("world_model")
if loaded_world_model is not None and not hasattr(loaded_world_model, "__path__"):
    # Drop incorrectly loaded module objects so package import can succeed.
    sys.modules.pop("world_model", None)

from world_model.config import InferScriptConfig, apply_namespace_overrides, load_infer_config
from world_model.data import build_lerobot_dataloader, prepare_packed_batch
from world_model.data.schema import PreparedPackedBatch
from world_model.eval import infer_future_videos_chunkwise
from world_model.latents import WanVAE
from world_model.models.wan_vace_conditioning import ActionTokenEncoder
from world_model.models.wan_vace_factory import build_wan_vace_runtime_modules
from world_model.models.wan_vace_world_model import WanVACEWorldModel


def _config_parser() -> argparse.ArgumentParser:
    """Create parser for bootstrap config argument."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    return parser


def _build_parser(defaults: InferScriptConfig) -> argparse.ArgumentParser:
    """Create full CLI parser using dataclass defaults."""
    parser = argparse.ArgumentParser(description=__doc__, parents=[_config_parser()])
    parser.add_argument(
        "--checkpoint",
        default=defaults.checkpoint,
        help="Optional local fine-tune checkpoint .pt to overlay on the pretrained Wan VACE transformer.",
    )
    parser.add_argument(
        "--video-path",
        default=defaults.video_path,
        help="Optional local video file. If set, dataset loading is skipped.",
    )
    parser.add_argument("--start-frame", type=int, default=defaults.start_frame)
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--video-key", default=defaults.video_key)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--context-len", type=int, default=defaults.context_len)
    parser.add_argument("--horizon-len", type=int, default=defaults.horizon_len)
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--subset-size", type=int, default=defaults.subset_size)
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--integration-steps", type=int, default=defaults.integration_steps)
    parser.add_argument("--num-vis-frames", type=int, default=defaults.num_vis_frames)
    parser.add_argument(
        "--load-pretrained-backbone",
        action="store_true",
        default=defaults.load_pretrained_backbone,
    )
    parser.add_argument("--no-load-pretrained-backbone", dest="load_pretrained_backbone", action="store_false")
    parser.add_argument("--wan-vace-model-id", default=defaults.wan_vace_model_id)
    parser.add_argument("--wan-vace-subfolder", default=defaults.wan_vace_subfolder)
    parser.add_argument("--wan-num-attention-heads", type=int, default=defaults.wan_num_attention_heads)
    parser.add_argument("--wan-attention-head-dim", type=int, default=defaults.wan_attention_head_dim)
    parser.add_argument("--wan-text-dim", type=int, default=defaults.wan_text_dim)
    parser.add_argument("--wan-freq-dim", type=int, default=defaults.wan_freq_dim)
    parser.add_argument("--wan-ffn-dim", type=int, default=defaults.wan_ffn_dim)
    parser.add_argument("--wan-num-layers", type=int, default=defaults.wan_num_layers)
    parser.add_argument("--vace-layers", type=int, nargs="+", default=list(defaults.vace_layers))
    parser.add_argument("--control-scale", type=float, default=defaults.control_scale)
    parser.add_argument("--mask-channels", type=int, default=defaults.mask_channels)
    parser.add_argument(
        "--action-path",
        default=defaults.action_path,
        help="Optional .npy action tensor with shape [A], [T,A], or [1,T,A].",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=defaults.action_dim,
        help="Action dim when --action-path is not set. 0 infers from checkpoint.",
    )
    parser.add_argument(
        "--action-value",
        type=float,
        default=defaults.action_value,
        help="Fill value for synthetic actions when --action-path is not set.",
    )
    parser.add_argument("--disable-proprio", action="store_true", default=defaults.disable_proprio)
    parser.add_argument("--enable-proprio", dest="disable_proprio", action="store_false")
    parser.add_argument("--disable-amp", action="store_true", default=defaults.disable_amp)
    parser.add_argument("--enable-amp", dest="disable_amp", action="store_false")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _load_args() -> InferScriptConfig:
    """Load YAML config and apply CLI overrides into final infer config."""
    config_args, _ = _config_parser().parse_known_args()
    defaults = load_infer_config(config_args.config)
    parser = _build_parser(defaults)
    args = parser.parse_args()
    return apply_namespace_overrides(defaults, args)


def _set_seed(seed: int) -> None:
    """Set Python and torch RNG seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    """Load a checkpoint payload from disk."""
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict")
    return payload


def _infer_action_dim_from_checkpoint(ckpt: dict[str, object]) -> int:
    """Infer action feature size from saved action-encoder state dict."""
    state_dict = ckpt.get("action_encoder_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing action_encoder_state_dict")

    for key in ("net.0.weight", "net.1.weight"):
        value = state_dict.get(key)
        if torch.is_tensor(value):
            if key == "net.0.weight":
                return int(value.shape[0])
            return int(value.shape[1])

    raise ValueError("Unable to infer action_dim from checkpoint action_encoder_state_dict")


def _load_video_clip(video_path: str | Path, start_frame: int, total_frames: int) -> torch.Tensor:
    """Load a contiguous clip from a local video as `[1,T,C,H,W]` uint8 tensor."""
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    video = iio.imread(Path(video_path))
    if video.ndim == 3:
        video = video[None, ...]
    if video.ndim != 4:
        raise ValueError(f"Expected video array [T,H,W,C], got shape {tuple(video.shape)}")

    num_frames = int(video.shape[0])
    end_frame = start_frame + total_frames
    if end_frame > num_frames:
        raise ValueError(
            f"Requested frames [{start_frame}:{end_frame}] exceed video length {num_frames}. "
            "Reduce --start-frame or use smaller context/horizon."
        )

    clip = video[start_frame:end_frame]
    if clip.shape[-1] == 4:
        clip = clip[..., :3]
    if clip.shape[-1] != 3:
        raise ValueError(f"Expected RGB video with C=3, got shape {tuple(clip.shape)}")

    clip_np = np.ascontiguousarray(clip)
    clip_t = torch.from_numpy(clip_np)
    if clip_t.dtype != torch.uint8:
        clip_t = clip_t.float()
    clip_t = clip_t.permute(0, 3, 1, 2).unsqueeze(0)
    return clip_t


def _load_action_tensor(
    *,
    action_path: str,
    action_dim: int,
    action_value: float,
    device: torch.device,
) -> torch.Tensor:
    """Load action conditioning tensor as `[1,A]` or `[1,T,A]`."""
    if action_path:
        action_np = np.load(Path(action_path))
        action_t = torch.from_numpy(np.ascontiguousarray(action_np))
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        elif action_t.ndim == 2:
            action_t = action_t.unsqueeze(0)
        elif action_t.ndim == 3 and action_t.shape[0] == 1:
            pass
        else:
            raise ValueError(
                f"Unsupported action array shape {tuple(action_t.shape)}; expected [A], [T,A], or [1,T,A]."
            )
        return action_t.to(device=device, dtype=torch.float32)

    if action_dim <= 0:
        raise ValueError("action_dim must be positive when --action-path is not provided")
    return torch.full((1, action_dim), fill_value=action_value, dtype=torch.float32, device=device)


def _prepare_from_local_video(
    *,
    cfg: InferScriptConfig,
    ckpt: dict[str, object] | None,
    vae: WanVAE,
    device: torch.device,
) -> Any:
    """Build a prepared batch from a local video clip plus action conditioning."""
    total_frames = cfg.context_len + cfg.horizon_len
    video_btchw = _load_video_clip(cfg.video_path, cfg.start_frame, total_frames).to(device)

    if cfg.action_dim > 0:
        action_dim = cfg.action_dim
    elif ckpt is not None:
        action_dim = _infer_action_dim_from_checkpoint(ckpt)
    else:
        action_dim = 0
    action = _load_action_tensor(
        action_path=cfg.action_path,
        action_dim=action_dim,
        action_value=cfg.action_value,
        device=device,
    )
    batch = {
        cfg.video_key: video_btchw,
        "action": action,
    }
    return prepare_packed_batch(
        batch=batch,
        encoder=vae,
        device=device,
        video_key=cfg.video_key,
        context_len=cfg.context_len,
        horizon_len=cfg.horizon_len,
        proprio_mode="last",
    )


def build_runtime_modules(
    *,
    cfg: InferScriptConfig,
    prepared: PreparedPackedBatch,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[WanVACEWorldModel, ActionTokenEncoder, None]:
    """Build Wan VACE runtime modules and optionally overlay a local fine-tune checkpoint."""
    return build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=device,
        checkpoint=checkpoint,
    )


def _save_grid(
    *,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    output_path: Path,
    num_frames: int,
) -> None:
    """Save a two-row ground-truth vs generated frame grid to disk."""
    pred_frames = pred_video[0].detach().cpu()
    target_frames = target_video[0].detach().cpu()
    vis_frames = min(num_frames, pred_frames.shape[0], target_frames.shape[0])
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
    draw.text((16, frame_h // 2), "Ground-truth", fill=(30, 30, 30))
    draw.text((24, frame_h + gap + frame_h // 2), "Generated", fill=(30, 30, 30))

    for idx in range(vis_frames):
        gt = (target_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        pred = (pred_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        canvas.paste(Image.fromarray(gt), (margin + idx * frame_w, 0))
        canvas.paste(Image.fromarray(pred), (margin + idx * frame_w, frame_h + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def main() -> None:
    """Run chunkwise autoregressive inference from pretrained Wan VACE weights."""
    cfg = _load_args()
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _load_checkpoint(cfg.checkpoint, device=device) if cfg.checkpoint else None

    vae = WanVAE.from_pretrained(device=device, deterministic=True)
    if cfg.video_path:
        prepared = _prepare_from_local_video(
            cfg=cfg,
            ckpt=ckpt,
            vae=vae,
            device=device,
        )
    else:
        loader = build_lerobot_dataloader(
            repo_id=cfg.repo_id,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            dt=cfg.dt,
            batch_size=cfg.batch_size,
            subset_size=cfg.subset_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        batch = next(iter(loader))
        prepared = prepare_packed_batch(
            batch=batch,
            encoder=vae,
            device=device,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            proprio_mode="last",
        )

    model, action_encoder, proprio_encoder = build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=device,
        checkpoint=ckpt,
    )

    model.eval()
    action_encoder.eval()
    if proprio_encoder is not None:
        proprio_encoder.eval()

    action_tokens = action_encoder(prepared.a_plan)
    pred_future_video = infer_future_videos_chunkwise(
        model,
        z_past_video=prepared.z_past_video,
        future_steps=prepared.z_future_video.shape[2],
        action_tokens=action_tokens,
        k=cfg.k,
        integration_steps=cfg.integration_steps,
    )

    pred_video = vae.decode(pred_future_video, output_layout="BTCHW", output_range="zero_to_one")
    target_video = vae.decode(prepared.z_future_video, output_layout="BTCHW", output_range="zero_to_one")

    grid_path = output_dir / "comparison_grid.png"
    _save_grid(
        pred_video=pred_video,
        target_video=target_video,
        output_path=grid_path,
        num_frames=cfg.num_vis_frames,
    )
    print(f"Saved comparison grid: {grid_path}")


if __name__ == "__main__":
    main()
