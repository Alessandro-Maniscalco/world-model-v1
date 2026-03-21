"""Run Wan VACE world-model inference and export GT-vs-generated grids.

This entrypoint uses typed YAML-backed config plus CLI overrides and shared
batch preparation utilities.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import os
import random
from pathlib import Path
import sys
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_vace import prompt_clean
from transformers import AutoTokenizer, UMT5EncoderModel

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
from world_model.data import build_lerobot_dataloader, prepare_packed_batch, preprocess_video_for_vae
from world_model.data.schema import PreparedPackedBatch
from world_model.eval import infer_future_videos_chunkwise
from world_model.latents import WanVAE
from world_model.models.wan_vace_conditioning import (
    ActionControlProjector,
    ActionTokenEncoder,
    NullActionControlProjector,
    NullConditioningEncoder,
)
from world_model.models.wan_vace_factory import build_wan_vace_runtime_modules
from world_model.models.wan_vace_world_model import WanVACEWorldModel


def _config_parser() -> argparse.ArgumentParser:
    """Create parser for bootstrap config argument."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path; defaults to configs/eval/infer_world_model.yaml.",
    )
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
    parser.add_argument(
        "--chunk-schedule-mode",
        choices=("k_plus_one", "k_chunks"),
        default=defaults.chunk_schedule_mode,
        help="Interpret k as K+1 total chunks or exactly K total chunks during rollout.",
    )
    parser.add_argument("--integration-steps", type=int, default=defaults.integration_steps)
    parser.add_argument(
        "--num-vis-frames",
        type=int,
        default=defaults.num_vis_frames,
        help="Maximum frames to render per output grid; 0 shows all available frames.",
    )
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
    parser.add_argument("--trainable-backbone", choices=("full", "vace", "head", "lora"), default=defaults.trainable_backbone)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--lora-target-modules", nargs="+", default=list(defaults.lora_target_modules))
    parser.add_argument("--frame-height", type=int, default=defaults.frame_height, help="resize frames to this height before VAE encoding (0=no resize)")
    parser.add_argument("--frame-width", type=int, default=defaults.frame_width, help="resize frames to this width before VAE encoding (0=no resize)")
    parser.add_argument(
        "--conditioning-mode",
        choices=("none", "action", "prompt"),
        default=defaults.conditioning_mode,
        help="Use null tokens, action-plan tokens, or prompt tokens for cross-attention conditioning.",
    )
    parser.add_argument(
        "--action-conditioning-window",
        choices=("chunk", "full"),
        default=defaults.action_conditioning_window,
        help="Use only the active action chunk or the full future plan on every denoising call.",
    )
    parser.add_argument(
        "--action-input-layernorm",
        action="store_true",
        default=defaults.action_input_layernorm,
        help="Normalize each action token with LayerNorm before projection.",
    )
    parser.add_argument(
        "--no-action-input-layernorm",
        dest="action_input_layernorm",
        action="store_false",
        help="Disable action-token input LayerNorm to preserve action magnitude information.",
    )
    parser.add_argument(
        "--action-mlp-dim",
        type=int,
        default=defaults.action_mlp_dim,
        help="Optional hidden width for a two-layer action-token MLP (0 keeps the legacy single linear projection).",
    )
    parser.add_argument(
        "--action-mlp-residual",
        action="store_true",
        default=defaults.action_mlp_residual,
        help="Add the optional action-token MLP as a residual path on top of the legacy linear projection.",
    )
    parser.add_argument(
        "--no-action-mlp-residual",
        dest="action_mlp_residual",
        action="store_false",
        help="Use the optional action-token MLP as a replacement path instead of a residual augmentation.",
    )
    parser.add_argument(
        "--action-order-conditioning",
        action="store_true",
        default=defaults.action_order_conditioning,
        help="Add learned continuous position features to action tokens before temporal mixing.",
    )
    parser.add_argument(
        "--no-action-order-conditioning",
        dest="action_order_conditioning",
        action="store_false",
        help="Disable learned action-order features and keep order-unaware token projections.",
    )
    parser.add_argument(
        "--action-backbone-added-kv-mode",
        choices=("none", "reuse_action_tokens"),
        default=defaults.action_backbone_added_kv_mode,
        help="Optionally mirror action tokens into Wan's added-K/V image-conditioning path.",
    )
    parser.add_argument(
        "--action-control-prior-scale",
        type=float,
        default=defaults.action_control_prior_scale,
        help="Scale for the action-derived latent control prior added to future VACE filler latents.",
    )
    parser.add_argument(
        "--action-control-prior-mode",
        choices=("reactive_only", "dual_fill"),
        default=defaults.action_control_prior_mode,
        help="Inject the action-derived latent control prior into only the reactive future branch or both future control branches.",
    )
    parser.add_argument(
        "--action-control-projector-init-mode",
        choices=("zero", "linear_default"),
        default=defaults.action_control_projector_init_mode,
        help="Initialization mode for the action-to-latent control projector when no projector weights are available in a checkpoint.",
    )
    parser.add_argument(
        "--action-control-projector-observed-context-mode",
        choices=("none", "last_frame"),
        default=defaults.action_control_projector_observed_context_mode,
        help="Optional observed-latent context pooled into the action-control projector before future broadcast.",
    )
    parser.add_argument(
        "--action-hidden-state-bias-scale",
        type=float,
        default=defaults.action_hidden_state_bias_scale,
        help="Scale for adding the action-derived latent control signal directly to future latent hidden states before the Wan backbone.",
    )
    parser.add_argument(
        "--action-temporal-difference-scale",
        type=float,
        default=defaults.action_temporal_difference_scale,
        help="Optional residual scale for projecting step-to-step action deltas alongside the raw action plan.",
    )
    parser.add_argument(
        "--action-temporal-mixer-kernel-size",
        type=int,
        default=defaults.action_temporal_mixer_kernel_size,
        help="Optional odd depthwise temporal kernel over projected action tokens (0 disables).",
    )
    parser.add_argument(
        "--action-temporal-mixer-scale",
        type=float,
        default=defaults.action_temporal_mixer_scale,
        help="Residual scale for the optional temporal action-token mixer.",
    )
    parser.add_argument(
        "--action-token-scale",
        type=float,
        default=defaults.action_token_scale,
        help="Final gain applied to projected action tokens before Wan cross-attention.",
    )
    parser.add_argument(
        "--prompt",
        default=defaults.prompt,
        help="Prompt text used when --conditioning-mode prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=defaults.negative_prompt,
        help="Negative prompt used for classifier-free guidance when --conditioning-mode prompt.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=defaults.guidance_scale,
        help="Classifier-free guidance scale for prompt conditioning.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=defaults.max_sequence_length,
        help="Maximum tokenizer sequence length for prompt conditioning.",
    )
    parser.add_argument(
        "--single-chunk-rollout",
        action="store_true",
        default=defaults.single_chunk_rollout,
        help="Roll out the entire future window as one chunk while keeping the chunked inference path.",
    )
    parser.add_argument(
        "--multi-chunk-rollout",
        dest="single_chunk_rollout",
        action="store_false",
    )
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


def _validate_infer_config(cfg: InferScriptConfig) -> None:
    """Reject inference configurations that cannot produce meaningful outputs."""
    if cfg.num_vis_frames < 0:
        raise ValueError(f"num_vis_frames must be >= 0, got {cfg.num_vis_frames}")
    if cfg.chunk_schedule_mode not in {"k_plus_one", "k_chunks"}:
        raise ValueError(
            "chunk_schedule_mode must be 'k_plus_one' or 'k_chunks', got "
            f"{cfg.chunk_schedule_mode!r}"
        )
    if cfg.action_control_prior_scale < 0.0:
        raise ValueError(
            f"action_control_prior_scale must be >= 0, got {cfg.action_control_prior_scale}"
        )
    if cfg.action_control_prior_mode not in {"reactive_only", "dual_fill"}:
        raise ValueError(
            "action_control_prior_mode must be 'reactive_only' or 'dual_fill', got "
            f"{cfg.action_control_prior_mode!r}"
        )
    if cfg.action_hidden_state_bias_scale < 0.0:
        raise ValueError(
            f"action_hidden_state_bias_scale must be >= 0, got {cfg.action_hidden_state_bias_scale}"
        )
    if cfg.action_control_projector_observed_context_mode not in {"none", "last_frame"}:
        raise ValueError(
            "action_control_projector_observed_context_mode must be 'none' or 'last_frame', got "
            f"{cfg.action_control_projector_observed_context_mode!r}"
        )
    if cfg.action_backbone_added_kv_mode not in {"none", "reuse_action_tokens"}:
        raise ValueError(
            "action_backbone_added_kv_mode must be 'none' or 'reuse_action_tokens', got "
            f"{cfg.action_backbone_added_kv_mode!r}"
        )
    if cfg.action_token_scale < 0.0:
        raise ValueError(f"action_token_scale must be >= 0, got {cfg.action_token_scale}")
    if cfg.conditioning_mode == "action" and not cfg.checkpoint:
        raise ValueError(
            "Action conditioning requires --checkpoint because the action encoder is random otherwise. "
            "Use --conditioning-mode prompt for pretrained-backbone smoke tests."
        )


def _resolve_effective_infer_config(cfg: InferScriptConfig) -> InferScriptConfig:
    """Promote checkpoint-free prompt smoke tests to a more VACE-like sampling setup."""
    if cfg.checkpoint or cfg.conditioning_mode not in ("none", "prompt"):
        return cfg

    resolved = cfg
    if not resolved.single_chunk_rollout:
        resolved = replace(resolved, single_chunk_rollout=True)
    if resolved.integration_steps < 50:
        resolved = replace(resolved, integration_steps=50)
    return resolved


def _restore_runtime_config_from_checkpoint(cfg: InferScriptConfig, ckpt: dict[str, object] | None) -> InferScriptConfig:
    """Adopt saved runtime settings from a checkpoint when the current config still uses defaults."""
    if ckpt is None:
        return cfg
    extra_state = ckpt.get("extra_state")
    if not isinstance(extra_state, dict):
        return cfg
    saved_cfg = extra_state.get("config")
    if not isinstance(saved_cfg, dict):
        return cfg

    defaults = InferScriptConfig()
    update_keys = (
        "trainable_backbone",
        "conditioning_mode",
        "control_scale",
        "mask_channels",
        "vace_layers",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "action_input_layernorm",
        "action_mlp_dim",
        "action_mlp_residual",
        "action_conditioning_window",
        "action_order_conditioning",
        "action_backbone_added_kv_mode",
        "action_control_prior_scale",
        "action_control_prior_mode",
        "action_hidden_state_bias_scale",
        "action_control_projector_observed_context_mode",
        "action_token_latent_aux_loss_scale",
        "action_temporal_difference_scale",
        "action_temporal_mixer_kernel_size",
        "action_temporal_mixer_scale",
        "action_token_scale",
        "chunk_schedule_mode",
    )
    updates: dict[str, Any] = {}
    for key in update_keys:
        if key not in saved_cfg or getattr(cfg, key) != getattr(defaults, key):
            continue
        value = saved_cfg[key]
        if key in {"vace_layers", "lora_target_modules"}:
            value = tuple(value)
        updates[key] = value
    if not updates:
        return cfg
    return replace(cfg, **updates)


def _set_seed(seed: int) -> None:
    """Set Python and torch RNG seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_runtime_dtype(*, device: torch.device, disable_amp: bool) -> torch.dtype:
    """Choose an inference dtype that fits the active device and AMP setting."""
    if device.type != "cuda" or disable_amp:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(*, device: torch.device, disable_amp: bool, dtype: torch.dtype):
    """Build the appropriate autocast context for inference on the active device."""
    if device.type != "cuda" or disable_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


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
) -> tuple[PreparedPackedBatch, torch.Tensor]:
    """Build a prepared batch from a local clip and return the source video tensor."""
    total_frames = cfg.context_len + cfg.horizon_len
    video_btchw = _load_video_clip(cfg.video_path, cfg.start_frame, total_frames).to(device)
    video_btchw = preprocess_video_for_vae(
        video_btchw,
        frame_height=cfg.frame_height,
        frame_width=cfg.frame_width,
    )

    if cfg.conditioning_mode in ("none", "prompt"):
        action_dim = max(int(cfg.action_dim), 1)
    elif cfg.action_dim > 0:
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
    prepared = prepare_packed_batch(
        batch=batch,
        encoder=vae,
        device=device,
        video_key=cfg.video_key,
        context_len=cfg.context_len,
        horizon_len=cfg.horizon_len,
        frame_height=cfg.frame_height,
        frame_width=cfg.frame_width,
    )
    return prepared, video_btchw


def _to_zero_one(video_btchw: torch.Tensor) -> torch.Tensor:
    """Convert video tensor to float `[0,1]` while preserving `BTCHW` layout."""
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


def _resample_video_time(video_btchw: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Nearest-resample `BTCHW` video to `target_steps` frames."""
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


def build_runtime_modules(
    *,
    cfg: InferScriptConfig,
    prepared: PreparedPackedBatch,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[
    WanVACEWorldModel,
    ActionTokenEncoder | NullConditioningEncoder,
    ActionControlProjector | NullActionControlProjector,
]:
    """Build Wan VACE runtime modules and optionally overlay a local fine-tune checkpoint."""
    return build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=device,
        checkpoint=checkpoint,
    )


def _load_flow_match_scheduler(cfg: InferScriptConfig) -> FlowMatchEulerDiscreteScheduler:
    """Load the upstream Wan flow-matching scheduler config."""
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.wan_vace_model_id,
        subfolder="scheduler",
        local_files_only=_offline_mode_enabled(),
    )


def _load_prompt_encoder(cfg: InferScriptConfig) -> tuple[Any, UMT5EncoderModel]:
    """Load the upstream tokenizer and UMT5 text encoder for prompt conditioning."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.wan_vace_model_id,
        subfolder="tokenizer",
        local_files_only=_offline_mode_enabled(),
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.wan_vace_model_id,
        subfolder="text_encoder",
        local_files_only=_offline_mode_enabled(),
    )
    text_encoder.eval()
    return tokenizer, text_encoder


@torch.no_grad()
def build_prompt_conditioning_tokens(
    *,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    tokenizer: Any,
    text_encoder: UMT5EncoderModel,
    encoder_device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode upstream-style prompt embeddings for Wan cross-attention."""
    prompt_embeds = _get_t5_prompt_embeds(
        prompt=[prompt] * batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        encoder_device=encoder_device,
        output_device=output_device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
    )
    negative_prompt_embeds = None
    if guidance_scale > 1.0:
        negative_prompt_embeds = _get_t5_prompt_embeds(
            prompt=[negative_prompt] * batch_size,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            encoder_device=encoder_device,
            output_device=output_device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
    return prompt_embeds, negative_prompt_embeds


@torch.no_grad()
def _get_t5_prompt_embeds(
    *,
    prompt: list[str],
    tokenizer: Any,
    text_encoder: UMT5EncoderModel,
    encoder_device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    """Mirror the diffusers Wan prompt-embedding path for inference-time prompts."""
    cleaned_prompt = [prompt_clean(text) for text in prompt]
    text_inputs = tokenizer(
        cleaned_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(encoder_device)
    attention_mask = text_inputs.attention_mask.to(encoder_device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids, attention_mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(device=output_device, dtype=dtype)
    trimmed = [hidden[:seq_len] for hidden, seq_len in zip(prompt_embeds, seq_lens)]
    return torch.stack(
        [torch.cat([hidden, hidden.new_zeros(max_sequence_length - hidden.size(0), hidden.size(1))]) for hidden in trimmed],
        dim=0,
    )


def _offline_mode_enabled() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _uses_chunk_conditioning(cfg: InferScriptConfig) -> bool:
    """Decide whether inference should slice action tokens per rollout chunk."""
    if cfg.conditioning_mode == "prompt":
        return False
    if cfg.conditioning_mode == "action":
        return cfg.action_conditioning_window == "chunk"
    return True


def _save_grid(
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
    vis_frames = _resolve_visualized_frame_count(
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
        gt = (target_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        pred = (pred_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        canvas.paste(Image.fromarray(gt), (margin + idx * frame_w, 0))
        canvas.paste(Image.fromarray(pred), (margin + idx * frame_w, frame_h + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_strip(
    *,
    video: torch.Tensor,
    output_path: Path,
    num_frames: int,
    label: str,
) -> None:
    """Save a one-row frame strip for a single video."""
    frames = video[0].detach().float().cpu()
    vis_frames = _resolve_visualized_frame_count(
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
        frame = (frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        canvas.paste(Image.fromarray(frame), (margin + idx * frame_w, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _resolve_visualized_frame_count(*, requested_frames: int, available_frames: int) -> int:
    """Resolve `0 => all` and clamp frame-visualization requests to availability."""
    if requested_frames < 0:
        raise ValueError(f"requested_frames must be >= 0, got {requested_frames}")
    if available_frames < 0:
        raise ValueError(f"available_frames must be >= 0, got {available_frames}")
    if requested_frames == 0:
        return available_frames
    return min(requested_frames, available_frames)


def _build_frame_report(
    *,
    cfg: InferScriptConfig,
    prepared: PreparedPackedBatch,
    source_video: torch.Tensor,
    raw_future: torch.Tensor,
    raw_future_aligned: torch.Tensor,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
) -> dict[str, object]:
    """Build a compact frame/latent accounting report for saved inference artifacts."""
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
            _resolve_visualized_frame_count(
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


def _save_frame_report(report: dict[str, object], output_path: Path) -> None:
    """Persist a JSON report describing raw/latent/decoded frame counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _build_sharpness_report(
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


def _save_json_report(report: dict[str, object], output_path: Path) -> None:
    """Persist a generic JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _release_sampling_modules(*modules: torch.nn.Module, device: torch.device) -> None:
    """Move completed sampling modules off the active accelerator before VAE decode."""
    if device.type != "cuda":
        return
    for module in modules:
        module.to("cpu")
    torch.cuda.empty_cache()


def _release_vae_after_prepare(vae: WanVAE, *, device: torch.device) -> None:
    """Move the Wan VAE off GPU once latent preparation has finished."""
    if device.type != "cuda":
        return
    vae.vae.to("cpu")
    torch.cuda.empty_cache()


def _decode_future_videos(
    *,
    vae: WanVAE,
    pred_future_video: torch.Tensor,
    target_future_video: torch.Tensor,
    device: torch.device,
    disable_amp: bool,
    runtime_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode future latent videos, falling back to CPU to reduce GPU peak memory."""
    decode_device = device
    decode_dtype = runtime_dtype
    if device.type == "cuda":
        vae.vae.to(device="cpu", dtype=torch.float32)
        pred_future_video = pred_future_video.to("cpu")
        target_future_video = target_future_video.to("cpu")
        torch.cuda.empty_cache()
        decode_device = torch.device("cpu")
        decode_dtype = torch.float32

    with _autocast_context(device=decode_device, disable_amp=disable_amp, dtype=decode_dtype):
        pred_video = vae.decode(pred_future_video, output_layout="BTCHW", output_range="zero_to_one")
        target_video = vae.decode(target_future_video, output_layout="BTCHW", output_range="zero_to_one")
    return pred_video, target_video


@torch.no_grad()
def main() -> None:
    """Run chunkwise autoregressive inference from pretrained Wan VACE weights."""
    cfg = _load_args()
    ckpt = _load_checkpoint(cfg.checkpoint, device=torch.device("cpu")) if cfg.checkpoint else None
    cfg = _restore_runtime_config_from_checkpoint(cfg, ckpt)
    _validate_infer_config(cfg)
    cfg = _resolve_effective_infer_config(cfg)
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_dtype = _select_runtime_dtype(device=device, disable_amp=cfg.disable_amp)

    vae = WanVAE.from_pretrained(
        device=device,
        deterministic=True,
        torch_dtype=runtime_dtype,
    )
    if cfg.video_path:
        prepared, source_video = _prepare_from_local_video(
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
        source_video = batch[cfg.video_key].to(device)
        source_video = preprocess_video_for_vae(
            source_video,
            frame_height=cfg.frame_height,
            frame_width=cfg.frame_width,
        )
        batch = dict(batch)
        batch[cfg.video_key] = source_video
        prepared = prepare_packed_batch(
            batch=batch,
            encoder=vae,
            device=device,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            frame_height=cfg.frame_height,
            frame_width=cfg.frame_width,
        )

    _release_vae_after_prepare(vae, device=device)
    model, action_encoder, action_control_projector = build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=device,
        checkpoint=ckpt,
    )
    if device.type == "cuda" and not cfg.disable_amp:
        model = model.to(device=device, dtype=runtime_dtype)
        action_encoder = action_encoder.to(device=device, dtype=runtime_dtype)
        action_control_projector = action_control_projector.to(device=device, dtype=runtime_dtype)
    scheduler = _load_flow_match_scheduler(cfg)

    model.eval()
    action_encoder.eval()
    action_control_projector.eval()

    if cfg.conditioning_mode == "prompt":
        tokenizer, text_encoder = _load_prompt_encoder(cfg)
        backbone_dtype = next(model.backbone.parameters()).dtype
        prompt_encoder_device = torch.device("cpu")
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens, negative_cross_attention_tokens = build_prompt_conditioning_tokens(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                batch_size=prepared.z_past_video.shape[0],
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                encoder_device=prompt_encoder_device,
                output_device=device,
                dtype=backbone_dtype,
                guidance_scale=cfg.guidance_scale,
                max_sequence_length=cfg.max_sequence_length,
            )
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        future_action_control_prior = None
        image_attention_tokens = None
    elif cfg.conditioning_mode == "action":
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens = action_encoder(prepared.a_plan)
            image_attention_tokens = (
                cross_attention_tokens
                if cfg.action_backbone_added_kv_mode == "reuse_action_tokens"
                else None
            )
            future_action_control_prior = None
            if cfg.action_control_prior_scale > 0.0 or cfg.action_hidden_state_bias_scale > 0.0:
                future_action_control_prior = action_control_projector(
                    prepared.a_plan,
                    latent_height=prepared.z_future_video.shape[3],
                    latent_width=prepared.z_future_video.shape[4],
                    observed_latents=(
                        prepared.z_past_video
                        if cfg.action_control_projector_observed_context_mode != "none"
                        else None
                    ),
                )
        negative_cross_attention_tokens = None
    else:
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens = action_encoder(prepared.a_plan)
        image_attention_tokens = None
        negative_cross_attention_tokens = None
        future_action_control_prior = None

    with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
        pred_future_video = infer_future_videos_chunkwise(
            model,
            z_past_video=prepared.z_past_video,
            future_steps=prepared.z_future_video.shape[2],
            cross_attention_tokens=cross_attention_tokens,
            image_attention_tokens=image_attention_tokens,
            future_action_control_prior=future_action_control_prior,
            k=cfg.k,
            chunk_schedule_mode=cfg.chunk_schedule_mode,
            integration_steps=cfg.integration_steps,
            negative_cross_attention_tokens=negative_cross_attention_tokens,
            guidance_scale=cfg.guidance_scale,
            chunk_conditioning=_uses_chunk_conditioning(cfg),
            single_chunk_rollout=cfg.single_chunk_rollout,
            scheduler=scheduler,
        )

    del cross_attention_tokens
    del image_attention_tokens
    del negative_cross_attention_tokens
    del future_action_control_prior
    _release_sampling_modules(model, action_encoder, action_control_projector, device=device)
    pred_video, target_video = _decode_future_videos(
        vae=vae,
        pred_future_video=pred_future_video,
        target_future_video=prepared.z_future_video,
        device=device,
        disable_amp=cfg.disable_amp,
        runtime_dtype=runtime_dtype,
    )

    grid_path = output_dir / "comparison_grid.png"
    _save_grid(
        pred_video=pred_video,
        target_video=target_video,
        output_path=grid_path,
        num_frames=cfg.num_vis_frames,
        top_label="VAE roundtrip",
        bottom_label="Generated",
    )
    print(f"Saved comparison grid: {grid_path}")

    raw_video = _to_zero_one(source_video)
    raw_future = raw_video[:, cfg.context_len:cfg.context_len + cfg.horizon_len]
    raw_grid_path = output_dir / "raw_future_grid.png"
    _save_strip(
        video=raw_future,
        output_path=raw_grid_path,
        num_frames=cfg.num_vis_frames,
        label="Raw future",
    )
    print(f"Saved raw future grid: {raw_grid_path}")
    raw_future_aligned = _resample_video_time(raw_future, target_video.shape[1])
    vae_grid_path = output_dir / "vae_roundtrip_future_grid.png"
    _save_grid(
        pred_video=target_video,
        target_video=raw_future_aligned,
        output_path=vae_grid_path,
        num_frames=cfg.num_vis_frames,
        top_label="Raw future aligned",
        bottom_label="VAE roundtrip",
    )
    print(f"Saved VAE blur check grid: {vae_grid_path}")
    frame_report = _build_frame_report(
        cfg=cfg,
        prepared=prepared,
        source_video=source_video,
        raw_future=raw_future,
        raw_future_aligned=raw_future_aligned,
        pred_video=pred_video,
        target_video=target_video,
    )
    report_path = output_dir / "frame_report.json"
    _save_frame_report(frame_report, report_path)
    print(f"Saved frame report: {report_path}")
    sharpness_report = _build_sharpness_report(
        raw_future_aligned=raw_future_aligned,
        target_video=target_video,
        pred_video=pred_video,
    )
    sharpness_report_path = output_dir / "sharpness_report.json"
    _save_json_report(sharpness_report, sharpness_report_path)
    print(f"Saved sharpness report: {sharpness_report_path}")
    print(
        "Frame counts: "
        f"requested raw future={frame_report['raw_future_frames']} "
        f"latent future={frame_report['latent_future_steps']} "
        f"decoded future={frame_report['decoded_roundtrip_future_frames']} "
        f"visualized={frame_report['visualized_frames']}"
    )
    print(
        "Sharpness ratios vs VAE roundtrip: "
        f"generated={sharpness_report['relative_to_vae_roundtrip']['generated']:.3f} "
        f"raw={sharpness_report['relative_to_vae_roundtrip']['raw_future_aligned']:.3f}"
    )


if __name__ == "__main__":
    main()
