"""Run Wan VACE world-model inference and export GT-vs-generated grids.

This entrypoint uses typed YAML-backed config plus CLI overrides and shared
batch preparation utilities.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
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

from world_model.chunking import normalize_chunk_schedule_mode
from world_model.config import InferScriptConfig, apply_namespace_overrides, load_infer_config
from world_model.data import build_lerobot_dataloader, prepare_packed_batch, preprocess_video_for_vae
from world_model.data.schema import PreparedPackedBatch
from world_model.eval.artifacts import (
    build_frame_report as _build_frame_report,
    build_sharpness_report as _build_sharpness_report,
    resample_video_time as _resample_video_time,
    resolve_visualized_frame_count as _resolve_visualized_frame_count,
    save_grid as _save_grid,
    save_json_report as _save_json_report,
    save_strip as _save_strip,
    select_runtime_dtype as _select_runtime_dtype,
    to_zero_one as _to_zero_one,
)
from world_model.eval import infer_future_videos_chunkwise
from world_model.latents import WanVAE
from world_model.models.wan_vace_conditioning import (
    ActionTokenEncoder,
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
        choices=("k_chunks",),
        default=defaults.chunk_schedule_mode,
        help="Interpret k as exactly K total chunks during rollout.",
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
    parser.add_argument(
        "--future-control-fill-mode",
        choices=("gray", "last_context_frame"),
        default=defaults.future_control_fill_mode,
        help="Fill masked future VACE control slots with gray templates or the last observed latent frame.",
    )
    parser.add_argument("--mask-channels", type=int, default=defaults.mask_channels)
    parser.add_argument("--trainable-backbone", choices=("full", "vace", "head", "lora"), default=defaults.trainable_backbone)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--lora-target-modules", nargs="+", default=list(defaults.lora_target_modules))
    parser.add_argument("--frame-height", type=int, default=defaults.frame_height, help="resize frames to this height before VAE encoding (0=no resize)")
    parser.add_argument("--frame-width", type=int, default=defaults.frame_width, help="resize frames to this width before VAE encoding (0=no resize)")
    parser.add_argument(
        "--future-latent-residual-mode",
        choices=("none", "last_context_frame"),
        default=defaults.future_latent_residual_mode,
        help="Optionally sample future latents in residual coordinates relative to the last observed latent frame.",
    )
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
        "--action-output-zero-init",
        action="store_true",
        default=defaults.action_output_zero_init,
        help="Start fresh action conditioning as an exact no-op by zero-initializing the final token projection.",
    )
    parser.add_argument(
        "--no-action-output-zero-init",
        dest="action_output_zero_init",
        action="store_false",
        help="Disable zero-init on the final action-token projection and inject learned action tokens immediately.",
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
    cfg = apply_namespace_overrides(defaults, args)
    return replace(
        cfg,
        chunk_schedule_mode=normalize_chunk_schedule_mode(cfg.chunk_schedule_mode),
    )


def _validate_infer_config(cfg: InferScriptConfig) -> None:
    """Reject inference configurations that cannot produce meaningful outputs."""
    if cfg.num_vis_frames < 0:
        raise ValueError(f"num_vis_frames must be >= 0, got {cfg.num_vis_frames}")
    normalize_chunk_schedule_mode(cfg.chunk_schedule_mode)
    if cfg.future_control_fill_mode not in {"gray", "last_context_frame"}:
        raise ValueError(
            "future_control_fill_mode must be 'gray' or 'last_context_frame', got "
            f"{cfg.future_control_fill_mode!r}"
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
        "future_control_fill_mode",
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
        "action_token_latent_aux_loss_scale",
        "action_temporal_difference_scale",
        "action_temporal_mixer_kernel_size",
        "action_temporal_mixer_scale",
        "action_token_scale",
        "action_output_zero_init",
        "chunk_schedule_mode",
        "future_latent_residual_mode",
    )
    updates: dict[str, Any] = {}
    for key in update_keys:
        if key not in saved_cfg or getattr(cfg, key) != getattr(defaults, key):
            continue
        value = saved_cfg[key]
        if key in {"vace_layers", "lora_target_modules"}:
            value = tuple(value)
        if key == "chunk_schedule_mode":
            value = normalize_chunk_schedule_mode(value)
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
            if value.ndim == 1:
                return int(value.shape[0])
            return int(value.shape[-1])

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


def build_runtime_modules(
    *,
    cfg: InferScriptConfig,
    prepared: PreparedPackedBatch,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[
    WanVACEWorldModel,
    ActionTokenEncoder | NullConditioningEncoder,
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


def _save_frame_report(report: dict[str, object], output_path: Path) -> None:
    """Persist a JSON report describing raw/latent/decoded frame counts."""
    _save_json_report(report, output_path)


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
    past_video_latents: torch.Tensor,
    pred_future_video: torch.Tensor,
    target_future_video: torch.Tensor,
    context_len: int,
    future_frame_count: int,
    device: torch.device,
    disable_amp: bool,
    runtime_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode future latents with their context so Wan reconstructs the full future horizon."""
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")
    if future_frame_count <= 0:
        raise ValueError(f"future_frame_count must be positive, got {future_frame_count}")

    pred_full_latents = torch.cat([past_video_latents, pred_future_video], dim=2)
    target_full_latents = torch.cat([past_video_latents, target_future_video], dim=2)
    decode_device = device
    decode_dtype = runtime_dtype
    if device.type == "cuda":
        vae.vae.to(device="cpu", dtype=torch.float32)
        pred_full_latents = pred_full_latents.to("cpu")
        target_full_latents = target_full_latents.to("cpu")
        torch.cuda.empty_cache()
        decode_device = torch.device("cpu")
        decode_dtype = torch.float32

    with _autocast_context(device=decode_device, disable_amp=disable_amp, dtype=decode_dtype):
        pred_video = vae.decode(pred_full_latents, output_layout="BTCHW", output_range="zero_to_one")
        target_video = vae.decode(target_full_latents, output_layout="BTCHW", output_range="zero_to_one")

    future_start = context_len
    future_end = context_len + future_frame_count
    if pred_video.shape[1] < future_end or target_video.shape[1] < future_end:
        raise ValueError(
            "Decoded full video is shorter than the requested future slice: "
            f"pred_frames={int(pred_video.shape[1])}, target_frames={int(target_video.shape[1])}, "
            f"future_end={future_end}."
        )
    return pred_video[:, future_start:future_end], target_video[:, future_start:future_end]


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
    model, action_encoder = build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=device,
        checkpoint=ckpt,
    )
    if device.type == "cuda" and not cfg.disable_amp:
        model = model.to(device=device, dtype=runtime_dtype)
        action_encoder = action_encoder.to(device=device, dtype=runtime_dtype)
    scheduler = _load_flow_match_scheduler(cfg)

    model.eval()
    action_encoder.eval()

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
        image_attention_tokens = None
    elif cfg.conditioning_mode == "action":
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens = action_encoder(prepared.a_plan)
            image_attention_tokens = (
                cross_attention_tokens
                if cfg.action_backbone_added_kv_mode == "reuse_action_tokens"
                else None
            )
        negative_cross_attention_tokens = None
    else:
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens = action_encoder(prepared.a_plan)
        image_attention_tokens = None
        negative_cross_attention_tokens = None

    with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
        pred_future_video = infer_future_videos_chunkwise(
            model,
            z_past_video=prepared.z_past_video,
            future_steps=prepared.z_future_video.shape[2],
            cross_attention_tokens=cross_attention_tokens,
            image_attention_tokens=image_attention_tokens,
            k=cfg.k,
            chunk_schedule_mode=cfg.chunk_schedule_mode,
            integration_steps=cfg.integration_steps,
            future_latent_residual_mode=cfg.future_latent_residual_mode,
            negative_cross_attention_tokens=negative_cross_attention_tokens,
            guidance_scale=cfg.guidance_scale,
            chunk_conditioning=_uses_chunk_conditioning(cfg),
            single_chunk_rollout=cfg.single_chunk_rollout,
            scheduler=scheduler,
        )

    del cross_attention_tokens
    del image_attention_tokens
    del negative_cross_attention_tokens
    _release_sampling_modules(model, action_encoder, device=device)
    pred_video, target_video = _decode_future_videos(
        vae=vae,
        past_video_latents=prepared.z_past_video,
        pred_future_video=pred_future_video,
        target_future_video=prepared.z_future_video,
        context_len=cfg.context_len,
        future_frame_count=cfg.horizon_len,
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
