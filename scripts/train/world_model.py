"""Train the world model with chunkwise teacher-forced flow matching.

This entrypoint uses typed YAML-backed config plus CLI overrides and the
canonical shared data-preparation pipeline.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import itertools
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch
from torch import nn

# Ensure local `src/` package imports work when run as `python scripts/train/world_model.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    # Prevent this file (`world_model.py`) from shadowing the `world_model` package.
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
from world_model.config import TrainScriptConfig, apply_namespace_overrides, load_train_config
from world_model.data import (
    build_lerobot_dataloader,
    load_local_video_clip,
    prepare_packed_batch,
    resolve_lerobot_episode_ids,
    split_train_validation_episode_ids,
)
from world_model.data.schema import PreparedPackedBatch
from world_model.latents import WanVAE
from world_model.models import WanVACEWorldModel
from world_model.models.wan_vace_factory import (
    build_conditioning_encoder_for_model,
    build_wan_vace_model_from_config,
    _load_action_encoder_state_dict,
)
from world_model.models.wan_vace_conditioning import (
    ActionTokenEncoder,
    NullConditioningEncoder,
)
from world_model.training import (
    append_jsonl,
    chunkwise_teacher_forcing_loss,
    save_checkpoint,
    train_chunkwise_batch,
)
from world_model.training.chunkwise_training import (
    _compute_action_token_latent_aux_loss,
)


def _config_parser() -> argparse.ArgumentParser:
    """Create parser for config-file bootstrap args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path; defaults to configs/train/world_model.yaml.",
    )
    return parser


def _build_parser(defaults: TrainScriptConfig) -> argparse.ArgumentParser:
    """Create full CLI parser using dataclass defaults."""
    parser = argparse.ArgumentParser(description=__doc__, parents=[_config_parser()])
    parser.add_argument("--resume-from", default=defaults.resume_from, help="Optional training checkpoint .pt to resume model, action encoder, optimizer, and step state.")
    parser.add_argument("--video-path", default=defaults.video_path, help="Optional local video file for fast single-clip training.")
    parser.add_argument("--start-frame", type=int, default=defaults.start_frame)
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--episodes", type=int, nargs="*", default=list(defaults.episodes))
    parser.add_argument("--video-key", default=defaults.video_key)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--context-len", type=int, default=defaults.context_len, help="frame-time context length (l)")
    parser.add_argument("--horizon-len", type=int, default=defaults.horizon_len, help="frame-time horizon length (H)")
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--k", type=int, default=defaults.k, help="Chunk-count parameter for latent-time scheduling")
    parser.add_argument(
        "--chunk-schedule-mode",
        choices=("k_chunks",),
        default=defaults.chunk_schedule_mode,
        help="Interpret k as exactly K total chunks.",
    )
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--auto-stop-check-every", type=int, default=defaults.auto_stop_check_every)
    parser.add_argument(
        "--auto-stop-min-relative-improvement",
        type=float,
        default=defaults.auto_stop_min_relative_improvement,
    )
    parser.add_argument(
        "--validation-enabled",
        action="store_true",
        default=defaults.validation_enabled,
        help="Enable held-out validation-loss checks during training.",
    )
    parser.add_argument(
        "--no-validation-enabled",
        dest="validation_enabled",
        action="store_false",
        help="Disable held-out validation-loss checks during training.",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        nargs="*",
        default=list(defaults.validation_episodes),
        help="Optional held-out validation episode ids. Defaults to a deterministic tail split.",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=defaults.validation_every,
        help="Run validation every N training steps, independently of checkpoint saves.",
    )
    parser.add_argument(
        "--validation-split-ratio",
        type=float,
        default=defaults.validation_split_ratio,
        help="Default held-out episode ratio when validation episodes are not provided explicitly.",
    )
    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=defaults.validation_max_batches,
        help="Maximum number of validation batches to score at each checkpoint-time validation pass.",
    )
    parser.add_argument(
        "--validation-patience-checks",
        type=int,
        default=defaults.validation_patience_checks,
        help="Number of consecutive non-improving validation checks allowed before stopping; 0 disables validation early stopping.",
    )
    parser.add_argument(
        "--validation-min-relative-improvement",
        type=float,
        default=defaults.validation_min_relative_improvement,
        help="Minimum relative validation-loss improvement required to reset validation patience.",
    )
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    parser.add_argument("--weight-mode", choices=["uniform", "snr", "clipped_snr"], default=defaults.weight_mode)
    parser.add_argument(
        "--motion-loss-alpha",
        type=float,
        default=defaults.motion_loss_alpha,
        help="Extra weight applied to moving latent regions; 0 disables motion-aware weighting.",
    )
    parser.add_argument(
        "--motion-loss-max-weight",
        type=float,
        default=defaults.motion_loss_max_weight,
        help="Optional cap on the per-region motion-aware loss multiplier; 0 disables capping.",
    )
    parser.add_argument(
        "--motion-loss-excess-only",
        action="store_true",
        default=defaults.motion_loss_excess_only,
        help="Only add motion-aware loss bonus above the per-sample mean motion level.",
    )
    parser.add_argument(
        "--no-motion-loss-excess-only",
        dest="motion_loss_excess_only",
        action="store_false",
        help="Apply motion-aware loss bonus to all motion magnitudes, including average regions.",
    )
    parser.add_argument(
        "--future-loss-early-bias",
        type=float,
        default=defaults.future_loss_early_bias,
        help="Optional linear loss bonus for earlier future timesteps; 0 disables temporal early bias.",
    )
    parser.add_argument(
        "--future-chunk-early-bias",
        type=float,
        default=defaults.future_chunk_early_bias,
        help="Optional linear loss bonus for earlier autoregressive chunks; 0 disables chunk-position bias.",
    )
    parser.add_argument(
        "--future-latent-residual-mode",
        choices=("none", "last_context_frame"),
        default=defaults.future_latent_residual_mode,
        help="Optionally predict future latents relative to the last observed latent frame instead of absolute latents.",
    )
    parser.add_argument(
        "--teacher-forcing-observation-mode",
        choices=("full_prefix", "past_only", "predicted_prefix"),
        default=defaults.teacher_forcing_observation_mode,
        help="Whether later teacher-forced chunks observe the true future prefix or only the true past.",
    )
    parser.add_argument(
        "--teacher-forcing-future-input-mode",
        choices=("full_suffix", "active_chunk"),
        default=defaults.teacher_forcing_future_input_mode,
        help="Whether teacher forcing denoises the full future suffix or only the active chunk to match rollout.",
    )
    parser.add_argument("--t-min", type=float, default=defaults.t_min)
    parser.add_argument("--t-max", type=float, default=defaults.t_max)
    parser.add_argument("--disable-amp", action="store_true", default=defaults.disable_amp)
    parser.add_argument("--enable-amp", dest="disable_amp", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=defaults.gradient_checkpointing)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
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
    parser.add_argument("--conditioning-mode", choices=("none", "action"), default=defaults.conditioning_mode)
    parser.add_argument(
        "--action-conditioning-window",
        choices=("chunk", "full"),
        default=defaults.action_conditioning_window,
        help="Use only the active chunk's action tokens or the full future plan on every denoising call.",
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
        "--action-token-latent-aux-loss-scale",
        type=float,
        default=defaults.action_token_latent_aux_loss_scale,
        help="Train-only auxiliary loss scale for matching projected action tokens to per-step clean future latent summaries.",
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
    parser.add_argument("--frame-height", type=int, default=defaults.frame_height, help="resize frames to this height before VAE encoding (0=no resize)")
    parser.add_argument("--frame-width", type=int, default=defaults.frame_width, help="resize frames to this width before VAE encoding (0=no resize)")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--checkpoint-early-every", type=int, default=defaults.checkpoint_early_every)
    parser.add_argument("--checkpoint-early-until", type=int, default=defaults.checkpoint_early_until)
    parser.add_argument("--subset-size", type=int, default=defaults.subset_size, help="0 uses full dataset")
    parser.add_argument("--overfit-one-batch", action="store_true", default=defaults.overfit_one_batch)
    parser.add_argument("--no-overfit-one-batch", dest="overfit_one_batch", action="store_false")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _load_args() -> TrainScriptConfig:
    """Load YAML config and apply CLI overrides into a final train config."""
    config_args, _ = _config_parser().parse_known_args()
    defaults = load_train_config(config_args.config)
    parser = _build_parser(defaults)
    args = parser.parse_args()
    cfg = apply_namespace_overrides(defaults, args)
    return replace(
        cfg,
        chunk_schedule_mode=normalize_chunk_schedule_mode(cfg.chunk_schedule_mode),
    )


def _set_seed(seed: int) -> None:
    """Set Python and torch RNG seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_runtime_dtype(*, device: torch.device, disable_amp: bool) -> torch.dtype:
    """Choose the mixed-precision dtype for training on the active device."""
    if device.type != "cuda" or disable_amp:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _training_autocast_context(*, device: torch.device, disable_amp: bool, dtype: torch.dtype):
    """Build the autocast context used by eval-mode loss checks."""
    if device.type != "cuda" or disable_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _validate_chunk_schedule(cfg: TrainScriptConfig, prepared_batch: PreparedPackedBatch) -> None:
    """Fail fast when latent-time chunking cannot satisfy the configured schedule."""
    chunk_schedule_mode = normalize_chunk_schedule_mode(cfg.chunk_schedule_mode)
    min_future_steps = cfg.k
    future_steps = prepared_batch.horizon_latent_steps
    if future_steps < min_future_steps:
        raise ValueError(
            "Invalid latent-time schedule: "
            f"raw horizon_len={cfg.horizon_len} compressed to horizon_latent_steps={future_steps}, "
            f"but k={cfg.k} with chunk_schedule_mode={chunk_schedule_mode!r} "
            f"requires at least {min_future_steps} latent future steps. "
            "Increase horizon_len or reduce k."
        )


def _validate_auto_stop_config(cfg: TrainScriptConfig) -> None:
    """Reject inconsistent blockwise continuation settings before entering the train loop."""
    if cfg.auto_stop_check_every < 0:
        raise ValueError(f"auto_stop_check_every must be >= 0, got {cfg.auto_stop_check_every}.")
    if cfg.auto_stop_min_relative_improvement < 0.0:
        raise ValueError(
            "auto_stop_min_relative_improvement must be >= 0, got "
            f"{cfg.auto_stop_min_relative_improvement}."
        )
    if cfg.checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be >= 0, got {cfg.checkpoint_every}.")
    if cfg.checkpoint_early_every < 0:
        raise ValueError(f"checkpoint_early_every must be >= 0, got {cfg.checkpoint_early_every}.")
    if cfg.checkpoint_early_until < 0:
        raise ValueError(f"checkpoint_early_until must be >= 0, got {cfg.checkpoint_early_until}.")
    if cfg.validation_every <= 0:
        raise ValueError(f"validation_every must be positive, got {cfg.validation_every}.")
    if cfg.validation_max_batches <= 0:
        raise ValueError(
            f"validation_max_batches must be positive, got {cfg.validation_max_batches}."
        )
    if cfg.validation_patience_checks < 0:
        raise ValueError(
            "validation_patience_checks must be >= 0, got "
            f"{cfg.validation_patience_checks}."
        )
    if cfg.validation_min_relative_improvement < 0.0:
        raise ValueError(
            "validation_min_relative_improvement must be >= 0, got "
            f"{cfg.validation_min_relative_improvement}."
        )
    if cfg.future_loss_early_bias < 0.0:
        raise ValueError(
            "future_loss_early_bias must be >= 0, got "
            f"{cfg.future_loss_early_bias}."
        )
    if cfg.future_chunk_early_bias < 0.0:
        raise ValueError(
            "future_chunk_early_bias must be >= 0, got "
            f"{cfg.future_chunk_early_bias}."
        )
    if cfg.future_latent_residual_mode not in {"none", "last_context_frame"}:
        raise ValueError(
            "future_latent_residual_mode must be 'none' or 'last_context_frame', got "
            f"{cfg.future_latent_residual_mode!r}."
        )
    if cfg.future_control_fill_mode not in {"gray", "last_context_frame"}:
        raise ValueError(
            "future_control_fill_mode must be 'gray' or 'last_context_frame', got "
            f"{cfg.future_control_fill_mode!r}."
        )
    if cfg.teacher_forcing_observation_mode not in {
        "full_prefix",
        "past_only",
        "predicted_prefix",
    }:
        raise ValueError(
            "teacher_forcing_observation_mode must be 'full_prefix', 'past_only', or "
            f"'predicted_prefix', got {cfg.teacher_forcing_observation_mode!r}."
        )
    if cfg.teacher_forcing_future_input_mode not in {"full_suffix", "active_chunk"}:
        raise ValueError(
            "teacher_forcing_future_input_mode must be 'full_suffix' or "
            f"'active_chunk', got {cfg.teacher_forcing_future_input_mode!r}."
        )
    normalize_chunk_schedule_mode(cfg.chunk_schedule_mode)
    if cfg.action_backbone_added_kv_mode not in {"none", "reuse_action_tokens"}:
        raise ValueError(
            "action_backbone_added_kv_mode must be 'none' or 'reuse_action_tokens', got "
            f"{cfg.action_backbone_added_kv_mode!r}."
        )
    if cfg.action_token_latent_aux_loss_scale < 0.0:
        raise ValueError(
            "action_token_latent_aux_loss_scale must be >= 0, got "
            f"{cfg.action_token_latent_aux_loss_scale}."
        )
    if cfg.action_token_scale < 0.0:
        raise ValueError(f"action_token_scale must be >= 0, got {cfg.action_token_scale}.")
    if cfg.validation_enabled and cfg.video_path:
        raise ValueError("Validation loss is not supported for local-video training in v1.")


def _format_episode_preview(episode_ids: list[int]) -> str:
    """Render a compact, stable episode-id preview for stdout."""
    if not episode_ids:
        return "[]"
    if len(episode_ids) <= 8:
        return str(episode_ids)
    return f"{episode_ids[:4]} ... {episode_ids[-4:]}"


def _relative_block_improvement(*, previous_mean_loss: float, current_mean_loss: float) -> float:
    """Compute a bounded relative mean-loss change between consecutive training blocks.

    Loss reductions keep the usual `(previous - current) / previous` value so
    improvement thresholds behave exactly as before. Regressions divide by the
    larger of the two losses so the reported negative change stays within
    `[-1, 0)` instead of growing past `-1` when the current loss is much larger
    than the previous best.
    """
    reference_loss = max(previous_mean_loss, current_mean_loss)
    if reference_loss <= 0.0:
        return 0.0
    return (previous_mean_loss - current_mean_loss) / reference_loss


def _update_validation_tracking(
    *,
    best_val_loss: float | None,
    current_val_loss: float,
    bad_checks: int,
    min_relative_improvement: float,
) -> tuple[float, int, float | None]:
    """Update best validation loss and patience state after one validation check."""
    if best_val_loss is None:
        return current_val_loss, 0, None

    relative_improvement = _relative_block_improvement(
        previous_mean_loss=best_val_loss,
        current_mean_loss=current_val_loss,
    )
    if relative_improvement >= min_relative_improvement:
        return current_val_loss, 0, relative_improvement
    return best_val_loss, bad_checks + 1, relative_improvement


def _update_validation_best_only(
    *,
    best_val_loss: float | None,
    current_val_loss: float,
) -> tuple[float, float | None]:
    """Track the lowest observed validation loss without advancing patience."""
    if best_val_loss is None:
        return current_val_loss, None

    relative_improvement = _relative_block_improvement(
        previous_mean_loss=best_val_loss,
        current_mean_loss=current_val_loss,
    )
    return min(best_val_loss, current_val_loss), relative_improvement


def _should_continue_after_block(
    *,
    block_mean_losses: list[float],
    min_relative_improvement: float,
) -> tuple[bool, float | None]:
    """Decide whether to continue training after a completed block."""
    if len(block_mean_losses) < 2:
        return True, None

    improvement = _relative_block_improvement(
        previous_mean_loss=block_mean_losses[-2],
        current_mean_loss=block_mean_losses[-1],
    )
    return improvement >= min_relative_improvement, improvement


def _should_save_checkpoint(cfg: TrainScriptConfig, step: int) -> bool:
    """Decide whether the current step should emit a checkpoint."""
    if step <= 0:
        return False
    if (
        cfg.checkpoint_early_every > 0
        and step <= cfg.checkpoint_early_until
        and step % cfg.checkpoint_early_every == 0
    ):
        return True
    if cfg.checkpoint_every <= 0:
        return False
    return step % cfg.checkpoint_every == 0


def _should_run_validation(cfg: TrainScriptConfig, step: int) -> bool:
    """Decide whether the current step should emit validation metrics."""
    if not cfg.validation_enabled or step <= 0:
        return False
    return step % cfg.validation_every == 0


def _load_training_checkpoint(path: str | Path) -> dict[str, object]:
    """Load one training checkpoint payload from disk onto CPU."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Training checkpoint payload must be a dict")
    return payload


def _resume_training_state(
    *,
    checkpoint: dict[str, object],
    model: nn.Module,
    action_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, bool]:
    """Restore training modules from checkpoint and report step plus optimizer restore status."""
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint missing model_state_dict")
    action_state = checkpoint.get("action_encoder_state_dict")
    if not isinstance(action_state, dict):
        raise ValueError("Checkpoint missing action_encoder_state_dict")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint missing optimizer_state_dict")

    step = checkpoint.get("step")
    if not isinstance(step, int):
        raise ValueError("Checkpoint missing integer step")
    if step < 0:
        raise ValueError(f"Checkpoint step must be >= 0, got {step}")

    model.load_state_dict(model_state)
    loaded_partial_action_state = _load_action_encoder_state_dict(
        action_encoder=action_encoder,
        action_state=action_state,
    )
    if loaded_partial_action_state:
        return step, False
    optimizer.load_state_dict(optimizer_state)
    return step, True


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, *, device: torch.device) -> None:
    """Move optimizer state tensors to the active device after a CPU checkpoint load."""
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _configure_trainable_parameters(
    cfg: TrainScriptConfig,
    model: WanVACEWorldModel,
    action_encoder: nn.Module,
) -> list[nn.Parameter]:
    """Apply the requested trainable-parameter policy and return optimizer params."""
    for parameter in model.parameters():
        parameter.requires_grad_(cfg.trainable_backbone == "full")

    if cfg.trainable_backbone in {"vace", "head", "lora"}:
        backbone = model.backbone
        module_names = ("vace_patch_embedding", "norm_out", "proj_out")
        if cfg.trainable_backbone == "vace":
            module_names = ("vace_patch_embedding", "vace_blocks", "norm_out", "proj_out")
        for module_name in module_names:
            module = getattr(backbone, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        scale_shift_table = getattr(backbone, "scale_shift_table", None)
        if isinstance(scale_shift_table, nn.Parameter):
            scale_shift_table.requires_grad_(True)
        if cfg.trainable_backbone == "lora":
            for name, parameter in backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)

    for parameter in action_encoder.parameters():
        parameter.requires_grad_(True)

    chained_parameters = itertools.chain(
        model.parameters(),
        action_encoder.parameters(),
    )
    parameters = [parameter for parameter in chained_parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters remain after applying trainable_backbone policy")
    return parameters


def build_model_from_config(cfg: TrainScriptConfig, prepared_batch: PreparedPackedBatch) -> WanVACEWorldModel:
    """Build the Wan VACE world-model adapter for training."""
    return build_wan_vace_model_from_config(cfg, prepared_batch)


def build_action_encoder_from_config(
    cfg: TrainScriptConfig,
    prepared_batch: PreparedPackedBatch,
    model: WanVACEWorldModel,
) -> ActionTokenEncoder | NullConditioningEncoder:
    """Build the configured cross-attention encoder matching the backbone text width."""
    return build_conditioning_encoder_for_model(cfg, prepared_batch, model)


def _move_train_modules_to_runtime(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module]:
    """Move train-time modules onto the active device and runtime dtype."""
    return (
        model.to(device=device, dtype=runtime_dtype),
        action_encoder.to(device=device, dtype=runtime_dtype),
    )


@torch.no_grad()
def _evaluate_loss(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    z_past_video: torch.Tensor,
    z_future_video: torch.Tensor,
    a_plan: torch.Tensor,
    k: int,
    action_conditioning_window: str,
    teacher_forcing_observation_mode: str,
    teacher_forcing_future_input_mode: str,
    chunk_schedule_mode: str,
    action_backbone_added_kv_mode: str,
    action_token_latent_aux_loss_scale: float,
    t_min: float,
    t_max: float,
    weight_mode: str,
    motion_loss_alpha: float,
    motion_loss_max_weight: float,
    motion_loss_excess_only: bool,
    future_latent_residual_mode: str,
    future_loss_early_bias: float,
    future_chunk_early_bias: float,
    device: torch.device,
    disable_amp: bool,
    runtime_dtype: torch.dtype,
) -> float:
    """Compute one eval-mode chunkwise loss for overfit diagnostics."""
    model.eval()
    action_encoder.eval()

    with _training_autocast_context(device=device, disable_amp=disable_amp, dtype=runtime_dtype):
        action_tokens = action_encoder(a_plan)
        action_image_tokens = (
            action_tokens
            if action_backbone_added_kv_mode == "reuse_action_tokens"
            else None
        )
        loss = chunkwise_teacher_forcing_loss(
            model,
            z_past_video=z_past_video,
            z_future_video=z_future_video,
            action_tokens=action_tokens,
            action_image_tokens=action_image_tokens,
            action_conditioning_window=action_conditioning_window,
            teacher_forcing_observation_mode=teacher_forcing_observation_mode,
            teacher_forcing_future_input_mode=teacher_forcing_future_input_mode,
            chunk_schedule_mode=chunk_schedule_mode,
            k=k,
            t_min=t_min,
            t_max=t_max,
            weight_mode=weight_mode,
            motion_loss_alpha=motion_loss_alpha,
            motion_loss_max_weight=motion_loss_max_weight,
            motion_loss_excess_only=motion_loss_excess_only,
            future_latent_residual_mode=future_latent_residual_mode,
            future_loss_early_bias=future_loss_early_bias,
            future_chunk_early_bias=future_chunk_early_bias,
        )
        loss = loss + (
            action_token_latent_aux_loss_scale
            * _compute_action_token_latent_aux_loss(
                action_encoder=action_encoder,
                action_tokens=action_tokens,
                z_past_video=z_past_video,
                z_future_video=z_future_video,
                future_latent_residual_mode=future_latent_residual_mode,
            )
        )
    return float(loss.detach().cpu().item())


@torch.no_grad()
def _evaluate_validation_loss(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    validation_loader,
    encoder: WanVAE,
    cfg: TrainScriptConfig,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[float, int]:
    """Average validation loss over a deterministic prefix of held-out batches."""
    total_loss = 0.0
    num_batches = 0
    for batch in itertools.islice(validation_loader, cfg.validation_max_batches):
        prepared = prepare_packed_batch(
            batch=batch,
            encoder=encoder,
            device=device,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            frame_height=cfg.frame_height,
            frame_width=cfg.frame_width,
            allow_missing_action=(cfg.conditioning_mode == "none"),
        )
        total_loss += _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            z_past_video=prepared.z_past_video,
            z_future_video=prepared.z_future_video,
            a_plan=prepared.a_plan,
            k=cfg.k,
            action_conditioning_window=cfg.action_conditioning_window,
            teacher_forcing_observation_mode=cfg.teacher_forcing_observation_mode,
            teacher_forcing_future_input_mode=cfg.teacher_forcing_future_input_mode,
            chunk_schedule_mode=cfg.chunk_schedule_mode,
            action_backbone_added_kv_mode=cfg.action_backbone_added_kv_mode,
            action_token_latent_aux_loss_scale=cfg.action_token_latent_aux_loss_scale,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
            future_latent_residual_mode=cfg.future_latent_residual_mode,
            future_loss_early_bias=cfg.future_loss_early_bias,
            future_chunk_early_bias=cfg.future_chunk_early_bias,
            device=device,
            disable_amp=cfg.disable_amp,
            runtime_dtype=runtime_dtype,
        )
        num_batches += 1

    if num_batches <= 0:
        raise ValueError("Validation loader produced no batches")
    return total_loss / num_batches, num_batches


def _build_checkpoint_extra_state(
    *,
    cfg: TrainScriptConfig,
    best_val_loss: float | None,
    val_bad_checks: int,
) -> dict[str, object]:
    """Build checkpoint extra state, including resumable validation tracking."""
    return {
        "config": asdict(cfg),
        "validation_state": {
            "best_val_loss": best_val_loss,
            "val_bad_checks": int(val_bad_checks),
        },
    }


def _load_validation_state_from_checkpoint(
    checkpoint: dict[str, object],
) -> tuple[float | None, int]:
    """Recover resumable validation tracking state from a checkpoint payload."""
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        return None, 0
    validation_state = extra_state.get("validation_state")
    if not isinstance(validation_state, dict):
        return None, 0

    best_val_loss = validation_state.get("best_val_loss")
    if best_val_loss is not None:
        best_val_loss = float(best_val_loss)
    val_bad_checks = int(validation_state.get("val_bad_checks", 0))
    if val_bad_checks < 0:
        val_bad_checks = 0
    return best_val_loss, val_bad_checks


def main() -> None:
    """Run chunkwise world-model training."""
    cfg = _load_args()
    _set_seed(cfg.seed)
    _validate_auto_stop_config(cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_dtype = _select_runtime_dtype(device=device, disable_amp=cfg.disable_amp)
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and not cfg.disable_amp and runtime_dtype == torch.float16),
    )
    print(f"Device: {device}", flush=True)
    print(
        f"Training config: steps={cfg.max_steps} batch={cfg.batch_size} "
        f"k={cfg.k} l={cfg.context_len} H={cfg.horizon_len}",
        flush=True,
    )
    print(f"Training dtype: {runtime_dtype}", flush=True)
    if cfg.resume_from:
        print(f"Resume checkpoint: {cfg.resume_from}", flush=True)

    vae = WanVAE.from_pretrained(device=device, deterministic=True, torch_dtype=runtime_dtype)
    loader = None
    data_iter = None
    validation_loader = None
    train_episode_ids: list[int] | tuple[int, ...] = list(cfg.episodes)
    validation_episode_ids: list[int] = []
    if cfg.video_path:
        if cfg.conditioning_mode != "none":
            raise ValueError("Local-video training currently supports only conditioning_mode=none")
        total_frames = cfg.context_len + cfg.horizon_len
        first_batch = {
            cfg.video_key: load_local_video_clip(
                cfg.video_path,
                start_frame=cfg.start_frame,
                total_frames=total_frames,
            )
        }
    else:
        if cfg.validation_enabled:
            available_episode_ids = resolve_lerobot_episode_ids(repo_id=cfg.repo_id)
            train_episode_ids, validation_episode_ids = split_train_validation_episode_ids(
                available_episode_ids=available_episode_ids,
                requested_episode_ids=cfg.episodes,
                validation_episode_ids=cfg.validation_episodes,
                validation_split_ratio=cfg.validation_split_ratio,
            )
            print(
                f"Validation split: train_episodes={len(train_episode_ids)} "
                f"val_episodes={len(validation_episode_ids)}",
                flush=True,
            )
            print(
                f"Train episodes preview: {_format_episode_preview(list(train_episode_ids))}",
                flush=True,
            )
            print(
                f"Validation episodes preview: {_format_episode_preview(validation_episode_ids)}",
                flush=True,
            )

        loader = build_lerobot_dataloader(
            repo_id=cfg.repo_id,
            episodes=train_episode_ids,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            dt=cfg.dt,
            batch_size=cfg.batch_size,
            subset_size=cfg.subset_size,
            shuffle=not cfg.overfit_one_batch,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
        data_iter = iter(loader)
        first_batch = next(data_iter)
        if cfg.validation_enabled:
            validation_loader = build_lerobot_dataloader(
                repo_id=cfg.repo_id,
                episodes=validation_episode_ids,
                video_key=cfg.video_key,
                context_len=cfg.context_len,
                horizon_len=cfg.horizon_len,
                dt=cfg.dt,
                batch_size=cfg.batch_size,
                subset_size=0,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
            )
    prepared = prepare_packed_batch(
        batch=first_batch,
        encoder=vae,
        device=device,
        video_key=cfg.video_key,
        context_len=cfg.context_len,
        horizon_len=cfg.horizon_len,
        frame_height=cfg.frame_height,
        frame_width=cfg.frame_width,
        allow_missing_action=(cfg.conditioning_mode == "none"),
    )
    _validate_chunk_schedule(cfg, prepared)
    print(
        "Latent window: "
        f"context={prepared.context_latent_steps} "
        f"future={prepared.horizon_latent_steps} "
        f"total={prepared.total_latent_steps}",
        flush=True,
    )
    cached_prepared = prepared if (cfg.overfit_one_batch or cfg.video_path) else None
    if cached_prepared is not None and device.type == "cuda":
        vae.vae.to("cpu")
        torch.cuda.empty_cache()

    model = build_model_from_config(cfg, prepared)
    action_encoder = build_action_encoder_from_config(cfg, prepared, model)
    model, action_encoder = _move_train_modules_to_runtime(
        model=model,
        action_encoder=action_encoder,
        device=device,
        runtime_dtype=runtime_dtype,
    )

    parameter_groups = _configure_trainable_parameters(
        cfg,
        model,
        action_encoder,
    )
    trainable_param_count = sum(parameter.numel() for parameter in parameter_groups)
    print(
        f"Trainable backbone mode: {cfg.trainable_backbone} ({trainable_param_count} params)",
        flush=True,
    )
    optimizer = torch.optim.AdamW(parameter_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    resumed_step = 0
    restored_optimizer_state = True
    best_val_loss: float | None = None
    val_bad_checks = 0
    if cfg.resume_from:
        checkpoint = _load_training_checkpoint(cfg.resume_from)
        resumed_step, restored_optimizer_state = _resume_training_state(
            checkpoint=checkpoint,
            model=model,
            action_encoder=action_encoder,
            optimizer=optimizer,
        )
        best_val_loss, val_bad_checks = _load_validation_state_from_checkpoint(checkpoint)
        _optimizer_state_to_device(optimizer, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Resumed training state from step={resumed_step:06d}", flush=True)
        if not restored_optimizer_state:
            print(
                "Resume note: checkpoint predates one or more optional conditioning modules, "
                "so optimizer state was not restored.",
                flush=True,
            )
        if best_val_loss is not None:
            print(
                f"Resumed validation state: best_val_loss={best_val_loss:.6f} "
                f"val_bad_checks={val_bad_checks}",
                flush=True,
            )
        del checkpoint

    cached_batch = first_batch if (cfg.overfit_one_batch or cfg.video_path) else None

    overfit_start_loss = None
    completed_steps = resumed_step
    block_losses: list[float] = []
    block_mean_losses: list[float] = []
    if cfg.overfit_one_batch:
        overfit_start_loss = _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            z_past_video=prepared.z_past_video,
            z_future_video=prepared.z_future_video,
            a_plan=prepared.a_plan,
            k=cfg.k,
            action_conditioning_window=cfg.action_conditioning_window,
            teacher_forcing_observation_mode=cfg.teacher_forcing_observation_mode,
            teacher_forcing_future_input_mode=cfg.teacher_forcing_future_input_mode,
            chunk_schedule_mode=cfg.chunk_schedule_mode,
            action_backbone_added_kv_mode=cfg.action_backbone_added_kv_mode,
            action_token_latent_aux_loss_scale=cfg.action_token_latent_aux_loss_scale,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
            future_latent_residual_mode=cfg.future_latent_residual_mode,
            future_loss_early_bias=cfg.future_loss_early_bias,
            future_chunk_early_bias=cfg.future_chunk_early_bias,
            device=device,
            disable_amp=cfg.disable_amp,
            runtime_dtype=runtime_dtype,
        )
        print(f"Overfit baseline loss: {overfit_start_loss:.6f}", flush=True)

    if cfg.max_steps <= resumed_step:
        print(
            f"Requested max_steps={cfg.max_steps} is not greater than resumed step={resumed_step}; "
            "skipping optimizer updates.",
            flush=True,
        )

    for step in itertools.count(start=resumed_step + 1):
        if step > cfg.max_steps:
            break
        started = time.time()

        if cached_batch is None:
            assert data_iter is not None
            assert loader is not None
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
        else:
            batch = cached_batch

        if cached_prepared is None:
            prepared = prepare_packed_batch(
                batch=batch,
                encoder=vae,
                device=device,
                video_key=cfg.video_key,
                context_len=cfg.context_len,
                horizon_len=cfg.horizon_len,
                frame_height=cfg.frame_height,
                frame_width=cfg.frame_width,
                allow_missing_action=(cfg.conditioning_mode == "none"),
            )
        else:
            prepared = cached_prepared

        metrics = train_chunkwise_batch(
            model=model,
            action_encoder=action_encoder,
            optimizer=optimizer,
            z_past_video=prepared.z_past_video,
            z_future_video=prepared.z_future_video,
            a_plan=prepared.a_plan,
            k=cfg.k,
            action_conditioning_window=cfg.action_conditioning_window,
            teacher_forcing_observation_mode=cfg.teacher_forcing_observation_mode,
            teacher_forcing_future_input_mode=cfg.teacher_forcing_future_input_mode,
            chunk_schedule_mode=cfg.chunk_schedule_mode,
            action_backbone_added_kv_mode=cfg.action_backbone_added_kv_mode,
            action_token_latent_aux_loss_scale=cfg.action_token_latent_aux_loss_scale,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
            future_latent_residual_mode=cfg.future_latent_residual_mode,
            future_loss_early_bias=cfg.future_loss_early_bias,
            future_chunk_early_bias=cfg.future_chunk_early_bias,
            grad_clip_norm=cfg.grad_clip_norm,
            amp_dtype=(None if cfg.disable_amp or device.type != "cuda" else runtime_dtype),
            grad_scaler=grad_scaler,
        )

        step_time_s = time.time() - started
        should_save_checkpoint = _should_save_checkpoint(cfg, step)
        log_payload = metrics.to_log_dict(step=step)
        log_payload["lr"] = float(optimizer.param_groups[0]["lr"])
        log_payload["step_time_s"] = float(step_time_s)
        completed_steps = step
        block_losses.append(metrics.loss)

        if step % cfg.log_every == 0 or step == 1:
            print(
                f"step={step:06d} loss={metrics.loss:.6f} grad={metrics.grad_norm:.4f} "
                f"time={step_time_s:.3f}s chunks={metrics.per_chunk_losses}",
                flush=True,
            )

        stop_for_validation = False
        should_run_validation = _should_run_validation(cfg, step)

        if should_run_validation:
            assert validation_loader is not None
            val_loss, val_num_batches = _evaluate_validation_loss(
                model=model,
                action_encoder=action_encoder,
                validation_loader=validation_loader,
                encoder=vae,
                cfg=cfg,
                device=device,
                runtime_dtype=runtime_dtype,
            )
            if cfg.validation_patience_checks > 0:
                best_val_loss, val_bad_checks, val_relative_improvement = _update_validation_tracking(
                    best_val_loss=best_val_loss,
                    current_val_loss=val_loss,
                    bad_checks=val_bad_checks,
                    min_relative_improvement=cfg.validation_min_relative_improvement,
                )
            else:
                best_val_loss, val_relative_improvement = _update_validation_best_only(
                    best_val_loss=best_val_loss,
                    current_val_loss=val_loss,
                )
                val_bad_checks = 0
            log_payload["val_loss"] = float(val_loss)
            log_payload["val_num_batches"] = int(val_num_batches)
            log_payload["best_val_loss"] = float(best_val_loss)
            log_payload["val_relative_improvement"] = (
                None if val_relative_improvement is None else float(val_relative_improvement)
            )
            log_payload["val_bad_checks"] = int(val_bad_checks)
            improvement_text = (
                "n/a"
                if val_relative_improvement is None
                else f"{val_relative_improvement:.4f}"
            )
            print(
                f"validation step={step:06d} val_loss={val_loss:.6f} "
                f"best_val_loss={best_val_loss:.6f} "
                f"relative_improvement={improvement_text} "
                f"bad_checks={val_bad_checks}/{cfg.validation_patience_checks}",
                flush=True,
            )
            stop_for_validation = (
                cfg.validation_patience_checks > 0
                and val_bad_checks >= cfg.validation_patience_checks
            )

        if should_save_checkpoint:
            path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                action_encoder=action_encoder,
                optimizer=optimizer,
                extra_state=_build_checkpoint_extra_state(
                    cfg=cfg,
                    best_val_loss=best_val_loss,
                    val_bad_checks=val_bad_checks,
                ),
            )
            print(f"checkpoint={path}", flush=True)

        append_jsonl(metrics_path, log_payload)

        if cfg.auto_stop_check_every > 0 and step % cfg.auto_stop_check_every == 0:
            block_mean_loss = sum(block_losses) / len(block_losses)
            block_mean_losses.append(block_mean_loss)
            should_continue, improvement = _should_continue_after_block(
                block_mean_losses=block_mean_losses,
                min_relative_improvement=cfg.auto_stop_min_relative_improvement,
            )
            if improvement is None:
                print(
                    "auto-stop block summary: "
                    f"steps={step - len(block_losses) + 1:06d}-{step:06d} "
                    f"mean_loss={block_mean_loss:.6f}; continuing (first block)",
                    flush=True,
                )
            else:
                print(
                    "auto-stop block summary: "
                    f"steps={step - len(block_losses) + 1:06d}-{step:06d} "
                    f"mean_loss={block_mean_loss:.6f} "
                    f"relative_improvement={improvement:.4f} "
                    f"threshold={cfg.auto_stop_min_relative_improvement:.4f}",
                    flush=True,
                )
            block_losses = []
            if not should_continue:
                print(f"auto-stop triggered at step={step:06d}", flush=True)
                break
        if stop_for_validation:
            print(f"validation early-stop triggered at step={step:06d}", flush=True)
            break

    final_ckpt = save_checkpoint(
        output_dir=output_dir,
        step=completed_steps,
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        extra_state=_build_checkpoint_extra_state(
            cfg=cfg,
            best_val_loss=best_val_loss,
            val_bad_checks=val_bad_checks,
        ),
    )
    print(f"final_checkpoint={final_ckpt}", flush=True)

    if cfg.overfit_one_batch:
        overfit_end_loss = _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            z_past_video=prepared.z_past_video,
            z_future_video=prepared.z_future_video,
            a_plan=prepared.a_plan,
            k=cfg.k,
            action_conditioning_window=cfg.action_conditioning_window,
            teacher_forcing_observation_mode=cfg.teacher_forcing_observation_mode,
            teacher_forcing_future_input_mode=cfg.teacher_forcing_future_input_mode,
            chunk_schedule_mode=cfg.chunk_schedule_mode,
            action_backbone_added_kv_mode=cfg.action_backbone_added_kv_mode,
            action_token_latent_aux_loss_scale=cfg.action_token_latent_aux_loss_scale,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
            future_latent_residual_mode=cfg.future_latent_residual_mode,
            future_loss_early_bias=cfg.future_loss_early_bias,
            future_chunk_early_bias=cfg.future_chunk_early_bias,
            device=device,
            disable_amp=cfg.disable_amp,
            runtime_dtype=runtime_dtype,
        )
        assert overfit_start_loss is not None
        print(
            f"overfit_loss_start={overfit_start_loss:.6f} "
            f"overfit_loss_end={overfit_end_loss:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
