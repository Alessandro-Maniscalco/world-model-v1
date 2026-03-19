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
from dataclasses import asdict
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

from world_model.config import TrainScriptConfig, apply_namespace_overrides, load_train_config
from world_model.data import build_lerobot_dataloader, load_local_video_clip, prepare_packed_batch
from world_model.data.schema import PreparedPackedBatch
from world_model.latents import WanVAE
from world_model.models import WanVACEWorldModel
from world_model.models.wan_vace_factory import (
    build_conditioning_encoder_for_model,
    build_wan_vace_model_from_config,
    _load_action_encoder_state_dict,
)
from world_model.models.wan_vace_conditioning import ActionTokenEncoder, NullConditioningEncoder
from world_model.training import (
    append_jsonl,
    chunkwise_teacher_forcing_loss,
    save_checkpoint,
    train_chunkwise_batch,
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
    parser.add_argument("--k", type=int, default=defaults.k, help="K in K+1 chunk schedule")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--auto-stop-check-every", type=int, default=defaults.auto_stop_check_every)
    parser.add_argument(
        "--auto-stop-min-relative-improvement",
        type=float,
        default=defaults.auto_stop_min_relative_improvement,
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
    parser.add_argument("--mask-channels", type=int, default=defaults.mask_channels)
    parser.add_argument("--trainable-backbone", choices=("full", "vace", "head", "lora"), default=defaults.trainable_backbone)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--lora-target-modules", nargs="+", default=list(defaults.lora_target_modules))
    parser.add_argument("--conditioning-mode", choices=("none", "action"), default=defaults.conditioning_mode)
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
    return apply_namespace_overrides(defaults, args)


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
    """Fail fast when latent-time chunking cannot satisfy the configured K+1 schedule."""
    min_future_steps = cfg.k + 1
    future_steps = prepared_batch.horizon_latent_steps
    if future_steps < min_future_steps:
        raise ValueError(
            "Invalid latent-time schedule: "
            f"raw horizon_len={cfg.horizon_len} compressed to horizon_latent_steps={future_steps}, "
            f"but k={cfg.k} requires at least {min_future_steps} latent future steps. "
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


def _relative_block_improvement(*, previous_mean_loss: float, current_mean_loss: float) -> float:
    """Compute relative mean-loss improvement between consecutive training blocks."""
    if previous_mean_loss <= 0.0:
        return 0.0 if current_mean_loss >= previous_mean_loss else float("inf")
    return (previous_mean_loss - current_mean_loss) / previous_mean_loss


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

    parameters = [parameter for parameter in itertools.chain(model.parameters(), action_encoder.parameters()) if parameter.requires_grad]
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


@torch.no_grad()
def _evaluate_loss(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    z_past_video: torch.Tensor,
    z_future_video: torch.Tensor,
    a_plan: torch.Tensor,
    k: int,
    t_min: float,
    t_max: float,
    weight_mode: str,
    motion_loss_alpha: float,
    motion_loss_max_weight: float,
    motion_loss_excess_only: bool,
    device: torch.device,
    disable_amp: bool,
    runtime_dtype: torch.dtype,
) -> float:
    """Compute one eval-mode chunkwise loss for overfit diagnostics."""
    model.eval()
    action_encoder.eval()

    with _training_autocast_context(device=device, disable_amp=disable_amp, dtype=runtime_dtype):
        action_tokens = action_encoder(a_plan)
        loss = chunkwise_teacher_forcing_loss(
            model,
            z_past_video=z_past_video,
            z_future_video=z_future_video,
            action_tokens=action_tokens,
            k=k,
            t_min=t_min,
            t_max=t_max,
            weight_mode=weight_mode,
            motion_loss_alpha=motion_loss_alpha,
            motion_loss_max_weight=motion_loss_max_weight,
            motion_loss_excess_only=motion_loss_excess_only,
        )
    return float(loss.detach().cpu().item())


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
        loader = build_lerobot_dataloader(
            repo_id=cfg.repo_id,
            episodes=cfg.episodes,
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

    model = build_model_from_config(cfg, prepared).to(device)
    action_encoder = build_action_encoder_from_config(cfg, prepared, model).to(device)

    parameter_groups = _configure_trainable_parameters(cfg, model, action_encoder)
    trainable_param_count = sum(parameter.numel() for parameter in parameter_groups)
    print(
        f"Trainable backbone mode: {cfg.trainable_backbone} ({trainable_param_count} params)",
        flush=True,
    )
    optimizer = torch.optim.AdamW(parameter_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    resumed_step = 0
    restored_optimizer_state = True
    if cfg.resume_from:
        checkpoint = _load_training_checkpoint(cfg.resume_from)
        resumed_step, restored_optimizer_state = _resume_training_state(
            checkpoint=checkpoint,
            model=model,
            action_encoder=action_encoder,
            optimizer=optimizer,
        )
        _optimizer_state_to_device(optimizer, device=device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Resumed training state from step={resumed_step:06d}", flush=True)
        if not restored_optimizer_state:
            print(
                "Resume note: checkpoint predates the temporal action mixer, so optimizer state was not restored.",
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
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
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
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
            grad_clip_norm=cfg.grad_clip_norm,
            amp_dtype=(None if cfg.disable_amp or device.type != "cuda" else runtime_dtype),
            grad_scaler=grad_scaler,
        )

        step_time_s = time.time() - started
        log_payload = metrics.to_log_dict(step=step)
        log_payload["lr"] = float(optimizer.param_groups[0]["lr"])
        log_payload["step_time_s"] = float(step_time_s)
        append_jsonl(metrics_path, log_payload)
        completed_steps = step
        block_losses.append(metrics.loss)

        if step % cfg.log_every == 0 or step == 1:
            print(
                f"step={step:06d} loss={metrics.loss:.6f} grad={metrics.grad_norm:.4f} "
                f"time={step_time_s:.3f}s chunks={metrics.per_chunk_losses}",
                flush=True,
            )

        if _should_save_checkpoint(cfg, step):
            path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                action_encoder=action_encoder,
                optimizer=optimizer,
                extra_state={"config": asdict(cfg)},
            )
            print(f"checkpoint={path}", flush=True)

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

    final_ckpt = save_checkpoint(
        output_dir=output_dir,
        step=completed_steps,
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        extra_state={"config": asdict(cfg)},
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
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            motion_loss_alpha=cfg.motion_loss_alpha,
            motion_loss_max_weight=cfg.motion_loss_max_weight,
            motion_loss_excess_only=cfg.motion_loss_excess_only,
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
