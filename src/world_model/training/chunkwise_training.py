"""Chunkwise flow-matching train-step and persistence helpers.

This module contains optimizer-step orchestration, metrics shaping, JSONL
logging, and checkpoint save utilities.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from world_model.training.flow_matching import chunkwise_teacher_forcing_loss


@dataclass(frozen=True)
class ChunkwiseStepMetrics:
    """Metrics emitted from one chunkwise optimization step."""

    loss: float
    grad_norm: float
    per_chunk_losses: tuple[float, ...]
    per_chunk_lengths: tuple[int, ...]
    action_control_aux_loss: float = 0.0

    def to_log_dict(self, *, step: int) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable payload."""
        return {
            "step": int(step),
            "loss": float(self.loss),
            "grad_norm": float(self.grad_norm),
            "action_control_aux_loss": float(self.action_control_aux_loss),
            "per_chunk_losses": [float(x) for x in self.per_chunk_losses],
            "per_chunk_lengths": [int(x) for x in self.per_chunk_lengths],
        }


def train_chunkwise_batch(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    action_control_projector: nn.Module | None = None,
    optimizer: torch.optim.Optimizer,
    z_past_video: torch.Tensor,
    z_future_video: torch.Tensor,
    a_plan: torch.Tensor,
    k: int,
    action_conditioning_window: str = "chunk",
    teacher_forcing_observation_mode: str = "full_prefix",
    teacher_forcing_future_input_mode: str = "full_suffix",
    chunk_schedule_mode: str = "k_plus_one",
    action_backbone_added_kv_mode: str = "none",
    action_control_prior_scale: float = 0.0,
    action_control_projector_observed_context_mode: str = "none",
    action_hidden_state_bias_scale: float = 0.0,
    action_control_aux_loss_scale: float = 0.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    weight_mode: str = "uniform",
    motion_loss_alpha: float = 0.0,
    motion_loss_max_weight: float = 0.0,
    motion_loss_excess_only: bool = False,
    future_loss_early_bias: float = 0.0,
    future_chunk_early_bias: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    snr_clip_max: float = 5.0,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> ChunkwiseStepMetrics:
    """Run one optimizer step using chunkwise teacher-forced flow matching."""
    model.train()
    action_encoder.train()
    if action_control_projector is not None:
        action_control_projector.train()

    optimizer.zero_grad(set_to_none=True)
    trainable_params = list(model.parameters()) + list(action_encoder.parameters())
    if action_control_projector is not None:
        trainable_params += list(action_control_projector.parameters())
    autocast_context = _build_training_autocast_context(z_past_video=z_past_video, amp_dtype=amp_dtype)
    with autocast_context:
        action_tokens = action_encoder(a_plan)
        action_image_tokens = (
            action_tokens
            if str(action_backbone_added_kv_mode) == "reuse_action_tokens"
            else None
        )
        action_control_prior = None
        if (
            action_control_projector is not None
            and (
                action_control_prior_scale > 0.0
                or action_hidden_state_bias_scale > 0.0
                or action_control_aux_loss_scale > 0.0
            )
        ):
            action_control_prior = action_control_projector(
                a_plan,
                latent_height=z_future_video.shape[3],
                latent_width=z_future_video.shape[4],
                observed_latents=(
                    z_past_video if action_control_projector_observed_context_mode != "none" else None
                ),
            )
        info = chunkwise_teacher_forcing_loss(
            model,
            z_past_video=z_past_video,
            z_future_video=z_future_video,
            action_tokens=action_tokens,
            action_image_tokens=action_image_tokens,
            action_control_prior=action_control_prior,
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
            future_loss_early_bias=future_loss_early_bias,
            future_chunk_early_bias=future_chunk_early_bias,
            snr_clip_max=snr_clip_max,
            eps=eps,
            generator=generator,
            return_info=True,
        )
        action_control_aux_loss = _compute_action_control_aux_loss(
            action_control_prior=action_control_prior,
            z_future_video=z_future_video,
        )
        total_loss = info.loss + (action_control_aux_loss_scale * action_control_aux_loss)

    if grad_scaler is None:
        total_loss.backward()
    else:
        grad_scaler.scale(total_loss).backward()

    if grad_clip_norm is None:
        grad_norm = _compute_grad_norm(trainable_params)
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
    else:
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm).detach().cpu().item()
        )
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()

    return ChunkwiseStepMetrics(
        loss=float(total_loss.detach().cpu().item()),
        grad_norm=float(grad_norm),
        action_control_aux_loss=float(action_control_aux_loss.detach().cpu().item()),
        per_chunk_losses=info.per_chunk_losses,
        per_chunk_lengths=info.per_chunk_lengths,
    )


def save_checkpoint(
    *,
    output_dir: str | Path,
    step: int,
    model: nn.Module,
    action_encoder: nn.Module,
    action_control_projector: nn.Module | None = None,
    optimizer: torch.optim.Optimizer,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Persist a training checkpoint and return its path."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{int(step):07d}.pt"
    payload: dict[str, Any] = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "action_encoder_state_dict": action_encoder.state_dict(),
        "action_control_projector_state_dict": (
            None if action_control_projector is None else action_control_projector.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state:
        payload["extra_state"] = extra_state
    torch.save(payload, path)
    return path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object line to a metrics log file."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, sort_keys=True) + "\n")


def _compute_grad_norm(parameters: Any) -> float:
    """Compute L2 norm over gradients for a parameter iterable."""
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = float(param.grad.detach().norm(2).cpu().item())
        total_sq += norm * norm
    return total_sq ** 0.5


def _compute_action_control_aux_loss(
    *,
    action_control_prior: torch.Tensor | None,
    z_future_video: torch.Tensor,
) -> torch.Tensor:
    """Match the action-derived latent prior to the clean future latent summary."""
    if action_control_prior is None:
        return z_future_video.new_zeros(())
    target_summary = z_future_video.detach().mean(dim=(3, 4), keepdim=True)
    predicted_summary = action_control_prior.mean(dim=(3, 4), keepdim=True)
    return torch.nn.functional.mse_loss(predicted_summary, target_summary)


def _build_training_autocast_context(
    *,
    z_past_video: torch.Tensor,
    amp_dtype: torch.dtype | None,
):
    """Create an autocast context for mixed-precision training when requested."""
    if amp_dtype is None or z_past_video.device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)
