"""Chunkwise flow-matching train-step and persistence helpers.

This module contains optimizer-step orchestration, metrics shaping, JSONL
logging, and checkpoint save utilities.
"""

from __future__ import annotations

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

    def to_log_dict(self, *, step: int) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable payload."""
        return {
            "step": int(step),
            "loss": float(self.loss),
            "grad_norm": float(self.grad_norm),
            "per_chunk_losses": [float(x) for x in self.per_chunk_losses],
            "per_chunk_lengths": [int(x) for x in self.per_chunk_lengths],
        }


def train_chunkwise_batch(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    z_past: torch.Tensor | None = None,
    z_future: torch.Tensor | None = None,
    z_past_video: torch.Tensor | None = None,
    z_future_video: torch.Tensor | None = None,
    a_plan: torch.Tensor,
    k: int,
    proprio_encoder: nn.Module | None = None,
    q_last: torch.Tensor | None = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
    weight_mode: str = "uniform",
    grad_clip_norm: float | None = 1.0,
    snr_clip_max: float = 5.0,
    eps: float = 1e-6,
    generator: torch.Generator | None = None,
) -> ChunkwiseStepMetrics:
    """Run one optimizer step using chunkwise teacher-forced flow matching."""
    model.train()
    action_encoder.train()
    if proprio_encoder is not None:
        proprio_encoder.train()

    optimizer.zero_grad(set_to_none=True)
    trainable_params = list(model.parameters()) + list(action_encoder.parameters())
    if proprio_encoder is not None:
        trainable_params.extend(proprio_encoder.parameters())

    if z_past_video is not None or z_future_video is not None:
        if z_past_video is None or z_future_video is None:
            raise ValueError("z_past_video and z_future_video must be provided together")
        if proprio_encoder is not None:
            raise ValueError("proprio_encoder is not yet supported for structured latent-video training")
        action_tokens = action_encoder(a_plan)
        info = chunkwise_teacher_forcing_loss(
            model,
            z_past_video=z_past_video,
            z_future_video=z_future_video,
            action_tokens=action_tokens,
            k=k,
            t_min=t_min,
            t_max=t_max,
            weight_mode=weight_mode,
            snr_clip_max=snr_clip_max,
            eps=eps,
            generator=generator,
            return_info=True,
        )
    else:
        if z_past is None or z_future is None:
            raise ValueError("z_past and z_future must be provided for token-path training")
        action_conditioning = action_encoder(a_plan)
        proprio_conditioning = None
        if proprio_encoder is not None:
            if q_last is None:
                raise ValueError("q_last must be provided when proprio_encoder is enabled")
            proprio_conditioning = proprio_encoder(q_last)

        info = chunkwise_teacher_forcing_loss(
            model,
            z_past=z_past,
            z_future=z_future,
            action_conditioning=action_conditioning,
            proprio_conditioning=proprio_conditioning,
            k=k,
            t_min=t_min,
            t_max=t_max,
            weight_mode=weight_mode,
            snr_clip_max=snr_clip_max,
            eps=eps,
            generator=generator,
            return_info=True,
        )
    info.loss.backward()

    if grad_clip_norm is None:
        grad_norm = _compute_grad_norm(trainable_params)
    else:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm).detach().cpu().item()
        )
    optimizer.step()

    return ChunkwiseStepMetrics(
        loss=float(info.loss.detach().cpu().item()),
        grad_norm=float(grad_norm),
        per_chunk_losses=info.per_chunk_losses,
        per_chunk_lengths=info.per_chunk_lengths,
    )


def save_checkpoint(
    *,
    output_dir: str | Path,
    step: int,
    model: nn.Module,
    action_encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    proprio_encoder: nn.Module | None = None,
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
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if proprio_encoder is not None:
        payload["proprio_encoder_state_dict"] = proprio_encoder.state_dict()
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
