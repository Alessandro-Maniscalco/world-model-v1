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
    z_past_video: torch.Tensor,
    z_future_video: torch.Tensor,
    a_plan: torch.Tensor,
    k: int,
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

    optimizer.zero_grad(set_to_none=True)
    trainable_params = list(model.parameters()) + list(action_encoder.parameters())
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
