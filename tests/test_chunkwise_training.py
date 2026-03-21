"""Unit tests for checkpoint and metrics helper utilities."""

from __future__ import annotations

import json
from contextlib import nullcontext

import torch
import torch.nn as nn

from world_model.training import ChunkwiseStepMetrics, append_jsonl, save_checkpoint
from world_model.training.chunkwise_training import _build_training_autocast_context, _compute_grad_norm


def test_checkpoint_and_jsonl_helpers(tmp_path):
    """Persist checkpoints and append metric logs for the active training loop."""
    model = nn.Linear(4, 4)
    action_encoder = nn.Linear(6, 8)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(action_encoder.parameters()),
        lr=1e-3,
    )
    log_path = tmp_path / "metrics.jsonl"

    payload = {"step": 1, "loss": 0.123}
    append_jsonl(log_path, payload)
    line = log_path.read_text(encoding="utf-8").strip()
    assert json.loads(line) == payload

    ckpt = save_checkpoint(
        output_dir=tmp_path,
        step=1,
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
    )
    assert ckpt.exists()
    loaded = torch.load(ckpt, map_location="cpu")
    assert loaded["step"] == 1
    assert "model_state_dict" in loaded
    assert "action_encoder_state_dict" in loaded
    assert "optimizer_state_dict" in loaded


def test_chunkwise_step_metrics_to_log_dict_serializes_numbers() -> None:
    """Convert in-memory metrics to JSON-friendly scalar and list values."""
    metrics = ChunkwiseStepMetrics(loss=1.5, grad_norm=2.5, per_chunk_losses=(0.5, 1.0), per_chunk_lengths=(4, 4))

    payload = metrics.to_log_dict(step=7)

    assert payload == {
        "step": 7,
        "loss": 1.5,
        "grad_norm": 2.5,
        "action_control_aux_loss": 0.0,
        "per_chunk_losses": [0.5, 1.0],
        "per_chunk_lengths": [4, 4],
    }


def test_compute_grad_norm_ignores_parameters_without_gradients() -> None:
    """Compute the L2 norm using only parameters that currently have gradients."""
    first = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    second = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    first.grad = torch.tensor([3.0], dtype=torch.float32)
    second.grad = None

    grad_norm = _compute_grad_norm([first, second])

    assert grad_norm == 3.0


def test_build_training_autocast_context_is_nullcontext_on_cpu() -> None:
    """Skip autocast setup when training is not running on CUDA."""
    context = _build_training_autocast_context(
        z_past_video=torch.zeros(1, 1, 1, 1, 1),
        amp_dtype=torch.float16,
    )

    assert isinstance(context, nullcontext)
