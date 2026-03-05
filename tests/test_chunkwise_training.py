"""Unit tests for checkpoint and metrics helper utilities."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from world_model.training import append_jsonl, save_checkpoint


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
