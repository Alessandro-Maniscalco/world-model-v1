"""Unit tests for chunkwise training orchestration helpers."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from world_model.conditioning import ActionEncoder
from world_model.training import append_jsonl, save_checkpoint, train_chunkwise_batch


class _TinyChunkModel(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        *,
        noisy_future_chunk: torch.Tensor,
        past_clean_chunks: torch.Tensor,
        action_conditioning: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        proprio_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del past_clean_chunks, action_conditioning, timestep_t, block_causal_attention_mask, proprio_conditioning
        return self.proj(noisy_future_chunk)


def test_train_chunkwise_batch_updates_parameters():
    torch.manual_seed(0)
    b, t_past, t_future, d = 2, 3, 8, 4
    model = _TinyChunkModel(latent_dim=d)
    action_encoder = ActionEncoder(action_dim=6, hidden_dim=8, pool="mean")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(action_encoder.parameters()),
        lr=1e-2,
    )

    z_past = torch.randn(b, t_past, d)
    z_future = torch.randn(b, t_future, d)
    a_plan = torch.randn(b, t_future, 6)
    before = model.proj.weight.detach().clone()

    metrics = train_chunkwise_batch(
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        z_past=z_past,
        z_future=z_future,
        a_plan=a_plan,
        k=1,
        t_min=0.5,
        t_max=0.5,
    )

    assert metrics.loss > 0.0
    assert metrics.grad_norm > 0.0
    assert metrics.per_chunk_lengths == (4, 4)
    assert not torch.allclose(before, model.proj.weight.detach())


def test_train_chunkwise_batch_requires_q_last_with_proprio():
    model = _TinyChunkModel(latent_dim=4)
    action_encoder = ActionEncoder(action_dim=6, hidden_dim=8, pool="mean")
    proprio_encoder = nn.Linear(5, 8)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(action_encoder.parameters()) + list(proprio_encoder.parameters()),
        lr=1e-3,
    )

    with pytest.raises(ValueError, match="q_last must be provided"):
        train_chunkwise_batch(
            model=model,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            optimizer=optimizer,
            z_past=torch.randn(2, 3, 4),
            z_future=torch.randn(2, 8, 4),
            a_plan=torch.randn(2, 8, 6),
            q_last=None,
            k=1,
        )


def test_checkpoint_and_jsonl_helpers(tmp_path):
    model = _TinyChunkModel(latent_dim=4)
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
