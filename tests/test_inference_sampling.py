"""Tests for inference-time chunkwise sampling utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from world_model.eval import infer_future_tokens_chunkwise, tokens_to_latents


class _RecordingInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, int | tuple[int, ...]]] = []

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
        assert block_causal_attention_mask is not None
        self.calls.append(
            {
                "chunk_len": noisy_future_chunk.shape[1],
                "past_len": past_clean_chunks.shape[1],
                "mask_shape": tuple(block_causal_attention_mask.shape),
            }
        )
        del action_conditioning, timestep_t, proprio_conditioning
        return -noisy_future_chunk


def test_infer_future_tokens_chunkwise_shapes_and_calls():
    torch.manual_seed(0)
    model = _RecordingInferenceModel()
    z_past = torch.randn(2, 3, 4)
    action_cond = torch.randn(2, 8)
    out = infer_future_tokens_chunkwise(
        model,
        z_past=z_past,
        future_steps=8,
        action_conditioning=action_cond,
        k=1,
        integration_steps=20,
    )
    assert out.shape == (2, 8, 4)
    # K=1 means 2 chunks, each with 20 integration steps => 40 calls.
    assert len(model.calls) == 40
    assert model.calls[0]["past_len"] == 3
    # Second chunk starts after 4 predicted steps.
    assert model.calls[20]["past_len"] == 7


def test_infer_future_tokens_chunkwise_validates_inputs():
    model = _RecordingInferenceModel()
    with pytest.raises(ValueError, match="integration_steps must be positive"):
        infer_future_tokens_chunkwise(
            model,
            z_past=torch.randn(2, 3, 4),
            future_steps=8,
            action_conditioning=torch.randn(2, 8),
            k=1,
            integration_steps=0,
        )


def test_tokens_to_latents_reshapes_correctly():
    tokens = torch.randn(2, 5, 12)
    latents = tokens_to_latents(tokens, latent_shape=(3, 2, 2))
    assert latents.shape == (2, 3, 5, 2, 2)
