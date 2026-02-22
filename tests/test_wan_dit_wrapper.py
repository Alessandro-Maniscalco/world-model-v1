"""Tests for Wan DiT wrapper interfaces and conditioning behavior."""

from __future__ import annotations

import torch

from world_model.chunking import build_full_sequence_chunk_ids
from world_model.masking import build_block_causal_mask
from world_model.models import WanDiTWrapper


def test_wan_dit_wrapper_forward_returns_future_velocity_shape():
    torch.manual_seed(0)
    b, t_past, t_future, d = 2, 6, 4, 16
    model = WanDiTWrapper(
        hidden_dim=d,
        cond_dim=d,
        num_layers=2,
        num_heads=4,
        mixed_precision=False,
    ).eval()

    noisy_future = torch.randn(b, t_future, d)
    past_clean = torch.randn(b, t_past, d)
    action_cond = torch.randn(b, d)
    timestep = torch.rand(b)
    chunk_ids = build_full_sequence_chunk_ids(past_steps=t_past, future_steps=t_future, k=1)
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    with torch.no_grad():
        out = model(
            noisy_future_chunk=noisy_future,
            past_clean_chunks=past_clean,
            action_conditioning=action_cond,
            timestep_t=timestep,
            block_causal_attention_mask=mask,
        )

    assert out.shape == noisy_future.shape


def test_wan_dit_wrapper_supports_batched_block_causal_mask():
    torch.manual_seed(0)
    b, t_past, t_future, d = 2, 5, 3, 12
    model = WanDiTWrapper(
        hidden_dim=d,
        cond_dim=d,
        num_layers=1,
        num_heads=3,
        mixed_precision=False,
    ).eval()

    noisy_future = torch.randn(b, t_future, d)
    past_clean = torch.randn(b, t_past, d)
    action_cond = torch.randn(b, d)
    timestep = torch.rand(b)
    chunk_ids = build_full_sequence_chunk_ids(past_steps=t_past, future_steps=t_future, k=1)
    padding = torch.tensor([[False] * (t_past + t_future), [False, False, False, False, False, False, True, True]])
    mask = build_block_causal_mask(chunk_ids, padding_mask=padding, mask_format="additive")

    with torch.no_grad():
        out = model(
            noisy_future_chunk=noisy_future,
            past_clean_chunks=past_clean,
            action_conditioning=action_cond,
            timestep_t=timestep,
            block_causal_attention_mask=mask,
        )

    assert out.shape == (b, t_future, d)


def test_wan_dit_wrapper_supports_latent_dim_different_from_hidden_dim():
    torch.manual_seed(0)
    b, t_past, t_future = 2, 5, 4
    d_latent, d_hidden = 12, 20
    model = WanDiTWrapper(
        latent_dim=d_latent,
        hidden_dim=d_hidden,
        cond_dim=d_hidden,
        num_layers=2,
        num_heads=4,
        mixed_precision=False,
    ).eval()

    noisy_future = torch.randn(b, t_future, d_latent)
    past_clean = torch.randn(b, t_past, d_latent)
    action_cond = torch.randn(b, d_hidden)
    timestep = torch.rand(b)
    chunk_ids = build_full_sequence_chunk_ids(past_steps=t_past, future_steps=t_future, k=1)
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    with torch.no_grad():
        out = model(
            noisy_future_chunk=noisy_future,
            past_clean_chunks=past_clean,
            action_conditioning=action_cond,
            timestep_t=timestep,
            block_causal_attention_mask=mask,
        )

    assert out.shape == noisy_future.shape


def test_wan_dit_wrapper_proprio_none_equals_zero_vector():
    torch.manual_seed(0)
    b, t_past, t_future, d = 2, 4, 3, 10
    model = WanDiTWrapper(
        hidden_dim=d,
        cond_dim=d,
        num_layers=2,
        num_heads=2,
        mixed_precision=False,
    ).eval()

    noisy_future = torch.randn(b, t_future, d)
    past_clean = torch.randn(b, t_past, d)
    action_cond = torch.randn(b, d)
    zero_proprio = torch.zeros(b, d)
    timestep = torch.rand(b)
    chunk_ids = build_full_sequence_chunk_ids(past_steps=t_past, future_steps=t_future, k=1)
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    with torch.no_grad():
        out_none = model(
            noisy_future_chunk=noisy_future,
            past_clean_chunks=past_clean,
            action_conditioning=action_cond,
            timestep_t=timestep,
            block_causal_attention_mask=mask,
            proprio_conditioning=None,
        )
        out_zero = model(
            noisy_future_chunk=noisy_future,
            past_clean_chunks=past_clean,
            action_conditioning=action_cond,
            timestep_t=timestep,
            block_causal_attention_mask=mask,
            proprio_conditioning=zero_proprio,
        )

    assert out_none.shape == out_zero.shape == noisy_future.shape
    assert torch.allclose(out_none, out_zero, atol=1e-6, rtol=0.0)


def test_wan_dit_wrapper_can_load_pretrained_state(tmp_path):
    torch.manual_seed(0)
    d = 8
    model = WanDiTWrapper(
        hidden_dim=d,
        cond_dim=d,
        num_layers=1,
        num_heads=2,
        mixed_precision=False,
    )
    ckpt = tmp_path / "wan_dit_wrapper.pt"
    torch.save(model.state_dict(), ckpt)

    loaded = WanDiTWrapper.from_pretrained_state(
        ckpt,
        hidden_dim=d,
        cond_dim=d,
        num_layers=1,
        num_heads=2,
        mixed_precision=False,
    )
    for key, value in model.state_dict().items():
        assert torch.allclose(value, loaded.state_dict()[key], atol=0, rtol=0)


def test_dit_block_residual_gates_are_zero_initialized():
    model = WanDiTWrapper(
        hidden_dim=8,
        cond_dim=8,
        num_layers=2,
        num_heads=2,
        mixed_precision=False,
    )

    for block in model.blocks:
        assert torch.allclose(block.attn_gate, torch.zeros_like(block.attn_gate))
        assert torch.allclose(block.mlp_gate, torch.zeros_like(block.mlp_gate))
