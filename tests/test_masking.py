"""Tests for no-leak and block-causal attention mask behavior."""

import torch
import torch.nn as nn

from world_model.chunking import build_full_sequence_chunk_ids
from world_model.masking import MaskSpec, build_no_future_leak_mask
from world_model.masking import build_block_causal_mask


class TinyAttnBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        return y


def test_no_future_leak_mask_blocks_future_columns():
    spec = MaskSpec(n_past=2, n_current=1, n_future=3)
    mask = build_no_future_leak_mask(spec, device=torch.device("cpu"))

    assert mask.shape == (6, 6)
    assert torch.isinf(mask[:3, 3:]).all()
    assert (mask[:3, :3] == 0).all()


def test_mask_prevents_future_leakage_for_past_and_current_positions():
    torch.manual_seed(0)
    spec = MaskSpec(n_past=8, n_current=4, n_future=8)
    block = TinyAttnBlock().eval()

    b, d = 2, 64
    l = spec.total_len
    x = torch.randn(b, l, d)
    x_changed = x.clone()
    start = spec.n_past + spec.n_current
    x_changed[:, start:] = torch.randn_like(x_changed[:, start:])

    mask = build_no_future_leak_mask(spec, device=torch.device("cpu"))
    out_a = block(x, attn_mask=mask)
    out_b = block(x_changed, attn_mask=mask)

    keep = spec.n_past + spec.n_current
    diff = (out_a[:, :keep] - out_b[:, :keep]).abs().max().item()
    assert diff < 1e-6


def test_block_causal_mask_from_chunk_ids_blocks_future_chunks():
    chunk_ids = torch.tensor([-1, -1, 0, 0, 1, 1], dtype=torch.long)
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    assert mask.shape == (6, 6)
    # Past and chunk 0 must not attend chunk 1.
    assert torch.isinf(mask[:4, 4:]).all()
    # Chunk 1 can attend all previous chunks.
    assert (mask[4:, :4] == 0).all()


def test_block_causal_mask_supports_padding_masks():
    chunk_ids = torch.tensor([-1, 0, 1, 1], dtype=torch.long)
    padding = torch.tensor([[False, False, False, True], [False, True, False, True]])

    mask = build_block_causal_mask(chunk_ids, padding_mask=padding, mask_format="additive")
    assert mask.shape == (2, 4, 4)

    # In batch 0 token 3 is padded => key column 3 and query row 3 are fully masked.
    assert torch.isinf(mask[0, :, 3]).all()
    assert torch.isinf(mask[0, 3, :]).all()

    # In batch 1 tokens 1 and 3 are padded.
    assert torch.isinf(mask[1, :, 1]).all()
    assert torch.isinf(mask[1, :, 3]).all()


def test_block_causal_mask_works_with_schedule_chunk_ids():
    chunk_ids = build_full_sequence_chunk_ids(past_steps=3, future_steps=5, k=1, past_chunk_id=-1)
    mask = build_block_causal_mask(chunk_ids, mask_format="bool")

    assert mask.shape == (8, 8)
    # Rows up to current chunk (past + future chunk 0) cannot attend future chunk 1.
    assert mask[:6, 6:].all()
