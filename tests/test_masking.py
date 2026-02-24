"""Tests for block-causal attention mask behavior."""

import torch

from world_model.chunking import build_full_sequence_chunk_ids
from world_model.masking import build_block_causal_mask


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
