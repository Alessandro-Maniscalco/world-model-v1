"""Public masking API for attention leakage prevention."""

from world_model.masking.block_causal import build_block_causal_mask

__all__ = ["build_block_causal_mask"]
