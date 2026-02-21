"""Public masking API for attention leakage prevention."""

from world_model.masking.attention import MaskSpec, build_no_future_leak_mask
from world_model.masking.block_causal import build_block_causal_mask

__all__ = ["MaskSpec", "build_no_future_leak_mask", "build_block_causal_mask"]
