"""Block-causal attention masks defined over latent-time chunk ids.

Masks forbid attention from a token to any token in a strictly future chunk.
"""

from __future__ import annotations

from typing import Literal

import torch


MaskFormat = Literal["additive", "bool"]


def build_block_causal_mask(
    chunk_ids: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    mask_format: MaskFormat = "additive",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a block-causal attention mask from per-token chunk ids.

    Args:
        chunk_ids: Token chunk ids, shape [L]. Query i may only attend keys j
            where chunk_ids[j] <= chunk_ids[i].
        padding_mask: Optional boolean padding mask with shape [B, L], where
            True marks padded tokens. If provided, output is batched [B, L, L].
        mask_format: "additive" for 0/-inf mask, "bool" for a disallow mask.
        device: Optional output device.
    """
    if chunk_ids.ndim != 1:
        raise ValueError(f"chunk_ids must have shape [L], got {tuple(chunk_ids.shape)}")

    ids = chunk_ids.to(device=device, dtype=torch.long)
    disallow = ids.unsqueeze(0) > ids.unsqueeze(1)  # [L, L]

    if padding_mask is not None:
        if padding_mask.ndim != 2:
            raise ValueError(f"padding_mask must have shape [B,L], got {tuple(padding_mask.shape)}")
        if padding_mask.shape[1] != ids.shape[0]:
            raise ValueError(
                f"padding_mask length L={padding_mask.shape[1]} does not match chunk_ids L={ids.shape[0]}"
            )

        pad = padding_mask.to(device=device, dtype=torch.bool)
        disallow = disallow.unsqueeze(0).expand(pad.shape[0], -1, -1)

        # Mask padded keys for all queries and also mask padded query rows.
        disallow = disallow | pad.unsqueeze(1) | pad.unsqueeze(2)

    if mask_format == "bool":
        return disallow
    if mask_format == "additive":
        return _bool_to_additive(disallow)

    raise ValueError(f"Unsupported mask_format: {mask_format}")


def _bool_to_additive(disallow: torch.Tensor) -> torch.Tensor:
    """Convert boolean disallow mask into additive 0/-inf attention mask."""
    mask = torch.zeros(disallow.shape, dtype=torch.float32, device=disallow.device)
    mask[disallow] = float("-inf")
    return mask
