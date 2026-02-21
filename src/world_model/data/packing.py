"""Compatibility wrappers for packing latent-time training batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model.data.pack import pack_latent_window


@dataclass(frozen=True)
class PackedWorldModelBatch:
    """Packed world-model tensors for context/horizon training."""

    z_past: torch.Tensor
    z_future: torch.Tensor
    a_past: torch.Tensor
    q_last: torch.Tensor | None


def pack_world_model_batch(
    z_tokens: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor | None,
    context_len: int,
    horizon_len: int,
) -> PackedWorldModelBatch:
    """Pack latent tokens and conditioning into context/horizon tensors."""
    packed = pack_latent_window(
        z_tokens=z_tokens,
        actions=actions,
        proprio=proprio,
        context_steps=context_len,
        horizon_steps=horizon_len,
        proprio_mode="last",
    )
    return PackedWorldModelBatch(
        z_past=packed.z_past,
        z_future=packed.z_future,
        a_past=packed.a_past,
        q_last=packed.q_cond,
    )
