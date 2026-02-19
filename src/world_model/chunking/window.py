import torch

from world_model.data.packing import pack_world_model_batch
from world_model.data.schema import PackedBatch


def chunk_into_past_and_future(
    z_tokens: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor | None,
    context_len: int,
    horizon_len: int,
) -> PackedBatch:
    # Chunking delegates to the canonical packer to keep one source of truth.
    return pack_world_model_batch(
        z_tokens=z_tokens,
        actions=actions,
        proprio=proprio,
        context_len=context_len,
        horizon_len=horizon_len,
    )
