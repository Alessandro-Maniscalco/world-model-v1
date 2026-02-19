from dataclasses import dataclass

import torch


@dataclass
class PackedBatch:
    # Past conditioning window
    z_past: torch.Tensor      # [B, context_len, z_dim]
    a_past: torch.Tensor      # [B, context_len, a_dim]
    q_last: torch.Tensor | None  # [B, q_dim] or None

    # Prediction target window
    z_future: torch.Tensor    # [B, horizon_len, z_dim]


def validate_packed_batch(batch: PackedBatch, context_len: int, horizon_len: int) -> None:
    if batch.z_past.ndim != 3:
        raise ValueError(f"z_past must be 3D [B,T,Z], got {batch.z_past.shape}")
    if batch.a_past.ndim != 3:
        raise ValueError(f"a_past must be 3D [B,T,A], got {batch.a_past.shape}")
    if batch.z_future.ndim != 3:
        raise ValueError(f"z_future must be 3D [B,T,Z], got {batch.z_future.shape}")

    if batch.z_past.shape[1] != context_len:
        raise ValueError(f"z_past expected context_len={context_len}, got {batch.z_past.shape[1]}")
    if batch.a_past.shape[1] != context_len:
        raise ValueError(f"a_past expected context_len={context_len}, got {batch.a_past.shape[1]}")
    if batch.z_future.shape[1] != horizon_len:
        raise ValueError(f"z_future expected horizon_len={horizon_len}, got {batch.z_future.shape[1]}")

    if batch.z_past.shape[0] != batch.a_past.shape[0] or batch.z_past.shape[0] != batch.z_future.shape[0]:
        raise ValueError("Batch dimension mismatch across packed tensors")

    if batch.q_last is not None and batch.q_last.ndim != 2:
        raise ValueError(f"q_last must be 2D [B,Q], got {batch.q_last.shape}")
