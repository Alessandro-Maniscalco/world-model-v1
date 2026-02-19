import torch

from world_model.data.schema import PackedBatch, validate_packed_batch


def flatten_latents_per_timestep(latents: torch.Tensor) -> torch.Tensor:
    """
    latents: [B, C_lat, T_lat, H_lat, W_lat]
    returns: [B, T_lat, z_dim]
    """
    if latents.ndim != 5:
        raise ValueError(f"Expected 5D latents [B,C,T,H,W], got {latents.shape}")
    b, c, t, h, w = latents.shape
    z = latents.permute(0, 2, 1, 3, 4).contiguous()
    return z.reshape(b, t, c * h * w)


def pack_world_model_batch(
    z_tokens: torch.Tensor,
    actions: torch.Tensor,
    proprio: torch.Tensor | None,
    context_len: int,
    horizon_len: int,
) -> PackedBatch:
    """
    Convert aligned sequence tensors into model-ready past/future windows.

    Inputs:
      z_tokens: [B, T, Z]
      actions:  [B, T, A]
      proprio:  [B, T, Q] or None
    """
    if z_tokens.ndim != 3 or actions.ndim != 3:
        raise ValueError("z_tokens and actions must be 3D [B,T,*]")

    total_len = context_len + horizon_len
    if z_tokens.shape[1] < total_len or actions.shape[1] < total_len:
        raise ValueError(
            f"Need at least T={total_len} timesteps, got z={z_tokens.shape[1]}, a={actions.shape[1]}"
        )

    z_window = z_tokens[:, :total_len]
    a_window = actions[:, :total_len]
    q_window = proprio[:, :total_len] if proprio is not None else None

    z_past = z_window[:, :context_len]
    z_future = z_window[:, context_len:context_len + horizon_len]
    a_past = a_window[:, :context_len]
    q_last = q_window[:, context_len - 1] if q_window is not None else None

    packed = PackedBatch(z_past=z_past, a_past=a_past, q_last=q_last, z_future=z_future)
    validate_packed_batch(packed, context_len=context_len, horizon_len=horizon_len)
    return packed
