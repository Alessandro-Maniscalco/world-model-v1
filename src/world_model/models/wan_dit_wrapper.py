"""Wan DiT-style wrapper with AdaLN-zero conditioning interfaces.

Builds a DiT-like latent transformer that predicts velocity on the noisy future chunk.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from world_model.conditioning import AdaLNZero


LEGACY_ONLY = True


class _DiTBlock(nn.Module):
    """Minimal DiT-style block with AdaLN conditioning."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Initialize one DiT-style self-attention + MLP residual block."""
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        mlp_dim = int(hidden_dim * mlp_ratio)
        self.ada_attn = AdaLNZero(hidden_dim=hidden_dim, cond_dim=hidden_dim)
        self.num_heads = int(num_heads)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ada_mlp = AdaLNZero(hidden_dim=hidden_dim, cond_dim=hidden_dim)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.mlp_gate = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Apply conditioned attention/MLP residual updates to token features."""
        h = self.ada_attn(x, cond)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=_prepare_attn_mask(attn_mask, batch_size=x.shape[0], num_heads=self.num_heads),
            need_weights=False,
        )
        x = x + self.attn_gate * attn_out
        x = x + self.mlp_gate * self.mlp(self.ada_mlp(x, cond))
        return x


def _prepare_attn_mask(
    attn_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor | None:
    """Normalize 2D/3D attention masks to PyTorch MHA expected layout."""
    if attn_mask is None:
        return None
    if attn_mask.ndim == 2:
        return attn_mask
    if attn_mask.ndim == 3:
        if attn_mask.shape[0] != batch_size:
            raise ValueError(
                f"3D attn_mask first dim must be batch size {batch_size}, got {attn_mask.shape[0]}"
            )
        return attn_mask.repeat_interleave(num_heads, dim=0)
    raise ValueError(f"attn_mask must be 2D or 3D, got {tuple(attn_mask.shape)}")


class WanDiTWrapper(nn.Module):
    """Wrapper around a DiT backbone for velocity prediction on future latents."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        latent_dim: int | None = None,
        num_layers: int = 6,
        num_heads: int = 8,
        cond_dim: int | None = None,
        timestep_dim: int = 1,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize latent-token projections, DiT blocks, and velocity head."""
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if latent_dim is None:
            latent_dim = hidden_dim
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if timestep_dim <= 0:
            raise ValueError(f"timestep_dim must be positive, got {timestep_dim}")
        if cond_dim is None:
            cond_dim = hidden_dim
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}")

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.cond_dim = int(cond_dim)
        self.timestep_dim = int(timestep_dim)
        self.mixed_precision = bool(mixed_precision)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.amp_dtype = amp_dtype

        self.latent_to_hidden: nn.Module
        self.hidden_to_latent: nn.Module
        if self.latent_dim == self.hidden_dim:
            self.latent_to_hidden = nn.Identity()
            self.hidden_to_latent = nn.Identity()
        else:
            self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
            self.hidden_to_latent = nn.Linear(self.hidden_dim, self.latent_dim)

        self.action_proj = nn.Linear(self.cond_dim, self.hidden_dim, bias=False)
        self.proprio_proj = nn.Linear(self.cond_dim, self.hidden_dim, bias=False)
        self.timestep_proj = nn.Sequential(
            nn.Linear(256, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, self.hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [_DiTBlock(hidden_dim=self.hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.final_norm = AdaLNZero(hidden_dim=self.hidden_dim, cond_dim=self.cond_dim)
        self.velocity_head = nn.Linear(self.hidden_dim, self.latent_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    @classmethod
    def from_pretrained_state(
        cls,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "WanDiTWrapper":
        """Build wrapper and load pretrained weights from a local checkpoint path."""
        model = cls(**kwargs)
        model.load_pretrained_weights(
            checkpoint_path=checkpoint_path,
            strict=strict,
            map_location=map_location,
        )
        return model

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> None:
        """Load pretrained backbone weights from a local path."""
        loaded = torch.load(Path(checkpoint_path), map_location=map_location)
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state_dict = loaded["state_dict"]
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise ValueError("Checkpoint must be a state_dict or a dict with key 'state_dict'")

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if not strict and (missing or unexpected):
            # Keep this explicit in logs/tests when partially loading large upstream checkpoints.
            print(f"WanDiTWrapper partial load: missing={len(missing)} unexpected={len(unexpected)}")

    def forward(
        self,
        *,
        noisy_future_chunk: torch.Tensor,
        past_clean_chunks: torch.Tensor,
        action_conditioning: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        proprio_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field `u_theta` for the noisy future chunk.

        Args:
            noisy_future_chunk: `[B, T_future, D_latent]` noisy latents.
            past_clean_chunks: `[B, T_past, D_latent]` clean teacher-forced latents.
            action_conditioning: `[B, C]` action embedding for AdaLN conditioning only.
            timestep_t: `[B]` or `[B, timestep_dim]` flow timestep.
            block_causal_attention_mask: additive (0/-inf) or bool mask with shape
                `[L, L]` or `[B, L, L]`, where `L = T_past + T_future`.
            proprio_conditioning: optional `[B, C]` proprio embedding for AdaLN only.
        """
        self._validate_inputs(
            noisy_future_chunk=noisy_future_chunk,
            past_clean_chunks=past_clean_chunks,
            action_conditioning=action_conditioning,
            proprio_conditioning=proprio_conditioning,
            timestep_t=timestep_t,
            block_causal_attention_mask=block_causal_attention_mask,
        )

        device_type = noisy_future_chunk.device.type
        amp_ctx = (
            torch.autocast(device_type=device_type, dtype=self.amp_dtype)
            if self.mixed_precision and device_type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            x_full = torch.cat([past_clean_chunks, noisy_future_chunk], dim=1)
            x_full = self.latent_to_hidden(x_full)
            t_len = x_full.shape[1]
            x_full = x_full + self.pos_emb[:, :t_len, :]

            cond = self._build_conditioning(
                action_conditioning=action_conditioning,
                proprio_conditioning=proprio_conditioning,
                timestep_t=timestep_t,
            )

            x_full = self._run_backbone(
                x_full=x_full,
                cond=cond,
                block_causal_attention_mask=block_causal_attention_mask,
            )

            future_len = noisy_future_chunk.shape[1]
            pred_future = x_full[:, -future_len:, :]
            pred_hidden = self.final_norm(pred_future, cond)
            return self.velocity_head(pred_hidden)

    def _run_backbone(
        self,
        *,
        x_full: torch.Tensor,
        cond: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run transformer block stack with optional gradient checkpointing."""
        x = x_full
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and x.requires_grad:
                x = checkpoint(
                    block,
                    x,
                    cond,
                    block_causal_attention_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, block_causal_attention_mask)
        return x

    def _build_conditioning(
        self,
        *,
        action_conditioning: torch.Tensor,
        proprio_conditioning: torch.Tensor | None,
        timestep_t: torch.Tensor,
    ) -> torch.Tensor:
        """Combine action/proprio/timestep inputs into one conditioning embedding."""
        action_emb = self.action_proj(action_conditioning)
        t_emb = self._sinusoidal_timestep(timestep_t, dim=256)
        cond = action_emb + self.timestep_proj(t_emb)
        if proprio_conditioning is not None:
            cond = cond + self.proprio_proj(proprio_conditioning)
        return cond

    def _sinusoidal_timestep(self, t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
        """Compute sinusoidal embeddings for flow timesteps in `[0,1]`."""
        if t.ndim > 1:
            t = t.squeeze()
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t = t * 1000.0  # Scale [0, 1] t up to [0, 1000] typical of diffusion
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def _normalize_timestep(self, timestep_t: torch.Tensor) -> torch.Tensor:
        """Normalize timestep tensors to `[B,timestep_dim]` float layout."""
        if timestep_t.ndim == 1:
            return timestep_t.unsqueeze(-1).to(dtype=torch.float32)
        if timestep_t.ndim == 2 and timestep_t.shape[1] == self.timestep_dim:
            return timestep_t.to(dtype=torch.float32)
        raise ValueError(
            f"timestep_t must be [B] or [B,{self.timestep_dim}], got {tuple(timestep_t.shape)}"
        )

    def _validate_inputs(
        self,
        *,
        noisy_future_chunk: torch.Tensor,
        past_clean_chunks: torch.Tensor,
        action_conditioning: torch.Tensor,
        proprio_conditioning: torch.Tensor | None,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
    ) -> None:
        """Validate model input tensor shapes and mask dimensions."""
        if noisy_future_chunk.ndim != 3:
            raise ValueError(
                f"noisy_future_chunk must be [B,T_future,D], got {tuple(noisy_future_chunk.shape)}"
            )
        if past_clean_chunks.ndim != 3:
            raise ValueError(
                f"past_clean_chunks must be [B,T_past,D], got {tuple(past_clean_chunks.shape)}"
            )
        if noisy_future_chunk.shape[0] != past_clean_chunks.shape[0]:
            raise ValueError("Batch size mismatch between noisy_future_chunk and past_clean_chunks")
        if noisy_future_chunk.shape[2] != self.latent_dim or past_clean_chunks.shape[2] != self.latent_dim:
            raise ValueError(f"Latent dim D must match latent_dim={self.latent_dim}")

        if action_conditioning.ndim != 2:
            raise ValueError(
                f"action_conditioning must be [B,C], got {tuple(action_conditioning.shape)}"
            )
        if action_conditioning.shape[0] != noisy_future_chunk.shape[0]:
            raise ValueError("Batch size mismatch between action_conditioning and latent chunks")
        if action_conditioning.shape[1] != self.cond_dim:
            raise ValueError(
                f"action_conditioning C={action_conditioning.shape[1]} must match cond_dim={self.cond_dim}"
            )

        if proprio_conditioning is not None:
            if proprio_conditioning.ndim != 2:
                raise ValueError(
                    f"proprio_conditioning must be [B,C], got {tuple(proprio_conditioning.shape)}"
                )
            if proprio_conditioning.shape[0] != noisy_future_chunk.shape[0]:
                raise ValueError("Batch size mismatch between proprio_conditioning and latent chunks")
            if proprio_conditioning.shape[1] != self.cond_dim:
                raise ValueError(
                    f"proprio_conditioning C={proprio_conditioning.shape[1]} must match cond_dim={self.cond_dim}"
                )

        if timestep_t.shape[0] != noisy_future_chunk.shape[0]:
            raise ValueError("Batch size mismatch between timestep_t and latent chunks")

        if block_causal_attention_mask is None:
            return

        total_len = past_clean_chunks.shape[1] + noisy_future_chunk.shape[1]
        if block_causal_attention_mask.ndim == 2:
            if block_causal_attention_mask.shape != (total_len, total_len):
                raise ValueError(
                    "2D block_causal_attention_mask must have shape [L,L] "
                    f"with L={total_len}, got {tuple(block_causal_attention_mask.shape)}"
                )
            return

        if block_causal_attention_mask.ndim == 3:
            bsz = noisy_future_chunk.shape[0]
            if block_causal_attention_mask.shape != (bsz, total_len, total_len):
                raise ValueError(
                    "3D block_causal_attention_mask must have shape [B,L,L] "
                    f"with B={bsz}, L={total_len}, got {tuple(block_causal_attention_mask.shape)}"
                )
            return

        raise ValueError(
            "block_causal_attention_mask must be None, [L,L], or [B,L,L], "
            f"got {tuple(block_causal_attention_mask.shape)}"
        )
