"""Leakage check for block-causal masks built from latent-time chunk ids."""

from __future__ import annotations

import random
from pathlib import Path
import sys

import torch
import torch.nn as nn

# Ensure local `src/` package imports work when run as `python scripts/check/masking_leakage.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from world_model.chunking import build_chunk_schedule
from world_model.masking import build_block_causal_mask


class TinyAttnBlock(nn.Module):
    """Minimal attention block used to validate masking behavior."""

    def __init__(self, d_model: int = 256, n_heads: int = 8) -> None:
        """Initialize a simple MHA + FFN residual block."""
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Run masked self-attention followed by residual FFN."""
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln(x + y)
        x = self.ln(x + self.ff(x))
        return x


def assert_no_leak(out_a: torch.Tensor, out_b: torch.Tensor, keep_len: int, atol: float = 1e-6) -> None:
    """Assert outputs for past+current tokens are unchanged after future perturbation."""
    diff = (out_a[:, :keep_len] - out_b[:, :keep_len]).abs().max().item()
    if diff > atol:
        raise AssertionError(f"Leak detected: max abs diff on past+current = {diff} > {atol}")
    print(f"PASS: no leak into past+current, max abs diff = {diff:.3e}")


def assert_leak_exists(out_a: torch.Tensor, out_b: torch.Tensor, keep_len: int, min_diff: float = 1e-5) -> None:
    """Assert unmasked attention exhibits measurable future leakage."""
    diff = (out_a[:, :keep_len] - out_b[:, :keep_len]).abs().max().item()
    if diff < min_diff:
        raise AssertionError(f"Expected leak but did not observe it: diff = {diff} < {min_diff}")
    print(f"PASS: leak present without mask, max abs diff = {diff:.3e}")


@torch.no_grad()
def main() -> None:
    """Run positive/negative leakage controls for block-causal masking."""
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    clip_len = 8
    dt = 0.1
    deltas = [-(clip_len - 1 - i) * dt for i in range(clip_len)]
    ds = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps={"observation.images.image": deltas},
        video_backend="pyav",
    )
    _ = ds[0]["observation.images.image"]

    n_past = 64
    n_current = 32
    n_future = 64
    batch_size = 2
    feature_dim = 256
    sequence_len = n_past + n_current + n_future
    keep_len = n_past + n_current

    block = TinyAttnBlock(d_model=feature_dim, n_heads=8).to(device).eval()

    schedule = build_chunk_schedule(future_steps=n_current + n_future, k=2, device=device)
    first_chunk_size = schedule.boundaries[0][1]
    keep_len = n_past + first_chunk_size

    chunk_ids = torch.cat(
        [
            torch.full((n_past,), -1, dtype=torch.long, device=device),
            schedule.chunk_ids,
        ]
    )

    x_base = torch.randn(batch_size, sequence_len, feature_dim, device=device)
    x_future_changed = x_base.clone()
    x_future_changed[:, keep_len:] = torch.randn_like(x_future_changed[:, keep_len:])

    attn_mask = build_block_causal_mask(chunk_ids, mask_format="additive")
    out_a = block(x_base, attn_mask=attn_mask)
    out_b = block(x_future_changed, attn_mask=attn_mask)
    assert_no_leak(out_a, out_b, keep_len, atol=1e-6)

    out_a_unmasked = block(x_base, attn_mask=None)
    out_b_unmasked = block(x_future_changed, attn_mask=None)
    assert_leak_exists(out_a_unmasked, out_b_unmasked, keep_len, min_diff=1e-5)

    print("Masking leakage test completed successfully.")


if __name__ == "__main__":
    main()
