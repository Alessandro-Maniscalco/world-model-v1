"""Leakage check for block-causal masks built from latent-time chunk ids.

Validates no information flow from future chunks into past/current tokens.
"""

import random

import torch
import torch.nn as nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from world_model.chunking import build_full_sequence_chunk_ids, build_k_plus_one_schedule
from world_model.masking import build_block_causal_mask


class TinyAttnBlock(nn.Module):
    """
    Minimal block for mask testing.
    Uses MultiheadAttention in batch-first mode on token sequences [B, L, D].
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        # x: [B, L, D]
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln(x + y)
        x = self.ln(x + self.ff(x))
        return x


def assert_no_leak(out_a: torch.Tensor, out_b: torch.Tensor, keep_len: int, atol: float = 1e-6):
    """
    Checks that outputs for past+current positions are unchanged when future tokens change.
    """
    diff = (out_a[:, :keep_len] - out_b[:, :keep_len]).abs().max().item()
    if diff > atol:
        raise AssertionError(f"Leak detected: max abs diff on past+current = {diff} > {atol}")
    print(f"PASS: no leak into past+current, max abs diff = {diff:.3e}")


def assert_leak_exists(out_a: torch.Tensor, out_b: torch.Tensor, keep_len: int, min_diff: float = 1e-5):
    """
    Negative control: without a mask, changing future tokens should change earlier outputs.
    """
    diff = (out_a[:, :keep_len] - out_b[:, :keep_len]).abs().max().item()
    if diff < min_diff:
        raise AssertionError(f"Expected leak but did not observe it: diff = {diff} < {min_diff}")
    print(f"PASS: leak present without mask, max abs diff = {diff:.3e}")


@torch.no_grad()
def main():
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # You said you need this in your next code path
    # This dataset line is included to confirm the pyav backend works in your environment.
    # The mask test itself does not depend on the dataset contents.
    clip_len = 8
    dt = 0.1
    deltas = [-(clip_len - 1 - i) * dt for i in range(clip_len)]
    ds = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps={"observation.images.image": deltas},
        video_backend="pyav",
    )
    _ = ds[0]["observation.images.image"]  # forces a decode

    # Token sequence specification:
    # [past_clean tokens] + [future chunk 0 (current noisy)] + [future chunk 1 (future hidden)]
    n_past = 64
    n_current = 32
    n_future = 64

    B = 2
    D = 256
    L = n_past + n_current + n_future
    keep_len = n_past + n_current

    block = TinyAttnBlock(d_model=D, n_heads=8).to(device).eval()

    # Build chunk ids from K+1 scheduler and prepend past ids.
    # Note: K+1 scheduler with k=1 splits future_steps=96 into two chunks of 48.
    # We must align keep_len with real chunk boundaries.
    schedule = build_k_plus_one_schedule(future_steps=n_current + n_future, k=1, device=device)
    first_chunk_size = schedule.boundaries[0][1]
    keep_len = n_past + first_chunk_size

    chunk_ids = torch.cat([
        torch.full((n_past,), -1, dtype=torch.long, device=device),
        schedule.chunk_ids
    ])

    # Construct two sequences that are identical except for future tokens
    x_base = torch.randn(B, L, D, device=device)

    x_future_changed = x_base.clone()
    future_start = keep_len
    x_future_changed[:, future_start:] = torch.randn_like(x_future_changed[:, future_start:])

    # 1) With block-causal mask: outputs for past+current must not change
    attn_mask = build_block_causal_mask(chunk_ids, mask_format="additive")
    out_a = block(x_base, attn_mask=attn_mask)
    out_b = block(x_future_changed, attn_mask=attn_mask)
    assert_no_leak(out_a, out_b, keep_len, atol=1e-6)

    # 2) Negative control: without mask, outputs should change
    out_a2 = block(x_base, attn_mask=None)
    out_b2 = block(x_future_changed, attn_mask=None)
    assert_leak_exists(out_a2, out_b2, keep_len, min_diff=1e-5)

    print("Masking leakage test completed successfully.")


if __name__ == "__main__":
    main()
