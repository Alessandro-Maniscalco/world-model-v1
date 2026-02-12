# scripts/check_masking_leakage.py
#
# Goal
# Confirm your teacher forcing attention mask prevents any information flow
# from future tokens into past or current tokens.
#
# What this script does
# 1. Builds a token sequence: [past_clean_tokens | current_noisy_tokens | future_tokens]
# 2. Runs a tiny Transformer attention block with an attention mask that blocks attending to future tokens
# 3. Replaces future tokens with random noise and checks that outputs for past+current DO NOT change
# 4. Also runs a negative control: without the mask, outputs SHOULD change

import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class MaskSpec:
    n_past: int
    n_current: int
    n_future: int


def build_no_future_leak_mask(spec: MaskSpec, device: torch.device) -> torch.Tensor:
    """
    Returns an additive attention mask for nn.MultiheadAttention with shape [L, L],
    where mask[i, j] = -inf means query position i cannot attend to key position j.

    Allowed:
      past attends to past
      current attends to past and current
      future can attend anywhere (does not matter for leakage check)

    Disallowed:
      past attends to future
      current attends to future

    This matches your stated rule:
      current noisy chunk can attend to clean previous chunks, but must not leak future.
    """
    L = spec.n_past + spec.n_current + spec.n_future
    mask = torch.zeros((L, L), device=device, dtype=torch.float32)

    past_end = spec.n_past
    cur_end = spec.n_past + spec.n_current

    # Block attending to future columns for past and current query rows
    if spec.n_future > 0:
        mask[:cur_end, cur_end:] = float("-inf")

    return mask


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


def assert_no_leak(out_a: torch.Tensor, out_b: torch.Tensor, spec: MaskSpec, atol: float = 1e-6):
    """
    Checks that outputs for past+current positions are unchanged when future tokens change.
    """
    L_keep = spec.n_past + spec.n_current
    diff = (out_a[:, :L_keep] - out_b[:, :L_keep]).abs().max().item()
    if diff > atol:
        raise AssertionError(f"Leak detected: max abs diff on past+current = {diff} > {atol}")
    print(f"PASS: no leak into past+current, max abs diff = {diff:.3e}")


def assert_leak_exists(out_a: torch.Tensor, out_b: torch.Tensor, spec: MaskSpec, min_diff: float = 1e-5):
    """
    Negative control: without a mask, changing future tokens should change earlier outputs.
    """
    L_keep = spec.n_past + spec.n_current
    diff = (out_a[:, :L_keep] - out_b[:, :L_keep]).abs().max().item()
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

    # Token sequence specification
    # Interpret these as chunks of tokens:
    # past_clean: tokens from z^{(1:k-1)}_1
    # current_noisy: tokens from z^{(k)}_{t_k}
    # future: tokens that must be invisible (next chunks or future timesteps)
    spec = MaskSpec(n_past=64, n_current=32, n_future=64)

    B = 2
    D = 256
    L = spec.n_past + spec.n_current + spec.n_future

    block = TinyAttnBlock(d_model=D, n_heads=8).to(device).eval()

    # Construct two sequences that are identical except for future tokens
    x_base = torch.randn(B, L, D, device=device)

    x_future_changed = x_base.clone()
    future_start = spec.n_past + spec.n_current
    x_future_changed[:, future_start:] = torch.randn_like(x_future_changed[:, future_start:])

    # 1) With correct mask: outputs for past+current must not change
    attn_mask = build_no_future_leak_mask(spec, device=device)
    out_a = block(x_base, attn_mask=attn_mask)
    out_b = block(x_future_changed, attn_mask=attn_mask)
    assert_no_leak(out_a, out_b, spec, atol=1e-6)

    # 2) Negative control: without mask, outputs should change
    out_a2 = block(x_base, attn_mask=None)
    out_b2 = block(x_future_changed, attn_mask=None)
    assert_leak_exists(out_a2, out_b2, spec, min_diff=1e-5)

    print("Masking leakage test completed successfully.")


if __name__ == "__main__":
    main()
