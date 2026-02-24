"""Attention mask builders for latent-time causal constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskSpec:
    """Mask partition sizes for [past, current, future] tokens."""

    n_past: int
    n_current: int
    n_future: int

    @property
    def total_len(self) -> int:
        """Return total token count across past/current/future partitions."""
        return self.n_past + self.n_current + self.n_future


def build_no_future_leak_mask(spec: MaskSpec, device: torch.device | None = None) -> torch.Tensor:
    """Build additive attention mask that blocks future-token columns for past/current queries."""
    _validate_mask_spec(spec)

    total = spec.total_len
    keep = spec.n_past + spec.n_current
    future_start = keep

    mask = torch.zeros((total, total), dtype=torch.float32, device=device)
    if spec.n_future > 0:
        mask[:keep, future_start:] = float("-inf")
    return mask


def _validate_mask_spec(spec: MaskSpec) -> None:
    """Validate non-negative partition sizes and non-empty total length."""
    if spec.n_past < 0 or spec.n_current < 0 or spec.n_future < 0:
        raise ValueError(
            "MaskSpec counts must be non-negative "
            f"(got n_past={spec.n_past}, n_current={spec.n_current}, n_future={spec.n_future})"
        )
    if spec.total_len <= 0:
        raise ValueError("MaskSpec must include at least one token")
