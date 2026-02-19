from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskSpec:
    n_past: int
    n_current: int
    n_future: int

    @property
    def total_len(self) -> int:
        return self.n_past + self.n_current + self.n_future


def build_no_future_leak_mask(spec: MaskSpec, device: torch.device) -> torch.Tensor:
    """
    Additive attention mask for nn.MultiheadAttention with shape [L, L].
    mask[i, j] = -inf means query i cannot attend to key j.

    Policy:
      - past: can attend to past only
      - current: can attend to past/current, not future
      - future: unconstrained for this leakage check
    """
    mask = torch.zeros((spec.total_len, spec.total_len), device=device, dtype=torch.float32)
    cur_end = spec.n_past + spec.n_current

    if spec.n_future > 0:
        mask[:cur_end, cur_end:] = float("-inf")

    return mask
