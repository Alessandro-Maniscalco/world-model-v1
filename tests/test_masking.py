import torch
import torch.nn as nn

from world_model.masking import MaskSpec, build_no_future_leak_mask


class TinyAttnBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        return y


def test_no_future_leak_mask_blocks_future_columns():
    spec = MaskSpec(n_past=2, n_current=1, n_future=3)
    mask = build_no_future_leak_mask(spec, device=torch.device("cpu"))

    assert mask.shape == (6, 6)
    assert torch.isinf(mask[:3, 3:]).all()
    assert (mask[:3, :3] == 0).all()


def test_mask_prevents_future_leakage_for_past_and_current_positions():
    torch.manual_seed(0)
    spec = MaskSpec(n_past=8, n_current=4, n_future=8)
    block = TinyAttnBlock().eval()

    b, d = 2, 64
    l = spec.total_len
    x = torch.randn(b, l, d)
    x_changed = x.clone()
    start = spec.n_past + spec.n_current
    x_changed[:, start:] = torch.randn_like(x_changed[:, start:])

    mask = build_no_future_leak_mask(spec, device=torch.device("cpu"))
    out_a = block(x, attn_mask=mask)
    out_b = block(x_changed, attn_mask=mask)

    keep = spec.n_past + spec.n_current
    diff = (out_a[:, :keep] - out_b[:, :keep]).abs().max().item()
    assert diff < 1e-6
