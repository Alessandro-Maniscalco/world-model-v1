"""Tests for AdaLN-Zero conditioning."""

import pytest
import torch

from world_model.conditioning import AdaLNZero


def test_adaln_zero_is_identity_over_layernorm_at_init():
    torch.manual_seed(0)
    b, t, d = 3, 5, 8
    x = torch.randn(b, t, d)
    cond = torch.randn(b, d)
    mod = AdaLNZero(hidden_dim=d).eval()

    with torch.no_grad():
        out = mod(x, cond)
        expected = mod.norm(x)

    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=0, rtol=0)


def test_adaln_zero_applies_shift_and_scale_from_conditioning():
    torch.manual_seed(0)
    b, t, d = 2, 4, 6
    x = torch.randn(b, t, d)
    cond = torch.randn(b, d)
    mod = AdaLNZero(hidden_dim=d).eval()

    with torch.no_grad():
        mod.modulation.bias[:d].fill_(0.25)  # shift
        mod.modulation.bias[d:].fill_(-0.5)  # scale => multiply by 0.5
        out = mod(x, cond)

    expected = mod.norm(x) * 0.5 + 0.25
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)


def test_adaln_zero_validates_shapes():
    mod = AdaLNZero(hidden_dim=4, cond_dim=3)

    with pytest.raises(ValueError, match="x must have at least 2 dims"):
        mod(torch.randn(4), torch.randn(1, 3))

    with pytest.raises(ValueError, match="x last dim"):
        mod(torch.randn(2, 5), torch.randn(2, 3))

    with pytest.raises(ValueError, match=r"cond must be \[B,C\]"):
        mod(torch.randn(2, 4), torch.randn(2, 1, 3))

    with pytest.raises(ValueError, match="cond last dim"):
        mod(torch.randn(2, 4), torch.randn(2, 4))

    with pytest.raises(ValueError, match="cond batch"):
        mod(torch.randn(2, 4), torch.randn(3, 3))
