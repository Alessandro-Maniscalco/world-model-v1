"""Tests for proprio projection and ablation behavior."""

import pytest
import torch

from world_model.conditioning import AdaLNZero, ProprioEncoder


def test_proprio_encoder_matches_direct_projection():
    torch.manual_seed(0)
    b, q, d = 4, 7, 13
    proprio = torch.randn(b, q)
    enc = ProprioEncoder(proprio_dim=q, hidden_dim=d, mlp_dim=None, dropout=0.0).eval()

    with torch.no_grad():
        out = enc(proprio)
        manual = enc.net(proprio)

    assert out.shape == (b, d)
    assert torch.allclose(out, manual, atol=0, rtol=0)


def test_proprio_encoder_disabled_returns_zeros_with_same_batch_shape():
    torch.manual_seed(0)
    b, q, d = 3, 5, 11
    proprio = torch.randn(b, q)
    enc = ProprioEncoder(proprio_dim=q, hidden_dim=d, enabled=False, mlp_dim=17, dropout=0.1).eval()

    with torch.no_grad():
        out = enc(proprio)

    assert out.shape == (b, d)
    assert torch.count_nonzero(out) == 0


def test_proprio_encoder_validates_input_shapes():
    enc = ProprioEncoder(proprio_dim=4, hidden_dim=8)

    with pytest.raises(ValueError, match="got None"):
        enc(None)

    with pytest.raises(ValueError, match=r"proprio must be \[B,Q\]"):
        enc(torch.randn(2, 3, 4))

    with pytest.raises(ValueError, match="does not match proprio_dim"):
        enc(torch.randn(2, 5))


def test_toggling_proprio_only_changes_conditioning_pathway_and_preserves_shapes():
    torch.manual_seed(0)
    b, t, d, q = 2, 3, 8, 5
    x = torch.randn(b, t, d)
    proprio = torch.randn(b, q)

    adaln = AdaLNZero(hidden_dim=d).eval()
    # Force a non-trivial conditioning effect so enabled/disabled diverge.
    with torch.no_grad():
        adaln.modulation.weight.fill_(0.05)
        adaln.modulation.bias.fill_(0.01)

    enc_enabled = ProprioEncoder(proprio_dim=q, hidden_dim=d, enabled=True, dropout=0.0).eval()
    enc_disabled = ProprioEncoder(proprio_dim=q, hidden_dim=d, enabled=False, dropout=0.0).eval()

    with torch.no_grad():
        cond_enabled = enc_enabled(proprio)
        cond_disabled = enc_disabled(proprio)
        y_enabled = adaln(x, cond_enabled)
        y_disabled = adaln(x, cond_disabled)

    # Backbone activations remain the same tensor and output shapes stay stable.
    assert x.shape == (b, t, d)
    assert y_enabled.shape == x.shape
    assert y_disabled.shape == x.shape

    # Toggle affects only conditioning values.
    assert torch.count_nonzero(cond_disabled) == 0
    assert torch.count_nonzero(cond_enabled) > 0
    assert not torch.allclose(y_enabled, y_disabled)
