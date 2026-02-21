"""Tests for action-plan pooling and projection."""

import pytest
import torch

from world_model.conditioning import ActionEncoder


def test_action_encoder_mean_pool_matches_manual_mean():
    torch.manual_seed(0)
    b, t, a, d = 3, 5, 7, 11
    a_plan = torch.randn(b, t, a)

    enc = ActionEncoder(action_dim=a, hidden_dim=d, pool="mean", mlp_dim=None, dropout=0.0).eval()

    with torch.no_grad():
        out = enc(a_plan)
        manual = enc.net(a_plan.mean(dim=1))

    assert out.shape == (b, d)
    assert torch.allclose(out, manual, atol=0, rtol=0)


def test_action_encoder_masked_mean_ignores_padded_steps():
    torch.manual_seed(0)
    b, t, a, d = 2, 4, 3, 8
    a_plan = torch.randn(b, t, a)

    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    a_plan_masked = a_plan.clone()
    a_plan_masked[~valid] = 1000.0

    enc = ActionEncoder(action_dim=a, hidden_dim=d, pool="mean", mlp_dim=None, dropout=0.0).eval()

    with torch.no_grad():
        out_a = enc(a_plan, valid_mask=valid)
        out_b = enc(a_plan_masked, valid_mask=valid)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_action_encoder_last_pool_with_mask_uses_last_valid_step():
    torch.manual_seed(0)
    b, t, a, d = 2, 4, 3, 8
    a_plan = torch.randn(b, t, a)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])

    enc = ActionEncoder(action_dim=a, hidden_dim=d, pool="last", mlp_dim=None, dropout=0.0).eval()

    with torch.no_grad():
        out = enc(a_plan, valid_mask=valid)
        expected = enc.net(torch.stack((a_plan[0, 1], a_plan[1, 2]), dim=0))

    assert out.shape == (b, d)
    assert torch.allclose(out, expected, atol=0, rtol=0)


def test_action_encoder_flatten_requires_fixed_horizon_steps():
    enc = ActionEncoder(action_dim=2, hidden_dim=5, pool="flatten", horizon_steps=3, mlp_dim=None)
    a_plan = torch.randn(1, 3, 2)
    out = enc(a_plan)
    assert out.shape == (1, 5)

    with pytest.raises(ValueError, match="Expected a_plan T=3"):
        enc(torch.randn(1, 4, 2))


def test_action_encoder_validates_input_shapes():
    enc = ActionEncoder(action_dim=2, hidden_dim=5)

    with pytest.raises(ValueError, match=r"a_plan must be \[B,T,A\]"):
        enc(torch.randn(2, 3))

    with pytest.raises(ValueError, match="does not match action_dim"):
        enc(torch.randn(2, 3, 4))

    with pytest.raises(ValueError, match=r"valid_mask must be \[B,T\]"):
        enc(torch.randn(2, 3, 2), valid_mask=torch.ones(2, 3, 1, dtype=torch.bool))
