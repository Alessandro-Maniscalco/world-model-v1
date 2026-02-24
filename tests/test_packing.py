import torch

from world_model.data import pack_latent_window


def test_pack_world_model_batch_shapes():
    b, t, z, a, q = 2, 16, 32, 7, 5
    context_len, horizon_len = 8, 8

    z_tokens = torch.randn(b, t, z)
    actions = torch.randn(b, t, a)
    proprio = torch.randn(b, t, q)

    packed = pack_latent_window(
        z_tokens=z_tokens,
        actions=actions,
        proprio=proprio,
        context_steps=context_len,
        horizon_steps=horizon_len,
        proprio_mode="last",
    )

    assert packed.z_past.shape == (b, context_len, z)
    assert packed.a_plan.shape == (b, horizon_len, a)
    assert packed.z_future.shape == (b, horizon_len, z)
    assert packed.q_cond is not None
    assert packed.q_cond.shape == (b, q)
