import torch

from world_model.data import flatten_latents_per_timestep, pack_latent_window


def test_flatten_latents_per_timestep_shape():
    latents = torch.randn(2, 4, 5, 3, 2)  # [B,C,T,H,W]
    tokens = flatten_latents_per_timestep(latents)
    assert tokens.shape == (2, 5, 24)


def test_pack_latent_window_splits_in_latent_time_and_aligns_actions():
    b, t_lat, z = 1, 6, 3
    context_steps, horizon_steps = 4, 2

    z_tokens = torch.arange(b * t_lat * z, dtype=torch.float32).reshape(b, t_lat, z)
    actions = torch.tensor([[[0.0], [10.0], [20.0]]])  # T_a=3, needs alignment to 6

    packed = pack_latent_window(
        z_tokens=z_tokens,
        actions=actions,
        proprio=None,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
    )

    assert packed.z_past.shape == (1, 4, 3)
    assert packed.z_future.shape == (1, 2, 3)
    assert packed.a_past.shape == (1, 4, 1)

    # nearest-neighbor indices from T_a=3 -> T=6 are [0,0,1,1,2,2]
    expected_a_past = torch.tensor([[[0.0], [0.0], [10.0], [10.0]]])
    assert torch.equal(packed.a_past, expected_a_past)


def test_pack_latent_window_optional_proprio_last_and_past():
    z_tokens = torch.randn(2, 8, 5)
    actions = torch.randn(2, 8, 4)
    proprio = torch.arange(2 * 8 * 2, dtype=torch.float32).reshape(2, 8, 2)

    packed_last = pack_latent_window(
        z_tokens=z_tokens,
        actions=actions,
        proprio=proprio,
        context_steps=5,
        horizon_steps=3,
        proprio_mode="last",
    )
    assert packed_last.q_cond is not None
    assert packed_last.q_cond.shape == (2, 2)
    assert torch.equal(packed_last.q_cond, proprio[:, 4])

    packed_past = pack_latent_window(
        z_tokens=z_tokens,
        actions=actions,
        proprio=proprio,
        context_steps=5,
        horizon_steps=3,
        proprio_mode="past",
    )
    assert packed_past.q_cond is not None
    assert packed_past.q_cond.shape == (2, 5, 2)
    assert torch.equal(packed_past.q_cond, proprio[:, :5])
