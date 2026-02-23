"""Unit tests for flow-matching training utilities."""

import pytest
import torch
import torch.nn as nn

from world_model.training.flow_matching import (
    chunkwise_teacher_forcing_loss,
    make_noisy_and_target,
    sample_t,
    w,
)


class _RecordingChunkModel(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.latent_dim = latent_dim
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        *,
        noisy_future_chunk: torch.Tensor,
        past_clean_chunks: torch.Tensor,
        action_conditioning: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor | None,
        proprio_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert block_causal_attention_mask is not None
        self.calls.append(
            {
                "past_len": torch.tensor(past_clean_chunks.shape[1]),
                "future_len": torch.tensor(noisy_future_chunk.shape[1]),
                "mask": block_causal_attention_mask.detach().clone(),
            }
        )
        return self.scale * noisy_future_chunk


def test_sample_t_shape_and_bounds():
    torch.manual_seed(0)
    t = sample_t(batch_size=16, t_min=0.2, t_max=0.8)
    assert t.shape == (16,)
    assert (t >= 0.2).all()
    assert (t <= 0.8).all()


def test_sample_t_supports_fixed_value_range():
    t = sample_t(batch_size=4, t_min=0.5, t_max=0.5)
    assert torch.allclose(t, torch.full((4,), 0.5))


def test_make_noisy_and_target_matches_linear_path_with_given_noise():
    z_clean = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    noise = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    t = torch.tensor([0.25, 0.75])

    z_t, v_target = make_noisy_and_target(z_clean, t, noise=noise)

    expected_0 = 0.75 * noise[0] + 0.25 * z_clean[0]
    expected_1 = 0.25 * noise[1] + 0.75 * z_clean[1]
    expected_z_t = torch.stack((expected_0, expected_1), dim=0)
    expected_v = z_clean - noise

    assert torch.allclose(z_t, expected_z_t)
    assert torch.allclose(v_target, expected_v)


def test_make_noisy_and_target_returns_expected_endpoint_states():
    torch.manual_seed(0)
    z_clean = torch.randn(3, 5, 7)
    noise = torch.randn_like(z_clean)

    t_zero = torch.zeros(3)
    t_one = torch.ones(3)
    z_zero, _ = make_noisy_and_target(z_clean, t_zero, noise=noise)
    z_one, _ = make_noisy_and_target(z_clean, t_one, noise=noise)

    assert torch.allclose(z_zero, noise)
    assert torch.allclose(z_one, z_clean)


def test_weight_function_modes():
    t = torch.tensor([0.0, 0.5, 0.9])

    uniform = w(t, mode="uniform")
    snr = w(t, mode="snr", eps=1e-6)
    clipped = w(t, mode="clipped_snr", snr_clip_max=2.0, eps=1e-6)

    assert torch.allclose(uniform, torch.ones_like(t))
    assert snr[0] == pytest.approx(0.0, abs=1e-6)
    assert snr[2] > snr[1]
    assert clipped.max() <= 2.0


def test_flow_matching_validates_shapes_and_params():
    with pytest.raises(ValueError, match="batch_size"):
        sample_t(batch_size=0)

    with pytest.raises(ValueError, match="Expected 0 <= t_min"):
        sample_t(batch_size=2, t_min=0.9, t_max=0.2)

    with pytest.raises(ValueError, match=r"z_clean must be at least rank-2"):
        make_noisy_and_target(torch.randn(4), torch.rand(4))

    with pytest.raises(ValueError, match=r"t must have shape \[B\]"):
        make_noisy_and_target(torch.randn(2, 3), torch.rand(2, 1))

    with pytest.raises(ValueError, match="must be a floating tensor"):
        w(torch.tensor([0, 1], dtype=torch.long))

    with pytest.raises(ValueError, match="Unsupported weight mode"):
        w(torch.tensor([0.1], dtype=torch.float32), mode="bad")


def test_chunkwise_teacher_forcing_loss_iterates_chunks_and_backprops():
    torch.manual_seed(0)
    b, t_past, t_future, d = 2, 3, 8, 4
    model = _RecordingChunkModel(latent_dim=d)
    z_past = torch.randn(b, t_past, d)
    z_future = torch.randn(b, t_future, d)
    action = torch.randn(b, 6)

    info = chunkwise_teacher_forcing_loss(
        model,
        z_past=z_past,
        z_future=z_future,
        action_conditioning=action,
        k=1,
        t_min=0.3,
        t_max=0.3,
        return_info=True,
    )
    info.loss.backward()

    assert len(model.calls) == 2
    assert info.per_chunk_lengths == (4, 4)
    assert model.calls[0]["past_len"].item() == t_past
    assert model.calls[1]["past_len"].item() == t_past + 4
    assert torch.isfinite(info.loss)
    assert model.scale.grad is not None
    assert model.scale.grad.abs().item() > 0.0


def test_chunkwise_teacher_forcing_mask_blocks_future_chunks():
    torch.manual_seed(0)
    b, t_past, t_future, d = 1, 2, 9, 3
    model = _RecordingChunkModel(latent_dim=d)
    z_past = torch.randn(b, t_past, d)
    z_future = torch.randn(b, t_future, d)
    action = torch.randn(b, 5)

    _ = chunkwise_teacher_forcing_loss(
        model,
        z_past=z_past,
        z_future=z_future,
        action_conditioning=action,
        k=2,
        t_min=0.4,
        t_max=0.4,
    )

    # For k=2 and t_future=9, boundaries are [(0, 3), (3, 6), (6, 9)].
    first_mask = model.calls[0]["mask"]
    second_mask = model.calls[1]["mask"]

    # In call 0, current chunk rows cannot attend to keys from future chunks.
    assert torch.isinf(first_mask[t_past : t_past + 3, t_past + 3 :]).all()
    # In call 1, rows for chunk 1 cannot attend chunk 2 keys.
    assert torch.isinf(second_mask[t_past + 3 : t_past + 6, t_past + 6 :]).all()


def test_chunkwise_teacher_forcing_validates_shapes():
    with pytest.raises(ValueError, match="z_past must be"):
        chunkwise_teacher_forcing_loss(
            _RecordingChunkModel(latent_dim=3),
            z_past=torch.randn(2, 3),
            z_future=torch.randn(2, 4, 3),
            action_conditioning=torch.randn(2, 5),
            k=1,
        )
