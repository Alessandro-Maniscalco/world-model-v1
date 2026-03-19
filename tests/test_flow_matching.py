"""Unit tests for flow-matching utility helpers."""

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import pytest
import torch
from world_model.training.flow_matching import (
    _compute_motion_loss_weight,
    chunkwise_teacher_forcing_loss,
    make_noisy_and_target,
    normalized_t_to_scheduler_timestep,
    sample_t,
    w,
)


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

    expected_0 = 0.75 * z_clean[0] + 0.25 * noise[0]
    expected_1 = 0.25 * z_clean[1] + 0.75 * noise[1]
    expected_z_t = torch.stack((expected_0, expected_1), dim=0)
    expected_v = noise - z_clean

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

    assert torch.allclose(z_zero, z_clean)
    assert torch.allclose(z_one, noise)


def test_normalized_t_to_scheduler_timestep_matches_flowmatch_scale():
    t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    scaled = normalized_t_to_scheduler_timestep(t)

    assert torch.allclose(scaled, torch.tensor([0.0, 500.0, 1000.0]))


def test_make_noisy_and_target_matches_euler_scheduler_direction():
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(1)
    clean = torch.tensor([[2.0]], dtype=torch.float32)
    noise = torch.tensor([[0.0]], dtype=torch.float32)
    sigma = torch.tensor([1.0], dtype=torch.float32)

    noisy, target = make_noisy_and_target(clean, sigma, noise=noise)
    denoised = scheduler.step(target, scheduler.timesteps[0], noisy, return_dict=False)[0]

    assert torch.allclose(noisy, noise)
    assert torch.allclose(denoised, clean)


def test_weight_function_modes():
    t = torch.tensor([0.1, 0.5, 0.9])

    uniform = w(t, mode="uniform")
    snr = w(t, mode="snr", eps=1e-6)
    clipped = w(t, mode="clipped_snr", snr_clip_max=2.0, eps=1e-6)

    assert torch.allclose(uniform, torch.ones_like(t))
    assert snr[0] > snr[1] > snr[2]
    assert clipped[0] == pytest.approx(2.0, abs=1e-6)
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

    with pytest.raises(ValueError, match="must be a floating tensor"):
        normalized_t_to_scheduler_timestep(torch.tensor([0, 1], dtype=torch.long))

    with pytest.raises(ValueError, match="Unsupported weight mode"):
        w(torch.tensor([0.1], dtype=torch.float32), mode="bad")


def test_chunkwise_teacher_forcing_loss_rejects_negative_motion_loss_alpha():
    """Reject negative motion weighting because it would downweight moving regions."""
    model = _ChunkActionWindowRecorder()

    with pytest.raises(ValueError, match="motion_loss_alpha"):
        chunkwise_teacher_forcing_loss(
            model,
            z_past_video=torch.randn(1, 2, 2, 2, 2),
            z_future_video=torch.randn(1, 2, 2, 2, 2),
            action_tokens=torch.randn(1, 2, 1),
            k=1,
            motion_loss_alpha=-0.1,
        )


def test_chunkwise_teacher_forcing_loss_rejects_fractional_motion_loss_cap():
    """Reject motion-loss caps below 1 because they would downweight the base loss."""
    model = _ChunkActionWindowRecorder()

    with pytest.raises(ValueError, match="motion_loss_max_weight"):
        chunkwise_teacher_forcing_loss(
            model,
            z_past_video=torch.randn(1, 2, 2, 2, 2),
            z_future_video=torch.randn(1, 2, 2, 2, 2),
            action_tokens=torch.randn(1, 2, 1),
            k=1,
            motion_loss_max_weight=0.5,
        )


class _ChunkActionWindowRecorder:
    """Record per-chunk action-token windows during teacher forcing."""

    def __init__(self) -> None:
        """Initialize captured action-window storage."""
        self.action_windows: list[torch.Tensor] = []

    def __call__(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture the active action window and emit zero velocities."""
        del observed_video, timestep_t, block_causal_attention_mask, observed_mask, control_hidden_states_scale
        self.action_windows.append(action_tokens.detach().clone())
        return torch.zeros_like(noisy_future_video)


def test_chunkwise_teacher_forcing_uses_current_chunk_action_window():
    """Use only the active chunk's action tokens in each teacher-forced call."""
    model = _ChunkActionWindowRecorder()
    action_tokens = torch.arange(5, dtype=torch.float32).view(1, 5, 1)

    chunkwise_teacher_forcing_loss(
        model,
        z_past_video=torch.randn(1, 2, 3, 1, 1),
        z_future_video=torch.randn(1, 2, 5, 1, 1),
        action_tokens=action_tokens,
        k=2,
        t_min=0.4,
        t_max=0.4,
    )

    assert len(model.action_windows) == 3
    assert torch.equal(model.action_windows[0], action_tokens[:, 0:2])
    assert torch.equal(model.action_windows[1], action_tokens[:, 2:4])
    assert torch.equal(model.action_windows[2], action_tokens[:, 4:5])


class _ZeroVelocityModel:
    """Return zero velocity predictions for deterministic loss comparisons."""

    def __call__(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ignore inputs and emit zeros with the same shape as the noisy suffix."""
        del observed_video, action_tokens, timestep_t, block_causal_attention_mask, observed_mask, control_hidden_states_scale
        return torch.zeros_like(noisy_future_video)


def test_motion_loss_weight_upweights_moving_regions():
    """Assign larger per-pixel weights to moving latent regions when enabled."""
    z_past = torch.zeros(1, 1, 1, 1, 1)
    clean_chunk = torch.tensor([[[[[0.0]], [[3.0]]]]], dtype=torch.float32)

    weight_uniform = _compute_motion_loss_weight(
        observed_video=z_past,
        clean_chunk=clean_chunk,
        alpha=0.0,
    )
    weight_motion = _compute_motion_loss_weight(
        observed_video=z_past,
        clean_chunk=clean_chunk,
        alpha=1.0,
    )

    assert torch.allclose(weight_uniform, torch.ones_like(clean_chunk))
    assert float(weight_motion[0, 0, 1, 0, 0]) > float(weight_motion[0, 0, 0, 0, 0])


def test_motion_loss_weight_cap_limits_peak_weight_without_removing_motion_bias():
    """Keep moving regions upweighted while preventing unbounded motion-loss spikes."""
    z_past = torch.zeros(1, 1, 1, 1, 1)
    clean_chunk = torch.tensor([[[[[0.0]], [[9.0]]]]], dtype=torch.float32)

    weight_capped = _compute_motion_loss_weight(
        observed_video=z_past,
        clean_chunk=clean_chunk,
        alpha=1.0,
        max_weight=2.0,
    )

    assert float(weight_capped[0, 0, 1, 0, 0]) == pytest.approx(2.0)
    assert float(weight_capped[0, 0, 1, 0, 0]) > float(weight_capped[0, 0, 0, 0, 0])
