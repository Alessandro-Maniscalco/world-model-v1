"""Unit tests for flow-matching utility helpers."""

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import pytest
import torch
from world_model.training.flow_matching import (
    _compute_future_chunk_early_weight,
    _compute_future_loss_early_weight,
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


def test_chunkwise_teacher_forcing_loss_rejects_negative_future_loss_early_bias():
    """Reject negative early-horizon loss bias because it would downweight early frames."""
    model = _ChunkActionWindowRecorder()

    with pytest.raises(ValueError, match="future_loss_early_bias"):
        chunkwise_teacher_forcing_loss(
            model,
            z_past_video=torch.randn(1, 2, 2, 2, 2),
            z_future_video=torch.randn(1, 2, 2, 2, 2),
            action_tokens=torch.randn(1, 2, 1),
            k=1,
            future_loss_early_bias=-0.1,
        )


def test_chunkwise_teacher_forcing_loss_rejects_negative_future_chunk_early_bias():
    """Reject negative early-chunk bias because it would downweight earlier chunks."""
    model = _ChunkActionWindowRecorder()

    with pytest.raises(ValueError, match="future_chunk_early_bias"):
        chunkwise_teacher_forcing_loss(
            model,
            z_past_video=torch.randn(1, 2, 2, 2, 2),
            z_future_video=torch.randn(1, 2, 2, 2, 2),
            action_tokens=torch.randn(1, 2, 1),
            k=1,
            future_chunk_early_bias=-0.1,
        )


def test_compute_future_chunk_early_weight_emphasizes_earlier_chunks() -> None:
    """Give the first autoregressive chunk the largest scalar weight."""
    early = _compute_future_chunk_early_weight(
        chunk_index=0,
        num_chunks=3,
        bias=0.6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    middle = _compute_future_chunk_early_weight(
        chunk_index=1,
        num_chunks=3,
        bias=0.6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    late = _compute_future_chunk_early_weight(
        chunk_index=2,
        num_chunks=3,
        bias=0.6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert early.item() == pytest.approx(1.6)
    assert middle.item() == pytest.approx(1.3)
    assert late.item() == pytest.approx(1.0)


def test_chunkwise_teacher_forcing_loss_chunk_bias_favors_earlier_chunk_errors(monkeypatch) -> None:
    """Increase the loss more when the same error sits in an earlier chunk."""

    def _fake_make_noisy_and_target(z_clean: torch.Tensor, t: torch.Tensor, *, noise=None):
        del t, noise
        return z_clean.clone(), z_clean.clone()

    class _ZeroPredictor:
        """Return zero velocity so the target magnitude controls the loss."""

        def __call__(
            self,
            *,
            noisy_future_video: torch.Tensor,
            observed_video: torch.Tensor,
            action_tokens: torch.Tensor,
            timestep_t: torch.Tensor,
            block_causal_attention_mask: torch.Tensor,
            observed_mask: torch.Tensor | None = None,
            future_action_control_prior: torch.Tensor | None = None,
            control_hidden_states_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del (
                observed_video,
                action_tokens,
                timestep_t,
                block_causal_attention_mask,
                observed_mask,
                future_action_control_prior,
                control_hidden_states_scale,
            )
            return torch.zeros_like(noisy_future_video)

    monkeypatch.setattr("world_model.training.flow_matching.make_noisy_and_target", _fake_make_noisy_and_target)

    common_kwargs = {
        "model": _ZeroPredictor(),
        "z_past_video": torch.zeros(1, 1, 1, 1, 1),
        "action_tokens": torch.zeros(1, 3, 1),
        "k": 2,
        "t_min": 0.0,
        "t_max": 0.0,
    }
    early_error = torch.tensor([[[[[1.0]], [[0.0]], [[0.0]]]]])
    late_error = torch.tensor([[[[[0.0]], [[0.0]], [[1.0]]]]])

    unbiased_early = chunkwise_teacher_forcing_loss(
        z_future_video=early_error,
        future_chunk_early_bias=0.0,
        **common_kwargs,
    )
    unbiased_late = chunkwise_teacher_forcing_loss(
        z_future_video=late_error,
        future_chunk_early_bias=0.0,
        **common_kwargs,
    )
    biased_early = chunkwise_teacher_forcing_loss(
        z_future_video=early_error,
        future_chunk_early_bias=1.0,
        **common_kwargs,
    )
    biased_late = chunkwise_teacher_forcing_loss(
        z_future_video=late_error,
        future_chunk_early_bias=1.0,
        **common_kwargs,
    )

    assert unbiased_early.item() == pytest.approx(unbiased_late.item())
    assert biased_early.item() > biased_late.item()


class _ChunkActionWindowRecorder:
    """Record per-chunk action-token windows during teacher forcing."""

    def __init__(self) -> None:
        """Initialize captured action-window storage."""
        self.action_windows: list[torch.Tensor] = []
        self.action_control_priors: list[torch.Tensor | None] = []
        self.observed_videos: list[torch.Tensor] = []
        self.noisy_future_videos: list[torch.Tensor] = []

    def __call__(
        self,
        *,
        noisy_future_video: torch.Tensor,
        observed_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_t: torch.Tensor,
        block_causal_attention_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        future_action_control_prior: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture the active action window and emit zero velocities."""
        del timestep_t, block_causal_attention_mask, observed_mask, control_hidden_states_scale
        self.noisy_future_videos.append(noisy_future_video.detach().clone())
        self.action_windows.append(action_tokens.detach().clone())
        self.observed_videos.append(observed_video.detach().clone())
        self.action_control_priors.append(
            None if future_action_control_prior is None else future_action_control_prior.detach().clone()
        )
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


def test_chunkwise_teacher_forcing_can_use_full_action_plan_on_every_chunk() -> None:
    """Reuse the full future action plan when full-plan conditioning is requested."""
    model = _ChunkActionWindowRecorder()
    action_tokens = torch.arange(5, dtype=torch.float32).view(1, 5, 1)

    chunkwise_teacher_forcing_loss(
        model,
        z_past_video=torch.randn(1, 2, 3, 1, 1),
        z_future_video=torch.randn(1, 2, 5, 1, 1),
        action_tokens=action_tokens,
        action_conditioning_window="full",
        k=2,
        t_min=0.4,
        t_max=0.4,
    )

    assert len(model.action_windows) == 3
    assert all(torch.equal(window, action_tokens) for window in model.action_windows)


def test_chunkwise_teacher_forcing_can_match_rollout_with_active_chunk_future_inputs() -> None:
    """Restrict teacher forcing to the active chunk so future inputs match rollout shape."""
    full_suffix_model = _ChunkActionWindowRecorder()
    active_chunk_model = _ChunkActionWindowRecorder()
    action_tokens = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    common_kwargs = {
        "z_past_video": torch.randn(1, 2, 3, 1, 1),
        "z_future_video": torch.randn(1, 2, 5, 1, 1),
        "action_tokens": action_tokens,
        "k": 2,
        "t_min": 0.4,
        "t_max": 0.4,
    }

    chunkwise_teacher_forcing_loss(
        full_suffix_model,
        teacher_forcing_future_input_mode="full_suffix",
        **common_kwargs,
    )
    chunkwise_teacher_forcing_loss(
        active_chunk_model,
        teacher_forcing_future_input_mode="active_chunk",
        **common_kwargs,
    )

    assert [video.shape[2] for video in full_suffix_model.noisy_future_videos] == [5, 3, 1]
    assert [video.shape[2] for video in active_chunk_model.noisy_future_videos] == [2, 2, 1]


def test_chunkwise_teacher_forcing_can_hide_future_prefix_from_later_chunks() -> None:
    """Hide the ground-truth future prefix from later chunks when requested."""
    full_prefix_model = _ChunkActionWindowRecorder()
    past_only_model = _ChunkActionWindowRecorder()
    z_past_video = torch.tensor([[[[[10.0]], [[11.0]]]]])
    z_future_video = torch.tensor([[[[[20.0]], [[21.0]], [[22.0]], [[23.0]], [[24.0]]]]])
    action_tokens = torch.arange(5, dtype=torch.float32).view(1, 5, 1)

    chunkwise_teacher_forcing_loss(
        full_prefix_model,
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        action_tokens=action_tokens,
        teacher_forcing_observation_mode="full_prefix",
        k=2,
        t_min=0.4,
        t_max=0.4,
    )
    chunkwise_teacher_forcing_loss(
        past_only_model,
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        action_tokens=action_tokens,
        teacher_forcing_observation_mode="past_only",
        k=2,
        t_min=0.4,
        t_max=0.4,
    )

    assert len(full_prefix_model.observed_videos) == 3
    assert torch.equal(full_prefix_model.observed_videos[0], z_past_video)
    assert torch.equal(
        full_prefix_model.observed_videos[1],
        torch.cat((z_past_video, z_future_video[:, :, :2, :, :]), dim=2),
    )
    assert torch.equal(
        full_prefix_model.observed_videos[2],
        torch.cat((z_past_video, z_future_video[:, :, :4, :, :]), dim=2),
    )
    assert all(torch.equal(observed, z_past_video) for observed in past_only_model.observed_videos)


def test_chunkwise_teacher_forcing_can_use_predicted_prefix_for_later_chunks(monkeypatch) -> None:
    """Feed detached predicted chunks back as the observed prefix when requested."""

    def _fake_make_noisy_and_target(
        z_clean: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del t, noise
        return z_clean.clone(), torch.zeros_like(z_clean)

    class _PredictedPrefixRecorder:
        """Record observed prefixes while returning a constant velocity field."""

        def __init__(self) -> None:
            """Initialize observed-prefix storage."""
            self.observed_videos: list[torch.Tensor] = []

        def __call__(
            self,
            *,
            noisy_future_video: torch.Tensor,
            observed_video: torch.Tensor,
            action_tokens: torch.Tensor,
            timestep_t: torch.Tensor,
            block_causal_attention_mask: torch.Tensor,
            observed_mask: torch.Tensor | None = None,
            future_action_control_prior: torch.Tensor | None = None,
            control_hidden_states_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Capture observed prefixes and return a constant unit velocity."""
            del (
                action_tokens,
                timestep_t,
                block_causal_attention_mask,
                observed_mask,
                future_action_control_prior,
                control_hidden_states_scale,
            )
            self.observed_videos.append(observed_video.detach().clone())
            return torch.ones_like(noisy_future_video)

    monkeypatch.setattr("world_model.training.flow_matching.make_noisy_and_target", _fake_make_noisy_and_target)

    model = _PredictedPrefixRecorder()
    z_past_video = torch.tensor([[[[[10.0]], [[11.0]]]]])
    z_future_video = torch.tensor([[[[[20.0]], [[21.0]], [[22.0]], [[23.0]], [[24.0]]]]])
    action_tokens = torch.arange(5, dtype=torch.float32).view(1, 5, 1)

    chunkwise_teacher_forcing_loss(
        model,
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        action_tokens=action_tokens,
        teacher_forcing_observation_mode="predicted_prefix",
        k=2,
        t_min=0.5,
        t_max=0.5,
    )

    assert len(model.observed_videos) == 3
    assert torch.equal(model.observed_videos[0], z_past_video)
    assert torch.equal(
        model.observed_videos[1],
        torch.cat((z_past_video, z_future_video[:, :, :2, :, :] - 0.5), dim=2),
    )
    assert torch.equal(
        model.observed_videos[2],
        torch.cat((z_past_video, z_future_video[:, :, :4, :, :] - 0.5), dim=2),
    )


def test_chunkwise_teacher_forcing_aligns_action_control_prior_to_suffix_modes() -> None:
    """Align chunk-mode priors to the active chunk and full-mode priors to the future suffix."""
    chunk_model = _ChunkActionWindowRecorder()
    full_model = _ChunkActionWindowRecorder()
    action_tokens = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    action_control_prior = torch.arange(1 * 2 * 4 * 1 * 1, dtype=torch.float32).view(1, 2, 4, 1, 1)

    chunkwise_teacher_forcing_loss(
        chunk_model,
        z_past_video=torch.randn(1, 2, 2, 1, 1),
        z_future_video=torch.randn(1, 2, 4, 1, 1),
        action_tokens=action_tokens,
        action_control_prior=action_control_prior,
        action_conditioning_window="chunk",
        k=1,
        t_min=0.4,
        t_max=0.4,
    )
    chunkwise_teacher_forcing_loss(
        full_model,
        z_past_video=torch.randn(1, 2, 2, 1, 1),
        z_future_video=torch.randn(1, 2, 4, 1, 1),
        action_tokens=action_tokens,
        action_control_prior=action_control_prior,
        action_conditioning_window="full",
        k=1,
        t_min=0.4,
        t_max=0.4,
    )

    assert chunk_model.action_control_priors[0] is not None
    expected_first_chunk_prior = torch.cat(
        [
            action_control_prior[:, :, 0:2],
            torch.zeros_like(action_control_prior[:, :, 0:2]),
        ],
        dim=2,
    )
    assert torch.equal(chunk_model.action_control_priors[0], expected_first_chunk_prior)
    assert chunk_model.action_control_priors[1] is not None
    assert chunk_model.action_control_priors[1].shape[2] == 2
    assert torch.equal(chunk_model.action_control_priors[1], action_control_prior[:, :, 2:4])
    assert torch.equal(full_model.action_control_priors[0], action_control_prior[:, :, 0:4])
    assert torch.equal(full_model.action_control_priors[1], action_control_prior[:, :, 2:4])


def test_chunkwise_teacher_forcing_trims_full_plan_action_control_prior_for_active_chunk_inputs() -> None:
    """Trim full-plan action priors to the active chunk when rollout-matched inputs are used."""
    model = _ChunkActionWindowRecorder()
    action_tokens = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    action_control_prior = torch.arange(1 * 2 * 4 * 1 * 1, dtype=torch.float32).view(1, 2, 4, 1, 1)

    chunkwise_teacher_forcing_loss(
        model,
        z_past_video=torch.randn(1, 2, 2, 1, 1),
        z_future_video=torch.randn(1, 2, 4, 1, 1),
        action_tokens=action_tokens,
        action_control_prior=action_control_prior,
        action_conditioning_window="full",
        teacher_forcing_future_input_mode="active_chunk",
        k=1,
        t_min=0.4,
        t_max=0.4,
    )

    assert all(window.shape[1] == 4 for window in model.action_windows)
    assert [video.shape[2] for video in model.noisy_future_videos] == [2, 2]
    assert torch.equal(model.action_control_priors[0], action_control_prior[:, :, 0:2])
    assert torch.equal(model.action_control_priors[1], action_control_prior[:, :, 2:4])


def test_chunkwise_teacher_forcing_rejects_unknown_observation_mode() -> None:
    """Reject unsupported observation modes before teacher forcing begins."""
    model = _ChunkActionWindowRecorder()

    with pytest.raises(ValueError, match="teacher_forcing_observation_mode"):
        chunkwise_teacher_forcing_loss(
            model,
            z_past_video=torch.randn(1, 2, 2, 1, 1),
            z_future_video=torch.randn(1, 2, 2, 1, 1),
            action_tokens=torch.randn(1, 2, 1),
            teacher_forcing_observation_mode="bad",
            k=1,
        )


def test_chunkwise_teacher_forcing_rejects_unknown_future_input_mode() -> None:
    """Reject unsupported future-input modes before teacher forcing begins."""
    model = _ChunkActionWindowRecorder()

    with pytest.raises(ValueError, match="teacher_forcing_future_input_mode"):
        chunkwise_teacher_forcing_loss(
            model,
            z_past_video=torch.randn(1, 2, 2, 1, 1),
            z_future_video=torch.randn(1, 2, 2, 1, 1),
            action_tokens=torch.randn(1, 2, 1),
            teacher_forcing_future_input_mode="bad",
            k=1,
        )


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
        future_action_control_prior: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ignore inputs and emit zeros with the same shape as the noisy suffix."""
        del (
            observed_video,
            action_tokens,
            timestep_t,
            block_causal_attention_mask,
            observed_mask,
            future_action_control_prior,
            control_hidden_states_scale,
        )
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


def test_motion_loss_weight_excess_only_leaves_average_motion_at_base_weight():
    """Upweight only above-average motion regions when excess-only weighting is enabled."""
    z_past = torch.zeros(1, 1, 1, 1, 1)
    clean_chunk = torch.tensor([[[[[1.0]], [[3.0]]]]], dtype=torch.float32)

    weight_default = _compute_motion_loss_weight(
        observed_video=z_past,
        clean_chunk=clean_chunk,
        alpha=1.0,
    )
    weight_excess = _compute_motion_loss_weight(
        observed_video=z_past,
        clean_chunk=clean_chunk,
        alpha=1.0,
        excess_only=True,
    )

    assert float(weight_default[0, 0, 0, 0, 0]) > 1.0
    assert float(weight_excess[0, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert float(weight_excess[0, 0, 1, 0, 0]) > 1.0


def test_future_loss_early_weight_prefers_earlier_future_positions():
    """Apply larger loss weights to earlier future steps when enabled."""
    weight = _compute_future_loss_early_weight(
        start=1,
        end=4,
        total_future_steps=5,
        bias=0.5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert weight.shape == (1, 1, 3, 1, 1)
    assert float(weight[0, 0, 0, 0, 0]) > float(weight[0, 0, 1, 0, 0]) > float(weight[0, 0, 2, 0, 0])
    assert float(weight[0, 0, 0, 0, 0]) == pytest.approx(1.375)
    assert float(weight[0, 0, 2, 0, 0]) == pytest.approx(1.125)
