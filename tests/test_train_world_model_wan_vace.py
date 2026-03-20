"""Tests for the Wan VACE training entrypoint wiring."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

from world_model.config import TrainScriptConfig, load_train_config
from world_model.data.schema import PreparedPackedBatch
from world_model.models import WanVACEWorldModel
from world_model.models.wan_vace_conditioning import (
    ActionControlProjector,
    ActionTokenEncoder,
    NullActionControlProjector,
    NullConditioningEncoder,
)
from world_model.training import ChunkwiseStepMetrics


def test_train_script_builds_wan_vace_world_model_from_config() -> None:
    """Build the Wan VACE model and null-conditioning encoder from train config."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
        control_scale=0.75,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, NullConditioningEncoder)
    assert model.mask_channels == 4
    assert model.control_scale == 0.75
    assert model.backbone.config.text_dim == 16
    assert model.backbone.config.in_channels == 16
    assert model.backbone.config.vace_in_channels == 36
    assert tuple(model.backbone.config.vace_layers) == (0, 1)
    assert action_encoder.hidden_dim == 16


def test_train_script_builds_action_encoder_when_requested() -> None:
    """Keep the action-token encoder available for the later action-conditioning stage."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)

    assert isinstance(action_encoder, ActionTokenEncoder)


def test_train_script_builds_action_control_projector_when_requested() -> None:
    """Keep the action-derived latent prior projector available for ordered-plan runs."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        action_control_prior_scale=1.0,
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    projector = train_script.build_action_control_projector_from_config(cfg, prepared, model)

    assert isinstance(projector, ActionControlProjector)


def test_train_script_builds_action_encoder_with_mlp_when_requested() -> None:
    """Allow the train config to request a deeper action-token encoder."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_mlp_dim=10,
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.net[1].out_features == 10
    assert action_encoder.net[4].out_features == 16


def test_train_script_builds_action_encoder_with_residual_mlp_when_requested() -> None:
    """Allow the train config to request a residual action-token MLP."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_mlp_dim=10,
        action_mlp_residual=True,
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.mlp_residual is True
    assert action_encoder.residual_net is not None
    assert action_encoder.net[1].out_features == 16
    assert action_encoder.residual_net[0].out_features == 10


def test_train_script_builds_action_encoder_with_temporal_difference_scale_when_requested() -> None:
    """Allow the train config to request temporal-difference-aware action tokens."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_temporal_difference_scale=0.75,
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.temporal_difference_scale == pytest.approx(0.75)


def test_train_script_builds_action_encoder_with_temporal_mixer_when_requested() -> None:
    """Allow the train config to request a temporal mixer over action tokens."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_temporal_mixer_kernel_size=3,
        action_temporal_mixer_scale=0.5,
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.temporal_mixer is not None
    assert action_encoder.temporal_mixer_scale == pytest.approx(0.5)


def test_train_script_parser_omits_legacy_dit_shape_flags() -> None:
    """Avoid exposing removed non-VACE backbone shape flags."""
    train_script = _load_train_script_module()
    parser = train_script._build_parser(load_train_config())
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--resume-from" in option_strings
    assert "--motion-loss-alpha" in option_strings
    assert "--motion-loss-max-weight" in option_strings
    assert "--motion-loss-excess-only" in option_strings
    assert "--action-conditioning-window" in option_strings
    assert "--action-order-conditioning" in option_strings
    assert "--action-control-prior-scale" in option_strings
    assert "--teacher-forcing-observation-mode" in option_strings
    assert "--action-temporal-difference-scale" in option_strings
    assert "--action-temporal-mixer-kernel-size" in option_strings
    assert "--action-temporal-mixer-scale" in option_strings
    assert "--hidden-dim" not in option_strings
    assert "--num-layers" not in option_strings
    assert "--num-heads" not in option_strings


def test_train_script_uses_bf16_when_cuda_supports_it(monkeypatch) -> None:
    """Prefer bf16 autocast on CUDA when the runtime reports support."""
    train_script = _load_train_script_module()

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    dtype = train_script._select_runtime_dtype(device=torch.device("cuda"), disable_amp=False)

    assert dtype == torch.bfloat16


def test_train_script_falls_back_to_fp16_without_bf16_support(monkeypatch) -> None:
    """Fall back to fp16 autocast on CUDA devices that lack bf16 support."""
    train_script = _load_train_script_module()

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    dtype = train_script._select_runtime_dtype(device=torch.device("cuda"), disable_amp=False)

    assert dtype == torch.float16


def test_train_script_disables_amp_outside_cuda() -> None:
    """Keep training in fp32 when AMP is disabled or CUDA is unavailable."""
    train_script = _load_train_script_module()

    assert train_script._select_runtime_dtype(device=torch.device("cpu"), disable_amp=False) == torch.float32
    assert train_script._select_runtime_dtype(device=torch.device("cuda"), disable_amp=True) == torch.float32


def test_train_script_validates_latent_schedule_for_k_plus_one_chunking() -> None:
    """Explain latent-time chunking failures before the train loop starts."""
    train_script = _load_train_script_module()
    cfg = TrainScriptConfig(k=1, horizon_len=4)
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 1, 8, 8),
        a_plan=torch.randn(1, 1, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=3,
        context_latent_steps=2,
        horizon_latent_steps=1,
    )

    with pytest.raises(ValueError, match="horizon_latent_steps=1"):
        train_script._validate_chunk_schedule(cfg, prepared)


def test_train_script_accepts_valid_latent_schedule() -> None:
    """Allow training to proceed when latent future steps can cover K+1 chunks."""
    train_script = _load_train_script_module()
    cfg = TrainScriptConfig(k=1, horizon_len=8)
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )

    train_script._validate_chunk_schedule(cfg, prepared)


def test_train_script_accepts_disabled_auto_stop() -> None:
    """Allow the default disabled auto-stop configuration."""
    train_script = _load_train_script_module()
    train_script._validate_auto_stop_config(TrainScriptConfig())


def test_train_script_can_resume_training_state(tmp_path: Path) -> None:
    """Restore model, action encoder, optimizer, and completed step from a saved checkpoint."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    source_model = train_script.build_model_from_config(cfg, prepared)
    source_action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, source_model)
    source_parameters = train_script._configure_trainable_parameters(cfg, source_model, source_action_encoder)
    source_optimizer = torch.optim.AdamW(source_parameters, lr=1e-3)

    for parameter in source_model.parameters():
        parameter.data.fill_(0.25)
    for parameter in source_action_encoder.parameters():
        parameter.data.fill_(-0.5)

    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        {
            "step": 123,
            "model_state_dict": source_model.state_dict(),
            "action_encoder_state_dict": source_action_encoder.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
        },
        checkpoint_path,
    )

    resumed_model = train_script.build_model_from_config(cfg, prepared)
    resumed_action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, resumed_model)
    resumed_parameters = train_script._configure_trainable_parameters(cfg, resumed_model, resumed_action_encoder)
    resumed_optimizer = torch.optim.AdamW(resumed_parameters, lr=1e-3)

    checkpoint = train_script._load_training_checkpoint(checkpoint_path)
    resumed_step, restored_optimizer_state = train_script._resume_training_state(
        checkpoint=checkpoint,
        model=resumed_model,
        action_encoder=resumed_action_encoder,
        optimizer=resumed_optimizer,
    )
    train_script._optimizer_state_to_device(resumed_optimizer, device=torch.device("cpu"))

    assert resumed_step == 123
    assert restored_optimizer_state is True
    first_model_key = next(iter(source_model.state_dict()))
    first_action_key = next(iter(source_action_encoder.state_dict()))
    assert torch.allclose(resumed_model.state_dict()[first_model_key], source_model.state_dict()[first_model_key])
    assert torch.allclose(
        resumed_action_encoder.state_dict()[first_action_key],
        source_action_encoder.state_dict()[first_action_key],
    )


def test_train_script_can_resume_old_action_checkpoint_without_temporal_mixer_optimizer_state(tmp_path: Path) -> None:
    """Allow older checkpoints to seed a new temporal mixer while keeping a fresh optimizer."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )
    old_cfg = TrainScriptConfig(
        conditioning_mode="action",
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )
    new_cfg = TrainScriptConfig(
        conditioning_mode="action",
        action_temporal_mixer_kernel_size=3,
        action_temporal_mixer_scale=0.5,
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    source_model = train_script.build_model_from_config(old_cfg, prepared)
    source_action_encoder = train_script.build_action_encoder_from_config(old_cfg, prepared, source_model)
    source_parameters = train_script._configure_trainable_parameters(old_cfg, source_model, source_action_encoder)
    source_optimizer = torch.optim.AdamW(source_parameters, lr=1e-3)

    checkpoint_path = tmp_path / "resume_old.pt"
    torch.save(
        {
            "step": 123,
            "model_state_dict": source_model.state_dict(),
            "action_encoder_state_dict": source_action_encoder.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
        },
        checkpoint_path,
    )

    resumed_model = train_script.build_model_from_config(new_cfg, prepared)
    resumed_action_encoder = train_script.build_action_encoder_from_config(new_cfg, prepared, resumed_model)
    resumed_parameters = train_script._configure_trainable_parameters(new_cfg, resumed_model, resumed_action_encoder)
    resumed_optimizer = torch.optim.AdamW(resumed_parameters, lr=1e-3)

    checkpoint = train_script._load_training_checkpoint(checkpoint_path)
    resumed_step, restored_optimizer_state = train_script._resume_training_state(
        checkpoint=checkpoint,
        model=resumed_model,
        action_encoder=resumed_action_encoder,
        optimizer=resumed_optimizer,
    )

    assert resumed_step == 123
    assert restored_optimizer_state is False
    assert resumed_action_encoder.temporal_mixer is not None
    assert torch.count_nonzero(resumed_action_encoder.temporal_mixer.weight) == 0


def test_train_script_rejects_missing_resume_checkpoint(tmp_path: Path) -> None:
    """Fail clearly when --resume-from points to a nonexistent file."""
    train_script = _load_train_script_module()
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Training checkpoint not found"):
        train_script._load_training_checkpoint(missing_path)


def test_train_script_allows_auto_stop_check_beyond_max_steps() -> None:
    """Allow auto-stop intervals longer than the configured training run."""
    train_script = _load_train_script_module()

    train_script._validate_auto_stop_config(
        TrainScriptConfig(max_steps=1000, auto_stop_check_every=5000),
    )


def test_train_script_uses_piecewise_checkpoint_schedule_when_configured() -> None:
    """Save more frequently during early steps before falling back to the long-run interval."""
    train_script = _load_train_script_module()
    cfg = TrainScriptConfig(
        checkpoint_every=500,
        checkpoint_early_every=100,
        checkpoint_early_until=500,
    )

    save_steps = [step for step in range(1, 1601) if train_script._should_save_checkpoint(cfg, step)]

    assert save_steps == [100, 200, 300, 400, 500, 1000, 1500]


def test_train_script_keeps_legacy_checkpoint_schedule_when_early_window_is_disabled() -> None:
    """Preserve the old fixed-interval checkpoint behavior by default."""
    train_script = _load_train_script_module()
    cfg = TrainScriptConfig(checkpoint_every=100)

    save_steps = [step for step in range(1, 351) if train_script._should_save_checkpoint(cfg, step)]

    assert save_steps == [100, 200, 300]


def test_train_script_runs_validation_on_its_own_schedule() -> None:
    """Allow validation cadence to be more frequent than checkpoint cadence."""
    train_script = _load_train_script_module()
    cfg = TrainScriptConfig(validation_enabled=True, validation_every=50)

    validation_steps = [step for step in range(1, 161) if train_script._should_run_validation(cfg, step)]

    assert validation_steps == [50, 100, 150]


def test_train_script_continues_after_first_auto_stop_block() -> None:
    """Always continue after the first completed block because no prior block exists yet."""
    train_script = _load_train_script_module()

    should_continue, improvement = train_script._should_continue_after_block(
        block_mean_losses=[0.8],
        min_relative_improvement=0.05,
    )

    assert should_continue is True
    assert improvement is None


def test_train_script_stops_when_block_improvement_is_too_small() -> None:
    """Stop once the latest block fails to improve enough over the prior block mean."""
    train_script = _load_train_script_module()

    should_continue, improvement = train_script._should_continue_after_block(
        block_mean_losses=[0.50, 0.49],
        min_relative_improvement=0.05,
    )

    assert should_continue is False
    assert improvement == pytest.approx(0.02)


def test_train_script_continues_when_block_improvement_clears_threshold() -> None:
    """Continue when the latest completed block improves enough over the prior block."""
    train_script = _load_train_script_module()

    should_continue, improvement = train_script._should_continue_after_block(
        block_mean_losses=[0.50, 0.40],
        min_relative_improvement=0.05,
    )

    assert should_continue is True
    assert improvement == pytest.approx(0.20)


def test_train_script_rejects_validation_for_local_video_mode() -> None:
    """Fail fast when validation is requested for local-video training."""
    train_script = _load_train_script_module()

    with pytest.raises(ValueError, match="not supported for local-video training"):
        train_script._validate_auto_stop_config(
            TrainScriptConfig(
                video_path="clip.mp4",
                validation_enabled=True,
            )
        )


def test_train_script_rejects_nonpositive_validation_interval() -> None:
    """Fail fast when validation cadence would never produce a valid check."""
    train_script = _load_train_script_module()

    with pytest.raises(ValueError, match="validation_every must be positive"):
        train_script._validate_auto_stop_config(TrainScriptConfig(validation_every=0))


def test_train_script_rejects_negative_future_loss_early_bias() -> None:
    """Fail fast when the early-horizon loss bias would downweight earlier frames."""
    train_script = _load_train_script_module()

    with pytest.raises(ValueError, match="future_loss_early_bias must be >= 0"):
        train_script._validate_auto_stop_config(TrainScriptConfig(future_loss_early_bias=-0.1))


def test_train_script_rejects_negative_future_chunk_early_bias() -> None:
    """Fail fast when the early-chunk bias would downweight earlier chunks."""
    train_script = _load_train_script_module()

    with pytest.raises(ValueError, match="future_chunk_early_bias must be >= 0"):
        train_script._validate_auto_stop_config(TrainScriptConfig(future_chunk_early_bias=-0.1))


def test_train_script_rejects_unknown_teacher_forcing_observation_mode() -> None:
    """Fail fast when the observation mode would silently change training semantics."""
    train_script = _load_train_script_module()

    with pytest.raises(ValueError, match="teacher_forcing_observation_mode must be"):
        train_script._validate_auto_stop_config(
            TrainScriptConfig(teacher_forcing_observation_mode="bad")
        )


def test_train_script_updates_validation_tracking_with_patience() -> None:
    """Reset or increment validation patience based on relative improvement."""
    train_script = _load_train_script_module()

    best, bad_checks, improvement = train_script._update_validation_tracking(
        best_val_loss=0.50,
        current_val_loss=0.48,
        bad_checks=2,
        min_relative_improvement=0.05,
    )
    assert best == pytest.approx(0.50)
    assert bad_checks == 3
    assert improvement == pytest.approx(0.04)

    best, bad_checks, improvement = train_script._update_validation_tracking(
        best_val_loss=0.50,
        current_val_loss=0.40,
        bad_checks=2,
        min_relative_improvement=0.05,
    )
    assert best == pytest.approx(0.40)
    assert bad_checks == 0
    assert improvement == pytest.approx(0.20)


def test_train_script_evaluates_validation_loss_with_fixed_batch_cap(monkeypatch) -> None:
    """Average only the configured prefix of validation batches."""
    train_script = _load_train_script_module()
    cfg = TrainScriptConfig(
        validation_enabled=True,
        validation_max_batches=2,
        context_len=5,
        horizon_len=4,
    )
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 1, 8, 8),
        a_plan=torch.randn(1, 1, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=3,
        context_latent_steps=2,
        horizon_latent_steps=1,
    )
    losses = iter([0.6, 0.3, 0.1])

    monkeypatch.setattr(train_script, "prepare_packed_batch", lambda **_: prepared)
    monkeypatch.setattr(train_script, "_evaluate_loss", lambda **_: next(losses))

    class _FakeEncoder:
        """Minimal encoder placeholder for validation-loss tests."""

    val_loss, num_batches = train_script._evaluate_validation_loss(
        model=torch.nn.Identity(),
        action_encoder=torch.nn.Identity(),
        validation_loader=[{"batch": 1}, {"batch": 2}, {"batch": 3}],
        encoder=_FakeEncoder(),
        cfg=cfg,
        device=torch.device("cpu"),
        runtime_dtype=torch.float32,
    )

    assert num_batches == 2
    assert val_loss == pytest.approx(0.45)


def test_train_script_restores_validation_state_from_checkpoint() -> None:
    """Recover best validation loss and patience counters from checkpoint extra state."""
    train_script = _load_train_script_module()

    best_val_loss, val_bad_checks = train_script._load_validation_state_from_checkpoint(
        {
            "extra_state": {
                "validation_state": {
                    "best_val_loss": 0.123,
                    "val_bad_checks": 2,
                }
            }
        }
    )

    assert best_val_loss == pytest.approx(0.123)
    assert val_bad_checks == 2


def test_train_script_logs_validation_metrics_on_validation_rows(monkeypatch, tmp_path: Path) -> None:
    """Emit validation fields on the independent validation cadence, not just checkpoints."""
    train_script = _load_train_script_module()
    output_dir = tmp_path / "train_run"
    captured_logs: list[dict[str, object]] = []

    cfg = TrainScriptConfig(
        output_dir=str(output_dir),
        repo_id="repo/x",
        video_key="video",
        context_len=5,
        horizon_len=4,
        batch_size=1,
        max_steps=2,
        checkpoint_every=2,
        checkpoint_early_every=0,
        checkpoint_early_until=0,
        validation_every=1,
        log_every=10,
        load_pretrained_backbone=False,
        validation_enabled=True,
        validation_episodes=(9,),
        validation_max_batches=2,
    )
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 1, 8, 8),
        a_plan=torch.randn(1, 1, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=3,
        context_latent_steps=2,
        horizon_latent_steps=1,
    )
    train_batches = [{"video": torch.randn(1, 9, 3, 8, 8), "action": torch.randn(1, 9, 6)}]
    val_batches = [{"video": torch.randn(1, 9, 3, 8, 8), "action": torch.randn(1, 9, 6)}]

    class _FakeEncoder:
        """Minimal VAE placeholder for patched training-loop tests."""

        def encode(self, video: torch.Tensor) -> torch.Tensor:
            """Return a structured latent tensor with valid Wan temporal packing."""
            batch_size = video.shape[0]
            return torch.zeros(batch_size, 16, 3, 8, 8)

    class _FakeModel(torch.nn.Module):
        """Small trainable model placeholder for patched training-loop tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _FakeActionEncoder(torch.nn.Module):
        """Small action encoder placeholder for patched training-loop tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _FakeActionControlProjector(torch.nn.Module):
        """Small action-control projector placeholder for patched training-loop tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    checkpoint_steps: list[int] = []

    def _fake_build_loader(**kwargs):
        if kwargs["episodes"] == [9]:
            return val_batches
        return train_batches

    monkeypatch.setattr(train_script, "_load_args", lambda: cfg)
    monkeypatch.setattr(train_script, "_set_seed", lambda seed: None)
    monkeypatch.setattr(train_script.WanVAE, "from_pretrained", lambda **_: _FakeEncoder())
    monkeypatch.setattr(train_script, "resolve_lerobot_episode_ids", lambda repo_id: [0, 1, 9])
    monkeypatch.setattr(train_script, "build_lerobot_dataloader", _fake_build_loader)
    monkeypatch.setattr(train_script, "prepare_packed_batch", lambda **_: prepared)
    monkeypatch.setattr(train_script, "_validate_chunk_schedule", lambda cfg, prepared: None)
    monkeypatch.setattr(train_script, "build_model_from_config", lambda cfg, prepared: _FakeModel())
    monkeypatch.setattr(
        train_script,
        "build_action_encoder_from_config",
        lambda cfg, prepared, model: _FakeActionEncoder(),
    )
    monkeypatch.setattr(
        train_script,
        "build_action_control_projector_from_config",
        lambda cfg, prepared, model: _FakeActionControlProjector(),
    )
    monkeypatch.setattr(
        train_script,
        "_configure_trainable_parameters",
        lambda cfg, model, action_encoder, action_control_projector=None: (
            list(model.parameters())
            + list(action_encoder.parameters())
            + ([] if action_control_projector is None else list(action_control_projector.parameters()))
        ),
    )
    monkeypatch.setattr(
        train_script,
        "train_chunkwise_batch",
        lambda **_: ChunkwiseStepMetrics(
            loss=0.5,
            grad_norm=0.1,
            per_chunk_losses=(0.5,),
            per_chunk_lengths=(1,),
        ),
    )
    monkeypatch.setattr(train_script, "_evaluate_validation_loss", lambda **_: (0.25, 1))
    monkeypatch.setattr(
        train_script,
        "append_jsonl",
        lambda path, payload: captured_logs.append(dict(payload)),
    )
    monkeypatch.setattr(
        train_script,
        "save_checkpoint",
        lambda **kwargs: checkpoint_steps.append(int(kwargs["step"])) or (output_dir / f"step_{int(kwargs['step']):07d}.pt"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_script.main()

    assert checkpoint_steps == [2, 2]
    assert len(captured_logs) == 2
    assert captured_logs[0]["val_loss"] == pytest.approx(0.25)
    assert captured_logs[0]["val_num_batches"] == 1
    assert captured_logs[1]["val_loss"] == pytest.approx(0.25)
    assert captured_logs[1]["val_num_batches"] == 1
    assert captured_logs[1]["best_val_loss"] == pytest.approx(0.25)
    assert captured_logs[1]["val_bad_checks"] == 1


def test_train_script_preserves_logs_when_validation_is_disabled(monkeypatch, tmp_path: Path) -> None:
    """Keep metrics rows free of validation fields when validation is disabled."""
    train_script = _load_train_script_module()
    output_dir = tmp_path / "train_run"
    captured_logs: list[dict[str, object]] = []

    cfg = TrainScriptConfig(
        output_dir=str(output_dir),
        repo_id="repo/x",
        video_key="video",
        context_len=5,
        horizon_len=4,
        batch_size=1,
        max_steps=2,
        checkpoint_every=2,
        checkpoint_early_every=0,
        checkpoint_early_until=0,
        log_every=10,
        load_pretrained_backbone=False,
        validation_enabled=False,
    )
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 1, 8, 8),
        a_plan=torch.randn(1, 1, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=3,
        context_latent_steps=2,
        horizon_latent_steps=1,
    )
    train_batches = [{"video": torch.randn(1, 9, 3, 8, 8), "action": torch.randn(1, 9, 6)}]

    class _FakeEncoder:
        """Minimal VAE placeholder for validation-disabled tests."""

        def encode(self, video: torch.Tensor) -> torch.Tensor:
            """Return a structured latent tensor with valid Wan temporal packing."""
            batch_size = video.shape[0]
            return torch.zeros(batch_size, 16, 3, 8, 8)

    class _FakeModel(torch.nn.Module):
        """Small trainable model placeholder for validation-disabled tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _FakeActionEncoder(torch.nn.Module):
        """Small action encoder placeholder for validation-disabled tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _FakeActionControlProjector(torch.nn.Module):
        """Small action-control projector placeholder for validation-disabled tests."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(train_script, "_load_args", lambda: cfg)
    monkeypatch.setattr(train_script, "_set_seed", lambda seed: None)
    monkeypatch.setattr(train_script.WanVAE, "from_pretrained", lambda **_: _FakeEncoder())
    monkeypatch.setattr(train_script, "build_lerobot_dataloader", lambda **kwargs: train_batches)
    monkeypatch.setattr(train_script, "prepare_packed_batch", lambda **_: prepared)
    monkeypatch.setattr(train_script, "_validate_chunk_schedule", lambda cfg, prepared: None)
    monkeypatch.setattr(train_script, "build_model_from_config", lambda cfg, prepared: _FakeModel())
    monkeypatch.setattr(
        train_script,
        "build_action_encoder_from_config",
        lambda cfg, prepared, model: _FakeActionEncoder(),
    )
    monkeypatch.setattr(
        train_script,
        "build_action_control_projector_from_config",
        lambda cfg, prepared, model: _FakeActionControlProjector(),
    )
    monkeypatch.setattr(
        train_script,
        "_configure_trainable_parameters",
        lambda cfg, model, action_encoder, action_control_projector=None: (
            list(model.parameters())
            + list(action_encoder.parameters())
            + ([] if action_control_projector is None else list(action_control_projector.parameters()))
        ),
    )
    monkeypatch.setattr(
        train_script,
        "train_chunkwise_batch",
        lambda **_: ChunkwiseStepMetrics(
            loss=0.5,
            grad_norm=0.1,
            per_chunk_losses=(0.5,),
            per_chunk_lengths=(1,),
        ),
    )
    monkeypatch.setattr(
        train_script,
        "append_jsonl",
        lambda path, payload: captured_logs.append(dict(payload)),
    )
    monkeypatch.setattr(
        train_script,
        "save_checkpoint",
        lambda **kwargs: output_dir / f"step_{int(kwargs['step']):07d}.pt",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    train_script.main()

    assert len(captured_logs) == 2
    assert all("val_loss" not in payload for payload in captured_logs)


def test_train_script_freezes_non_vace_backbone_params_in_vace_mode() -> None:
    """Keep optimizer state off the frozen backbone during local smoke runs."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )
    cfg = TrainScriptConfig(
        trainable_backbone="vace",
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)
    parameters = train_script._configure_trainable_parameters(cfg, model, action_encoder)
    trainable_names = {
        name for name, parameter in itertools.chain(model.named_parameters(), action_encoder.named_parameters()) if parameter.requires_grad
    }

    assert parameters
    assert "backbone.patch_embedding.weight" not in trainable_names
    assert "backbone.vace_patch_embedding.weight" in trainable_names
    assert "backbone.proj_out.weight" in trainable_names


def test_train_script_keeps_full_backbone_trainable_in_full_mode() -> None:
    """Retain the canonical full-fine-tune mode for larger runs."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )
    cfg = TrainScriptConfig(
        trainable_backbone="full",
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)
    train_script._configure_trainable_parameters(cfg, model, action_encoder)

    assert model.backbone.patch_embedding.weight.requires_grad is True
    assert model.backbone.vace_patch_embedding.weight.requires_grad is True


def test_train_script_keeps_vace_blocks_frozen_in_head_mode() -> None:
    """Use the smallest practical trainable subset for 16 GB smoke runs."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )
    cfg = TrainScriptConfig(
        trainable_backbone="head",
        load_pretrained_backbone=False,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)
    train_script._configure_trainable_parameters(cfg, model, action_encoder)

    assert model.backbone.vace_patch_embedding.weight.requires_grad is True
    assert model.backbone.proj_out.weight.requires_grad is True
    assert next(model.backbone.vace_blocks[0].parameters()).requires_grad is False


def test_train_script_enables_lora_parameters_without_unfreezing_full_backbone() -> None:
    """Use LoRA adapters plus the small output modules for the ALOHA full-dataset path."""
    train_script = _load_train_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )
    cfg = TrainScriptConfig(
        trainable_backbone="lora",
        load_pretrained_backbone=False,
        conditioning_mode="action",
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"),
    )

    model = train_script.build_model_from_config(cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(cfg, prepared, model)
    parameters = train_script._configure_trainable_parameters(cfg, model, action_encoder)
    trainable_names = {
        name for name, parameter in itertools.chain(model.named_parameters(), action_encoder.named_parameters()) if parameter.requires_grad
    }

    assert parameters
    assert any("lora_" in name for name in trainable_names)
    assert "backbone.patch_embedding.weight" not in trainable_names
    assert "backbone.vace_patch_embedding.weight" in trainable_names
    assert any(name.startswith("backbone.vace_blocks.0.attn1.to_q") and "lora_" in name for name in trainable_names)


def _load_train_script_module():
    """Load the train script module without executing the CLI entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "world_model.py"
    spec = importlib.util.spec_from_file_location("test_train_world_model_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
