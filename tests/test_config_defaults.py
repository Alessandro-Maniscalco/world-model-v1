"""Tests for canonical YAML-backed config defaults."""

from world_model.config import (
    DEFAULT_INFER_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    load_infer_config,
    load_train_config,
)


def test_train_config_defaults_load_from_canonical_yaml() -> None:
    """Load the canonical train preset when no explicit path is supplied."""
    cfg = load_train_config()

    assert DEFAULT_TRAIN_CONFIG_PATH.exists()
    assert cfg.load_pretrained_backbone is True
    assert cfg.wan_vace_model_id == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert cfg.wan_vace_subfolder == "transformer"
    assert cfg.trainable_backbone == "full"
    assert cfg.conditioning_mode == "none"
    assert cfg.action_conditioning_window == "chunk"
    assert cfg.chunk_schedule_mode == "k_plus_one"
    assert cfg.action_order_conditioning is False
    assert cfg.action_backbone_added_kv_mode == "none"
    assert cfg.action_control_prior_scale == 0.0
    assert cfg.action_control_prior_mode == "reactive_only"
    assert cfg.action_control_projector_init_mode == "zero"
    assert cfg.action_control_projector_observed_context_mode == "none"
    assert cfg.action_hidden_state_bias_scale == 0.0
    assert cfg.action_control_aux_loss_scale == 0.0
    assert cfg.action_token_latent_aux_loss_scale == 0.0
    assert cfg.action_token_scale == 1.0
    assert cfg.future_latent_residual_mode == "none"
    assert cfg.future_chunk_early_bias == 0.0
    assert cfg.teacher_forcing_observation_mode == "full_prefix"
    assert cfg.teacher_forcing_future_input_mode == "full_suffix"
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.validation_enabled is True
    assert cfg.validation_every == 50
    assert cfg.validation_episodes == ()
    assert cfg.validation_split_ratio == 0.1
    assert cfg.validation_max_batches == 8
    assert cfg.validation_patience_checks == 0
    assert cfg.validation_min_relative_improvement == 0.01
    assert cfg.lora_target_modules == (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
        "proj_in",
        "proj_out",
    )


def test_infer_config_defaults_load_from_canonical_yaml() -> None:
    """Load the canonical eval preset when no explicit path is supplied."""
    cfg = load_infer_config()

    assert DEFAULT_INFER_CONFIG_PATH.exists()
    assert cfg.load_pretrained_backbone is True
    assert cfg.wan_vace_model_id == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert cfg.wan_vace_subfolder == "transformer"
    assert cfg.trainable_backbone == "full"
    assert cfg.conditioning_mode == "none"
    assert cfg.action_conditioning_window == "chunk"
    assert cfg.chunk_schedule_mode == "k_plus_one"
    assert cfg.action_order_conditioning is False
    assert cfg.action_backbone_added_kv_mode == "none"
    assert cfg.action_control_prior_scale == 0.0
    assert cfg.action_control_prior_mode == "reactive_only"
    assert cfg.action_control_projector_init_mode == "zero"
    assert cfg.action_control_projector_observed_context_mode == "none"
    assert cfg.action_hidden_state_bias_scale == 0.0
    assert cfg.action_token_latent_aux_loss_scale == 0.0
    assert cfg.action_token_scale == 1.0
    assert cfg.future_latent_residual_mode == "none"
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_target_modules == (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
        "proj_in",
        "proj_out",
    )
    assert cfg.guidance_scale == 5.0
    assert cfg.num_vis_frames == 0
    assert cfg.frame_height == 0
    assert cfg.frame_width == 0
    assert cfg.single_chunk_rollout is False
