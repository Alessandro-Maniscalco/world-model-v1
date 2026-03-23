"""Tests for action-only resume wiring in the world-model train entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from world_model.config import TrainScriptConfig
from world_model.data.schema import PreparedPackedBatch


def test_train_script_keeps_backbone_frozen_in_none_mode_while_training_action_encoder() -> None:
    """Use `trainable_backbone=none` to train only the action-conditioning stack."""
    train_script = _load_train_script_module()
    prepared = _build_prepared_batch()
    cfg = TrainScriptConfig(
        conditioning_mode="action",
        trainable_backbone="none",
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

    assert parameters
    assert model.backbone.patch_embedding.weight.requires_grad is False
    assert model.backbone.vace_patch_embedding.weight.requires_grad is False
    assert action_encoder.net[1].weight.requires_grad is True


def test_resume_training_state_skips_empty_null_action_state_for_action_encoder() -> None:
    """Allow action-only tuning to resume from a none-conditioned checkpoint."""
    train_script = _load_train_script_module()
    prepared = _build_prepared_batch()
    common = dict(
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

    none_cfg = TrainScriptConfig(trainable_backbone="full", conditioning_mode="none", **common)
    none_model = train_script.build_model_from_config(none_cfg, prepared)
    none_action_encoder = train_script.build_action_encoder_from_config(none_cfg, prepared, none_model)
    none_params = train_script._configure_trainable_parameters(none_cfg, none_model, none_action_encoder)
    none_optimizer = train_script._build_optimizer(none_cfg, none_params)
    checkpoint = {
        "model_state_dict": none_model.state_dict(),
        "action_encoder_state_dict": none_action_encoder.state_dict(),
        "optimizer_state_dict": none_optimizer.state_dict(),
        "step": 350,
        "extra_state": {"config": {"conditioning_mode": "none"}},
    }

    action_cfg = TrainScriptConfig(trainable_backbone="none", conditioning_mode="action", **common)
    action_model = train_script.build_model_from_config(action_cfg, prepared)
    action_encoder = train_script.build_action_encoder_from_config(action_cfg, prepared, action_model)
    original_weight = action_encoder.net[1].weight.detach().clone()
    action_params = train_script._configure_trainable_parameters(action_cfg, action_model, action_encoder)
    action_optimizer = train_script._build_optimizer(action_cfg, action_params)

    resumed_step, restored_optimizer_state = train_script._resume_training_state(
        checkpoint=checkpoint,
        model=action_model,
        action_encoder=action_encoder,
        optimizer=action_optimizer,
    )

    assert resumed_step == 350
    assert restored_optimizer_state is False
    assert torch.equal(action_encoder.net[1].weight, original_weight)


def _build_prepared_batch() -> PreparedPackedBatch:
    """Create a tiny packed batch for train-script wiring tests."""
    return PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )


def _load_train_script_module():
    """Load the train script module without executing the CLI entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "world_model.py"
    spec = importlib.util.spec_from_file_location("test_train_world_model_action_resume_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
