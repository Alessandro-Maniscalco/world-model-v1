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
from world_model.models.wan_vace_conditioning import ActionTokenEncoder, NullConditioningEncoder


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


def test_train_script_parser_omits_legacy_dit_shape_flags() -> None:
    """Avoid exposing removed non-VACE backbone shape flags."""
    train_script = _load_train_script_module()
    parser = train_script._build_parser(load_train_config())
    option_strings = {option for action in parser._actions for option in action.option_strings}

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


def test_train_script_rejects_invalid_auto_stop_config() -> None:
    """Reject impossible blockwise continuation settings before training starts."""
    train_script = _load_train_script_module()

    with pytest.raises(ValueError, match="max_steps must be >= auto_stop_check_every"):
        train_script._validate_auto_stop_config(
            TrainScriptConfig(max_steps=1000, auto_stop_check_every=5000),
        )


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
