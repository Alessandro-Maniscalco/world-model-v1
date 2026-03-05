"""Tests for the Wan VACE training entrypoint wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from world_model.config import TrainScriptConfig, load_train_config
from world_model.data.schema import PreparedPackedBatch
from world_model.models import WanVACEWorldModel
from world_model.models.wan_vace_conditioning import ActionTokenEncoder


def test_train_script_builds_wan_vace_world_model_from_config() -> None:
    """Build the Wan VACE model and action-token encoder from train config."""
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
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert model.mask_channels == 4
    assert model.control_scale == 0.75
    assert model.backbone.config.text_dim == 16
    assert model.backbone.config.in_channels == 16
    assert model.backbone.config.vace_in_channels == 36
    assert tuple(model.backbone.config.vace_layers) == (0, 1)
    assert action_encoder.hidden_dim == 16


def test_train_script_parser_omits_legacy_dit_shape_flags() -> None:
    """Avoid exposing removed non-VACE backbone shape flags."""
    train_script = _load_train_script_module()
    parser = train_script._build_parser(load_train_config())
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--hidden-dim" not in option_strings
    assert "--num-layers" not in option_strings
    assert "--num-heads" not in option_strings


def _load_train_script_module():
    """Load the train script module without executing the CLI entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "world_model.py"
    spec = importlib.util.spec_from_file_location("test_train_world_model_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
