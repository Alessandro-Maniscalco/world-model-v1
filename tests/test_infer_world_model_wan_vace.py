"""Tests for the Wan VACE inference entrypoint wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from world_model.config import InferScriptConfig
from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import ActionTokenEncoder
from world_model.models.wan_vace_world_model import WanVACEWorldModel


def test_infer_script_builds_wan_vace_runtime_modules_without_checkpoint() -> None:
    """Build Wan VACE inference modules without requiring a local fine-tune checkpoint."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        z_past=torch.randn(2, 2, 16 * 8 * 8),
        z_future=torch.randn(2, 4, 16 * 8 * 8),
        a_plan=torch.randn(2, 4, 6),
        q_last=None,
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = InferScriptConfig(
        load_pretrained_backbone=False,
        disable_proprio=True,
        wan_num_attention_heads=2,
        wan_attention_head_dim=8,
        wan_text_dim=16,
        wan_freq_dim=8,
        wan_ffn_dim=32,
        wan_num_layers=2,
        vace_layers=(0, 1),
        mask_channels=4,
    )

    model, action_encoder, proprio_encoder = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert proprio_encoder is None


def test_infer_script_parser_omits_legacy_dit_shape_flags() -> None:
    """Avoid exposing unused legacy DiT width/depth CLI flags on the VACE path."""
    infer_script = _load_infer_script_module()
    parser = infer_script._build_parser(InferScriptConfig())
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--hidden-dim" not in option_strings
    assert "--num-layers" not in option_strings
    assert "--num-heads" not in option_strings


def _load_infer_script_module():
    """Load the infer script module without executing the CLI entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "infer_world_model.py"
    spec = importlib.util.spec_from_file_location("test_infer_world_model_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
