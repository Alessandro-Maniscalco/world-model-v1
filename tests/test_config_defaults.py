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


def test_infer_config_defaults_load_from_canonical_yaml() -> None:
    """Load the canonical eval preset when no explicit path is supplied."""
    cfg = load_infer_config()

    assert DEFAULT_INFER_CONFIG_PATH.exists()
    assert cfg.load_pretrained_backbone is True
    assert cfg.wan_vace_model_id == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    assert cfg.wan_vace_subfolder == "transformer"
    assert cfg.conditioning_mode == "action"
    assert cfg.guidance_scale == 5.0
    assert cfg.single_chunk_rollout is False
