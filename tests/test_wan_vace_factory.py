"""Tests for shared Wan VACE model and checkpoint builders."""

from __future__ import annotations

import builtins
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from world_model.config import InferScriptConfig, load_infer_config, load_train_config
from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import (
    ActionControlProjector,
    ActionTokenEncoder,
    NullActionControlProjector,
    NullConditioningEncoder,
)
from world_model.models.wan_vace_world_model import WanVACEWorldModel
from world_model.models import wan_vace_factory


def test_build_runtime_modules_loads_pretrained_backbone_by_default(monkeypatch) -> None:
    """Use canonical Diffusers Wan VACE weights by default for train and infer configs."""
    prepared = _make_prepared_batch()
    calls: list[tuple[str, str | None]] = []

    class _FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                text_dim=32,
                in_channels=16,
                vace_in_channels=36,
                vace_layers=(0, 1),
            )

    def _fake_from_pretrained(
        model_id: str,
        subfolder: str | None = None,
        local_files_only: bool = False,
    ):
        calls.append((model_id, subfolder, local_files_only))
        return _FakeBackbone()

    monkeypatch.setattr(wan_vace_factory.WanVACETransformer3DModel, "from_pretrained", _fake_from_pretrained)

    assert load_train_config().load_pretrained_backbone is True
    defaults = load_infer_config()
    assert defaults.load_pretrained_backbone is True

    model, action_encoder, action_control_projector = wan_vace_factory.build_wan_vace_runtime_modules(
        InferScriptConfig(
            load_pretrained_backbone=defaults.load_pretrained_backbone,
            wan_vace_model_id=defaults.wan_vace_model_id,
            wan_vace_subfolder=defaults.wan_vace_subfolder,
            mask_channels=4,
        ),
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, NullConditioningEncoder)
    assert isinstance(action_control_projector, NullActionControlProjector)
    assert model.control_black_latents is not None
    assert model.control_gray_latents is not None
    assert calls == [("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "transformer", False)]


def test_build_runtime_modules_respects_action_input_layernorm_flag() -> None:
    """Propagate the action-input normalization flag into the runtime encoder."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
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

    _, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.input_layernorm is False


def test_build_runtime_modules_respects_action_mlp_dim_flag() -> None:
    """Propagate the optional action-token MLP width into the runtime encoder."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_mlp_dim=12,
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

    _, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.net[1].out_features == 12
    assert action_encoder.net[4].out_features == 16


def test_build_runtime_modules_respects_action_mlp_residual_flag() -> None:
    """Propagate the residual action-MLP mode into the runtime encoder."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_mlp_dim=12,
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

    _, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.mlp_residual is True
    assert action_encoder.residual_net is not None
    assert action_encoder.net[1].out_features == 16
    assert action_encoder.residual_net[0].out_features == 12


def test_build_runtime_modules_respects_action_temporal_difference_scale() -> None:
    """Propagate the temporal-difference action residual scale into the runtime encoder."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
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

    _, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.temporal_difference_scale == pytest.approx(0.75)


def test_build_runtime_modules_respects_action_token_scale() -> None:
    """Propagate post-projection action-token gain into the runtime encoder."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
        action_token_scale=2.0,
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

    _, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.token_scale == pytest.approx(2.0)


def test_build_runtime_modules_respects_action_temporal_mixer_settings() -> None:
    """Propagate temporal action-mixer settings into the runtime encoder."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
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

    _, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.temporal_mixer is not None
    assert action_encoder.temporal_mixer_kernel_size == 3
    assert action_encoder.temporal_mixer_scale == pytest.approx(0.5)


def test_build_runtime_modules_respects_action_order_and_control_prior_flags() -> None:
    """Build ordered action tokens plus a latent prior projector for action-conditioned runs."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_order_conditioning=True,
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

    model, action_encoder, action_control_projector = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.order_conditioning is True
    assert model.action_control_prior_mode == "reactive_only"
    assert isinstance(action_control_projector, ActionControlProjector)


def test_build_runtime_modules_respects_action_control_prior_mode() -> None:
    """Propagate latent control-prior mode into the runtime world-model wrapper."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_control_prior_scale=1.0,
        action_control_prior_mode="dual_fill",
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

    model, _, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert model.action_control_prior_mode == "dual_fill"


def test_build_runtime_modules_respects_action_hidden_state_bias_scale() -> None:
    """Propagate latent hidden-state bias scale into the runtime world-model wrapper."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
        conditioning_mode="action",
        action_hidden_state_bias_scale=0.75,
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

    model, _, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert model.action_hidden_state_bias_scale == pytest.approx(0.75)


def test_build_runtime_modules_applies_local_checkpoint_overlay() -> None:
    """Overlay local fine-tune weights on top of the Wan VACE runtime modules."""
    prepared = _make_prepared_batch()
    cfg = InferScriptConfig(
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
    model, action_encoder, action_control_projector = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    model_state = {key: torch.full_like(value, 0.25) for key, value in model.state_dict().items()}
    action_state = {key: torch.full_like(value, -0.5) for key, value in action_encoder.state_dict().items()}
    projector_state = {
        key: torch.full_like(value, 0.75)
        for key, value in action_control_projector.state_dict().items()
    }
    checkpoint = {
        "model_state_dict": model_state,
        "action_encoder_state_dict": action_state,
        "action_control_projector_state_dict": projector_state,
    }

    loaded_model, loaded_action_encoder, loaded_action_control_projector = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=checkpoint,
    )

    first_model_key = next(iter(model_state))
    first_action_key = next(iter(action_state))
    first_projector_key = next(iter(projector_state))
    assert torch.allclose(loaded_model.state_dict()[first_model_key], model_state[first_model_key])
    assert torch.allclose(loaded_action_encoder.state_dict()[first_action_key], action_state[first_action_key])
    assert torch.allclose(
        loaded_action_control_projector.state_dict()[first_projector_key],
        projector_state[first_projector_key],
    )


def test_build_runtime_modules_allows_older_action_checkpoint_without_temporal_mixer() -> None:
    """Allow old action-encoder checkpoints to load when the new temporal mixer is enabled."""
    prepared = _make_prepared_batch()
    old_cfg = InferScriptConfig(
        conditioning_mode="action",
        action_input_layernorm=False,
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
    new_cfg = replace(
        old_cfg,
        action_temporal_mixer_kernel_size=3,
        action_temporal_mixer_scale=0.5,
    )
    model, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        old_cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "action_encoder_state_dict": action_encoder.state_dict(),
    }

    _, loaded_action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        new_cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=checkpoint,
    )

    assert isinstance(loaded_action_encoder, ActionTokenEncoder)
    assert loaded_action_encoder.temporal_mixer is not None
    assert torch.count_nonzero(loaded_action_encoder.temporal_mixer.weight) == 0


def test_build_runtime_modules_allows_older_checkpoint_without_control_projector() -> None:
    """Treat missing control-prior weights as an allowed old-checkpoint case."""
    prepared = _make_prepared_batch()
    old_cfg = InferScriptConfig(
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
    new_cfg = replace(old_cfg, action_control_prior_scale=1.0)
    model, action_encoder, _ = wan_vace_factory.build_wan_vace_runtime_modules(
        old_cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "action_encoder_state_dict": action_encoder.state_dict(),
    }

    _, _, action_control_projector = wan_vace_factory.build_wan_vace_runtime_modules(
        new_cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=checkpoint,
    )

    assert isinstance(action_control_projector, ActionControlProjector)
    assert torch.count_nonzero(action_control_projector.projection.weight) == 0


def test_build_runtime_modules_forwards_offline_mode_to_pretrained_load(monkeypatch) -> None:
    """Load pretrained backbones in local-files-only mode when offline env vars are set."""
    prepared = _make_prepared_batch()
    calls: list[bool] = []

    class _FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(text_dim=32, in_channels=16, vace_in_channels=36, vace_layers=(0,))

    def _fake_from_pretrained(*args, local_files_only: bool = False, **kwargs):
        del args, kwargs
        calls.append(local_files_only)
        return _FakeBackbone()

    monkeypatch.setattr(wan_vace_factory.WanVACETransformer3DModel, "from_pretrained", _fake_from_pretrained)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    wan_vace_factory.build_wan_vace_runtime_modules(
        InferScriptConfig(mask_channels=4),
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert calls == [True]


def test_expected_control_channels_matches_inactive_reactive_plus_mask() -> None:
    """Count both fill-latent streams plus the expanded mask channels."""
    assert wan_vace_factory._expected_control_channels(latent_channels=16, mask_channels=64) == 96


def test_merge_runtime_backbone_config_restores_saved_defaults_only() -> None:
    """Apply checkpoint config only to fields the runtime has not overridden."""
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "mask_channels": 8,
                "vace_layers": [1, 3],
                "lora_target_modules": ["to_q", "to_v"],
            }
        }
    }

    merged = wan_vace_factory._merge_runtime_backbone_config(cfg=cfg, checkpoint=checkpoint)

    assert merged.conditioning_mode == "action"
    assert merged.mask_channels == 8
    assert merged.vace_layers == (1, 3)
    assert merged.lora_target_modules == ("to_q", "to_v")


def test_merge_runtime_backbone_config_keeps_explicit_runtime_overrides() -> None:
    """Do not overwrite runtime choices that already differ from defaults."""
    cfg = InferScriptConfig(conditioning_mode="action", mask_channels=4)
    checkpoint = {"extra_state": {"config": {"conditioning_mode": "none", "mask_channels": 99}}}

    merged = wan_vace_factory._merge_runtime_backbone_config(cfg=cfg, checkpoint=checkpoint)

    assert merged.conditioning_mode == "action"
    assert merged.mask_channels == 4


@pytest.mark.parametrize(
    ("cfg", "pattern"),
    [
        (SimpleNamespace(lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_target_modules=("to_q",)), "lora_rank"),
        (SimpleNamespace(lora_rank=4, lora_alpha=0, lora_dropout=0.0, lora_target_modules=("to_q",)), "lora_alpha"),
        (
            SimpleNamespace(lora_rank=4, lora_alpha=16, lora_dropout=-0.1, lora_target_modules=("to_q",)),
            "lora_dropout",
        ),
        (
            SimpleNamespace(lora_rank=4, lora_alpha=16, lora_dropout=0.0, lora_target_modules=()),
            "lora_target_modules",
        ),
    ],
)
def test_attach_lora_adapters_validates_config_before_import(cfg: SimpleNamespace, pattern: str) -> None:
    """Reject invalid LoRA settings before attempting to import PEFT."""
    backbone = SimpleNamespace(add_adapter=lambda config: None)

    with pytest.raises(ValueError, match=pattern):
        wan_vace_factory._attach_lora_adapters(backbone=backbone, cfg=cfg)


def test_attach_lora_adapters_requires_peft(monkeypatch) -> None:
    """Raise a clear error when LoRA is requested without the PEFT dependency."""
    backbone = SimpleNamespace(add_adapter=lambda config: None)
    cfg = SimpleNamespace(lora_rank=4, lora_alpha=16, lora_dropout=0.0, lora_target_modules=("to_q",))
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "peft":
            raise ImportError("missing peft")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="peft is required"):
        wan_vace_factory._attach_lora_adapters(backbone=backbone, cfg=cfg)


def test_build_wan_vace_model_from_config_rejects_backbone_channel_mismatch(monkeypatch) -> None:
    """Reject pretrained backbones whose latent channel counts do not match prepared data."""
    prepared = _make_prepared_batch()

    class _FakeBackbone(torch.nn.Module):
        def __init__(self, *, in_channels: int, vace_in_channels: int) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                text_dim=32,
                in_channels=in_channels,
                vace_in_channels=vace_in_channels,
                vace_layers=(0,),
            )

    cfg = InferScriptConfig(mask_channels=4)

    monkeypatch.setattr(
        wan_vace_factory.WanVACETransformer3DModel,
        "from_pretrained",
        lambda *args, **kwargs: _FakeBackbone(in_channels=8, vace_in_channels=36),
    )
    with pytest.raises(ValueError, match="in_channels=8 does not match latent channels=16"):
        wan_vace_factory.build_wan_vace_model_from_config(cfg, prepared)

    monkeypatch.setattr(
        wan_vace_factory.WanVACETransformer3DModel,
        "from_pretrained",
        lambda *args, **kwargs: _FakeBackbone(in_channels=16, vace_in_channels=12),
    )
    with pytest.raises(ValueError, match="vace_in_channels=12 does not match expected control channels=36"):
        wan_vace_factory.build_wan_vace_model_from_config(cfg, prepared)


def _make_prepared_batch() -> PreparedPackedBatch:
    """Build a compact prepared batch fixture for Wan VACE module tests."""
    return PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
        control_black_latents=torch.full((2, 16, 6, 8, 8), -1.0),
        control_gray_latents=torch.full((2, 16, 6, 8, 8), 0.5),
    )
