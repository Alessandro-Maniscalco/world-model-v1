"""Tests for shared Wan VACE model and checkpoint builders."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from world_model.config import InferScriptConfig, TrainScriptConfig
from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import ActionTokenEncoder
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

    def _fake_from_pretrained(model_id: str, subfolder: str | None = None):
        calls.append((model_id, subfolder))
        return _FakeBackbone()

    monkeypatch.setattr(wan_vace_factory.WanVACETransformer3DModel, "from_pretrained", _fake_from_pretrained)

    assert TrainScriptConfig().load_pretrained_backbone is True
    assert InferScriptConfig().load_pretrained_backbone is True

    model, action_encoder, proprio_encoder = wan_vace_factory.build_wan_vace_runtime_modules(
        InferScriptConfig(mask_channels=4),
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert proprio_encoder is None
    assert calls == [("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "transformer")]


def test_build_runtime_modules_applies_local_checkpoint_overlay() -> None:
    """Overlay local fine-tune weights on top of the Wan VACE runtime modules."""
    prepared = _make_prepared_batch()
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
    model, action_encoder, proprio_encoder = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )
    assert proprio_encoder is None

    model_state = {key: torch.full_like(value, 0.25) for key, value in model.state_dict().items()}
    action_state = {key: torch.full_like(value, -0.5) for key, value in action_encoder.state_dict().items()}
    checkpoint = {
        "model_state_dict": model_state,
        "action_encoder_state_dict": action_state,
    }

    loaded_model, loaded_action_encoder, loaded_proprio_encoder = wan_vace_factory.build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=torch.device("cpu"),
        checkpoint=checkpoint,
    )

    assert loaded_proprio_encoder is None
    first_model_key = next(iter(model_state))
    first_action_key = next(iter(action_state))
    assert torch.allclose(loaded_model.state_dict()[first_model_key], model_state[first_model_key])
    assert torch.allclose(loaded_action_encoder.state_dict()[first_action_key], action_state[first_action_key])


def _make_prepared_batch() -> PreparedPackedBatch:
    """Build a compact prepared batch fixture for Wan VACE module tests."""
    return PreparedPackedBatch(
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
