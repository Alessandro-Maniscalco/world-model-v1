"""Tests for the Wan VACE inference entrypoint wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from world_model.config import InferScriptConfig, load_infer_config
from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import (
    ActionControlProjector,
    ActionTokenEncoder,
    NullActionControlProjector,
    NullConditioningEncoder,
)
from world_model.models.wan_vace_world_model import WanVACEWorldModel


def test_infer_script_builds_wan_vace_runtime_modules_without_checkpoint() -> None:
    """Build Wan VACE inference modules without requiring a local fine-tune checkpoint."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = InferScriptConfig(
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

    model, action_encoder, action_control_projector = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, NullConditioningEncoder)
    assert isinstance(action_control_projector, NullActionControlProjector)


def test_infer_script_parser_omits_legacy_dit_shape_flags() -> None:
    """Avoid exposing removed non-VACE backbone shape flags."""
    infer_script = _load_infer_script_module()
    parser = infer_script._build_parser(load_infer_config())
    option_strings = {option for action in parser._actions for option in action.option_strings}

    assert "--action-temporal-difference-scale" in option_strings
    assert "--action-temporal-mixer-kernel-size" in option_strings
    assert "--action-temporal-mixer-scale" in option_strings
    assert "--action-conditioning-window" in option_strings
    assert "--chunk-schedule-mode" in option_strings
    assert "--action-order-conditioning" in option_strings
    assert "--action-control-prior-scale" in option_strings
    assert "--action-token-scale" in option_strings
    assert "--action-control-prior-mode" in option_strings
    assert "--action-hidden-state-bias-scale" in option_strings
    assert "--hidden-dim" not in option_strings
    assert "--num-layers" not in option_strings
    assert "--num-heads" not in option_strings


def test_build_prompt_conditioning_tokens_match_batch_and_sequence_length() -> None:
    """Build prompt embeddings that match the requested batch and max sequence length."""
    infer_script = _load_infer_script_module()

    class _FakeTokenizer:
        def __call__(self, prompt, **kwargs):
            del kwargs
            batch = len(prompt)
            return SimpleNamespace(
                input_ids=torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)[:batch],
                attention_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)[:batch],
            )

    class _FakeTextEncoder:
        dtype = torch.float32

        def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            del attention_mask
            hidden = input_ids.unsqueeze(-1).repeat(1, 1, 3).float()
            return SimpleNamespace(last_hidden_state=hidden)

    prompt_embeds, negative_prompt_embeds = infer_script.build_prompt_conditioning_tokens(
        prompt="bird",
        negative_prompt="blur",
        batch_size=2,
        tokenizer=_FakeTokenizer(),
        text_encoder=_FakeTextEncoder(),
        encoder_device=torch.device("cpu"),
        output_device=torch.device("cpu"),
        dtype=torch.float32,
        guidance_scale=5.0,
        max_sequence_length=4,
    )

    assert prompt_embeds.shape == (2, 4, 3)
    assert negative_prompt_embeds is not None
    assert negative_prompt_embeds.shape == (2, 4, 3)
    assert prompt_embeds.dtype == torch.float32
    assert prompt_embeds.device.type == "cpu"


def test_infer_script_rejects_action_conditioning_without_checkpoint() -> None:
    """Require a trained checkpoint when using action conditioning at inference time."""
    infer_script = _load_infer_script_module()

    with pytest.raises(ValueError, match="Action conditioning requires --checkpoint"):
        infer_script._validate_infer_config(
            InferScriptConfig(
                checkpoint="",
                conditioning_mode="action",
            )
        )


def test_infer_script_builds_action_encoder_when_requested() -> None:
    """Keep action-conditioning inference wiring available for later checkpoints."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
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

    model, action_encoder, action_control_projector = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert isinstance(action_control_projector, ActionControlProjector)


def test_infer_script_builds_action_encoder_with_mlp_when_requested() -> None:
    """Allow the infer config to request a deeper action-token encoder."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = InferScriptConfig(
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

    model, action_encoder, _ = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.net[1].out_features == 10
    assert action_encoder.net[4].out_features == 16


def test_infer_script_builds_action_encoder_with_residual_mlp_when_requested() -> None:
    """Allow the infer config to request a residual action-token MLP."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
    cfg = InferScriptConfig(
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

    model, action_encoder, _ = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.mlp_residual is True
    assert action_encoder.residual_net is not None
    assert action_encoder.net[1].out_features == 16
    assert action_encoder.residual_net[0].out_features == 10


def test_infer_script_builds_action_encoder_with_temporal_difference_scale_when_requested() -> None:
    """Allow the infer config to request temporal-difference-aware action tokens."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
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

    model, action_encoder, _ = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.temporal_difference_scale == pytest.approx(0.75)


def test_infer_script_builds_action_encoder_with_token_scale_when_requested() -> None:
    """Allow the infer config to scale projected action tokens directly."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
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

    model, action_encoder, _ = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.token_scale == pytest.approx(2.0)


def test_infer_script_builds_action_encoder_with_temporal_mixer_when_requested() -> None:
    """Allow the infer config to request a temporal mixer over action tokens."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
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

    model, action_encoder, _ = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(model, WanVACEWorldModel)
    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.temporal_mixer is not None
    assert action_encoder.temporal_mixer_scale == pytest.approx(0.5)


def test_infer_script_builds_action_ordered_prior_modules_when_requested() -> None:
    """Expose both ordered action tokens and a latent prior projector for action inference."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(2, 16, 2, 8, 8),
        z_future_video=torch.randn(2, 16, 4, 8, 8),
        a_plan=torch.randn(2, 4, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=6,
        context_latent_steps=2,
        horizon_latent_steps=4,
    )
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

    _, action_encoder, action_control_projector = infer_script.build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=torch.device("cpu"),
        checkpoint=None,
    )

    assert isinstance(action_encoder, ActionTokenEncoder)
    assert action_encoder.order_conditioning is True
    assert isinstance(action_control_projector, ActionControlProjector)


def test_infer_script_restores_lora_runtime_settings_from_checkpoint_defaults() -> None:
    """Reuse saved LoRA and conditioning settings when the infer config still uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "trainable_backbone": "lora",
                "conditioning_mode": "action",
                "lora_rank": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.1,
                "lora_target_modules": ["to_q", "to_k", "to_v"],
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.trainable_backbone == "lora"
    assert restored.conditioning_mode == "action"
    assert restored.lora_rank == 4
    assert restored.lora_alpha == 8
    assert restored.lora_dropout == pytest.approx(0.1)
    assert restored.lora_target_modules == ("to_q", "to_k", "to_v")


def test_infer_script_restores_action_temporal_difference_scale_from_checkpoint_defaults() -> None:
    """Reuse saved temporal-difference action settings when infer config uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_temporal_difference_scale": 0.75,
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_temporal_difference_scale == pytest.approx(0.75)


def test_infer_script_restores_action_token_scale_from_checkpoint_defaults() -> None:
    """Reuse saved token-scale settings when infer config uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_token_scale": 2.0,
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_token_scale == pytest.approx(2.0)


def test_infer_script_restores_action_temporal_mixer_settings_from_checkpoint_defaults() -> None:
    """Reuse saved temporal-mixer settings when infer config uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_temporal_mixer_kernel_size": 3,
                "action_temporal_mixer_scale": 0.5,
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_temporal_mixer_kernel_size == 3
    assert restored.action_temporal_mixer_scale == pytest.approx(0.5)


def test_infer_script_restores_ordered_plan_settings_from_checkpoint_defaults() -> None:
    """Reuse saved ordered-plan inference settings when the runtime still uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_conditioning_window": "full",
                "action_order_conditioning": True,
                "action_control_prior_scale": 1.0,
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_conditioning_window == "full"
    assert restored.action_order_conditioning is True
    assert restored.action_control_prior_scale == pytest.approx(1.0)


def test_infer_script_restores_action_control_prior_mode_from_checkpoint_defaults() -> None:
    """Reuse saved latent-prior routing mode when the infer config still uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_control_prior_scale": 1.0,
                "action_control_prior_mode": "dual_fill",
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_control_prior_scale == pytest.approx(1.0)
    assert restored.action_control_prior_mode == "dual_fill"


def test_infer_script_restores_action_hidden_state_bias_scale_from_checkpoint_defaults() -> None:
    """Reuse saved hidden-state action bias scale when the infer config still uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_hidden_state_bias_scale": 0.75,
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_hidden_state_bias_scale == pytest.approx(0.75)


def test_infer_script_restores_action_control_projector_observed_context_mode_from_checkpoint_defaults() -> None:
    """Reuse saved projector observed-context mode when infer config still uses defaults."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig()
    checkpoint = {
        "extra_state": {
            "config": {
                "conditioning_mode": "action",
                "action_control_projector_observed_context_mode": "last_frame",
            }
        }
    }

    restored = infer_script._restore_runtime_config_from_checkpoint(cfg, checkpoint)

    assert restored.conditioning_mode == "action"
    assert restored.action_control_projector_observed_context_mode == "last_frame"


def test_infer_script_allows_zero_num_vis_frames_to_mean_show_all() -> None:
    """Treat `num_vis_frames=0` as a request to render every available frame."""
    infer_script = _load_infer_script_module()

    assert infer_script._resolve_visualized_frame_count(requested_frames=0, available_frames=6) == 6
    assert infer_script._resolve_visualized_frame_count(requested_frames=4, available_frames=6) == 4


def test_infer_script_builds_frame_report_with_raw_latent_and_decoded_counts() -> None:
    """Expose frame counts for raw, latent, and decoded future windows."""
    infer_script = _load_infer_script_module()
    cfg = InferScriptConfig(context_len=9, horizon_len=8, num_vis_frames=0)
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 3, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 6),
        latent_shape=(16, 8, 8),
        total_latent_steps=5,
        context_latent_steps=3,
        horizon_latent_steps=2,
    )
    source_video = torch.rand(1, 17, 3, 32, 32)
    raw_future = torch.rand(1, 8, 3, 32, 32)
    raw_future_aligned = torch.rand(1, 5, 3, 32, 32)
    pred_video = torch.rand(1, 5, 3, 32, 32)
    target_video = torch.rand(1, 5, 3, 32, 32)

    report = infer_script._build_frame_report(
        cfg=cfg,
        prepared=prepared,
        source_video=source_video,
        raw_future=raw_future,
        raw_future_aligned=raw_future_aligned,
        pred_video=pred_video,
        target_video=target_video,
    )

    assert report["raw_future_frames"] == 8
    assert report["latent_future_steps"] == 2
    assert report["decoded_roundtrip_future_frames"] == 5
    assert report["visualized_frames"] == 5


def test_infer_script_uses_vace_like_defaults_for_checkpoint_free_prompt_smoke_tests() -> None:
    """Promote checkpoint-free prompt inference to a less lossy single-chunk 50-step path."""
    infer_script = _load_infer_script_module()

    resolved = infer_script._resolve_effective_infer_config(
        InferScriptConfig(
            checkpoint="",
            conditioning_mode="prompt",
            integration_steps=20,
            single_chunk_rollout=False,
        )
    )

    assert resolved.integration_steps == 50
    assert resolved.single_chunk_rollout is True


def test_infer_script_switches_chunk_conditioning_for_full_plan_action_mode() -> None:
    """Disable per-chunk token slicing when ordered full-plan action conditioning is enabled."""
    infer_script = _load_infer_script_module()

    assert infer_script._uses_chunk_conditioning(InferScriptConfig(conditioning_mode="action")) is True
    assert (
        infer_script._uses_chunk_conditioning(
            InferScriptConfig(conditioning_mode="action", action_conditioning_window="full")
        )
        is False
    )
    assert infer_script._uses_chunk_conditioning(InferScriptConfig(conditioning_mode="prompt")) is False


def test_infer_script_builds_sharpness_report() -> None:
    """Report relative sharpness so generated blur can be compared to the VAE roundtrip."""
    infer_script = _load_infer_script_module()
    raw_future_aligned = torch.zeros(1, 2, 3, 4, 4)
    target_video = torch.zeros(1, 2, 3, 4, 4)
    pred_video = torch.zeros(1, 2, 3, 4, 4)
    pred_video[:, :, :, :, 2:] = 1.0
    target_video[:, :, :, :, 1:] = 1.0

    report = infer_script._build_sharpness_report(
        raw_future_aligned=raw_future_aligned,
        target_video=target_video,
        pred_video=pred_video,
    )

    assert "mean_gradient_energy" in report
    assert "relative_to_vae_roundtrip" in report
    assert report["mean_gradient_energy"]["generated"] >= 0.0


def test_infer_script_local_video_none_mode_ignores_checkpoint_action_dim(monkeypatch) -> None:
    """Allow null-conditioned checkpoint inference even when the action encoder has no learned weights."""
    infer_script = _load_infer_script_module()
    prepared = PreparedPackedBatch(
        z_past_video=torch.randn(1, 16, 2, 8, 8),
        z_future_video=torch.randn(1, 16, 2, 8, 8),
        a_plan=torch.randn(1, 2, 1),
        latent_shape=(16, 8, 8),
        total_latent_steps=4,
        context_latent_steps=2,
        horizon_latent_steps=2,
    )

    monkeypatch.setattr(
        infer_script,
        "_load_video_clip",
        lambda video_path, start_frame, total_frames: torch.zeros(1, total_frames, 3, 8, 8, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        infer_script,
        "preprocess_video_for_vae",
        lambda video_btchw, frame_height, frame_width: video_btchw,
    )

    captured: dict[str, torch.Tensor] = {}

    def _fake_prepare_packed_batch(**kwargs):
        captured["action"] = kwargs["batch"]["action"]
        return prepared

    monkeypatch.setattr(infer_script, "prepare_packed_batch", _fake_prepare_packed_batch)

    resolved, source_video = infer_script._prepare_from_local_video(
        cfg=InferScriptConfig(
            checkpoint="dummy.pt",
            conditioning_mode="none",
            video_path="clip.mp4",
            video_key="observation.images.exterior_1_left",
            context_len=9,
            horizon_len=8,
        ),
        ckpt={"action_encoder_state_dict": {}},
        vae=object(),
        device=torch.device("cpu"),
    )

    assert resolved is prepared
    assert source_video.shape == (1, 17, 3, 8, 8)
    assert captured["action"].shape == (1, 1)


def test_infer_script_decodes_on_cpu_after_cuda_sampling() -> None:
    """Move VAE decode to CPU so GPU inference can free memory before visualization."""
    infer_script = _load_infer_script_module()

    class _FakeVaeModule:
        def __init__(self) -> None:
            self.moves: list[tuple[str, torch.dtype | None]] = []

        def to(self, device=None, dtype=None):
            self.moves.append((str(device), dtype))
            return self

    class _FakeWanVAE:
        def __init__(self) -> None:
            self.vae = _FakeVaeModule()
            self.decode_inputs: list[tuple[str, torch.dtype]] = []

        def decode(self, latents, output_layout="BTCHW", output_range="zero_to_one"):
            del output_layout, output_range
            self.decode_inputs.append((latents.device.type, latents.dtype))
            return latents.float()

    fake_vae = _FakeWanVAE()
    pred_video, target_video = infer_script._decode_future_videos(
        vae=fake_vae,
        pred_future_video=torch.randn(1, 16, 2, 8, 8),
        target_future_video=torch.randn(1, 16, 2, 8, 8),
        device=torch.device("cuda"),
        disable_amp=False,
        runtime_dtype=torch.bfloat16,
    )

    assert fake_vae.vae.moves == [("cpu", torch.float32)]
    assert fake_vae.decode_inputs == [("cpu", torch.float32), ("cpu", torch.float32)]
    assert pred_video.device.type == "cpu"
    assert target_video.device.type == "cpu"


def test_infer_script_releases_vae_after_prepare_on_cuda() -> None:
    """Free GPU memory held by the VAE before loading the inference backbone."""
    infer_script = _load_infer_script_module()

    class _FakeVaeModule:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def to(self, device=None, dtype=None):
            del dtype
            self.moves.append(str(device))
            return self

    fake_vae = SimpleNamespace(vae=_FakeVaeModule())

    infer_script._release_vae_after_prepare(fake_vae, device=torch.device("cuda"))

    assert fake_vae.vae.moves == ["cpu"]


def _load_infer_script_module():
    """Load the infer script module without executing the CLI entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "infer_world_model.py"
    spec = importlib.util.spec_from_file_location("test_infer_world_model_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
