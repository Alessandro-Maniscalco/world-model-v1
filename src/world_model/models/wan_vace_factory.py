"""Shared builders for Wan VACE runtime modules and checkpoint overlays."""

from __future__ import annotations

import os
from typing import Any

import torch

from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import ActionTokenEncoder
from world_model.models.wan_vace_world_model import WanVACEWorldModel
from world_model.vendor.wan import WanVACETransformer3DModel


def build_wan_vace_model_from_config(cfg: Any, prepared_batch: PreparedPackedBatch) -> WanVACEWorldModel:
    """Build a Wan VACE world-model adapter from typed config and prepared batch data."""
    latent_channels = int(prepared_batch.latent_shape[0])
    expected_control_channels = _expected_control_channels(latent_channels=latent_channels, mask_channels=cfg.mask_channels)
    if cfg.load_pretrained_backbone:
        backbone = WanVACETransformer3DModel.from_pretrained(
            cfg.wan_vace_model_id,
            subfolder=cfg.wan_vace_subfolder or None,
            local_files_only=_offline_mode_enabled(),
        )
    else:
        backbone = WanVACETransformer3DModel(
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_attention_heads=cfg.wan_num_attention_heads,
            attention_head_dim=cfg.wan_attention_head_dim,
            text_dim=cfg.wan_text_dim,
            freq_dim=cfg.wan_freq_dim,
            ffn_dim=cfg.wan_ffn_dim,
            num_layers=cfg.wan_num_layers,
            vace_layers=list(cfg.vace_layers),
            vace_in_channels=expected_control_channels,
        )

    if getattr(backbone.config, "in_channels", latent_channels) != latent_channels:
        raise ValueError(
            f"Wan VACE in_channels={backbone.config.in_channels} does not match latent channels={latent_channels}"
        )
    if getattr(backbone.config, "vace_in_channels", expected_control_channels) != expected_control_channels:
        raise ValueError(
            f"Wan VACE vace_in_channels={backbone.config.vace_in_channels} does not match expected "
            f"control channels={expected_control_channels}"
        )
    if getattr(cfg, "gradient_checkpointing", False):
        if hasattr(backbone, "enable_gradient_checkpointing"):
            backbone.enable_gradient_checkpointing()
        else:
            backbone.gradient_checkpointing = True

    return WanVACEWorldModel(
        backbone=backbone,
        control_scale=cfg.control_scale,
        mask_channels=cfg.mask_channels,
    )


def build_action_token_encoder_for_model(
    prepared_batch: PreparedPackedBatch,
    model: WanVACEWorldModel,
) -> ActionTokenEncoder:
    """Build an action-token encoder matching the Wan backbone text width."""
    return ActionTokenEncoder(
        action_dim=int(prepared_batch.a_plan.shape[-1]),
        hidden_dim=int(model.backbone.config.text_dim),
    )


def apply_wan_vace_checkpoint_overlay(
    *,
    model: WanVACEWorldModel,
    action_encoder: ActionTokenEncoder,
    checkpoint: dict[str, object],
) -> None:
    """Overlay local fine-tune checkpoint weights onto runtime Wan VACE modules."""
    model_state = checkpoint.get("model_state_dict")
    action_state = checkpoint.get("action_encoder_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint missing model_state_dict")
    if not isinstance(action_state, dict):
        raise ValueError("Checkpoint missing action_encoder_state_dict")
    model.load_state_dict(model_state)
    action_encoder.load_state_dict(action_state)


def build_wan_vace_runtime_modules(
    cfg: Any,
    prepared_batch: PreparedPackedBatch,
    *,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[WanVACEWorldModel, ActionTokenEncoder]:
    """Build Wan VACE runtime modules and optionally overlay a local fine-tune checkpoint."""
    model = build_wan_vace_model_from_config(cfg, prepared_batch).to(device)
    action_encoder = build_action_token_encoder_for_model(prepared_batch, model).to(device)
    if checkpoint is not None:
        apply_wan_vace_checkpoint_overlay(
            model=model,
            action_encoder=action_encoder,
            checkpoint=checkpoint,
        )
    return model, action_encoder


def _expected_control_channels(*, latent_channels: int, mask_channels: int) -> int:
    """Compute the `[inactive; reactive; mask]` control-stream channel count."""
    return (2 * int(latent_channels)) + int(mask_channels)


def _offline_mode_enabled() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
