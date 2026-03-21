"""Shared builders for Wan VACE runtime modules and checkpoint overlays."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import torch

from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import (
    ActionControlProjector,
    ActionTokenEncoder,
    NullActionControlProjector,
    NullConditioningEncoder,
)
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
    if getattr(cfg, "trainable_backbone", "full") == "lora":
        _attach_lora_adapters(backbone=backbone, cfg=cfg)

    return WanVACEWorldModel(
        backbone=backbone,
        control_scale=cfg.control_scale,
        action_control_prior_scale=float(getattr(cfg, "action_control_prior_scale", 0.0)),
        action_control_prior_mode=str(getattr(cfg, "action_control_prior_mode", "reactive_only")),
        action_hidden_state_bias_scale=float(getattr(cfg, "action_hidden_state_bias_scale", 0.0)),
        mask_channels=cfg.mask_channels,
        control_black_latents=prepared_batch.control_black_latents,
        control_gray_latents=prepared_batch.control_gray_latents,
    )


def build_conditioning_encoder_for_model(
    cfg: Any,
    prepared_batch: PreparedPackedBatch,
    model: WanVACEWorldModel,
) -> ActionTokenEncoder | NullConditioningEncoder:
    """Build the configured cross-attention encoder matching the Wan text width."""
    if getattr(cfg, "conditioning_mode", "none") != "action":
        return NullConditioningEncoder(hidden_dim=int(model.backbone.config.text_dim))
    return ActionTokenEncoder(
        action_dim=int(prepared_batch.a_plan.shape[-1]),
        hidden_dim=int(model.backbone.config.text_dim),
        mlp_dim=_resolve_action_mlp_dim(cfg),
        mlp_residual=bool(getattr(cfg, "action_mlp_residual", False)),
        input_layernorm=bool(getattr(cfg, "action_input_layernorm", True)),
        order_conditioning=bool(getattr(cfg, "action_order_conditioning", False)),
        temporal_difference_scale=float(getattr(cfg, "action_temporal_difference_scale", 0.0)),
        temporal_mixer_kernel_size=int(getattr(cfg, "action_temporal_mixer_kernel_size", 0) or 0),
        temporal_mixer_scale=float(getattr(cfg, "action_temporal_mixer_scale", 0.0)),
        token_scale=float(getattr(cfg, "action_token_scale", 1.0)),
    )


def build_action_control_projector_for_model(
    cfg: Any,
    prepared_batch: PreparedPackedBatch,
    model: WanVACEWorldModel,
) -> ActionControlProjector | NullActionControlProjector:
    """Build the action-to-latent control-prior projector for action-conditioned runs."""
    if getattr(cfg, "conditioning_mode", "none") != "action":
        return NullActionControlProjector(latent_channels=int(model.backbone.config.in_channels))
    return ActionControlProjector(
        action_dim=int(prepared_batch.a_plan.shape[-1]),
        latent_channels=int(model.backbone.config.in_channels),
        init_mode=str(getattr(cfg, "action_control_projector_init_mode", "zero")),
    )


def build_action_token_encoder_for_model(
    prepared_batch: PreparedPackedBatch,
    model: WanVACEWorldModel,
) -> ActionTokenEncoder:
    """Build the legacy action-token encoder matching the Wan backbone text width."""
    return ActionTokenEncoder(
        action_dim=int(prepared_batch.a_plan.shape[-1]),
        hidden_dim=int(model.backbone.config.text_dim),
    )


def apply_wan_vace_checkpoint_overlay(
    *,
    model: WanVACEWorldModel,
    action_encoder: ActionTokenEncoder | NullConditioningEncoder,
    action_control_projector: ActionControlProjector | NullActionControlProjector,
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
    _load_action_encoder_state_dict(action_encoder=action_encoder, action_state=action_state)
    _load_action_control_projector_state_dict(
        action_control_projector=action_control_projector,
        projector_state=checkpoint.get("action_control_projector_state_dict"),
    )


def build_wan_vace_runtime_modules(
    cfg: Any,
    prepared_batch: PreparedPackedBatch,
    *,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[
    WanVACEWorldModel,
    ActionTokenEncoder | NullConditioningEncoder,
    ActionControlProjector | NullActionControlProjector,
]:
    """Build Wan VACE runtime modules and optionally overlay a local fine-tune checkpoint."""
    cfg = _merge_runtime_backbone_config(cfg=cfg, checkpoint=checkpoint)
    model = build_wan_vace_model_from_config(cfg, prepared_batch).to(device)
    action_encoder = build_conditioning_encoder_for_model(cfg, prepared_batch, model).to(device)
    action_control_projector = build_action_control_projector_for_model(cfg, prepared_batch, model).to(device)
    if checkpoint is not None:
        apply_wan_vace_checkpoint_overlay(
            model=model,
            action_encoder=action_encoder,
            action_control_projector=action_control_projector,
            checkpoint=checkpoint,
        )
    return model, action_encoder, action_control_projector


def _expected_control_channels(*, latent_channels: int, mask_channels: int) -> int:
    """Compute the `[inactive; reactive; mask]` control-stream channel count."""
    return (2 * int(latent_channels)) + int(mask_channels)


def _offline_mode_enabled() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _attach_lora_adapters(*, backbone: WanVACETransformer3DModel, cfg: Any) -> None:
    """Attach PEFT LoRA adapters to the Wan VACE backbone."""
    if int(cfg.lora_rank) <= 0:
        raise ValueError(f"lora_rank must be positive, got {cfg.lora_rank}")
    if int(cfg.lora_alpha) <= 0:
        raise ValueError(f"lora_alpha must be positive, got {cfg.lora_alpha}")
    if float(cfg.lora_dropout) < 0.0:
        raise ValueError(f"lora_dropout must be non-negative, got {cfg.lora_dropout}")
    target_modules = [str(module_name) for module_name in getattr(cfg, "lora_target_modules", ())]
    if not target_modules:
        raise ValueError("lora_target_modules must be non-empty when trainable_backbone=lora")

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("peft is required when trainable_backbone=lora.") from exc

    lora_config = LoraConfig(
        r=int(cfg.lora_rank),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    backbone.add_adapter(lora_config)
    if hasattr(backbone, "enable_adapters"):
        backbone.enable_adapters()


def _merge_runtime_backbone_config(cfg: Any, checkpoint: dict[str, object] | None) -> Any:
    """Restore train-time backbone settings from checkpoint metadata when available."""
    if checkpoint is None:
        return cfg
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        return cfg
    saved_cfg = extra_state.get("config")
    if not isinstance(saved_cfg, dict):
        return cfg

    default_cfg = _make_default_config_like(cfg)
    update_keys = (
        "trainable_backbone",
        "conditioning_mode",
        "control_scale",
        "mask_channels",
        "vace_layers",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "action_input_layernorm",
        "action_mlp_dim",
        "action_mlp_residual",
        "action_conditioning_window",
        "action_order_conditioning",
        "action_control_prior_scale",
        "action_control_prior_mode",
        "action_control_projector_init_mode",
        "action_hidden_state_bias_scale",
        "action_temporal_difference_scale",
        "action_temporal_mixer_kernel_size",
        "action_temporal_mixer_scale",
        "action_token_scale",
    )
    updates: dict[str, Any] = {}
    for key in update_keys:
        if not hasattr(cfg, key) or key not in saved_cfg or not hasattr(default_cfg, key):
            continue
        if getattr(cfg, key) != getattr(default_cfg, key):
            continue
        value = saved_cfg[key]
        if key in {"vace_layers", "lora_target_modules"}:
            value = tuple(value)
        updates[key] = value

    if not updates:
        return cfg
    cfg_dict = dict(vars(cfg))
    cfg_dict.update(updates)
    return SimpleNamespace(**cfg_dict)


def _make_default_config_like(cfg: Any) -> Any:
    """Construct a default config instance matching the runtime config type when possible."""
    cfg_type = type(cfg)
    try:
        return cfg_type()
    except TypeError:
        return SimpleNamespace()


def _resolve_action_mlp_dim(cfg: Any) -> int | None:
    """Resolve non-positive config values to the encoder's default linear projection path."""
    value = int(getattr(cfg, "action_mlp_dim", 0) or 0)
    return None if value <= 0 else value


def _load_action_encoder_state_dict(
    *,
    action_encoder: ActionTokenEncoder | NullConditioningEncoder,
    action_state: dict[str, torch.Tensor],
) -> bool:
    """Load action-encoder weights while tolerating missing optional module params."""
    if isinstance(action_encoder, NullConditioningEncoder):
        return False

    incompatible = action_encoder.load_state_dict(action_state, strict=False)
    unexpected_keys = set(incompatible.unexpected_keys)
    missing_keys = set(incompatible.missing_keys)
    allowed_missing = action_encoder.allowed_missing_state_dict_keys()
    if unexpected_keys:
        raise RuntimeError(f"Unexpected action-encoder checkpoint keys: {sorted(unexpected_keys)}")
    disallowed_missing = missing_keys - allowed_missing
    if disallowed_missing:
        raise RuntimeError(f"Missing required action-encoder checkpoint keys: {sorted(disallowed_missing)}")
    return bool(missing_keys)


def _load_action_control_projector_state_dict(
    *,
    action_control_projector: ActionControlProjector | NullActionControlProjector,
    projector_state: object,
) -> bool:
    """Load action-control-prior weights while tolerating old checkpoints without them."""
    if isinstance(action_control_projector, NullActionControlProjector):
        return False
    if projector_state is None:
        return True
    if not isinstance(projector_state, dict):
        raise ValueError("Checkpoint action_control_projector_state_dict must be a dict when present")

    incompatible = action_control_projector.load_state_dict(projector_state, strict=False)
    unexpected_keys = set(incompatible.unexpected_keys)
    missing_keys = set(incompatible.missing_keys)
    allowed_missing = action_control_projector.allowed_missing_state_dict_keys()
    if unexpected_keys:
        raise RuntimeError(f"Unexpected action-control-projector checkpoint keys: {sorted(unexpected_keys)}")
    disallowed_missing = missing_keys - allowed_missing
    if disallowed_missing:
        raise RuntimeError(
            "Missing required action-control-projector checkpoint keys: "
            f"{sorted(disallowed_missing)}"
        )
    return True
