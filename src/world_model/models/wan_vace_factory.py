"""Shared builders for Wan VACE runtime modules and checkpoint overlays."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import torch
from diffusers.pipelines.wan.pipeline_wan_vace import prompt_clean
from transformers import AutoTokenizer, UMT5EncoderModel

from world_model.data.schema import PreparedPackedBatch
from world_model.models.wan_vace_conditioning import (
    ActionTokenEncoder,
    NullConditioningEncoder,
)
from world_model.models.wan_vace_world_model import WanVACEWorldModel
from world_model.vendor.wan import WanVACETransformer3DModel


_NONE_CONDITIONING_TOKEN_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_NONE_CONDITIONING_MAX_SEQUENCE_LENGTH = 512


def build_wan_vace_model_from_config(cfg: Any, prepared_batch: PreparedPackedBatch) -> WanVACEWorldModel:
    """Build a Wan VACE world-model adapter from typed config and prepared batch data."""
    latent_channels = int(prepared_batch.latent_shape[0])
    expected_control_channels = _expected_control_channels(latent_channels=latent_channels, mask_channels=cfg.mask_channels)
    use_action_added_kv = _uses_action_added_kv(cfg)
    if cfg.load_pretrained_backbone:
        backbone = WanVACETransformer3DModel.from_pretrained(
            cfg.wan_vace_model_id,
            subfolder=cfg.wan_vace_subfolder or None,
            local_files_only=_offline_mode_enabled(),
        )
        if use_action_added_kv:
            backbone = _enable_action_added_kv_path(backbone=backbone)
    else:
        inner_dim = int(cfg.wan_num_attention_heads) * int(cfg.wan_attention_head_dim)
        backbone = WanVACETransformer3DModel(
            in_channels=latent_channels,
            out_channels=latent_channels,
            num_attention_heads=cfg.wan_num_attention_heads,
            attention_head_dim=cfg.wan_attention_head_dim,
            text_dim=cfg.wan_text_dim,
            freq_dim=cfg.wan_freq_dim,
            ffn_dim=cfg.wan_ffn_dim,
            num_layers=cfg.wan_num_layers,
            image_dim=(int(cfg.wan_text_dim) if use_action_added_kv else None),
            added_kv_proj_dim=(inner_dim if use_action_added_kv else None),
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
        if use_action_added_kv:
            _enable_action_added_kv_training(backbone=backbone)

    return WanVACEWorldModel(
        backbone=backbone,
        control_scale=cfg.control_scale,
        future_control_fill_mode=str(getattr(cfg, "future_control_fill_mode", "gray")),
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
        hidden_dim = int(model.backbone.config.text_dim)
        base_token = _build_pretrained_none_conditioning_token(cfg=cfg, hidden_dim=hidden_dim)
        return NullConditioningEncoder(
            hidden_dim=hidden_dim,
            base_token=base_token,
            trainable=base_token is not None,
        )
    return ActionTokenEncoder(
        action_dim=int(prepared_batch.a_plan.shape[-1]),
        hidden_dim=int(model.backbone.config.text_dim),
        latent_summary_channels=(
            int(model.backbone.config.in_channels)
            if float(getattr(cfg, "action_token_latent_aux_loss_scale", 0.0)) > 0.0
            else 0
        ),
        mlp_dim=_resolve_action_mlp_dim(cfg),
        mlp_residual=bool(getattr(cfg, "action_mlp_residual", False)),
        input_layernorm=bool(getattr(cfg, "action_input_layernorm", True)),
        order_conditioning=bool(getattr(cfg, "action_order_conditioning", False)),
        temporal_difference_scale=float(getattr(cfg, "action_temporal_difference_scale", 0.0)),
        temporal_mixer_kernel_size=int(getattr(cfg, "action_temporal_mixer_kernel_size", 0) or 0),
        temporal_mixer_scale=float(getattr(cfg, "action_temporal_mixer_scale", 0.0)),
        token_scale=float(getattr(cfg, "action_token_scale", 1.0)),
        output_zero_init=bool(getattr(cfg, "action_output_zero_init", True)),
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
    if _checkpoint_uses_fresh_action_encoder(
        checkpoint=checkpoint,
        action_encoder=action_encoder,
        action_state=action_state,
    ):
        return
    _load_action_encoder_state_dict(action_encoder=action_encoder, action_state=action_state)


def build_wan_vace_runtime_modules(
    cfg: Any,
    prepared_batch: PreparedPackedBatch,
    *,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[
    WanVACEWorldModel,
    ActionTokenEncoder | NullConditioningEncoder,
]:
    """Build Wan VACE runtime modules and optionally overlay a local fine-tune checkpoint."""
    original_cfg = cfg
    cfg = _merge_runtime_backbone_config(cfg=cfg, checkpoint=checkpoint)
    _apply_untouched_none_contract_defaults(cfg=cfg, checkpoint=checkpoint)
    _copy_runtime_contract_defaults(source_cfg=cfg, target_cfg=original_cfg)
    model = build_wan_vace_model_from_config(cfg, prepared_batch).to(device)
    action_encoder = build_conditioning_encoder_for_model(cfg, prepared_batch, model).to(device)
    if checkpoint is not None:
        apply_wan_vace_checkpoint_overlay(
            model=model,
            action_encoder=action_encoder,
            checkpoint=checkpoint,
        )
    return model, action_encoder


def _checkpoint_uses_fresh_action_encoder(
    *,
    checkpoint: dict[str, object],
    action_encoder: ActionTokenEncoder | NullConditioningEncoder,
    action_state: dict[str, torch.Tensor],
) -> bool:
    """Keep a fresh zero-init action encoder when probing action mode from a none checkpoint."""
    if isinstance(action_encoder, NullConditioningEncoder):
        return False
    if action_state and not _action_state_is_none_conditioning_token(action_state):
        return False
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        return False
    saved_cfg = extra_state.get("config")
    if not isinstance(saved_cfg, dict):
        return False
    return str(saved_cfg.get("conditioning_mode", "none")) == "none"


def _action_state_is_none_conditioning_token(action_state: dict[str, torch.Tensor]) -> bool:
    """Detect null-conditioning checkpoints that only save the reusable base token."""
    if not action_state:
        return True
    if set(action_state.keys()) != {"base_token"}:
        return False
    base_token = action_state["base_token"]
    return torch.is_tensor(base_token) and base_token.ndim == 2


def _expected_control_channels(*, latent_channels: int, mask_channels: int) -> int:
    """Compute the `[inactive; reactive; mask]` control-stream channel count."""
    return (2 * int(latent_channels)) + int(mask_channels)


def _offline_mode_enabled() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


@torch.no_grad()
def _build_pretrained_none_conditioning_token(*, cfg: Any, hidden_dim: int) -> torch.Tensor | None:
    """Initialize prompt-free null conditioning from Wan's empty-prompt token sequence."""
    if getattr(cfg, "conditioning_mode", "none") != "none":
        return None
    if not bool(getattr(cfg, "load_pretrained_backbone", False)):
        return None
    model_id = str(getattr(cfg, "wan_vace_model_id", "") or "")
    if not model_id:
        return None

    cache_key = (model_id, int(hidden_dim))
    cached = _NONE_CONDITIONING_TOKEN_CACHE.get(cache_key)
    if cached is not None:
        return cached.clone()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        local_files_only=_offline_mode_enabled(),
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        local_files_only=_offline_mode_enabled(),
    )
    text_encoder.eval()

    text_inputs = tokenizer(
        [prompt_clean("")],
        padding="max_length",
        max_length=_NONE_CONDITIONING_MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    attention_mask = text_inputs.attention_mask.to("cpu")
    seq_len = int(attention_mask.gt(0).sum(dim=1).long().item())
    prompt_embeds = text_encoder(
        text_inputs.input_ids.to("cpu"),
        attention_mask,
    ).last_hidden_state.to(device="cpu", dtype=torch.float32)
    prompt_tokens = prompt_embeds[0, :seq_len]
    if prompt_tokens.shape[-1] != hidden_dim:
        raise ValueError(
            f"Empty-prompt token width {prompt_tokens.shape[-1]} does not match hidden_dim={hidden_dim}"
        )
    padded_tokens = torch.cat(
        [
            prompt_tokens,
            prompt_tokens.new_zeros(
                _NONE_CONDITIONING_MAX_SEQUENCE_LENGTH - prompt_tokens.shape[0],
                prompt_tokens.shape[1],
            ),
        ],
        dim=0,
    )
    _NONE_CONDITIONING_TOKEN_CACHE[cache_key] = padded_tokens.detach().cpu()
    del text_encoder
    return _NONE_CONDITIONING_TOKEN_CACHE[cache_key].clone()


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
        "future_control_fill_mode",
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
        "action_backbone_added_kv_mode",
        "action_token_latent_aux_loss_scale",
        "action_temporal_difference_scale",
        "action_temporal_mixer_kernel_size",
        "action_temporal_mixer_scale",
        "action_token_scale",
        "action_output_zero_init",
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


def _apply_untouched_none_contract_defaults(*, cfg: Any, checkpoint: dict[str, object] | None) -> None:
    """Upgrade untouched pretrained none checkpoints to the validated dual-anchor contract."""
    if not _should_upgrade_untouched_none_contract(cfg=cfg, checkpoint=checkpoint):
        return
    if str(getattr(cfg, "future_control_fill_mode", "gray")) == "gray":
        cfg.future_control_fill_mode = "last_context_frame"
    if str(getattr(cfg, "future_latent_residual_mode", "none")) == "none":
        cfg.future_latent_residual_mode = "last_context_frame"


def _should_upgrade_untouched_none_contract(*, cfg: Any, checkpoint: dict[str, object] | None) -> bool:
    """Detect untouched pretrained none checkpoints that still carry the legacy unstable defaults."""
    if str(getattr(cfg, "conditioning_mode", "none")) != "none":
        return False
    if bool(getattr(cfg, "force_legacy_none_contract", False)):
        return False
    if not bool(getattr(cfg, "load_pretrained_backbone", False)):
        return False
    if str(getattr(cfg, "future_control_fill_mode", "gray")) != "gray":
        return False
    if str(getattr(cfg, "future_latent_residual_mode", "none")) != "none":
        return False
    return _checkpoint_saved_max_steps(checkpoint=checkpoint) == 0


def _checkpoint_saved_max_steps(*, checkpoint: dict[str, object] | None) -> int | None:
    """Read the saved train-step budget from checkpoint metadata when it is available."""
    if checkpoint is None:
        return None
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        return None
    saved_cfg = extra_state.get("config")
    if not isinstance(saved_cfg, dict):
        return None
    value = saved_cfg.get("max_steps")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _copy_runtime_contract_defaults(*, source_cfg: Any, target_cfg: Any) -> None:
    """Propagate effective none-contract defaults back to the caller-visible runtime config."""
    for key in ("future_control_fill_mode", "future_latent_residual_mode"):
        if hasattr(source_cfg, key):
            _set_runtime_cfg_attr(target_cfg=target_cfg, key=key, value=getattr(source_cfg, key))


def _set_runtime_cfg_attr(*, target_cfg: Any, key: str, value: Any) -> None:
    """Assign runtime config fields even when the caller uses a frozen dataclass config."""
    try:
        setattr(target_cfg, key, value)
    except (AttributeError, FrozenInstanceError, TypeError):
        object.__setattr__(target_cfg, key, value)


def _uses_action_added_kv(cfg: Any) -> bool:
    """Decide whether to mirror action tokens into Wan's added-K/V image-conditioning path."""
    return (
        getattr(cfg, "conditioning_mode", "none") == "action"
        and str(getattr(cfg, "action_backbone_added_kv_mode", "none")) == "reuse_action_tokens"
    )


def _enable_action_added_kv_path(backbone: WanVACETransformer3DModel) -> WanVACETransformer3DModel:
    """Rebuild a pretrained Wan VACE backbone with action-driven added-K/V image conditioning enabled."""
    config = backbone.config
    inner_dim = int(config.num_attention_heads) * int(config.attention_head_dim)
    upgraded = WanVACETransformer3DModel(
        patch_size=tuple(config.patch_size),
        num_attention_heads=int(config.num_attention_heads),
        attention_head_dim=int(config.attention_head_dim),
        in_channels=int(config.in_channels),
        out_channels=int(config.out_channels),
        text_dim=int(config.text_dim),
        freq_dim=int(config.freq_dim),
        ffn_dim=int(config.ffn_dim),
        num_layers=int(config.num_layers),
        cross_attn_norm=bool(config.cross_attn_norm),
        qk_norm=getattr(config, "qk_norm", "rms_norm_across_heads"),
        eps=float(config.eps),
        image_dim=int(config.text_dim),
        added_kv_proj_dim=inner_dim,
        rope_max_seq_len=int(getattr(config, "rope_max_seq_len", 1024)),
        pos_embed_seq_len=getattr(config, "pos_embed_seq_len", None),
        vace_layers=list(config.vace_layers),
        vace_in_channels=int(config.vace_in_channels),
    )
    incompatible = upgraded.load_state_dict(backbone.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected pretrained keys when enabling action added-K/V path: "
            f"{sorted(incompatible.unexpected_keys)}"
        )
    for key in incompatible.missing_keys:
        if key.startswith("condition_embedder.image_embedder."):
            continue
        if ".add_k_proj." in key or ".add_v_proj." in key:
            continue
        if ".norm_added_k." in key:
            continue
        raise ValueError(
            "Unexpected missing pretrained key when enabling action added-K/V path: "
            f"{key!r}"
        )
    _zero_init_new_action_added_kv_modules(backbone=upgraded)
    return upgraded


def _enable_action_added_kv_training(*, backbone: WanVACETransformer3DModel) -> None:
    """Keep newly introduced added-K/V image-conditioning weights trainable under LoRA."""
    for name, parameter in backbone.named_parameters():
        if "condition_embedder.image_embedder" in name or ".add_k_proj." in name or ".add_v_proj." in name:
            parameter.requires_grad = True


def _zero_init_new_action_added_kv_modules(*, backbone: WanVACETransformer3DModel) -> None:
    """Start newly added action-conditioning modules as an exact zero-token no-op."""
    image_embedder = getattr(getattr(backbone, "condition_embedder", None), "image_embedder", None)
    if image_embedder is not None:
        for parameter in image_embedder.parameters():
            torch.nn.init.zeros_(parameter)
    for name, parameter in backbone.named_parameters():
        if ".add_k_proj." in name or ".add_v_proj." in name:
            torch.nn.init.zeros_(parameter)


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
    if "output_proj.weight" in missing_keys:
        action_encoder.restore_legacy_output_projection()
    return bool(missing_keys)
