"""Typed script configuration and YAML loading helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train" / "world_model.yaml"
DEFAULT_INFER_CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "infer_world_model.yaml"


@dataclass(frozen=True)
class TrainScriptConfig:
    """Configuration for world-model training entrypoint."""

    resume_from: str = ""
    video_path: str = ""
    start_frame: int = 0
    repo_id: str = "lerobot/libero"
    episodes: tuple[int, ...] = ()
    video_key: str = "observation.images.image"
    output_dir: str = "runs/world_model_train"
    context_len: int = 9
    horizon_len: int = 8
    dt: float = 0.1
    batch_size: int = 2
    k: int = 1
    max_steps: int = 2000
    auto_stop_check_every: int = 0
    auto_stop_min_relative_improvement: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    weight_mode: str = "uniform"
    motion_loss_alpha: float = 0.0
    motion_loss_max_weight: float = 0.0
    motion_loss_excess_only: bool = False
    t_min: float = 0.0
    t_max: float = 1.0
    disable_amp: bool = False
    gradient_checkpointing: bool = False
    load_pretrained_backbone: bool = True
    wan_vace_model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    wan_vace_subfolder: str = "transformer"
    wan_num_attention_heads: int = 40
    wan_attention_head_dim: int = 128
    wan_text_dim: int = 4096
    wan_freq_dim: int = 256
    wan_ffn_dim: int = 13824
    wan_num_layers: int = 40
    vace_layers: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 35)
    control_scale: float = 1.0
    mask_channels: int = 64
    trainable_backbone: str = "full"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
        "proj_in",
        "proj_out",
    )
    conditioning_mode: str = "none"
    action_input_layernorm: bool = True
    action_mlp_dim: int = 0
    action_mlp_residual: bool = False
    action_temporal_difference_scale: float = 0.0
    frame_height: int = 0
    frame_width: int = 0
    num_workers: int = 0
    log_every: int = 10
    checkpoint_every: int = 100
    checkpoint_early_every: int = 0
    checkpoint_early_until: int = 0
    subset_size: int = 0
    overfit_one_batch: bool = False
    seed: int = 0


@dataclass(frozen=True)
class InferScriptConfig:
    """Configuration for world-model inference entrypoint."""

    checkpoint: str = ""
    video_path: str = ""
    start_frame: int = 0
    repo_id: str = "lerobot/libero"
    video_key: str = "observation.images.image"
    output_dir: str = "runs/infer_world_model"
    context_len: int = 9
    horizon_len: int = 8
    dt: float = 0.1
    batch_size: int = 1
    subset_size: int = 1
    k: int = 1
    integration_steps: int = 20
    num_vis_frames: int = 0
    load_pretrained_backbone: bool = True
    wan_vace_model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    wan_vace_subfolder: str = "transformer"
    wan_num_attention_heads: int = 40
    wan_attention_head_dim: int = 128
    wan_text_dim: int = 4096
    wan_freq_dim: int = 256
    wan_ffn_dim: int = 13824
    wan_num_layers: int = 40
    vace_layers: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 35)
    control_scale: float = 1.0
    mask_channels: int = 64
    trainable_backbone: str = "full"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
        "proj_in",
        "proj_out",
    )
    conditioning_mode: str = "none"
    action_input_layernorm: bool = True
    action_mlp_dim: int = 0
    action_mlp_residual: bool = False
    action_temporal_difference_scale: float = 0.0
    frame_height: int = 0
    frame_width: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 5.0
    max_sequence_length: int = 512
    single_chunk_rollout: bool = False
    action_path: str = ""
    action_dim: int = 0
    action_value: float = 0.0
    disable_amp: bool = False
    seed: int = 0


def load_train_config(path: str | Path | None = None) -> TrainScriptConfig:
    """Load train config from YAML, defaulting to the canonical train preset."""
    return _load_config(TrainScriptConfig, path, DEFAULT_TRAIN_CONFIG_PATH)


def load_infer_config(path: str | Path | None = None) -> InferScriptConfig:
    """Load infer config from YAML, defaulting to the canonical eval preset."""
    return _load_config(InferScriptConfig, path, DEFAULT_INFER_CONFIG_PATH)


def apply_namespace_overrides(config: T, namespace: Any) -> T:
    """Apply argparse namespace values that are not None on top of a dataclass config."""
    payload = asdict(config)
    valid = set(payload.keys())
    for key, value in vars(namespace).items():
        if key in valid and value is not None:
            payload[key] = value
    return type(config)(**payload)


def to_parser_defaults(config: Any) -> dict[str, Any]:
    """Convert dataclass config into parser default mapping."""
    return asdict(config)


def _load_config(cls: type[T], path: str | Path | None, default_path: Path) -> T:
    """Load dataclass `cls` from YAML at `path` or the canonical preset file."""
    path_obj = default_path if path is None else Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    payload = _load_yaml(path_obj)
    return _coerce_dataclass(cls, payload)


def _coerce_dataclass(cls: type[T], payload: dict[str, Any]) -> T:
    """Construct a dataclass instance from partial mapping payload."""
    allowed = {field_info.name for field_info in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    base = asdict(cls())
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        base[field_info.name] = _coerce_field_value(base[field_info.name], payload[field_info.name])
    return cls(**base)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file into a dict mapping."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --config YAML loading") from exc

    with path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at root of {path}, got {type(loaded).__name__}")
    return loaded


def _coerce_field_value(default_value: Any, value: Any) -> Any:
    """Convert YAML-loaded values into the shapes expected by typed config fields."""
    if isinstance(default_value, tuple) and isinstance(value, list):
        return tuple(value)
    return value
