"""Typed script configuration and YAML loading helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TrainScriptConfig:
    """Configuration for world-model training entrypoint."""

    repo_id: str = "lerobot/libero"
    video_key: str = "observation.images.image"
    output_dir: str = "runs/world_model_train"
    context_len: int = 10
    horizon_len: int = 8
    dt: float = 0.1
    batch_size: int = 2
    k: int = 1
    max_steps: int = 2000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    weight_mode: str = "uniform"
    t_min: float = 0.0
    t_max: float = 1.0
    num_layers: int = 6
    num_heads: int = 8
    hidden_dim: int = 1024
    disable_proprio: bool = False
    disable_amp: bool = False
    gradient_checkpointing: bool = False
    num_workers: int = 0
    log_every: int = 10
    checkpoint_every: int = 100
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
    context_len: int = 10
    horizon_len: int = 8
    dt: float = 0.1
    batch_size: int = 1
    subset_size: int = 1
    k: int = 1
    integration_steps: int = 20
    num_vis_frames: int = 5
    hidden_dim: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    action_path: str = ""
    action_dim: int = 0
    action_value: float = 0.0
    disable_proprio: bool = False
    disable_amp: bool = False
    seed: int = 0


def load_train_config(path: str | Path | None = None) -> TrainScriptConfig:
    """Load a training config dataclass from YAML, or defaults when path is None."""
    return _load_config(TrainScriptConfig, path)


def load_infer_config(path: str | Path | None = None) -> InferScriptConfig:
    """Load an inference config dataclass from YAML, or defaults when path is None."""
    return _load_config(InferScriptConfig, path)


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


def _load_config(cls: type[T], path: str | Path | None) -> T:
    """Load dataclass `cls` from YAML at `path`, or return default instance."""
    if path is None:
        return cls()

    path_obj = Path(path)
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
    base.update(payload)
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
