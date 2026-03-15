"""Behavior tests for typed YAML-backed config helpers."""

from __future__ import annotations

from argparse import Namespace

import pytest

from world_model.config import (
    InferScriptConfig,
    apply_namespace_overrides,
    load_infer_config,
    load_train_config,
    to_parser_defaults,
)


def test_apply_namespace_overrides_only_replaces_non_none_values() -> None:
    """Override config fields only when the argparse value is not `None`."""
    config = InferScriptConfig(prompt="base", guidance_scale=5.0)
    namespace = Namespace(prompt="updated", guidance_scale=None, unused_field="ignored")

    updated = apply_namespace_overrides(config, namespace)

    assert updated.prompt == "updated"
    assert updated.guidance_scale == 5.0


def test_to_parser_defaults_returns_dataclass_mapping() -> None:
    """Expose dataclass config fields as parser defaults."""
    defaults = to_parser_defaults(InferScriptConfig(prompt="hello"))

    assert defaults["prompt"] == "hello"
    assert defaults["guidance_scale"] == 5.0


def test_load_config_coerces_yaml_lists_to_tuples(tmp_path) -> None:
    """Convert YAML lists into tuple-backed config fields."""
    config_path = tmp_path / "infer.yaml"
    config_path.write_text(
        "vace_layers: [1, 3, 5]\n"
        "lora_target_modules: ['to_q', 'to_v']\n",
        encoding="utf-8",
    )

    cfg = load_infer_config(config_path)

    assert cfg.vace_layers == (1, 3, 5)
    assert cfg.lora_target_modules == ("to_q", "to_v")


def test_load_config_rejects_unknown_keys(tmp_path) -> None:
    """Fail fast when YAML includes fields outside the typed config schema."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("unknown_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config keys"):
        load_train_config(config_path)


def test_load_config_rejects_missing_files(tmp_path) -> None:
    """Raise a clear error when the requested config file does not exist."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_train_config(missing_path)


def test_load_config_rejects_non_mapping_yaml_root(tmp_path) -> None:
    """Reject YAML documents whose root node is not a mapping."""
    config_path = tmp_path / "infer.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping at root"):
        load_infer_config(config_path)
