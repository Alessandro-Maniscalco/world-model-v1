"""Tests for staged training artifact validation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from world_model.training.validation import format_stage_summary, validate_training_stage


def test_validate_training_stage_accepts_matching_metrics_and_checkpoint(tmp_path: Path) -> None:
    """Accept a stage when metrics and checkpoint agree on the expected final step."""
    output_dir = tmp_path / "run"
    _write_metrics(
        output_dir / "metrics.jsonl",
        [
            {"step": 100, "loss": 2.5},
            {"step": 200, "loss": 1.25},
        ],
    )
    _write_checkpoint(output_dir / "checkpoints" / "step_0000200.pt", step=200)

    summary = validate_training_stage(output_dir, expected_step=200)

    assert summary.last_step == 200
    assert summary.last_loss == 1.25
    assert summary.metric_rows == 2
    assert "expected_step=200" in format_stage_summary(summary)


def test_validate_training_stage_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """Fail clearly when the expected final checkpoint file is absent."""
    output_dir = tmp_path / "run"
    _write_metrics(output_dir / "metrics.jsonl", [{"step": 200, "loss": 1.25}])

    with pytest.raises(FileNotFoundError, match="Expected checkpoint not found"):
        validate_training_stage(output_dir, expected_step=200)


def test_validate_training_stage_rejects_step_mismatch(tmp_path: Path) -> None:
    """Reject a stage when the last logged step is not the expected one."""
    output_dir = tmp_path / "run"
    _write_metrics(output_dir / "metrics.jsonl", [{"step": 199, "loss": 1.25}])
    _write_checkpoint(output_dir / "checkpoints" / "step_0000200.pt", step=200)

    with pytest.raises(ValueError, match="Last logged step 199 does not match expected_step=200"):
        validate_training_stage(output_dir, expected_step=200)


def test_validate_training_stage_rejects_non_finite_loss(tmp_path: Path) -> None:
    """Reject a stage when the final loss is NaN or infinite."""
    output_dir = tmp_path / "run"
    _write_metrics(output_dir / "metrics.jsonl", [{"step": 200, "loss": math.nan}])
    _write_checkpoint(output_dir / "checkpoints" / "step_0000200.pt", step=200)

    with pytest.raises(ValueError, match="non-finite loss"):
        validate_training_stage(output_dir, expected_step=200)


def test_validate_training_stage_enforces_max_loss_when_requested(tmp_path: Path) -> None:
    """Allow callers to stop a staged run when loss has degraded too far."""
    output_dir = tmp_path / "run"
    _write_metrics(output_dir / "metrics.jsonl", [{"step": 200, "loss": 1.25}])
    _write_checkpoint(output_dir / "checkpoints" / "step_0000200.pt", step=200)

    with pytest.raises(ValueError, match="exceeds max_loss=1.000000"):
        validate_training_stage(output_dir, expected_step=200, max_loss=1.0)


def _write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write a compact JSONL metrics log for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{_json_line(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _write_checkpoint(path: Path, *, step: int) -> None:
    """Write a minimal checkpoint payload for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step}, path)


def _json_line(payload: dict[str, float | int]) -> str:
    """Serialize one metrics payload as a JSON object line."""
    import json

    return json.dumps(payload, sort_keys=True)
