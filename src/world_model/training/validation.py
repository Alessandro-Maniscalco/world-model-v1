"""Validate staged training outputs before a resumed run continues.

This module checks that metrics and checkpoints are present and consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TrainingStageSummary:
    """Summarize the final recorded state of one training stage."""

    output_dir: Path
    metrics_path: Path
    checkpoint_path: Path
    expected_step: int
    last_step: int
    last_loss: float
    metric_rows: int


def load_metrics_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL metrics rows from disk."""
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics log not found: {metrics_path}")

    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Metrics row {line_number} in {metrics_path} must be a JSON object"
                )
            rows.append(payload)
    return rows


def validate_training_stage(
    output_dir: str | Path,
    *,
    expected_step: int,
    require_exact_step: bool = True,
    max_loss: float | None = None,
) -> TrainingStageSummary:
    """Validate one stage output directory and return a concise summary."""
    if expected_step <= 0:
        raise ValueError(f"expected_step must be > 0, got {expected_step}")

    output_path = Path(output_dir)
    metrics_path = output_path / "metrics.jsonl"
    rows = load_metrics_rows(metrics_path)
    if not rows:
        raise ValueError(f"Metrics log is empty: {metrics_path}")

    last_row = rows[-1]
    last_step = _coerce_step(last_row.get("step"), metrics_path)
    last_loss = _coerce_loss(last_row.get("loss"), metrics_path)

    if require_exact_step:
        if last_step != expected_step:
            raise ValueError(
                f"Last logged step {last_step} does not match expected_step={expected_step}"
            )
    elif last_step < expected_step:
        raise ValueError(
            f"Last logged step {last_step} is smaller than expected_step={expected_step}"
        )

    if max_loss is not None and last_loss > max_loss:
        raise ValueError(
            f"Last logged loss {last_loss:.6f} exceeds max_loss={max_loss:.6f}"
        )

    checkpoint_path = output_path / "checkpoints" / f"step_{expected_step:07d}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a dict: {checkpoint_path}")
    checkpoint_step = checkpoint.get("step")
    if checkpoint_step != expected_step:
        raise ValueError(
            f"Checkpoint step {checkpoint_step!r} does not match expected_step={expected_step}"
        )

    return TrainingStageSummary(
        output_dir=output_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        expected_step=expected_step,
        last_step=last_step,
        last_loss=last_loss,
        metric_rows=len(rows),
    )


def format_stage_summary(summary: TrainingStageSummary) -> str:
    """Render a compact human-readable stage validation summary."""
    return (
        f"validated output_dir={summary.output_dir} "
        f"expected_step={summary.expected_step} "
        f"last_step={summary.last_step} "
        f"last_loss={summary.last_loss:.6f} "
        f"metric_rows={summary.metric_rows} "
        f"checkpoint={summary.checkpoint_path}"
    )


def _coerce_step(value: Any, metrics_path: Path) -> int:
    """Normalize the final logged step value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Final metrics row in {metrics_path} is missing integer step")
    return value


def _coerce_loss(value: Any, metrics_path: Path) -> float:
    """Normalize the final logged loss and reject non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Final metrics row in {metrics_path} is missing numeric loss")
    loss = float(value)
    if not math.isfinite(loss):
        raise ValueError(f"Final metrics row in {metrics_path} has non-finite loss={loss!r}")
    return loss
