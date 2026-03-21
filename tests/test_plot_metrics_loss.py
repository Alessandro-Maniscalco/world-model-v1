"""Tests for the training-loss plotting helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_plot_metrics_loss_loads_sparse_validation_series(tmp_path: Path) -> None:
    """Load training and sparse validation loss rows from one JSONL metrics file."""
    plot_script = _load_plot_script_module()
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"step": 1, "loss": 1.0}\n'
        '{"step": 2, "loss": 0.8, "val_loss": 0.9}\n'
        '{"step": 3, "loss": 0.7}\n',
        encoding="utf-8",
    )

    steps, losses, val_steps, val_losses = plot_script.load_step_loss_series(metrics_path)

    assert steps == [1, 2, 3]
    assert losses == [1.0, 0.8, 0.7]
    assert val_steps == [2]
    assert val_losses == [0.9]


def test_plot_metrics_loss_renders_without_validation_rows(tmp_path: Path) -> None:
    """Keep plotting backwards-compatible when metrics logs have no validation loss."""
    plot_script = _load_plot_script_module()
    output_path = tmp_path / "loss.png"

    plot_script.plot_loss(
        steps=[1, 2, 3],
        losses=[1.0, 0.8, 0.7],
        val_steps=[],
        val_losses=[],
        output_path=output_path,
        rolling_window=2,
        title="Loss",
    )

    assert output_path.exists()


def _load_plot_script_module():
    """Load the plotting script module without executing the CLI entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "check" / "plot_metrics_loss.py"
    spec = importlib.util.spec_from_file_location("test_plot_metrics_loss_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
