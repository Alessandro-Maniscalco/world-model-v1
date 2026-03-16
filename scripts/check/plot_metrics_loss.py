"""Plot training loss over optimization steps from a JSONL metrics file.

source .venv/bin/activate
python scripts/check/plot_metrics_loss.py \
  runs/hour_test_action/metrics.jsonl \
  --rolling-window 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for loss plotting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_path", type=Path, help="Path to the JSONL metrics file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults next to the metrics file.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help="Optional moving-average window size. Disabled when <= 1.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training Loss Over Steps",
        help="Plot title.",
    )
    return parser.parse_args()


def load_step_loss_series(metrics_path: Path) -> tuple[list[int], list[float]]:
    """Load `(step, loss)` pairs from a JSONL metrics file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    steps: list[int] = []
    losses: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "step" not in payload or "loss" not in payload:
                raise ValueError(
                    f"Line {line_number} in {metrics_path} is missing required keys 'step' and 'loss'."
                )
            steps.append(int(payload["step"]))
            losses.append(float(payload["loss"]))

    if not steps:
        raise ValueError(f"No metric rows found in {metrics_path}.")

    return steps, losses


def build_moving_average(values: list[float], window: int) -> list[float]:
    """Compute a simple trailing moving average for plotting."""
    if window <= 1:
        return values

    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return smoothed


def default_output_path(metrics_path: Path) -> Path:
    """Choose the default PNG output path for a metrics file."""
    return metrics_path.with_name(f"{metrics_path.stem}_loss.png")


def plot_loss(
    *,
    steps: list[int],
    losses: list[float],
    output_path: Path,
    rolling_window: int,
    title: str,
) -> None:
    """Render the loss-vs-step plot to disk."""
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(steps, losses, label="loss", linewidth=1.3, alpha=0.8)

    if rolling_window > 1:
        smoothed = build_moving_average(losses, rolling_window)
        axis.plot(steps, smoothed, label=f"moving avg ({rolling_window})", linewidth=2.0)

    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Load a metrics file and save a loss-over-steps plot."""
    args = parse_args()
    steps, losses = load_step_loss_series(args.metrics_path)
    output_path = args.output if args.output is not None else default_output_path(args.metrics_path)
    plot_loss(
        steps=steps,
        losses=losses,
        output_path=output_path,
        rolling_window=args.rolling_window,
        title=args.title,
    )
    print(f"Saved loss plot to: {output_path}")
    print(f"Loaded {len(steps)} points from: {args.metrics_path}")


if __name__ == "__main__":
    main()
