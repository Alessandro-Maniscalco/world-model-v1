"""Export and plot robot actions from an ALOHA LeRobot episode window.

Run:

source .venv/bin/activate
python scripts/check/preview_aloha_actions.py \
  --repo-id lerobot/aloha_static_fork_pick_up \
  --episode-index 0 \
  --frame-offset 0 \
  --num-frames 120 \
  --output-dir runs/check_aloha_fork_actions
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEROBOT_REPO_ID = "lerobot/aloha_static_fork_pick_up"
LEROBOT_EPISODE_INDEX = 0
LEROBOT_FRAME_OFFSET = 0
NUM_PREVIEW_FRAMES = 120
OUTPUT_DIR = Path("runs/check_aloha_fork_actions")


@dataclass(frozen=True)
class ActionPreview:
    """Store the extracted action window and optional aligned metadata."""

    actions: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray
    states: np.ndarray | None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ALOHA action preview export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=LEROBOT_REPO_ID)
    parser.add_argument("--episode-index", type=int, default=LEROBOT_EPISODE_INDEX)
    parser.add_argument("--frame-offset", type=int, default=LEROBOT_FRAME_OFFSET)
    parser.add_argument("--num-frames", type=int, default=NUM_PREVIEW_FRAMES)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    return parser.parse_args()


def _load_action_preview(
    *,
    repo_id: str,
    episode_index: int,
    frame_offset: int,
    num_frames: int,
    video_backend: str,
) -> ActionPreview:
    """Load a contiguous action/state window from a LeRobot episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}.")
    if frame_offset < 0:
        raise ValueError(f"frame_offset must be >= 0, got {frame_offset}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

    dataset = LeRobotDataset(
        repo_id,
        episodes=[episode_index],
        video_backend=video_backend,
    )
    end_index = frame_offset + num_frames
    if end_index > len(dataset):
        raise ValueError(
            f"Requested frames [{frame_offset}, {end_index - 1}] exceed episode length {len(dataset)}."
        )

    actions: list[np.ndarray] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    states: list[np.ndarray] = []
    saw_state = True
    for idx in range(frame_offset, end_index):
        sample = dataset[idx]
        action = sample.get("action")
        if action is None:
            raise KeyError(f"Sample at index {idx} is missing the 'action' field.")
        actions.append(action.detach().cpu().numpy().astype(np.float32, copy=False))
        frame_indices.append(int(sample.get("frame_index", idx)))
        timestamps.append(float(sample.get("timestamp", idx)))

        state = sample.get("observation.state")
        if state is None:
            saw_state = False
        elif saw_state:
            states.append(state.detach().cpu().numpy().astype(np.float32, copy=False))

    states_array = np.stack(states, axis=0) if saw_state and states else None
    return ActionPreview(
        actions=np.stack(actions, axis=0),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        states=states_array,
    )


def _write_actions_csv(*, preview: ActionPreview, output_path: Path) -> Path:
    """Write the extracted action window to a CSV file for spreadsheet inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_dim = int(preview.actions.shape[1])
    state_dim = 0 if preview.states is None else int(preview.states.shape[1])
    header = ["step", "frame_index", "timestamp"] + [f"action_{idx}" for idx in range(action_dim)]
    if state_dim > 0:
        header += [f"state_{idx}" for idx in range(state_dim)]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for step, (frame_index, timestamp, action_row) in enumerate(
            zip(preview.frame_indices, preview.timestamps, preview.actions, strict=True)
        ):
            row: list[float | int] = [step, int(frame_index), float(timestamp), *action_row.tolist()]
            if preview.states is not None:
                row.extend(preview.states[step].tolist())
            writer.writerow(row)
    return output_path


def _plot_actions(*, preview: ActionPreview, output_path: Path, show: bool) -> Path:
    """Save a per-dimension time-series plot for the action window."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_dim = int(preview.actions.shape[1])
    ncols = 2 if action_dim <= 8 else 3
    nrows = int(np.ceil(action_dim / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.8 * nrows), sharex=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    x_axis = preview.frame_indices

    for action_index in range(action_dim):
        row = action_index // ncols
        col = action_index % ncols
        axis = axes_array[row, col]
        axis.plot(x_axis, preview.actions[:, action_index], color="#c84c09", linewidth=1.8, label="action")
        if preview.states is not None and preview.states.shape[1] == action_dim:
            axis.plot(
                x_axis,
                preview.states[:, action_index],
                color="#1f77b4",
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label="state",
            )
        axis.set_title(f"dim {action_index}")
        axis.grid(alpha=0.3, linewidth=0.5)
        if row == nrows - 1:
            axis.set_xlabel("frame_index")
        axis.set_ylabel("value")

    for axis_index in range(action_dim, nrows * ncols):
        row = axis_index // ncols
        col = axis_index % ncols
        axes_array[row, col].axis("off")

    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right")
    figure.suptitle("ALOHA robot actions over time", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def _rounded_list(values: np.ndarray) -> list[float]:
    """Return values rounded for cleaner terminal summaries."""
    return [round(float(value), 4) for value in values]


def _print_summary(*, preview: ActionPreview) -> None:
    """Print compact numeric summaries for quick terminal inspection."""
    print(f"frames: {preview.actions.shape[0]}")
    print(f"frame range: {int(preview.frame_indices[0])} -> {int(preview.frame_indices[-1])}")
    print(f"action_dim: {preview.actions.shape[1]}")
    print("action min per dim:", _rounded_list(preview.actions.min(axis=0)))
    print("action max per dim:", _rounded_list(preview.actions.max(axis=0)))
    print("action mean per dim:", _rounded_list(preview.actions.mean(axis=0)))
    print("first action:", _rounded_list(preview.actions[0]))
    print("last action:", _rounded_list(preview.actions[-1]))
    if preview.states is not None:
        print(f"state_dim: {preview.states.shape[1]}")


def main() -> None:
    """Load an ALOHA episode window and save action CSV + plot artifacts."""
    args = _parse_args()
    preview = _load_action_preview(
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        frame_offset=args.frame_offset,
        num_frames=args.num_frames,
        video_backend=args.video_backend,
    )
    csv_path = _write_actions_csv(preview=preview, output_path=args.output_dir / "actions.csv")
    plot_path = _plot_actions(preview=preview, output_path=args.output_dir / "actions.png", show=args.show)
    _print_summary(preview=preview)
    print(f"Saved action CSV to: {csv_path}")
    print(f"Saved action plot to: {plot_path}")
    print(
        "Settings: "
        f"repo_id={args.repo_id}, episode_index={args.episode_index}, "
        f"frame_offset={args.frame_offset}, num_frames={args.num_frames}"
    )


if __name__ == "__main__":
    main()
