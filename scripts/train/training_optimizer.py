"""Run the staged training-optimization controller for the current repo workflow.

source .venv/bin/activate
python scripts/train/training_optimizer.py \
  --planner codex \
  --train-config configs/train/aloha_fork_pick_up.yaml \
  --memory-path docs/training_optimizer.md \
  --state-path runs/training_optimizer/controller_state.json \
  --iterations 1 \
  --max-real-runs 1 \
  --max-codex-calls 4 \
  --max-failed-runs 3 \
  --max-edit-cycles 1

"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_model.optimization.controller import (
    DEFAULT_MEMORY_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    run_training_optimization_loop,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the staged training-optimization controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner",
        choices=("codex", "deterministic"),
        default="codex",
        help="Planning mode. `codex` uses the local ChatGPT-authenticated Codex CLI; `deterministic` keeps the rule-based controller.",
    )
    parser.add_argument(
        "--codex-model",
        type=str,
        default=None,
        help="Optional Codex model override passed through to `codex exec --model`.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help="Base train YAML that defines the canonical runtime recipe.",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=DEFAULT_MEMORY_PATH,
        help="Markdown file used as persistent optimization memory.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Machine-readable controller state JSON path.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of controller iterations to execute in one invocation.",
    )
    parser.add_argument(
        "--max-real-runs",
        type=int,
        default=None,
        help="Optional hard cap on actual training/eval stages launched in one Codex loop.",
    )
    parser.add_argument(
        "--max-codex-calls",
        type=int,
        default=None,
        help="Optional hard cap on total `codex exec` calls in one Codex loop.",
    )
    parser.add_argument(
        "--max-failed-runs",
        type=int,
        default=3,
        help="Hard cap on failed experiment stages before the Codex loop stops.",
    )
    parser.add_argument(
        "--max-edit-cycles",
        type=int,
        default=4,
        help="Hard cap on automatic repo-edit cycles in one Codex loop.",
    )
    parser.add_argument(
        "--max-wall-clock-minutes",
        type=int,
        default=None,
        help="Optional wall-clock cap for one Codex loop invocation.",
    )
    parser.add_argument(
        "--stage-steps",
        type=int,
        default=None,
        help="Optional override for checkpoint spacing. Defaults to memory/config hints.",
    )
    parser.add_argument(
        "--eval-episode-index",
        type=int,
        default=0,
        help="Episode index used by checkpoint evaluation and reference preview export.",
    )
    parser.add_argument(
        "--eval-start-frame",
        type=int,
        default=60,
        help="Episode-local start frame used by checkpoint evaluation.",
    )
    parser.add_argument(
        "--reference-frame-offset",
        type=int,
        default=None,
        help="Optional reference-preview frame offset. Defaults to --eval-start-frame.",
    )
    parser.add_argument(
        "--reference-video",
        type=Path,
        default=None,
        help="Optional precomputed reference video for plausibility checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the next experiment without launching training or evaluation.",
    )
    return parser


def main() -> int:
    """Parse CLI flags and run the requested controller iterations."""
    args = build_parser().parse_args()
    run_training_optimization_loop(
        train_config_path=args.train_config,
        memory_path=args.memory_path,
        state_path=args.state_path,
        planner=args.planner,
        codex_model=args.codex_model,
        iterations=args.iterations,
        max_real_runs=args.max_real_runs,
        max_codex_calls=args.max_codex_calls,
        max_failed_runs=args.max_failed_runs,
        max_edit_cycles=args.max_edit_cycles,
        max_wall_clock_minutes=args.max_wall_clock_minutes,
        stage_step_override=args.stage_steps,
        eval_episode_index=args.eval_episode_index,
        eval_start_frame=args.eval_start_frame,
        reference_frame_offset=args.reference_frame_offset,
        reference_video=args.reference_video,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
