"""CLI for the shared-session Codex training optimizer.

source .venv/bin/activate
python scripts/train/training_optimizer.py --iterations 1
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
    DEFAULT_PROMPT_PATH,
    DEFAULT_STATE_PATH,
    derive_state_path_for_memory_path,
    render_controller_status,
    run_training_optimization_loop,
)
from world_model.optimization.codex_runner import DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS
from world_model.config import DEFAULT_TRAIN_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the shared-session training controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help="Base train YAML passed to Codex as the canonical training configuration.",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=DEFAULT_MEMORY_PATH,
        help="Mutable markdown memory file that Codex reads and updates.",
    )
    parser.add_argument(
        "--prompt-path",
        "--instructions-path",
        "--controller-prompt",
        "--controller_prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Static markdown prompt file that defines the controller workflow.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help=(
            "Optional machine-readable shared-session controller state JSON path. "
            "If omitted, the CLI derives one from --memory-path "
            f"(default memory keeps {DEFAULT_STATE_PATH})."
        ),
    )
    parser.add_argument(
        "--codex-model",
        type=str,
        default=None,
        help="Optional Codex model override passed through to `codex exec`.",
    )
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS,
        help="Timeout for each short in-session Codex turn.",
    )
    parser.add_argument(
        "--codex-session-id",
        type=str,
        default=None,
        help="Optional shared Codex session id to resume explicitly.",
    )
    parser.add_argument(
        "--codex-force-fresh-session",
        action="store_true",
        help="Ignore any persisted session id and start a fresh shared Codex session.",
    )
    parser.add_argument(
        "--codex-reuse-persisted-session",
        action="store_true",
        help="Reuse the last persisted shared Codex session from the state file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Maximum number of long external experiment commands to execute in this invocation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Let Codex decide the next long command without executing it.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the current controller status and exit.",
    )
    return parser


def main() -> int:
    """Parse CLI flags and run the shared-session controller."""
    args = build_parser().parse_args()
    state_path = args.state_path or derive_state_path_for_memory_path(args.memory_path)
    if args.status:
        print(render_controller_status(state_path))
        return 0
    run_training_optimization_loop(
        train_config_path=args.train_config,
        memory_path=args.memory_path,
        prompt_path=args.prompt_path,
        state_path=state_path,
        codex_model=args.codex_model,
        codex_timeout_seconds=args.codex_timeout_seconds,
        codex_session_id=args.codex_session_id,
        codex_force_fresh_session=args.codex_force_fresh_session,
        codex_reuse_persisted_session=args.codex_reuse_persisted_session,
        iterations=args.iterations,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
