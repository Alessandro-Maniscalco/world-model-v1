"""Download ALOHA fork-pick-up episode 0 if needed, then launch smoke training."""

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
loaded_world_model = sys.modules.get("world_model")
if loaded_world_model is not None and not hasattr(loaded_world_model, "__path__"):
    sys.modules.pop("world_model", None)

from world_model.training.aloha_episode0 import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_EPISODE,
    DEFAULT_REPO_ID,
    run_aloha_episode0_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the episode-0 training helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo to fetch and train.")
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE, help="Episode index to fetch and train.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Training config passed through to scripts/train/world_model.py.",
    )
    return parser


def main() -> None:
    """Parse CLI args and run the episode-0 ALOHA workflow."""
    args = build_parser().parse_args()
    run_aloha_episode0_workflow(
        repo_id=args.repo_id,
        episode=args.episode,
        config_path=Path(args.config).resolve(),
    )


if __name__ == "__main__":
    main()
