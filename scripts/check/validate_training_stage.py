"""Validate one training stage before continuing a staged resume workflow.

This script checks the final metrics row and expected checkpoint artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_model.training.validation import format_stage_summary, validate_training_stage


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for staged training validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Stage output directory to validate.")
    parser.add_argument("--expected-step", required=True, type=int, help="Expected final step for this stage.")
    parser.add_argument(
        "--allow-greater-step",
        action="store_true",
        help="Accept a last metrics step greater than --expected-step instead of requiring an exact match.",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=None,
        help="Optional upper bound for the final logged loss.",
    )
    return parser


def main() -> int:
    """Run validation and exit nonzero when the stage should not continue."""
    args = build_parser().parse_args()
    summary = validate_training_stage(
        args.output_dir,
        expected_step=args.expected_step,
        require_exact_step=not args.allow_greater_step,
        max_loss=args.max_loss,
    )
    print(format_stage_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
