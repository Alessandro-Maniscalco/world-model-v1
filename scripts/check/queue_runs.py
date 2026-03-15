"""Run a pasted list of shell commands sequentially.

Edit `_runs` below and each non-empty command will execute in order.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]

_runs = [
    """
    python scripts/check/sweep_local_repo_resolutions.py \
  --mode checkpoint \
  --checkpoint runs/test_full_multi_320x240_lora8_none/checkpoints/step_0000400.pt \
  --resolutions 320x240
    """,
    """
    python scripts/check/sweep_local_repo_resolutions.py \
  --mode checkpoint \
  --checkpoint runs/test_full_multi_320x240_lora8_none/checkpoints/step_0000800.pt \
  --resolutions 320x240
    """,
    """
    
    """
]

def _normalized_runs(raw_runs: list[str]) -> list[str]:
    """Strip indentation and drop blank pasted commands."""
    normalized: list[str] = []
    for run in raw_runs:
        command = textwrap.dedent(run).strip()
        if command:
            normalized.append(command)
    return normalized


RUNS = _normalized_runs(_runs)


def _run_queue(runs: list[str]) -> list[int]:
    """Run each queued shell command in order and collect return codes."""
    returncodes: list[int] = []
    for command in runs:
        print(f"Running:\n{command}\n")
        completed = subprocess.run(
            f"source .venv/bin/activate ; {command}",
            cwd=REPO_ROOT,
            executable="/bin/bash",
            shell=True,
            check=False,
        )
        returncodes.append(int(completed.returncode))
    return returncodes


def main() -> int:
    """Execute the pasted queue and report which runs failed."""
    returncodes = _run_queue(RUNS)
    success_count = sum(1 for code in returncodes if code == 0)
    print(f"Completed {success_count}/{len(returncodes)} queued runs successfully.")
    for index, (command, returncode) in enumerate(zip(RUNS, returncodes, strict=True), start=1):
        status = "ok" if returncode == 0 else "error"
        print(f"Run {index}: {status}")
        print(command)
    return 0 if all(code == 0 for code in returncodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
