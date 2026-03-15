"""Tests for the pasted queued local sweep helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

def _load_script_module():
    """Load the queue-runs script module from disk."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "check" / "queue_runs.py"
    spec = importlib.util.spec_from_file_location("queue_runs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def test_normalized_runs_strip_blank_entries() -> None:
    """Keep only non-empty pasted commands after dedenting them."""
    runs = script._normalized_runs(
        [
            """
            python first.py \
              --flag value
            """,
            """

            """,
            """
            python second.py
            """,
        ]
    )

    assert runs == [
        "python first.py               --flag value",
        "python second.py",
    ]


def test_run_queue_continues_after_failure(monkeypatch) -> None:
    """Run later commands even when an earlier queued command fails."""
    seen: list[str] = []

    class _Completed:
        """Minimal subprocess result stub for queue-run tests."""

        def __init__(self, returncode: int) -> None:
            """Store the desired subprocess exit code."""
            self.returncode = returncode

    def _fake_run(
        command: str,
        cwd: Path,
        executable: str,
        shell: bool,
        check: bool,
    ) -> _Completed:
        """Record commands and return a failing first run followed by success."""
        assert cwd == script.REPO_ROOT
        assert executable == "/bin/bash"
        assert shell is True
        assert check is False
        seen.append(command)
        return _Completed(returncode=1 if len(seen) == 1 else 0)

    monkeypatch.setattr(script.subprocess, "run", _fake_run)

    returncodes = script._run_queue(
        [
            "python first.py",
            "python second.py",
        ]
    )

    assert seen == [
        "source .venv/bin/activate ; python first.py",
        "source .venv/bin/activate ; python second.py",
    ]
    assert returncodes == [1, 0]
