"""Codex CLI helpers for structured local planning and inspection.

This module enforces ChatGPT-authenticated Codex usage and wraps `codex exec`
for structured JSON responses with optional image inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CODEX_BIN_CANDIDATES = (
    "/home/amaniscalco/.antigravity/extensions/openai.chatgpt-*/bin/*/codex",
)


@dataclass(frozen=True)
class CodexExecutionResult:
    """Capture one Codex CLI invocation and its structured payload."""

    command: tuple[str, ...]
    payload: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    stdout: str
    stderr: str


def resolve_codex_bin(codex_bin: str | Path | None = None) -> Path:
    """Resolve the local Codex CLI binary path."""
    if codex_bin is not None:
        candidate = Path(codex_bin)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        raise FileNotFoundError(f"Codex binary is not executable: {candidate}")

    detected = shutil_which("codex")
    if detected is not None:
        return Path(detected)

    for pattern in DEFAULT_CODEX_BIN_CANDIDATES:
        for raw_match in glob.glob(pattern):
            match = Path(raw_match)
            if match.exists() and os.access(match, os.X_OK):
                return match

    raise FileNotFoundError("Unable to locate the `codex` CLI binary.")


def ensure_codex_chatgpt_login(*, codex_bin: str | Path | None = None) -> Path:
    """Require a ChatGPT-authenticated Codex login before autonomous use."""
    resolved = resolve_codex_bin(codex_bin)
    completed = subprocess.run(
        [str(resolved), "login", "status"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Unable to verify Codex login status: {combined}")
    if "Logged in using ChatGPT" not in combined:
        raise RuntimeError(
            "Codex autonomous mode requires `codex login status` to report `Logged in using ChatGPT`."
        )
    return resolved


def run_codex_exec(
    *,
    prompt: str,
    schema: dict[str, Any],
    codex_bin: str | Path | None = None,
    model: str | None = None,
    images: list[Path] | None = None,
    cwd: str | Path = REPO_ROOT,
    extra_args: list[str] | None = None,
) -> CodexExecutionResult:
    """Run Codex non-interactively and parse the final structured JSON payload."""
    resolved = ensure_codex_chatgpt_login(codex_bin=codex_bin)
    output_dir = Path(cwd)
    temp_parent = REPO_ROOT / "runs" / "training_optimizer"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codex_runner_", dir=str(temp_parent)) as temp_dir:
        temp_root = Path(temp_dir)
        schema_path = temp_root / "schema.json"
        output_path = temp_root / "last_message.json"
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        command = [
            str(resolved),
            "exec",
            "--cd",
            str(output_dir),
            "--skip-git-repo-check",
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        if model is not None:
            command[2:2] = ["--model", model]
        if images:
            image_args: list[str] = []
            for image_path in images:
                image_args.extend(["-i", str(image_path)])
            command[2:2] = image_args
        if extra_args:
            command[2:2] = list(extra_args)

        completed = subprocess.run(
            command,
            cwd=output_dir,
            check=False,
            capture_output=True,
            text=True,
            input=prompt,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Codex exec failed with exit code "
                f"{completed.returncode}: stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )
        if not output_path.exists():
            raise FileNotFoundError("Codex exec did not write the structured output file.")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Codex structured output must be a JSON object.")
        events = tuple(_parse_codex_json_events(completed.stdout))
        return CodexExecutionResult(
            command=tuple(command),
            payload=payload,
            events=events,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _parse_codex_json_events(stdout: str) -> list[dict[str, Any]]:
    """Parse best-effort JSONL events emitted by `codex exec --json`."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def shutil_which(command: str) -> str | None:
    """Return the first executable on PATH for the given command name."""
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
