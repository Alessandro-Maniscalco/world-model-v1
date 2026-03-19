"""Codex CLI helpers for structured local planning and inspection.

This module enforces ChatGPT-authenticated Codex usage and wraps `codex exec`
for structured JSON responses with optional image inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
CODEX_HOME = Path.home() / ".codex"
CODEX_SESSION_INDEX_PATH = CODEX_HOME / "session_index.jsonl"
CODEX_SESSION_ROOT = CODEX_HOME / "sessions"
CODEX_ARCHIVED_SESSION_ROOT = CODEX_HOME / "archived_sessions"
CODEX_DEBUG_ROOT = REPO_ROOT / "runs" / "training_optimizer" / "debug"
DEFAULT_CODEX_BIN_CANDIDATES = (
    "/home/amaniscalco/.antigravity/extensions/openai.chatgpt-*/bin/*/codex",
)
DEFAULT_CODEX_LOGIN_TIMEOUT_SECONDS = 15
DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS = 1000
DEFAULT_CODEX_SESSION_DISCOVERY_WINDOW_MINUTES = 10


@dataclass(frozen=True)
class CodexExecutionResult:
    """Capture one Codex CLI invocation and its structured payload."""

    command: tuple[str, ...]
    payload: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    stdout: str
    stderr: str
    session_id: str | None
    session_reused: bool
    session_reset_reason: str | None


@dataclass(frozen=True)
class CodexSessionMetadata:
    """Describe one locally persisted Codex CLI session."""

    session_id: str
    cwd: str | None
    started_at: str | None
    updated_at: str | None
    session_path: str


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


def ensure_codex_chatgpt_login(
    *,
    codex_bin: str | Path | None = None,
    timeout_seconds: int = DEFAULT_CODEX_LOGIN_TIMEOUT_SECONDS,
) -> Path:
    """Require a ChatGPT-authenticated Codex login before autonomous use."""
    resolved = resolve_codex_bin(codex_bin)
    try:
        completed = subprocess.run(
            [str(resolved), "login", "status"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out after {timeout_seconds}s while checking `codex login status`."
        ) from exc
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
    timeout_seconds: int = DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS,
    session_id: str | None = None,
    fallback_to_new_session: bool = True,
    debug_metadata: dict[str, Any] | None = None,
) -> CodexExecutionResult:
    """Run Codex non-interactively and parse the final structured JSON payload."""
    resolved = ensure_codex_chatgpt_login(codex_bin=codex_bin)
    output_dir = Path(cwd)
    attempted_resume = session_id is not None
    start_started_at = datetime.now(timezone.utc)
    try:
        return _run_codex_exec_once(
            prompt=prompt,
            schema=schema,
            resolved=resolved,
            model=model,
            images=images,
            cwd=output_dir,
            extra_args=extra_args,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            discovery_started_at=start_started_at,
            session_reset_reason=None,
            debug_metadata=debug_metadata,
        )
    except RuntimeError as exc:
        if not attempted_resume or not fallback_to_new_session:
            raise
        _log_codex_status(
            f"Resume failed for session {session_id}; starting a fresh session instead. Error: {exc}"
        )
        return _run_codex_exec_once(
            prompt=prompt,
            schema=schema,
            resolved=resolved,
            model=model,
            images=images,
            cwd=output_dir,
            extra_args=extra_args,
            timeout_seconds=timeout_seconds,
            session_id=None,
            discovery_started_at=datetime.now(timezone.utc),
            session_reset_reason=f"resume_failed:{session_id}",
            debug_metadata=debug_metadata,
        )


def _run_codex_exec_once(
    *,
    prompt: str,
    schema: dict[str, Any],
    resolved: Path,
    model: str | None,
    images: list[Path] | None,
    cwd: Path,
    extra_args: list[str] | None,
    timeout_seconds: int,
    session_id: str | None,
    discovery_started_at: datetime,
    session_reset_reason: str | None,
    debug_metadata: dict[str, Any] | None,
) -> CodexExecutionResult:
    """Run one fresh or resumed Codex command and return parsed metadata."""
    temp_parent = REPO_ROOT / "runs" / "training_optimizer"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codex_runner_", dir=str(temp_parent)) as temp_dir:
        temp_root = Path(temp_dir)
        schema_path = temp_root / "schema.json"
        output_path = temp_root / "last_message.json"
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        command = _build_codex_command(
            codex_bin=resolved,
            cwd=cwd,
            schema_path=schema_path,
            output_path=output_path,
            model=model,
            images=images,
            extra_args=extra_args,
            session_id=session_id,
        )
        start_time = time.monotonic()
        debug_dir = _prepare_codex_debug_dir(command_mode="resume" if session_id is not None else "new")
        command_mode = "resume" if session_id is not None else "new session"
        _log_codex_status(
            "Running Codex command"
            f" ({command_mode}, images={0 if images is None else len(images)}, "
            f"model={model or 'default'}, timeout={timeout_seconds}s): {shlex.join(command)}"
        )
        _log_codex_debug_paths(debug_dir)
        _write_codex_debug_artifacts(
            debug_dir=debug_dir,
            prompt=prompt,
            schema=schema,
            command=command,
            session_id=session_id,
            images=images or [],
            status="started",
            debug_metadata=debug_metadata,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                input=prompt,
                env=os.environ.copy(),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed_seconds = time.monotonic() - start_time
            _log_codex_status(
                f"Codex command timed out after {elapsed_seconds:.1f}s (limit={timeout_seconds}s)."
            )
            _write_codex_debug_artifacts(
                debug_dir=debug_dir,
                prompt=prompt,
                schema=schema,
                command=command,
                session_id=session_id,
                images=images or [],
                status="timeout",
                stdout="",
                stderr=str(exc),
                debug_metadata=debug_metadata,
            )
            raise RuntimeError(f"Codex exec timed out after {timeout_seconds}s.") from exc
        elapsed_seconds = time.monotonic() - start_time
        if completed.returncode != 0:
            _log_codex_status(
                f"Codex command failed after {elapsed_seconds:.1f}s with exit code {completed.returncode}."
            )
            _write_codex_debug_artifacts(
                debug_dir=debug_dir,
                prompt=prompt,
                schema=schema,
                command=command,
                session_id=session_id,
                images=images or [],
                status="failed",
                stdout=completed.stdout,
                stderr=completed.stderr,
                debug_metadata=debug_metadata,
            )
            raise RuntimeError(
                "Codex exec failed with exit code "
                f"{completed.returncode}: stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
            )
        if not output_path.exists():
            raise FileNotFoundError("Codex exec did not write the structured output file.")
        payload = _load_codex_payload(output_path)
        events = tuple(_parse_codex_json_events(completed.stdout))
        discovered_session_id = session_id or _discover_latest_codex_session_id(
            cwd=cwd,
            started_at=discovery_started_at,
        )
        final_reply_text = output_path.read_text(encoding="utf-8")
        _write_codex_debug_artifacts(
            debug_dir=debug_dir,
            prompt=prompt,
            schema=schema,
            command=command,
            session_id=discovered_session_id,
            images=images or [],
            status="completed",
            stdout=completed.stdout,
            stderr=completed.stderr,
            final_reply=final_reply_text,
            payload=payload,
            debug_metadata=debug_metadata,
        )
        _log_codex_status(
            f"Codex command completed in {elapsed_seconds:.1f}s with {len(events)} JSON event(s); "
            f"session={discovered_session_id or 'unknown'}."
        )
        return CodexExecutionResult(
            command=tuple(command),
            payload=payload,
            events=events,
            stdout=completed.stdout,
            stderr=completed.stderr,
            session_id=discovered_session_id,
            session_reused=session_id is not None,
            session_reset_reason=session_reset_reason,
        )


def _build_codex_command(
    *,
    codex_bin: Path,
    cwd: Path,
    schema_path: Path,
    output_path: Path,
    model: str | None,
    images: list[Path] | None,
    extra_args: list[str] | None,
    session_id: str | None,
) -> list[str]:
    """Build the non-interactive Codex CLI command for a fresh or resumed turn."""
    if session_id is None:
        command = [
            str(codex_bin),
            "exec",
            "--cd",
            str(cwd),
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        option_insert_index = 2
    else:
        command = [
            str(codex_bin),
            "exec",
            "resume",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(output_path),
            session_id,
            "-",
        ]
        option_insert_index = 3
    if model is not None:
        command[option_insert_index:option_insert_index] = ["--model", model]
        option_insert_index += 2
    if images:
        image_args: list[str] = []
        for image_path in images:
            image_args.extend(["-i", str(image_path)])
        command[option_insert_index:option_insert_index] = image_args
        option_insert_index += len(image_args)
    if extra_args:
        command[option_insert_index:option_insert_index] = list(extra_args)
    return command


def _load_codex_payload(output_path: Path) -> dict[str, Any]:
    """Load the final Codex message and coerce it into a JSON object payload."""
    raw_text = output_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("Codex structured output file was empty.")
    payload = _parse_json_object_text(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Codex structured output must be a JSON object.")
    return payload


def _parse_json_object_text(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain text or a fenced JSON block."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        candidate_text = fenced_match.group(1) if fenced_match is not None else text
        start = candidate_text.find("{")
        end = candidate_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Codex last message did not contain a JSON object.") from None
        payload = json.loads(candidate_text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Codex structured output must be a JSON object.")
    return payload


def load_codex_session_metadata(session_id: str) -> CodexSessionMetadata | None:
    """Load persisted metadata for one Codex session id from the local CLI store."""
    session_path = _find_session_path(session_id)
    if session_path is None or not session_path.exists():
        return None
    with session_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return None
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if payload.get("type") != "session_meta" or not isinstance(payload.get("payload"), dict):
        return None
    meta = payload["payload"]
    return CodexSessionMetadata(
        session_id=str(meta.get("id", session_id)),
        cwd=str(meta.get("cwd")) if meta.get("cwd") is not None else None,
        started_at=str(meta.get("timestamp")) if meta.get("timestamp") is not None else None,
        updated_at=_lookup_session_updated_at(session_id),
        session_path=str(session_path),
    )


def _discover_latest_codex_session_id(*, cwd: Path, started_at: datetime) -> str | None:
    """Find the newest Codex session id for the given workspace after a command starts."""
    earliest = started_at - timedelta(minutes=DEFAULT_CODEX_SESSION_DISCOVERY_WINDOW_MINUTES)
    best_match: tuple[datetime, str] | None = None
    for session_id, updated_at in _iter_recent_session_index_entries():
        updated_dt = _parse_iso_datetime(updated_at)
        if updated_dt is None or updated_dt < earliest:
            continue
        metadata = load_codex_session_metadata(session_id)
        if metadata is None or metadata.cwd != str(cwd):
            continue
        started_dt = _parse_iso_datetime(metadata.started_at)
        if started_dt is not None and started_dt < earliest:
            continue
        if best_match is None or updated_dt > best_match[0]:
            best_match = (updated_dt, session_id)
    if best_match is None:
        for metadata, candidate_dt in _iter_recent_session_file_metadata(cwd=cwd, earliest=earliest):
            if best_match is None or candidate_dt > best_match[0]:
                best_match = (candidate_dt, metadata.session_id)
    return None if best_match is None else best_match[1]


def _iter_recent_session_index_entries() -> list[tuple[str, str]]:
    """Read recent session ids from the local Codex session index in newest-first order."""
    if not CODEX_SESSION_INDEX_PATH.exists():
        return []
    entries: list[tuple[str, str]] = []
    for line in CODEX_SESSION_INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = payload.get("id")
        updated_at = payload.get("updated_at")
        if isinstance(session_id, str) and isinstance(updated_at, str):
            entries.append((session_id, updated_at))
    entries.reverse()
    return entries


def _find_session_path(session_id: str) -> Path | None:
    """Resolve the local session transcript path for one Codex session id."""
    for root in (CODEX_SESSION_ROOT, CODEX_ARCHIVED_SESSION_ROOT):
        if not root.exists():
            continue
        matches = sorted(root.glob(f"**/*{session_id}.jsonl"))
        if matches:
            return matches[-1]
    return None


def _lookup_session_updated_at(session_id: str) -> str | None:
    """Look up the last-updated timestamp for one session id from the index."""
    for indexed_session_id, updated_at in _iter_recent_session_index_entries():
        if indexed_session_id == session_id:
            return updated_at
    return None


def _iter_recent_session_file_metadata(
    *,
    cwd: Path,
    earliest: datetime,
) -> list[tuple[CodexSessionMetadata, datetime]]:
    """Scan recent session transcripts directly when the session index is stale."""
    matches: list[tuple[CodexSessionMetadata, datetime]] = []
    for root in (CODEX_SESSION_ROOT, CODEX_ARCHIVED_SESSION_ROOT):
        if not root.exists():
            continue
        for session_path in root.glob("**/*.jsonl"):
            modified_at = datetime.fromtimestamp(session_path.stat().st_mtime, tz=timezone.utc)
            if modified_at < earliest:
                continue
            session_id = _session_id_from_path(session_path)
            if session_id is None:
                continue
            metadata = load_codex_session_metadata(session_id)
            if metadata is None or metadata.cwd != str(cwd):
                continue
            candidate_dt = _parse_iso_datetime(metadata.updated_at) or _parse_iso_datetime(metadata.started_at) or modified_at
            if candidate_dt < earliest:
                continue
            matches.append((metadata, candidate_dt))
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def _session_id_from_path(path: Path) -> str | None:
    """Extract the Codex session id suffix from a local transcript filename."""
    match = re.search(r"([0-9a-f]{8,}-[0-9a-f-]+)\.jsonl$", path.name)
    if match is None:
        return None
    return match.group(1)


def _parse_iso_datetime(raw_value: str | None) -> datetime | None:
    """Parse an ISO timestamp used by Codex session metadata."""
    if raw_value is None or not raw_value:
        return None
    normalized = raw_value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _prepare_codex_debug_dir(*, command_mode: str) -> Path:
    """Create one per-call debug directory for Codex prompt/reply inspection."""
    CODEX_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    debug_dir = CODEX_DEBUG_ROOT / f"{timestamp}_{command_mode}_{_short_hash(str(time.time_ns()))}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _write_codex_debug_artifacts(
    *,
    debug_dir: Path,
    prompt: str,
    schema: dict[str, Any],
    command: list[str],
    session_id: str | None,
    images: list[Path],
    status: str,
    stdout: str = "",
    stderr: str = "",
    final_reply: str = "",
    payload: dict[str, Any] | None = None,
    debug_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist prompt/reply debug artifacts for one Codex CLI invocation."""
    (debug_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (debug_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if final_reply:
        (debug_dir / "final_reply.txt").write_text(final_reply, encoding="utf-8")
    if payload is not None:
        (debug_dir / "final_payload.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if stdout:
        (debug_dir / "stdout.jsonl").write_text(stdout, encoding="utf-8")
    if stderr:
        (debug_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "session_id": session_id,
        "command": command,
        "images": [str(path) for path in images],
        "prompt_chars": len(prompt),
    }
    if debug_metadata:
        metadata.update(debug_metadata)
    (debug_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log_codex_debug_paths(debug_dir: Path) -> None:
    """Print the main debug artifact paths for one Codex CLI invocation."""
    _log_codex_status(f"prompt: {debug_dir / 'prompt.txt'}")
    _log_codex_status(f"final reply: {debug_dir / 'final_reply.txt'}")
    _log_codex_status(f"parsed payload: {debug_dir / 'final_payload.json'}")


def _short_hash(value: str) -> str:
    """Return a short stable hash fragment for debug-directory naming."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


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


def _log_codex_status(message: str) -> None:
    """Emit a flushed status line for local Codex CLI activity."""
    print(f"[codex-runner] {message}", flush=True)
