"""Shared-session Codex controller for long experiment execution.

This module keeps one persistent Codex session, lets Codex do short work
in-session, and only executes long experiment shell chains outside the chat.
"""

from __future__ import annotations

from datetime import datetime, timezone
import filecmp
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

from world_model.config import DEFAULT_TRAIN_CONFIG_PATH
from world_model.optimization import paths as optimization_paths
from world_model.optimization.codex_runner import (
    ensure_codex_chatgpt_login,
    run_codex_exec,
)


REPO_ROOT = optimization_paths.REPO_ROOT
DEFAULT_MEMORY_PATH = optimization_paths.default_memory_path(REPO_ROOT)
DEFAULT_PROMPT_PATH = optimization_paths.default_prompt_path(REPO_ROOT)
DEFAULT_STATE_PATH = optimization_paths.default_state_path(REPO_ROOT)
DEFAULT_CONTROLLER_CODEX_TIMEOUT_SECONDS = 25 * 60
DEFAULT_LOG_TAIL_CHARS = 4000
SNAPSHOT_EXCLUDED_PATH_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "runs",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".coverage",
    }
)


def derive_state_path_for_memory_path(memory_path: str | Path) -> Path:
    """Map a memory markdown path to its default controller-state JSON path."""
    return optimization_paths.derive_state_path_for_memory_path(
        memory_path,
        repo_root=REPO_ROOT,
    )


def run_training_optimization_loop(
    *,
    train_config_path: str | Path = DEFAULT_TRAIN_CONFIG_PATH,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    state_path: str | Path = DEFAULT_STATE_PATH,
    codex_model: str | None = None,
    codex_timeout_seconds: int = DEFAULT_CONTROLLER_CODEX_TIMEOUT_SECONDS,
    codex_session_id: str | None = None,
    codex_force_fresh_session: bool = False,
    codex_reuse_persisted_session: bool = False,
    iterations: int = 1,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run the shared-session Codex controller for up to `iterations` long commands."""
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if codex_timeout_seconds <= 0:
        raise ValueError(
            f"codex_timeout_seconds must be > 0, got {codex_timeout_seconds}"
        )

    train_config_path = Path(train_config_path)
    memory_path = _resolve_repo_relative_path(memory_path)
    prompt_path = _resolve_repo_relative_path(prompt_path)
    state_path = _resolve_repo_relative_path(state_path)
    if not _resolve_repo_relative_path(train_config_path).exists():
        raise FileNotFoundError(
            f"train_config_path does not exist: {_display_path(train_config_path)}"
        )

    _log_controller_status("Checking Codex ChatGPT login status.")
    ensure_codex_chatgpt_login()
    _log_controller_status(f"state: {_display_path(state_path)}")
    state = load_controller_state(state_path)
    state = _initialize_controller_state(
        state=state,
        codex_session_id=codex_session_id,
        codex_force_fresh_session=codex_force_fresh_session,
        codex_reuse_persisted_session=codex_reuse_persisted_session,
    )
    state["status"] = "running"
    state["phase"] = "in_session"
    save_controller_state(state_path, state)

    records: list[dict[str, Any]] = []
    allow_long_command = True
    external_runs_used = 0
    prompt_note = ""
    last_result: dict[str, Any] | None = None

    while True:
        decision, state = _run_shared_session_turn(
            train_config_path=train_config_path,
            memory_path=memory_path,
            prompt_path=prompt_path,
            state=state,
            state_path=state_path,
            codex_model=codex_model,
            codex_timeout_seconds=codex_timeout_seconds,
            allow_long_command=allow_long_command,
            phase_note=prompt_note,
            last_long_command_result=last_result,
        )

        if decision["action_type"] == "run_long_command" and isinstance(
            state.get("last_long_command_result"), dict
        ) and state["last_long_command_result"]:
            _append_current_invocation_run_summary(state, decision["summary"])
            save_controller_state(state_path, state)

        if decision["action_type"] == "stop":
            state["status"] = "stopped"
            state["phase"] = "idle"
            state["last_stop_reason"] = decision["stop"]["reason"] or decision["summary"]
            stop_summary_path = _write_stop_summary_report(
                state_path=Path(state_path),
                decision=decision,
                state=state,
            )
            state["last_stop_summary_path"] = str(stop_summary_path)
            save_controller_state(state_path, state)
            _log_stop_summary_path(stop_summary_path)
            break

        if dry_run:
            state["status"] = "stopped"
            state["phase"] = "idle"
            state["last_stop_reason"] = "dry_run requested before external command execution"
            save_controller_state(state_path, state)
            records.append(
                {
                    "dry_run": True,
                    "decision": decision,
                    "resume_command": _format_resume_command(state.get("session_id")),
                }
            )
            break

        if external_runs_used >= iterations:
            state["status"] = "stopped"
            state["phase"] = "idle"
            state["last_stop_reason"] = f"max_iterations={iterations} exhausted"
            save_controller_state(state_path, state)
            break

        command_result = _run_long_command(
            command=decision["long_command"]["command"],
            expected_artifacts=decision["long_command"]["expected_artifacts"],
        )
        external_runs_used += 1
        last_result = command_result
        state["phase"] = "post_run_validation"
        state["active_long_command"] = None
        state["last_long_command_result"] = command_result
        state["latest_artifacts"] = list(command_result["artifacts"])
        state.setdefault("history", []).append(
            {
                "entry_type": "long_command",
                "command": decision["long_command"]["command"],
                "reason": decision["long_command"]["reason"],
                "returncode": command_result["returncode"],
                "timestamp": command_result["completed_at"],
            }
        )
        save_controller_state(state_path, state)
        records.append(
            {
                "decision": decision,
                "long_command_result": command_result,
            }
        )
        prompt_note = ""

        if external_runs_used >= iterations:
            allow_long_command = False
            prompt_note = (
                f"No more long commands may run in this invocation because "
                f"max_iterations={iterations} has been reached. "
                "Finish validation, update memory, and return stop."
            )
        else:
            allow_long_command = True

    return records


def render_controller_status(state_path: str | Path = DEFAULT_STATE_PATH) -> str:
    """Render a human-readable status summary for the shared-session controller."""
    state = load_controller_state(state_path)
    lines = [
        f"status: {state['status']}",
        f"phase: {state['phase']}",
        f"session_id: {state['session_id'] or '(none)'}",
        f"resume_command: {state['resume_command'] or '(none)'}",
    ]
    if state["status"] != "running":
        lines.extend(
            [
                f"last_stop_reason: {state['last_stop_reason'] or '(none)'}",
                f"last_stop_summary_path: {state.get('last_stop_summary_path') or '(none)'}",
            ]
        )
    active_long_command = state.get("active_long_command")
    if isinstance(active_long_command, dict) and active_long_command:
        lines.append(f"active_long_command: {active_long_command.get('command', '')}")
    last_long_command_result = state.get("last_long_command_result")
    if isinstance(last_long_command_result, dict) and last_long_command_result:
        lines.append(
            "last_long_command_result: "
            f"returncode={last_long_command_result.get('returncode')} "
            f"completed_at={last_long_command_result.get('completed_at', '')}"
        )
    artifact_lines = [
        str(item.get("path", ""))
        for item in state.get("latest_artifacts", [])
        if isinstance(item, dict) and item.get("path")
    ]
    if artifact_lines:
        lines.append("latest_artifacts:")
        lines.extend(f"- {path}" for path in artifact_lines[:5])
    return "\n".join(lines)


def load_controller_state(path: str | Path) -> dict[str, Any]:
    """Load the controller state JSON, returning defaults when it is missing."""
    state_path = Path(path)
    if not state_path.exists():
        return _normalize_controller_state({})
    return _normalize_controller_state(
        json.loads(state_path.read_text(encoding="utf-8"))
    )


def save_controller_state(path: str | Path, state: dict[str, Any]) -> None:
    """Persist controller state JSON to disk."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(_normalize_controller_state(state), indent=2) + "\n",
        encoding="utf-8",
    )


def _initialize_controller_state(
    *,
    state: dict[str, Any],
    codex_session_id: str | None,
    codex_force_fresh_session: bool,
    codex_reuse_persisted_session: bool,
) -> dict[str, Any]:
    """Apply session overrides for one controller invocation."""
    if _should_reset_for_fresh_controller_start(
        codex_session_id=codex_session_id,
        codex_force_fresh_session=codex_force_fresh_session,
        codex_reuse_persisted_session=codex_reuse_persisted_session,
    ):
        updated = _normalize_controller_state({})
    else:
        updated = _normalize_controller_state(state)
    should_reset_invocation = codex_force_fresh_session or not _should_reuse_persisted_session(
        updated
    )
    if codex_force_fresh_session:
        updated["session_id"] = None
    elif should_reset_invocation:
        updated["session_id"] = None
    if should_reset_invocation:
        updated["current_invocation_run_summaries"] = []
    if codex_session_id is not None:
        updated["session_id"] = codex_session_id
    if _should_reuse_persisted_session(updated):
        updated = _recover_deleted_latest_result_state(updated)
    return updated


def _should_reset_for_fresh_controller_start(
    *,
    codex_session_id: str | None,
    codex_force_fresh_session: bool,
    codex_reuse_persisted_session: bool,
) -> bool:
    """Return whether this invocation should discard persisted controller state."""
    if isinstance(codex_session_id, str) and codex_session_id.strip():
        return False
    if codex_force_fresh_session:
        return True
    return not codex_reuse_persisted_session


def _run_shared_session_turn(
    *,
    train_config_path: Path,
    memory_path: Path,
    prompt_path: Path,
    state: dict[str, Any],
    state_path: Path,
    codex_model: str | None,
    codex_timeout_seconds: int,
    allow_long_command: bool,
    phase_note: str,
    last_long_command_result: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one in-session Codex turn with snapshot-based rollback protection."""
    prompt = _build_turn_prompt(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state=state,
        state_path=state_path,
        allow_long_command=allow_long_command,
        phase_note=phase_note,
        last_long_command_result=last_long_command_result,
        fresh_session=_should_send_full_prompt(state),
    )
    rollback_notice = ""
    session_id = state.get("session_id")

    for attempt in range(2):
        _log_controller_status("Snapshotting editable repo tree before Codex turn.")
        snapshot_dir, snapshot_paths = _snapshot_editable_tree()
        try:
            result = run_codex_exec(
                prompt=rollback_notice + prompt,
                schema=_turn_response_schema(),
                model=codex_model,
                cwd=REPO_ROOT,
                timeout_seconds=codex_timeout_seconds,
                session_id=session_id,
            )
            decision = _validate_turn_payload(result.payload, allow_long_command=allow_long_command)
            changed_files = _diff_editable_tree_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_paths=snapshot_paths,
            )
        except Exception:
            _restore_editable_tree_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_paths=snapshot_paths,
            )
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise

        session_id = result.session_id
        rollback_requested = decision["repo_edit_status"] == "rollback_requested"
        unvalidated_changes = bool(changed_files) and decision["repo_edit_status"] != "validated"

        if rollback_requested or unvalidated_changes:
            _restore_editable_tree_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_paths=snapshot_paths,
            )
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            reason = (
                "Codex requested rollback."
                if rollback_requested
                else "Codex edited repo files without returning `repo_edit_status=validated`."
            )
            rollback_notice = (
                f"The controller restored the editable-tree snapshot for this turn. {reason} "
                "The workspace is now back to the pre-turn state. Continue in the same session, "
                "redo only the short in-session work you still want to keep, and return one final JSON object only.\n\n"
            )
            continue

        shutil.rmtree(snapshot_dir, ignore_errors=True)
        updated = _normalize_controller_state(state)
        updated["status"] = "running"
        updated["phase"] = (
            "awaiting_external_run"
            if decision["action_type"] == "run_long_command"
            else "idle"
        )
        updated["session_id"] = result.session_id
        updated["active_long_command"] = (
            {
                "command": decision["long_command"]["command"],
                "reason": decision["long_command"]["reason"],
                "expected_artifacts": decision["long_command"]["expected_artifacts"],
                "requested_at": _utc_timestamp(),
            }
            if decision["action_type"] == "run_long_command"
            else None
        )
        updated["resume_command"] = _format_resume_command(updated.get("session_id"))
        updated.setdefault("history", []).append(
            {
                "entry_type": "session_turn",
                "action_type": decision["action_type"],
                "repo_edit_status": decision["repo_edit_status"],
                "summary": decision["summary"],
                "changed_files": changed_files,
                "timestamp": _utc_timestamp(),
            }
        )
        _log_resume_command(updated["resume_command"])
        save_controller_state(state_path, updated)
        return decision, updated

    raise RuntimeError(
        "Codex could not complete a validated in-session turn after rollback."
    )


def _build_turn_prompt(
    *,
    train_config_path: Path,
    memory_path: Path,
    prompt_path: Path,
    state: dict[str, Any],
    state_path: Path,
    allow_long_command: bool,
    phase_note: str,
    last_long_command_result: dict[str, Any] | None,
    fresh_session: bool,
) -> str:
    """Build the shared-session Codex prompt for one controller turn."""
    prompt_payload = {
        "train_config_path": _display_path(train_config_path),
        "memory_path": _display_path(memory_path),
        "prompt_path": _display_path(prompt_path),
        "state_path": _display_path(state_path),
        "allow_long_command": allow_long_command,
        "controller_status": str(state.get("status", "")),
        "controller_phase": str(state.get("phase", "")),
        "latest_result_available": isinstance(last_long_command_result, dict) and bool(last_long_command_result),
    }
    action_instruction = (
        "You may return `run_long_command` or `stop`."
        if allow_long_command
        else "You must return `stop` in this turn after finishing validation and memory updates."
    )
    note_block = "" if not phase_note else f"\n\nController note:\n{phase_note}"
    context_json = json.dumps(prompt_payload, indent=2, sort_keys=True)
    if fresh_session:
        return _build_fresh_session_prompt(
            train_config_path=train_config_path,
            memory_path=memory_path,
            prompt_path=prompt_path,
            state_path=state_path,
            action_instruction=action_instruction,
            note_block=note_block,
            context_json=context_json,
        )
    return _build_continuation_prompt(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
        action_instruction=action_instruction,
        note_block=note_block,
        context_json=context_json,
    )


def _build_fresh_session_prompt(
    *,
    train_config_path: Path,
    memory_path: Path,
    prompt_path: Path,
    state_path: Path,
    action_instruction: str,
    note_block: str,
    context_json: str,
) -> str:
    """Build the single fresh-session prompt template."""
    prompt_path_text = _display_path(prompt_path)
    memory_path_text = _display_path(memory_path)
    state_path_text = _display_path(state_path)
    train_config_path_text = _display_path(train_config_path)
    return f"""This is a shared session. Read {prompt_path_text}. The ##Goal is in {memory_path_text}.
Latest controller state: {state_path_text}.
Read {train_config_path_text} for the base training configuration.
{action_instruction}
Your final output should be a JSON object.{note_block}

Context JSON:
{context_json}""".strip()


def _build_continuation_prompt(
    *,
    train_config_path: Path,
    memory_path: Path,
    prompt_path: Path,
    state_path: Path,
    action_instruction: str,
    note_block: str,
    context_json: str,
) -> str:
    """Build the single continuation prompt template."""
    prompt_path_text = _display_path(prompt_path)
    memory_path_text = _display_path(memory_path)
    state_path_text = _display_path(state_path)
    train_config_path_text = _display_path(train_config_path)
    return f"""Continue the same shared Codex session. Make sure you are working 
    towards the goal in {memory_path_text}.

Context JSON:
{context_json}""".strip()


def _turn_response_schema() -> dict[str, Any]:
    """Return the JSON schema for the final Codex turn response."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action_type",
            "summary",
            "session_work_summary",
            "repo_edit_status",
            "long_command",
            "stop",
        ],
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["run_long_command", "stop"],
            },
            "summary": {"type": "string"},
            "session_work_summary": {
                "type": "array",
                "items": {"type": "string"},
            },
            "repo_edit_status": {
                "type": "string",
                "enum": ["none", "validated", "rollback_requested"],
            },
            "long_command": {
                "type": "object",
                "additionalProperties": False,
                "required": ["command", "reason", "expected_artifacts"],
                "properties": {
                    "command": {"type": "string"},
                    "reason": {"type": "string"},
                    "expected_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "stop": {
                "type": "object",
                "additionalProperties": False,
                "required": ["reason"],
                "properties": {
                    "reason": {"type": "string"},
                },
            },
        },
    }


def _validate_turn_payload(
    payload: dict[str, Any],
    *,
    allow_long_command: bool,
) -> dict[str, Any]:
    """Validate and normalize the final Codex turn payload."""
    if not isinstance(payload, dict):
        raise ValueError("Codex turn payload must be a JSON object.")
    action_type = str(payload.get("action_type", "")).strip()
    if action_type not in {"run_long_command", "stop"}:
        raise ValueError(f"Unsupported action_type: {action_type!r}")
    if action_type == "run_long_command" and not allow_long_command:
        raise ValueError("This turn does not allow another long command.")
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise ValueError("Codex turn response requires a non-empty `summary`.")
    session_work_summary = payload.get("session_work_summary", [])
    if not isinstance(session_work_summary, list) or not all(
        isinstance(item, str) for item in session_work_summary
    ):
        raise ValueError("`session_work_summary` must be a list of strings.")
    repo_edit_status = str(payload.get("repo_edit_status", "")).strip()
    if repo_edit_status not in {"none", "validated", "rollback_requested"}:
        raise ValueError(f"Unsupported repo_edit_status: {repo_edit_status!r}")

    long_command_section = payload.get("long_command", {})
    if not isinstance(long_command_section, dict):
        raise ValueError("`long_command` must be an object.")
    stop_section = payload.get("stop", {})
    if not isinstance(stop_section, dict):
        raise ValueError("`stop` must be an object.")

    normalized = {
        "action_type": action_type,
        "summary": summary,
        "session_work_summary": [item.strip() for item in session_work_summary if item.strip()],
        "repo_edit_status": repo_edit_status,
        "long_command": {
            "command": str(long_command_section.get("command", "")).strip(),
            "reason": str(long_command_section.get("reason", "")).strip(),
            "expected_artifacts": [
                str(item).strip()
                for item in long_command_section.get("expected_artifacts", [])
                if str(item).strip()
            ],
        },
        "stop": {
            "reason": str(stop_section.get("reason", "")).strip(),
        },
    }
    if action_type == "run_long_command":
        if not normalized["long_command"]["command"]:
            raise ValueError("`run_long_command` requires a non-empty command.")
        if not normalized["long_command"]["reason"]:
            raise ValueError("`run_long_command` requires a non-empty reason.")
    if action_type == "stop" and not normalized["stop"]["reason"]:
        normalized["stop"]["reason"] = summary
    return normalized


def _run_long_command(
    *,
    command: str,
    expected_artifacts: list[str],
) -> dict[str, Any]:
    """Execute one long shell command and mirror its logs to files and the terminal."""
    logs_dir = optimization_paths.controller_logs_root(REPO_ROOT)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_timestamp().replace(":", "-")
    stdout_log = logs_dir / f"{timestamp}_stdout.log"
    stderr_log = logs_dir / f"{timestamp}_stderr.log"
    _log_controller_status(f"Running long command: {command}")
    _log_controller_status(f"stdout log: {_display_path(stdout_log)}")
    _log_controller_status(f"stderr log: {_display_path(stderr_log)}")
    process_env = os.environ.copy()
    process_env["PYTHONUNBUFFERED"] = "1"
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=process_env,
        )
        stdout_thread = threading.Thread(
            target=_stream_process_output,
            args=(process.stdout, stdout_handle, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_process_output,
            args=(process.stderr, stderr_handle, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    artifacts = _resolve_expected_artifacts(
        expected_artifacts=expected_artifacts,
        extra_paths=[stdout_log, stderr_log],
    )
    return {
        "command": command,
        "returncode": returncode,
        "completed_at": _utc_timestamp(),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stdout_tail": _read_tail(stdout_log),
        "stderr_tail": _read_tail(stderr_log),
        "artifacts": artifacts,
    }


def _stream_process_output(
    stream: Any,
    log_handle: Any,
    terminal_handle: Any,
) -> None:
    """Mirror a subprocess text stream to both its log file and the live terminal."""
    if stream is None:
        return
    try:
        for chunk in iter(stream.readline, ""):
            if not chunk:
                break
            log_handle.write(chunk)
            log_handle.flush()
            terminal_handle.write(chunk)
            terminal_handle.flush()
    finally:
        stream.close()


def _resolve_expected_artifacts(
    *,
    expected_artifacts: list[str],
    extra_paths: list[Path],
) -> list[dict[str, Any]]:
    """Resolve expected artifact paths and record whether they exist yet."""
    artifacts: list[dict[str, Any]] = []
    for raw_path in expected_artifacts:
        path = _resolve_repo_relative_path(raw_path)
        artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "kind": "expected",
            }
        )
    for path in extra_paths:
        artifacts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "kind": "log",
            }
        )
    return artifacts


def _should_send_full_prompt(state: dict[str, Any]) -> bool:
    """Return whether the next turn should resend the full static prompt text."""
    session_id = state.get("session_id")
    return not isinstance(session_id, str) or not session_id.strip()


def _should_reuse_persisted_session(state: dict[str, Any]) -> bool:
    """Reuse a persisted session only while the controller is mid-loop."""
    session_id = state.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return False
    status = str(state.get("status", "")).strip().lower()
    phase = str(state.get("phase", "")).strip().lower()
    return status == "running" and phase in {
        "in_session",
        "awaiting_external_run",
        "post_run_validation",
    }


def _recover_deleted_latest_result_state(state: dict[str, Any]) -> dict[str, Any]:
    """Drop stale validation state when a successful latest result was deleted."""
    updated = _normalize_controller_state(state)
    if str(updated.get("phase", "")).strip().lower() != "post_run_validation":
        return updated
    if not _latest_successful_result_has_missing_expected_artifacts(updated):
        return updated

    updated["phase"] = "in_session"
    updated["active_long_command"] = None
    updated["last_long_command_result"] = None
    updated["latest_artifacts"] = []
    updated.setdefault("history", []).append(
        {
            "entry_type": "state_recovery",
            "reason": "deleted_latest_result_artifacts",
            "timestamp": _utc_timestamp(),
        }
    )
    return updated


def _latest_successful_result_has_missing_expected_artifacts(state: dict[str, Any]) -> bool:
    """Return whether a successful latest result now points at deleted outputs."""
    result = state.get("last_long_command_result")
    if not isinstance(result, dict) or not result:
        return False
    if int(result.get("returncode", 1)) != 0:
        return False
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        return False

    expected_artifact_paths = [
        Path(str(item.get("path", "")).strip())
        for item in artifacts
        if isinstance(item, dict)
        and str(item.get("kind", "")).strip() == "expected"
        and str(item.get("path", "")).strip()
    ]
    if not expected_artifact_paths:
        return False
    return any(not path.exists() for path in expected_artifact_paths)


def _normalize_controller_state(state: dict[str, Any]) -> dict[str, Any]:
    """Backfill the minimal shared-session controller state shape."""
    source = dict(state)
    return {
        "status": str(source.get("status", "idle")),
        "phase": str(source.get("phase", "idle")),
        "session_id": source.get("session_id"),
        "resume_command": str(
            source.get("resume_command", _format_resume_command(source.get("session_id")))
        ),
        "last_stop_reason": str(source.get("last_stop_reason", "")),
        "last_stop_summary_path": str(source.get("last_stop_summary_path", "")),
        "active_long_command": source.get("active_long_command"),
        "last_long_command_result": source.get("last_long_command_result"),
        "latest_artifacts": source.get("latest_artifacts", []),
        "current_invocation_run_summaries": source.get("current_invocation_run_summaries", []),
        "history": source.get("history", []),
    }


def _append_current_invocation_run_summary(
    state: dict[str, Any],
    summary: str,
) -> None:
    """Store one canonical per-run summary for the current invocation."""
    text = str(summary).strip()
    if not text:
        return
    summaries = state.setdefault("current_invocation_run_summaries", [])
    if isinstance(summaries, list):
        summaries.append(text)


def _format_resume_command(session_id: Any) -> str:
    """Format the `codex resume` command for the active session id."""
    if not isinstance(session_id, str) or not session_id.strip():
        return ""
    return f"codex resume {session_id.strip()}"


def _snapshot_editable_tree() -> tuple[Path, list[str]]:
    """Snapshot the editable repo tree to a temporary directory."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="controller_snapshot_"))
    snapshot_paths: list[str] = []
    for relative_path in _list_editable_files(REPO_ROOT):
        source_path = REPO_ROOT / relative_path
        destination_path = snapshot_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        snapshot_paths.append(relative_path)
    return snapshot_dir, snapshot_paths


def _restore_editable_tree_snapshot(
    *,
    snapshot_dir: Path,
    snapshot_paths: list[str],
) -> None:
    """Restore the editable repo tree from a snapshot directory."""
    snapshot_set = set(snapshot_paths)
    for relative_path in _list_editable_files(REPO_ROOT):
        if relative_path in snapshot_set:
            continue
        target_path = REPO_ROOT / relative_path
        if target_path.exists():
            target_path.unlink()
    for relative_path in snapshot_paths:
        source_path = snapshot_dir / relative_path
        target_path = REPO_ROOT / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _diff_editable_tree_snapshot(
    *,
    snapshot_dir: Path,
    snapshot_paths: list[str],
) -> list[str]:
    """Return repo-relative editable files that differ from the snapshot."""
    changed: set[str] = set()
    snapshot_set = set(snapshot_paths)
    current_paths = set(_list_editable_files(REPO_ROOT))

    for relative_path in snapshot_set ^ current_paths:
        changed.add(relative_path)
    for relative_path in snapshot_set & current_paths:
        current_path = REPO_ROOT / relative_path
        source_path = snapshot_dir / relative_path
        if not filecmp.cmp(source_path, current_path, shallow=False):
            changed.add(relative_path)
    return sorted(changed)


def _list_editable_files(root: Path) -> list[str]:
    """List editable repo files relative to `root`."""
    files: list[str] = []
    for current_root, dirnames, filenames in os.walk(root):
        current_root_path = Path(current_root)
        relative_root = current_root_path.relative_to(root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _should_skip_snapshot_path(relative_root / name)
        ]
        for filename in filenames:
            relative_path = relative_root / filename
            if _should_skip_snapshot_path(relative_path):
                continue
            files.append(str(relative_path))
    return sorted(files)


def _should_skip_snapshot_path(relative_path: Path) -> bool:
    """Return whether a repo-relative path should be excluded from snapshots."""
    return any(part in SNAPSHOT_EXCLUDED_PATH_PARTS for part in relative_path.parts)


def _read_tail(path: Path, *, max_chars: int = DEFAULT_LOG_TAIL_CHARS) -> str:
    """Read the last `max_chars` characters from a text file."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _resolve_repo_relative_path(raw_path: str | Path) -> Path:
    """Resolve a possibly repo-relative path inside the current repository."""
    return optimization_paths.resolve_repo_relative_path(raw_path, repo_root=REPO_ROOT)


def _display_path(path: Path) -> str:
    """Render an absolute path relative to the repo when possible."""
    return optimization_paths.display_repo_path(path, repo_root=REPO_ROOT)


def _utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log_controller_status(message: str) -> None:
    """Print one controller status line."""
    print(f"[controller] {message}", flush=True)


def _log_resume_command(resume_command: str) -> None:
    """Print the current `codex resume` command when available."""
    if not resume_command:
        return
    _log_controller_status(f"Resume shared session with: {resume_command}")


def _log_stop_summary_path(summary_path: Path) -> None:
    """Print only the markdown stop-summary path when the controller stops."""
    _log_controller_status(f"stop summary: {_display_path(summary_path)}")


def _write_stop_summary_report(
    *,
    state_path: Path,
    decision: dict[str, Any],
    state: dict[str, Any],
) -> Path:
    """Write one concise markdown stop report."""
    report_dir = state_path.parent / "stop_summaries"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{_utc_timestamp().replace(':', '-')}_stop_summary.md"
    summary = str(decision.get("summary", "")).strip()
    reason = str(decision.get("stop", {}).get("reason", "")).strip() or summary
    prior_run_summaries = [
        str(item).strip()
        for item in state.get("current_invocation_run_summaries", [])
        if str(item).strip()
    ]
    stop_work_items = [
        str(item).strip()
        for item in decision.get("session_work_summary", [])
        if str(item).strip()
    ]
    work_items = list(prior_run_summaries)
    if summary:
        work_items.append(summary)
    if stop_work_items:
        work_items.append(stop_work_items[-1])
    lines = [
        "# Controller Stop Summary",
        "",
        f"- Summary: {summary or '(none)'}",
        f"- Reason: {reason or '(none)'}",
        f"- Session ID: {state.get('session_id') or '(none)'}",
        "",
        "## Session Work",
    ]
    lines.extend(f"- {item}" for item in work_items) if work_items else lines.append("- (none)")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
