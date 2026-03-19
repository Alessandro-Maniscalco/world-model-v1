"""Tests for the local Codex CLI wrapper used by the optimizer loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import world_model.optimization.codex_runner as codex_runner
from world_model.optimization.codex_runner import (
    CodexSessionMetadata,
    _discover_latest_codex_session_id,
    ensure_codex_chatgpt_login,
    load_codex_session_metadata,
    run_codex_exec,
)


def test_ensure_codex_chatgpt_login_accepts_chatgpt_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Accept a local Codex binary only when login status reports ChatGPT auth."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)

    monkeypatch.setattr(
        codex_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Logged in using ChatGPT\n",
            stderr="",
        ),
    )

    assert ensure_codex_chatgpt_login(codex_bin=codex_bin) == codex_bin


def test_ensure_codex_chatgpt_login_rejects_non_chatgpt_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Fail closed when the local Codex login is missing or uses another auth mode."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)

    monkeypatch.setattr(
        codex_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Logged in using API key\n",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="Logged in using ChatGPT"):
        ensure_codex_chatgpt_login(codex_bin=codex_bin)


def test_run_codex_exec_parses_structured_output(monkeypatch, tmp_path: Path) -> None:
    """Capture Codex CLI command metadata, JSONL events, and the final payload."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(codex_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(codex_runner, "CODEX_DEBUG_ROOT", tmp_path / "runs" / "training_optimizer" / "debug")
    monkeypatch.setattr(codex_runner, "ensure_codex_chatgpt_login", lambda **_: codex_bin)
    monkeypatch.setattr(codex_runner, "_discover_latest_codex_session_id", lambda **_: "session-new")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        output_flag_index = command.index("--output-last-message") + 1
        output_path = Path(command[output_flag_index])
        output_path.write_text(json.dumps({"action_type": "stop", "stop": {"reason": "done"}}), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout='{"event":"start"}\nnot-json\n{"event":"done"}\n',
            stderr="",
        )

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)

    result = run_codex_exec(
        prompt="Inspect the latest run.",
        schema={"type": "object"},
        codex_bin=codex_bin,
        model="gpt-5",
        images=[image_path],
        cwd=tmp_path,
    )

    assert result.payload == {"action_type": "stop", "stop": {"reason": "done"}}
    assert result.events == ({"event": "start"}, {"event": "done"})
    assert "--model" in result.command
    assert "gpt-5" in result.command
    assert "-i" in result.command
    assert str(image_path) in result.command
    assert "--dangerously-bypass-approvals-and-sandbox" in result.command
    assert "--output-schema" in result.command
    assert "--output-last-message" in result.command
    assert result.command[-1] == "-"
    assert os.path.basename(result.command[0]) == "codex"
    assert result.session_id == "session-new"
    assert result.session_reused is False
    debug_dirs = sorted((tmp_path / "runs" / "training_optimizer" / "debug").iterdir())
    assert len(debug_dirs) == 1
    assert (debug_dirs[0] / "prompt.txt").read_text(encoding="utf-8") == "Inspect the latest run."
    assert json.loads((debug_dirs[0] / "final_payload.json").read_text(encoding="utf-8")) == {
        "action_type": "stop",
        "stop": {"reason": "done"},
    }
    assert (debug_dirs[0] / "final_reply.txt").read_text(encoding="utf-8").strip() == json.dumps(
        {"action_type": "stop", "stop": {"reason": "done"}}
    )
    metadata = json.loads((debug_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_chars"] == len("Inspect the latest run.")


def test_run_codex_exec_logs_debug_artifact_paths(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the main debug artifact paths before each Codex invocation starts."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    monkeypatch.setattr(codex_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(codex_runner, "CODEX_DEBUG_ROOT", tmp_path / "runs" / "training_optimizer" / "debug")
    monkeypatch.setattr(codex_runner, "ensure_codex_chatgpt_login", lambda **_: codex_bin)
    monkeypatch.setattr(codex_runner, "_discover_latest_codex_session_id", lambda **_: "session-new")

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        output_flag_index = command.index("--output-last-message") + 1
        output_path = Path(command[output_flag_index])
        output_path.write_text('{"action_type":"stop","stop":{"reason":"done"}}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"event":"done"}\n', stderr="")

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)

    run_codex_exec(
        prompt="Inspect the latest run.",
        schema={"type": "object"},
        codex_bin=codex_bin,
        cwd=tmp_path,
    )

    stdout = capsys.readouterr().out
    assert "[codex-runner] prompt:" in stdout
    assert "prompt.txt" in stdout
    assert "final_reply.txt" in stdout
    assert "final_payload.json" in stdout


def test_run_codex_exec_raises_on_timeout(monkeypatch, tmp_path: Path) -> None:
    """Fail cleanly when a Codex planning call exceeds the configured timeout."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    monkeypatch.setattr(codex_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(codex_runner, "ensure_codex_chatgpt_login", lambda **_: codex_bin)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        raise codex_runner.subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 7s"):
        run_codex_exec(
            prompt="Inspect the latest run.",
            schema={"type": "object"},
            codex_bin=codex_bin,
            cwd=tmp_path,
            timeout_seconds=7,
        )


def test_run_codex_exec_resumes_existing_session(monkeypatch, tmp_path: Path) -> None:
    """Use `codex exec resume` when a persisted session id is provided."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    monkeypatch.setattr(codex_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(codex_runner, "ensure_codex_chatgpt_login", lambda **_: codex_bin)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        output_flag_index = command.index("--output-last-message") + 1
        output_path = Path(command[output_flag_index])
        output_path.write_text('{"action_type":"stop","stop":{"reason":"done"}}', encoding="utf-8")
        assert command[:3] == [str(codex_bin), "exec", "resume"]
        assert "--dangerously-bypass-approvals-and-sandbox" in command
        assert "session-123" in command
        assert "--output-schema" not in command
        return SimpleNamespace(returncode=0, stdout='{"event":"done"}\n', stderr="")

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)

    result = run_codex_exec(
        prompt="Continue the same task.",
        schema={"type": "object"},
        codex_bin=codex_bin,
        cwd=tmp_path,
        session_id="session-123",
    )

    assert result.session_id == "session-123"
    assert result.session_reused is True


def test_run_codex_exec_falls_back_to_new_session_after_resume_failure(monkeypatch, tmp_path: Path) -> None:
    """Recover from a bad persisted session by starting a fresh Codex session."""
    codex_bin = tmp_path / "codex"
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    monkeypatch.setattr(codex_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(codex_runner, "ensure_codex_chatgpt_login", lambda **_: codex_bin)
    monkeypatch.setattr(codex_runner, "_discover_latest_codex_session_id", lambda **_: "session-fresh")

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if command[:3] == [str(codex_bin), "exec", "resume"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing session")
        output_flag_index = command.index("--output-last-message") + 1
        output_path = Path(command[output_flag_index])
        output_path.write_text('{"action_type":"stop","stop":{"reason":"done"}}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout='{"event":"done"}\n', stderr="")

    monkeypatch.setattr(codex_runner.subprocess, "run", fake_run)

    result = run_codex_exec(
        prompt="Continue the same task.",
        schema={"type": "object"},
        codex_bin=codex_bin,
        cwd=tmp_path,
        session_id="session-bad",
    )

    assert calls[0][:3] == [str(codex_bin), "exec", "resume"]
    assert calls[1][:2] == [str(codex_bin), "exec"]
    assert result.session_id == "session-fresh"
    assert result.session_reused is False
    assert result.session_reset_reason == "resume_failed:session-bad"


def test_load_codex_session_metadata_reads_local_session_store(monkeypatch, tmp_path: Path) -> None:
    """Load persisted session metadata from the local Codex session transcript."""
    session_file = tmp_path / "sessions" / "2026" / "03" / "15" / "rollout-test-session-123.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "session-123",
                    "cwd": "/tmp/demo",
                    "timestamp": "2026-03-15T22:00:00Z",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "session_index.jsonl"
    index_path.write_text(
        json.dumps({"id": "session-123", "updated_at": "2026-03-15T22:05:00Z"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_runner, "CODEX_SESSION_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(codex_runner, "CODEX_ARCHIVED_SESSION_ROOT", tmp_path / "archived")
    monkeypatch.setattr(codex_runner, "CODEX_SESSION_INDEX_PATH", index_path)

    metadata = load_codex_session_metadata("session-123")

    assert metadata == CodexSessionMetadata(
        session_id="session-123",
        cwd="/tmp/demo",
        started_at="2026-03-15T22:00:00Z",
        updated_at="2026-03-15T22:05:00Z",
        session_path=str(session_file),
    )


def test_discover_latest_codex_session_id_falls_back_to_session_files(monkeypatch, tmp_path: Path) -> None:
    """Recover the newest matching session from transcript files when the index is stale."""
    session_root = tmp_path / "sessions" / "2026" / "03" / "15"
    session_root.mkdir(parents=True, exist_ok=True)
    session_id = "019cf3aa-83d5-7901-873b-e5390878c801"
    session_file = session_root / f"rollout-2026-03-15T22-42-44-{session_id}.jsonl"
    session_file.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": session_id,
                    "cwd": str(tmp_path / "repo"),
                    "timestamp": "2026-03-15T22:42:44Z",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_runner, "CODEX_SESSION_ROOT", tmp_path / "sessions")
    monkeypatch.setattr(codex_runner, "CODEX_ARCHIVED_SESSION_ROOT", tmp_path / "archived")
    monkeypatch.setattr(codex_runner, "CODEX_SESSION_INDEX_PATH", tmp_path / "missing-index.jsonl")

    discovered = _discover_latest_codex_session_id(
        cwd=tmp_path / "repo",
        started_at=codex_runner.datetime(2026, 3, 15, 22, 43, 0, tzinfo=codex_runner.timezone.utc),
    )

    assert discovered == session_id
