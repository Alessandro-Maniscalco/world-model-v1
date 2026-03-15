"""Tests for the local Codex CLI wrapper used by the optimizer loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import world_model.optimization.codex_runner as codex_runner
from world_model.optimization.codex_runner import (
    ensure_codex_chatgpt_login,
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
    monkeypatch.setattr(codex_runner, "ensure_codex_chatgpt_login", lambda **_: codex_bin)

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
    assert "--output-schema" in result.command
    assert "--output-last-message" in result.command
    assert result.command[-1] == "-"
    assert os.path.basename(result.command[0]) == "codex"
