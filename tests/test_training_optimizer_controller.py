"""Tests for the shared-session Codex training controller."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import world_model.optimization.controller as controller_module


def test_legacy_training_optimizer_memory_alias_selects_fixed_anchor_workflow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Route the old README command onto the fixed-anchor memory, prompt, and state."""
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")
    legacy_memory_path = Path("docs/training_optimizer.md")
    default_prompt_path = Path("docs/controller_prompt.md")
    default_state_path = Path("runs/training_optimizer/controller_state.json")

    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "training_optimizer.md").write_text("# Legacy Memory\n", encoding="utf-8")
    (tmp_path / "docs" / "fixed_anchor_investigation.md").write_text(
        "# Fixed Memory\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "controller_prompt.md").write_text(
        "# Legacy Prompt\n\n## Validation\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "controller_prompt_fixed_anchor.md").write_text(
        "# Fixed Prompt\n\n## Validation\n",
        encoding="utf-8",
    )

    prompts: list[str] = []
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, prompt: str, **_: object) -> SimpleNamespace:
        prompts.append(prompt)
        return SimpleNamespace(
            payload=_stop_payload("done"),
            session_id="session-fixed-anchor",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=legacy_memory_path,
        prompt_path=default_prompt_path,
        state_path=default_state_path,
    )

    fixed_state_path = tmp_path / "runs" / "training_optimizer" / "fixed_anchor_controller_state.json"
    state = controller_module.load_controller_state(fixed_state_path)

    assert "First read and adopt docs/controller_prompt_fixed_anchor.md before doing any work." in prompts[0]
    assert "Use docs/fixed_anchor_investigation.md as the mutable optimization memory." in prompts[0]
    assert fixed_state_path.exists() is True
    assert state["last_stop_reason"] == "done"


def test_stopped_state_starts_fresh_session_on_new_invocation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Start a fresh session after a prior invocation already stopped cleanly."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    session_calls: list[str | None] = []

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, session_id: str | None = None, **_: object) -> SimpleNamespace:
        session_calls.append(session_id)
        return SimpleNamespace(
            payload=_stop_payload("done"),
            session_id="session-123",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )
    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )

    state = controller_module.load_controller_state(state_path)
    status_text = controller_module.render_controller_status(state_path)

    assert session_calls == [None, None]
    assert state["session_id"] == "session-123"
    assert state["resume_command"] == "codex resume session-123"
    assert "resume_command: codex resume session-123" in status_text


def test_controller_logs_state_path_at_run_start(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the controller state path before the first Codex turn starts."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)
    monkeypatch.setattr(
        controller_module,
        "run_codex_exec",
        lambda **_: SimpleNamespace(
            payload=_stop_payload("done"),
            session_id="session-123",
        ),
    )

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )

    stdout = capsys.readouterr().out
    assert "[controller] state: runs/training_optimizer/controller_state.json" in stdout


def test_controller_status_logging_flushes_immediately(monkeypatch) -> None:
    """Flush controller status prints so piped sessions show startup progress."""
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_print(*args: object, **kwargs: object) -> None:
        captured.append((args, dict(kwargs)))

    monkeypatch.setattr("builtins.print", fake_print)

    controller_module._log_controller_status("hello")

    assert captured == [(("[controller] hello",), {"flush": True})]


def test_controller_defaults_codex_timeout_to_twenty_five_minutes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Use a controller-local 25 minute timeout for in-session Codex turns."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    timeouts: list[int] = []

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, timeout_seconds: int, **_: object) -> SimpleNamespace:
        timeouts.append(timeout_seconds)
        return SimpleNamespace(
            payload=_stop_payload("done"),
            session_id="session-timeout",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )

    assert timeouts == [controller_module.DEFAULT_CONTROLLER_CODEX_TIMEOUT_SECONDS]
    assert controller_module.DEFAULT_CONTROLLER_CODEX_TIMEOUT_SECONDS == 1500


def test_missing_train_config_path_fails_fast(tmp_path: Path) -> None:
    """Raise a clear error before starting the controller when the config is missing."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="train_config_path does not exist"):
        controller_module.run_training_optimization_loop(
            train_config_path=tmp_path / "missing-train.yaml",
            memory_path=memory_path,
            prompt_path=prompt_path,
            state_path=state_path,
        )


def test_shared_session_loop_runs_one_full_external_command_cycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Execute one long command, then reopen the same session for validation."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")
    expected_mp4 = tmp_path / "runs" / "demo" / "comparison.mp4"
    expected_mp4.parent.mkdir(parents=True, exist_ok=True)

    prompts: list[str] = []
    session_calls: list[str | None] = []
    decisions = iter(
        [
            _run_long_command_payload(
                command=(
                    "mkdir -p runs/demo && "
                    "printf 'frame' > runs/demo/comparison.mp4 && "
                    "printf 'ok' >&2"
                ),
                expected_artifacts=["runs/demo/comparison.mp4"],
            ),
            _stop_payload("validation complete"),
        ]
    )

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, prompt: str, session_id: str | None = None, **_: object) -> SimpleNamespace:
        prompts.append(prompt)
        session_calls.append(session_id)
        return SimpleNamespace(
            payload=next(decisions),
            session_id="session-full-loop",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    records = controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
        iterations=1,
    )
    state = controller_module.load_controller_state(state_path)

    assert len(records) == 1
    assert session_calls == [None, "session-full-loop"]
    assert records[0]["decision"]["action_type"] == "run_long_command"
    assert records[0]["long_command_result"]["returncode"] == 0
    assert expected_mp4.exists() is True
    assert state["last_stop_reason"] == "validation complete"
    assert any(item["path"] == str(expected_mp4) and item["exists"] for item in state["latest_artifacts"])
    assert "First read and adopt docs/controller_prompt.md before doing any work." in prompts[0]
    assert "Read runs/training_optimizer/controller_state.json for the latest controller history" in prompts[0]
    assert "Read train.yaml for the base training configuration." in prompts[0]
    assert "Optimize for the best next action under long-run experiment cost" in prompts[0]
    assert "Ground validation summaries and next-action reasons in concrete video observations" in prompts[0]
    assert (
        "At the start of a fresh session, after reading the controller state and optimization memory, "
        "you may delete only clearly dominated checkpoints"
    ) in prompts[0]
    assert '"state_path": "runs/training_optimizer/controller_state.json"' in prompts[0]
    assert '"train_config_path": "train.yaml"' in prompts[0]
    assert '"state_summary"' not in prompts[0]
    assert '"latest_artifacts"' not in prompts[0]
    assert "Check docs/controller_prompt.md before deciding" in prompts[1]
    assert "Read runs/training_optimizer/controller_state.json for the latest controller history" in prompts[1]
    assert "- ### Motion-First Ranking" in prompts[1]
    assert "- ## Decision Rule" in prompts[1]
    assert "Choose the best next action for the overall long-run budget" in prompts[1]
    assert "Describe the reviewed videos concretely in your summary and reasoning" in prompts[1]
    assert "you may delete only clearly dominated checkpoints" not in prompts[1]
    assert '"state_path": "runs/training_optimizer/controller_state.json"' in prompts[1]
    assert '"latest_result_available": true' in prompts[1]
    assert '"state_summary"' not in prompts[1]


def test_stop_summary_markdown_is_written_after_validation_stop(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write a markdown stop summary and print only its path when the run ends."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    decisions = iter(
        [
            _run_long_command_payload(
                command="mkdir -p runs/demo && printf 'frame' > runs/demo/comparison.mp4",
                expected_artifacts=["runs/demo/comparison.mp4"],
            ),
            _stop_payload("validation complete"),
        ]
    )

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)
    monkeypatch.setattr(
        controller_module,
        "run_codex_exec",
        lambda **_: SimpleNamespace(
            payload=next(decisions),
            session_id="session-stop-summary",
        ),
    )

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
        iterations=1,
    )

    stdout = capsys.readouterr().out
    state = controller_module.load_controller_state(state_path)
    summary_path = Path(state["last_stop_summary_path"])
    summary_text = summary_path.read_text(encoding="utf-8")

    assert f"[controller] stop summary: {controller_module._display_path(summary_path)}" in stdout
    assert "[controller] summary:" not in stdout
    assert summary_path.exists() is True
    assert "# Controller Stop Summary" in summary_text
    assert "- Summary: validation complete" in summary_text
    assert "- Reason: validation complete" in summary_text
    assert "## Session Work" in summary_text
    assert "- completed short in-session work" in summary_text


def test_stop_summary_uses_current_invocation_run_summaries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Write one paragraph per completed run plus one final takeaway."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    decisions = iter(
        [
            _run_long_command_payload(
                command="mkdir -p runs/demo && printf 'one' > runs/demo/run1.mp4",
                expected_artifacts=["runs/demo/run1.mp4"],
                summary="plan-only summary for the first run",
            ),
            _run_long_command_payload(
                command="mkdir -p runs/demo && printf 'two' > runs/demo/run2.mp4",
                expected_artifacts=["runs/demo/run2.mp4"],
                summary="validated run one paragraph",
            ),
            _stop_payload(
                "validated run two paragraph",
                session_work_summary=[
                    "supporting stop detail",
                    "overall takeaway paragraph",
                ],
            ),
        ]
    )

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)
    monkeypatch.setattr(
        controller_module,
        "run_codex_exec",
        lambda **_: SimpleNamespace(
            payload=next(decisions),
            session_id="session-stop-summary",
        ),
    )

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
        iterations=2,
    )

    state = controller_module.load_controller_state(state_path)
    summary_path = Path(state["last_stop_summary_path"])
    summary_text = summary_path.read_text(encoding="utf-8")

    assert "validated run one paragraph" in summary_text
    assert "validated run two paragraph" in summary_text
    assert "overall takeaway paragraph" in summary_text
    assert "plan-only summary for the first run" not in summary_text


def test_initialize_controller_state_resets_or_preserves_run_summaries() -> None:
    """Reset summaries for fresh invocations and preserve them mid-loop."""
    reset_state = controller_module._initialize_controller_state(
        state={
            "status": "stopped",
            "phase": "idle",
            "current_invocation_run_summaries": ["stale summary"],
        },
        codex_session_id=None,
        codex_force_fresh_session=False,
    )
    preserved_state = controller_module._initialize_controller_state(
        state={
            "status": "running",
            "phase": "post_run_validation",
            "session_id": "session-mid-loop",
            "current_invocation_run_summaries": ["keep summary"],
        },
        codex_session_id=None,
        codex_force_fresh_session=False,
    )

    assert reset_state["current_invocation_run_summaries"] == []
    assert preserved_state["current_invocation_run_summaries"] == ["keep summary"]


def test_initialize_controller_state_drops_deleted_successful_latest_result(
    tmp_path: Path,
) -> None:
    """Reopen in-session work when the last successful result was deleted."""
    expected_artifact = tmp_path / "runs" / "demo" / "comparison.mp4"
    expected_artifact.parent.mkdir(parents=True, exist_ok=True)
    expected_artifact.write_text("frame", encoding="utf-8")

    recovered_state = controller_module._initialize_controller_state(
        state={
            "status": "running",
            "phase": "post_run_validation",
            "session_id": "session-mid-loop",
            "last_long_command_result": {
                "returncode": 0,
                "artifacts": [
                    {
                        "path": str(expected_artifact),
                        "exists": True,
                        "kind": "expected",
                    }
                ],
            },
            "latest_artifacts": [
                {
                    "path": str(expected_artifact),
                    "exists": True,
                    "kind": "expected",
                }
            ],
        },
        codex_session_id=None,
        codex_force_fresh_session=False,
    )
    assert recovered_state["phase"] == "post_run_validation"
    assert recovered_state["last_long_command_result"]["returncode"] == 0

    expected_artifact.unlink()

    recovered_state = controller_module._initialize_controller_state(
        state=recovered_state,
        codex_session_id=None,
        codex_force_fresh_session=False,
    )

    assert recovered_state["session_id"] == "session-mid-loop"
    assert recovered_state["phase"] == "in_session"
    assert recovered_state["active_long_command"] is None
    assert recovered_state["last_long_command_result"] is None
    assert recovered_state["latest_artifacts"] == []
    assert recovered_state["history"][-1]["entry_type"] == "state_recovery"
    assert recovered_state["history"][-1]["reason"] == "deleted_latest_result_artifacts"


def test_running_state_reuses_persisted_session_on_new_invocation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resume the persisted shared session when the controller is still mid-loop."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")
    controller_module.save_controller_state(
        state_path,
        {
            "session_id": "session-mid-loop",
            "status": "running",
            "phase": "post_run_validation",
            "last_stop_reason": "",
        },
    )

    session_calls: list[str | None] = []

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, session_id: str | None = None, **_: object) -> SimpleNamespace:
        session_calls.append(session_id)
        return SimpleNamespace(
            payload=_stop_payload("done"),
            session_id="session-mid-loop",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )

    assert session_calls == ["session-mid-loop"]


def test_render_controller_status_hides_last_stop_fields_while_running(
    tmp_path: Path,
) -> None:
    """Hide stale stop metadata while the controller is actively running."""
    state_path, _, _ = _make_controller_paths(tmp_path)
    controller_module.save_controller_state(
        state_path,
        {
            "status": "running",
            "phase": "in_session",
            "session_id": "session-mid-loop",
            "last_stop_reason": "old stop reason",
            "last_stop_summary_path": str(tmp_path / "runs" / "training_optimizer" / "stop.md"),
        },
    )

    status_text = controller_module.render_controller_status(state_path)

    assert "status: running" in status_text
    assert "last_stop_reason:" not in status_text
    assert "last_stop_summary_path:" not in status_text


def test_render_controller_status_shows_last_stop_fields_when_not_running(
    tmp_path: Path,
) -> None:
    """Keep last stop metadata visible once the controller is no longer running."""
    state_path, _, _ = _make_controller_paths(tmp_path)
    summary_path = tmp_path / "runs" / "training_optimizer" / "stop.md"
    controller_module.save_controller_state(
        state_path,
        {
            "status": "stopped",
            "phase": "idle",
            "last_stop_reason": "validation complete",
            "last_stop_summary_path": str(summary_path),
        },
    )

    status_text = controller_module.render_controller_status(state_path)

    assert "status: stopped" in status_text
    assert "last_stop_reason: validation complete" in status_text
    assert f"last_stop_summary_path: {summary_path}" in status_text


def test_stop_after_full_loop_is_honored_by_validation_turn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Stop after post-run validation when the shared session asks for it."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    decisions = iter(
        [
            _run_long_command_payload(
                command="mkdir -p runs/demo && printf 'frame' > runs/demo/out.mp4",
                expected_artifacts=["runs/demo/out.mp4"],
            ),
            _stop_payload("operator requested stop after full loop"),
        ]
    )

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)
    monkeypatch.setattr(
        controller_module,
        "run_codex_exec",
        lambda **_: SimpleNamespace(
            payload=next(decisions),
            session_id="session-stop",
        ),
    )

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
        iterations=1,
    )
    state = controller_module.load_controller_state(state_path)

    assert state["status"] == "stopped"
    assert state["last_stop_reason"] == "operator requested stop after full loop"


def test_external_command_failure_reopens_session_for_diagnosis(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resume the same session for diagnosis after a failing long command."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    prompts: list[str] = []
    session_calls: list[str | None] = []
    decisions = iter(
        [
            _run_long_command_payload(
                command="printf 'bad run' >&2 && exit 3",
                expected_artifacts=["runs/demo/missing.mp4"],
            ),
            _stop_payload("diagnosed failed run"),
        ]
    )

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, prompt: str, session_id: str | None = None, **_: object) -> SimpleNamespace:
        prompts.append(prompt)
        session_calls.append(session_id)
        return SimpleNamespace(
            payload=next(decisions),
            session_id="session-failure",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    records = controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
        iterations=1,
    )
    state = controller_module.load_controller_state(state_path)

    assert len(records) == 1
    assert records[0]["long_command_result"]["returncode"] == 3
    assert session_calls == [None, "session-failure"]
    assert '"state_path": "runs/training_optimizer/controller_state.json"' in prompts[1]
    assert '"latest_result_available": true' in prompts[1]
    assert '"last_long_command_result"' not in prompts[1]
    assert state["last_stop_reason"] == "diagnosed failed run"


def test_long_command_streams_output_to_terminal_and_logs(
    monkeypatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirror long-command stdout and stderr to both the terminal and log files."""
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)

    result = controller_module._run_long_command(
        command="printf 'step=10 loss=0.123\\n'; printf 'warning line\\n' >&2",
        expected_artifacts=[],
    )

    captured = capsys.readouterr()
    assert "step=10 loss=0.123" in captured.out
    assert "warning line" in captured.err
    assert "step=10 loss=0.123" in Path(result["stdout_log"]).read_text(encoding="utf-8")
    assert "warning line" in Path(result["stderr_log"]).read_text(encoding="utf-8")


def test_validated_in_session_edits_are_preserved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep direct shared-session edits when Codex marks them as validated."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")
    target_path = tmp_path / "src" / "example.py"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text('"""Fixture."""\n\nVALUE = "before"\n', encoding="utf-8")

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(**_: object) -> SimpleNamespace:
        target_path.write_text('"""Fixture."""\n\nVALUE = "after"\n', encoding="utf-8")
        return SimpleNamespace(
            payload=_stop_payload("validated edit", repo_edit_status="validated"),
            session_id="session-edit",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )
    state = controller_module.load_controller_state(state_path)

    assert 'VALUE = "after"' in target_path.read_text(encoding="utf-8")
    assert state["history"][-1]["changed_files"] == ["src/example.py"]


def test_rollback_requested_restores_preexisting_dirty_repo_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Restore the exact pre-turn file contents when the session requests rollback."""
    state_path, memory_path, prompt_path = _make_controller_paths(tmp_path)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")
    target_path = tmp_path / "src" / "example.py"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text('"""Fixture."""\n\nVALUE = "dirty-before-turn"\n', encoding="utf-8")

    prompts: list[str] = []
    call_count = {"value": 0}

    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: None)

    def fake_run_codex_exec(*, prompt: str, **_: object) -> SimpleNamespace:
        prompts.append(prompt)
        call_count["value"] += 1
        if call_count["value"] == 1:
            target_path.write_text('"""Fixture."""\n\nVALUE = "bad-edit"\n', encoding="utf-8")
            return SimpleNamespace(
                payload=_stop_payload("rollback this turn", repo_edit_status="rollback_requested"),
                session_id="session-rollback",
            )
        return SimpleNamespace(
            payload=_stop_payload("rollback complete"),
            session_id="session-rollback",
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        prompt_path=prompt_path,
        state_path=state_path,
    )

    assert call_count["value"] == 2
    assert 'VALUE = "dirty-before-turn"' in target_path.read_text(encoding="utf-8")
    assert "restored the editable-tree snapshot" in prompts[1]


def _make_controller_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the memory, prompt, and state paths for one controller test."""
    memory_path = tmp_path / "docs" / "complexity_ladder_training.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text("# Memory\n", encoding="utf-8")
    prompt_path = tmp_path / "docs" / "controller_prompt.md"
    prompt_path.write_text("# Prompt\n\n## Validation\n", encoding="utf-8")
    state_path = tmp_path / "runs" / "training_optimizer" / "controller_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    return state_path, memory_path, prompt_path


def _run_long_command_payload(
    *,
    command: str,
    expected_artifacts: list[str],
    summary: str = "run a long command",
) -> dict[str, object]:
    """Build a normalized `run_long_command` payload for controller tests."""
    return {
        "action_type": "run_long_command",
        "summary": summary,
        "session_work_summary": ["planned next experiment"],
        "repo_edit_status": "none",
        "long_command": {
            "command": command,
            "reason": "run training outside the session",
            "expected_artifacts": expected_artifacts,
        },
        "stop": {"reason": ""},
    }


def _stop_payload(
    reason: str,
    *,
    repo_edit_status: str = "none",
    session_work_summary: list[str] | None = None,
) -> dict[str, object]:
    """Build a normalized `stop` payload for controller tests."""
    return {
        "action_type": "stop",
        "summary": reason,
        "session_work_summary": session_work_summary or ["completed short in-session work"],
        "repo_edit_status": repo_edit_status,
        "long_command": {
            "command": "",
            "reason": "",
            "expected_artifacts": [],
        },
        "stop": {"reason": reason},
    }
