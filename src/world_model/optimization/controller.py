"""Controller helpers for staged training-optimization experiments.

This module plans conservative experiments from markdown memory, shells out to
the canonical train/eval/check scripts, and records auditable stage summaries.

source .venv/bin/activate && python scripts/train/training_optimizer.py --train-config configs/train/aloha_fork_pick_up.yaml --memory-path docs/training_optimizer.md
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import imageio.v2 as iio
import numpy as np

from world_model.optimization.codex_runner import (
    DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS,
    CodexExecutionResult,
    ensure_codex_chatgpt_login,
    load_codex_session_metadata,
    run_codex_exec,
)
from world_model.config import TrainScriptConfig, load_train_config
from world_model.training.validation import load_metrics_rows, validate_training_stage


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_SOURCE_PATH = Path(__file__).resolve()
DEFAULT_MEMORY_PATH = REPO_ROOT / "docs" / "training_optimizer.md"
DEFAULT_TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train" / "aloha_fork_pick_up.yaml"
DEFAULT_STATE_PATH = REPO_ROOT / "runs" / "training_optimizer" / "controller_state.json"
CONTROLLER_BULLET_PREFIX = "- [controller "
DEFAULT_LOOP_LOCK_TIMEOUT_SECONDS = 8 * 60 * 60
DEFAULT_MAX_FAILED_RUNS = 3
DEFAULT_MAX_CODEX_CALLS_MULTIPLIER = 4
DEFAULT_MAX_EDIT_CYCLES = 4
DEFAULT_CONTEXT_HISTORY_LIMIT = 20
DEFAULT_DECISION_HISTORY_LIMIT = 20
DEFAULT_HISTORY_SUMMARY_LIMIT = 3
DEFAULT_CONTINUATION_HISTORY_LIMIT = 1
DEFAULT_CONTINUATION_DECISION_LIMIT = 2
DEFAULT_CONTINUATION_EDIT_LIMIT = 1
DEFAULT_MAX_INSPECTION_ROUNDS = 2
DEFAULT_CODEX_MEMORY_MODE = "hybrid"
DEFAULT_CODEX_MAX_SESSION_TURNS = 0
DEFAULT_CODEX_MAX_SESSION_AGE_MINUTES = 180
DEFAULT_CODEX_MEMORY_SUMMARIZE_EVERY_TURNS = 3
DEFAULT_CODEX_CONTINUATION_PROMPT_CHAR_BUDGET = 16000
DEFAULT_CODEX_FRESH_PROMPT_CHAR_BUDGET = 22000
STOP_AFTER_STAGE_REQUEST_FILENAME = "stop_after_stage.request"
CODEX_VISUAL_REVIEW_SECTION = "Codex Visual Reviews"
DIAGNOSTIC_CONFIG_KEYS = (
    "overfit_one_batch",
    "video_path",
    "subset_size",
    "start_frame",
)
# controller-self-edit: policy begin
CONTROLLER_POLICY = {
    "fallback_lr_scale": 0.5,
    "improvement_threshold_floor": 0.02,
    "visual_review_focus": "generic",
}
# controller-self-edit: policy end
GENERIC_VISUAL_REVIEW_FOCUS_POINTS = (
    "blur",
    "brightness drift",
    "ghosting on moving objects",
    "motion path vs target",
)
GENERIC_VISUAL_REVIEW_FOCUS_NOTE = (
    "Focus on blur, brightness drift, ghosting on moving objects, and whether the motion path matches the target."
)
MOTION_VISUAL_REVIEW_FOCUS_POINTS = (
    "arm pose vs target",
    "tool path vs target",
    "contact dynamics",
    "motion collapse or lag",
)
MOTION_VISUAL_REVIEW_FOCUS_NOTE = (
    "Focus on arm pose, tool path, contact dynamics, and whether the generated motion stays aligned with the target."
)


@dataclass(frozen=True)
class MemoryHints:
    """Structured planning hints extracted from the optimization markdown."""

    overrides: dict[str, Any]
    locked_keys: tuple[str, ...]
    stage_step: int | None
    train_from_scratch: bool
    reasoning: tuple[str, ...]


@dataclass(frozen=True)
class RunProgress:
    """Summarize the resumable state already present in one run directory."""

    current_step: int
    checkpoint_path: Path | None
    metrics_path: Path | None


@dataclass(frozen=True)
class MetricsSummary:
    """Compact loss summary for one completed controller stage."""

    previous_step: int
    target_step: int
    last_loss: float
    best_loss: float
    stage_mean_loss: float
    trailing_mean_loss: float
    stage_row_count: int
    relative_stage_improvement: float | None


@dataclass(frozen=True)
class PlausibilitySummary:
    """Compact video-quality summary from the plausibility checker."""

    plausible: bool
    mean_frame_mae_rgb_0_255: float
    temporal_delta_ratio: float
    num_failing_frames: int
    video_flags: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentPlan:
    """Describe one explicit controller iteration."""

    experiment_name: str
    output_dir: Path
    overrides: dict[str, Any]
    resolved_config: dict[str, Any]
    current_step: int
    target_step: int
    stage_step: int
    resume_from: Path | None
    reasoning: tuple[str, ...]


@dataclass(frozen=True)
class ControllerPolicyEditProposal:
    """Describe one bounded self-edit to the controller's policy block."""

    key: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass(frozen=True)
class LoopBudget:
    """Track hard limits and current usage for autonomous Codex looping."""

    max_iterations: int
    max_real_runs: int
    max_codex_calls: int
    max_failed_runs: int
    max_edit_cycles: int
    max_wall_clock_minutes: int | None
    iterations_used: int
    real_runs_used: int
    codex_calls_used: int
    failed_runs_used: int
    edit_cycles_used: int
    started_at: str


def run_training_optimization_loop(
    *,
    train_config_path: str | Path = DEFAULT_TRAIN_CONFIG_PATH,
    memory_path: str | Path = DEFAULT_MEMORY_PATH,
    state_path: str | Path = DEFAULT_STATE_PATH,
    planner: str = "codex",
    codex_model: str | None = None,
    codex_timeout_seconds: int = DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS,
    codex_memory: str = DEFAULT_CODEX_MEMORY_MODE,
    codex_session_id: str | None = None,
    codex_max_session_turns: int = DEFAULT_CODEX_MAX_SESSION_TURNS,
    codex_max_session_age_minutes: int = DEFAULT_CODEX_MAX_SESSION_AGE_MINUTES,
    iterations: int = 1,
    max_real_runs: int | None = None,
    max_codex_calls: int | None = None,
    max_failed_runs: int = DEFAULT_MAX_FAILED_RUNS,
    max_edit_cycles: int = DEFAULT_MAX_EDIT_CYCLES,
    max_wall_clock_minutes: int | None = None,
    stage_step_override: int | None = None,
    eval_episode_index: int = 0,
    eval_start_frame: int = 60,
    reference_frame_offset: int | None = None,
    reference_video: str | Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run one or more conservative optimization iterations."""
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    planner_key = planner.strip().lower()
    if planner_key == "deterministic":
        return _run_deterministic_training_optimization_loop(
            train_config_path=Path(train_config_path),
            memory_path=Path(memory_path),
            state_path=Path(state_path),
            iterations=iterations,
            stage_step_override=stage_step_override,
            eval_episode_index=eval_episode_index,
            eval_start_frame=eval_start_frame,
            reference_frame_offset=reference_frame_offset,
            reference_video=Path(reference_video) if reference_video is not None else None,
            dry_run=dry_run,
        )
    if planner_key != "codex":
        raise ValueError(f"Unsupported planner {planner!r}; expected 'codex' or 'deterministic'.")
    return _run_codex_training_optimization_loop(
        train_config_path=Path(train_config_path),
        memory_path=Path(memory_path),
        state_path=Path(state_path),
        codex_model=codex_model,
        codex_timeout_seconds=codex_timeout_seconds,
        codex_memory=codex_memory,
        codex_session_id=codex_session_id,
        codex_max_session_turns=codex_max_session_turns,
        codex_max_session_age_minutes=codex_max_session_age_minutes,
        iterations=iterations,
        max_real_runs=max_real_runs,
        max_codex_calls=max_codex_calls,
        max_failed_runs=max_failed_runs,
        max_edit_cycles=max_edit_cycles,
        max_wall_clock_minutes=max_wall_clock_minutes,
        stage_step_override=stage_step_override,
        eval_episode_index=eval_episode_index,
        eval_start_frame=eval_start_frame,
        reference_frame_offset=reference_frame_offset,
        reference_video=Path(reference_video) if reference_video is not None else None,
        dry_run=dry_run,
    )


def _run_deterministic_training_optimization_loop(
    *,
    train_config_path: Path,
    memory_path: Path,
    state_path: Path,
    iterations: int,
    stage_step_override: int | None,
    eval_episode_index: int,
    eval_start_frame: int,
    reference_frame_offset: int | None,
    reference_video: Path | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Run the existing rule-based staged optimizer loop."""
    train_config = load_train_config(train_config_path)
    state = load_controller_state(state_path)
    records: list[dict[str, Any]] = []

    for _ in range(iterations):
        memory_text = memory_path.read_text(encoding="utf-8")
        codex_visual_gate = _latest_stage_codex_visual_gate(memory_text=memory_text, state=state)
        if codex_visual_gate is not None:
            raise RuntimeError(codex_visual_gate)
        plan = select_experiment_plan(
            train_config=train_config,
            memory_text=memory_text,
            state=state,
            stage_step_override=stage_step_override,
        )
        _print_plan(plan)
        if dry_run:
            break

        record = run_experiment_stage(
            plan=plan,
            train_config_path=train_config_path,
            memory_path=memory_path,
            state=state,
            eval_episode_index=eval_episode_index,
            eval_start_frame=eval_start_frame,
            reference_frame_offset=reference_frame_offset,
            reference_video=reference_video,
        )
        update_memory_file(memory_path, record)
        state = append_stage_record(state, record)
        save_controller_state(state_path, state)
        records.append(record)

    return records


def _run_codex_training_optimization_loop(
    *,
    train_config_path: Path,
    memory_path: Path,
    state_path: Path,
    codex_model: str | None,
    codex_timeout_seconds: int,
    codex_memory: str,
    codex_session_id: str | None,
    codex_max_session_turns: int,
    codex_max_session_age_minutes: int,
    iterations: int,
    max_real_runs: int | None,
    max_codex_calls: int | None,
    max_failed_runs: int,
    max_edit_cycles: int,
    max_wall_clock_minutes: int | None,
    stage_step_override: int | None,
    eval_episode_index: int,
    eval_start_frame: int,
    reference_frame_offset: int | None,
    reference_video: Path | None,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Run the autonomous Codex-authenticated experiment loop."""
    if codex_timeout_seconds <= 0:
        raise ValueError(f"codex_timeout_seconds must be > 0, got {codex_timeout_seconds}")
    memory_mode = _normalize_codex_memory_mode(codex_memory)
    if codex_max_session_turns < 0:
        raise ValueError(f"codex_max_session_turns must be >= 0, got {codex_max_session_turns}")
    if codex_max_session_age_minutes <= 0:
        raise ValueError(
            f"codex_max_session_age_minutes must be > 0, got {codex_max_session_age_minutes}"
        )
    _log_controller_status("Checking Codex ChatGPT login status.")
    ensure_codex_chatgpt_login()
    train_config = load_train_config(train_config_path)
    state = load_controller_state(state_path)
    state = _normalize_controller_state(state)
    _initialize_codex_memory_state(
        state=state,
        memory_mode=memory_mode,
        codex_model=codex_model,
        explicit_session_id=codex_session_id,
    )
    budget = _build_loop_budget(
        existing_budget=state.get("budget", {}),
        iterations=iterations,
        max_real_runs=max_real_runs,
        max_codex_calls=max_codex_calls,
        max_failed_runs=max_failed_runs,
        max_edit_cycles=max_edit_cycles,
        max_wall_clock_minutes=max_wall_clock_minutes,
    )
    state["budget"] = loop_budget_to_dict(budget)
    save_controller_state(state_path, state)
    _log_controller_status(
        "Starting Codex planner loop "
        f"(iterations={budget.max_iterations}, max_real_runs={budget.max_real_runs}, "
        f"max_codex_calls={budget.max_codex_calls}, max_edit_cycles={budget.max_edit_cycles}, "
        f"memory_mode={memory_mode})."
    )

    records: list[dict[str, Any]] = []
    pending_controller_edits: list[dict[str, Any]] = []

    with _controller_loop_lock(state_path):
        while True:
            stop_reason = _budget_stop_reason(budget)
            if stop_reason is not None:
                _log_controller_status(f"Stopping optimizer loop: {stop_reason}.")
                state["codex_state"]["last_stop_reason"] = stop_reason
                save_controller_state(state_path, state)
                break

            budget = _increment_loop_budget_counter(budget, "iterations_used")
            _log_controller_status(
                f"Planning iteration {budget.iterations_used}/{budget.max_iterations} "
                f"(real_runs={budget.real_runs_used}/{budget.max_real_runs}, "
                f"codex_calls={budget.codex_calls_used}/{budget.max_codex_calls})."
            )
            memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
            session_policy = _resolve_codex_session_policy(
                state=state,
                memory_mode=memory_mode,
                codex_model=codex_model,
                explicit_session_id=codex_session_id,
                max_session_turns=codex_max_session_turns,
                max_session_age_minutes=codex_max_session_age_minutes,
            )
            if memory_mode == "controller-only":
                _clear_codex_session_state(state, reason="controller_only_mode")
            elif session_policy["reuse_session"]:
                _log_controller_status(
                    f"Reusing Codex session {session_policy['session_id']} "
                    f"(turns={session_policy['session_turns']})."
                )
            else:
                _log_controller_status(
                    "Starting a fresh Codex session"
                    + (
                        f" ({session_policy['reset_reason']})."
                        if session_policy["reset_reason"]
                        else "."
                    )
                )
            visual_review_call_count = _ensure_latest_codex_visual_review(
                state=state,
                memory_path=memory_path,
                budget=budget,
                codex_model=codex_model,
                codex_timeout_seconds=codex_timeout_seconds,
                memory_mode=memory_mode,
                session_policy=session_policy,
            )
            if visual_review_call_count:
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=visual_review_call_count,
                )
                save_controller_state(state_path, _persist_budget(state, budget))
                memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
            codex_visual_gate = _latest_stage_codex_visual_gate(memory_text=memory_text, state=state)
            if codex_visual_gate is not None:
                stop_reason = codex_visual_gate
                _log_controller_status(f"Stopping optimizer loop: {stop_reason}.")
                state["codex_state"]["last_stop_reason"] = stop_reason
                save_controller_state(state_path, _persist_budget(state, budget))
                break
            context_bundle = build_codex_context_bundle(
                train_config=train_config,
                memory_text=memory_text,
                state=state,
                budget=budget,
                pending_controller_edits=pending_controller_edits,
                memory_mode=memory_mode,
                session_policy=session_policy,
            )
            _append_limited_history(
                state["context_history"],
                {
                    "timestamp": _utc_timestamp(),
                    "budget": loop_budget_to_dict(budget),
                    "history_count": len(state.get("history", [])),
                    "latest_experiment": context_bundle.get("latest_experiment"),
                },
                limit=DEFAULT_CONTEXT_HISTORY_LIMIT,
            )

            decision, decision_call_count, decision_result = request_codex_loop_decision(
                context_bundle=context_bundle,
                codex_model=codex_model,
                codex_timeout_seconds=codex_timeout_seconds,
                session_id=session_policy["session_id"] if session_policy["reuse_session"] else None,
            )
            budget = _increment_loop_budget_counter(
                budget,
                "codex_calls_used",
                count=decision_call_count,
            )
            _update_codex_session_state(
                state=state,
                result=decision_result,
                memory_mode=memory_mode,
                codex_model=codex_model,
                session_policy=session_policy,
            )
            _record_codex_decision(state, decision=decision, budget=budget)
            budget = _increment_loop_budget_counter(
                budget,
                "codex_calls_used",
                count=0
                if dry_run
                else _refresh_codex_memory_summary(
                    state=state,
                    decision=decision,
                    budget=budget,
                    codex_model=codex_model,
                    codex_timeout_seconds=codex_timeout_seconds,
                    memory_mode=memory_mode,
                ),
            )
            save_controller_state(state_path, _persist_budget(state, budget))
            _log_controller_status(
                f"Codex decision: {decision['action_type']} | {decision['analysis_summary']}"
            )

            inspection_context: dict[str, Any] | None = None
            inspection_rounds = 0
            stop_after_inspection = False
            while decision["action_type"] == "inspect_artifact":
                if inspection_rounds >= DEFAULT_MAX_INSPECTION_ROUNDS:
                    _log_controller_status(
                        f"Stopping after {DEFAULT_MAX_INSPECTION_ROUNDS} inspection round(s) in one iteration."
                    )
                    state["codex_state"]["last_stop_reason"] = (
                        f"max_inspection_rounds={DEFAULT_MAX_INSPECTION_ROUNDS} exhausted"
                    )
                    update_memory_with_codex_analysis_file(memory_path, decision=decision)
                    save_controller_state(state_path, _persist_budget(state, budget))
                    stop_after_inspection = True
                    break
                stop_reason = _budget_stop_reason(budget, include_iterations=False)
                if stop_reason is not None:
                    _log_controller_status(f"Stopping optimizer loop during inspection: {stop_reason}.")
                    state["codex_state"]["last_stop_reason"] = stop_reason
                    update_memory_with_codex_analysis_file(memory_path, decision=decision)
                    save_controller_state(state_path, _persist_budget(state, budget))
                    stop_after_inspection = True
                    break
                _log_controller_status(
                    f"Inspection round {inspection_rounds + 1}/{DEFAULT_MAX_INSPECTION_ROUNDS}: "
                    f"artifacts={len(decision['inspect_artifact']['artifact_paths'])}, "
                    f"code_paths={len(decision['inspect_artifact']['code_paths'])}."
                )
                for artifact_path in decision["inspect_artifact"]["artifact_paths"]:
                    _log_controller_status(f"  inspect artifact: {artifact_path}")
                for code_path in decision["inspect_artifact"]["code_paths"]:
                    _log_controller_status(f"  inspect code: {code_path}")
                inspection_context = prepare_codex_inspection_context(
                    request=decision["inspect_artifact"],
                    state=state,
                    session_id=state.get("codex_state", {}).get("session_id"),
                    memory_mode=memory_mode,
                )
                _log_controller_status(
                    "Prepared inspection context with "
                    f"{len(inspection_context['image_inputs'])} image attachment(s), "
                    f"{inspection_context['summary'].get('reused_code_paths', 0)} reused code path(s), and "
                    f"{inspection_context['summary'].get('reused_artifact_paths', 0)} reused artifact(s)."
                )
                _append_limited_history(
                    state["inspection_history"],
                    inspection_context["summary"],
                    limit=DEFAULT_DECISION_HISTORY_LIMIT,
                )
                decision, decision_call_count, decision_result = request_codex_loop_decision(
                    context_bundle=context_bundle,
                    codex_model=codex_model,
                    codex_timeout_seconds=codex_timeout_seconds,
                    session_id=state.get("codex_state", {}).get("session_id"),
                    inspection_context=inspection_context,
                )
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=decision_call_count,
                )
                _update_codex_session_state(
                    state=state,
                    result=decision_result,
                    memory_mode=memory_mode,
                    codex_model=codex_model,
                    session_policy=session_policy,
                )
                _record_codex_decision(state, decision=decision, budget=budget)
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=0
                    if dry_run
                    else _refresh_codex_memory_summary(
                        state=state,
                        decision=decision,
                        budget=budget,
                        codex_model=codex_model,
                        codex_timeout_seconds=codex_timeout_seconds,
                        memory_mode=memory_mode,
                    ),
                )
                save_controller_state(state_path, _persist_budget(state, budget))
                inspection_rounds += 1
                _log_controller_status(
                    f"Codex follow-up decision after inspection: {decision['action_type']} | "
                    f"{decision['analysis_summary']}"
                )
            if stop_after_inspection:
                break

            if dry_run:
                _log_controller_status("Dry-run mode enabled; stopping after planning.")
                save_controller_state(state_path, _persist_budget(state, budget))
                return [
                    {
                        "dry_run": True,
                        "budget": loop_budget_to_dict(budget),
                        "decision": decision,
                        "inspection_context": inspection_context,
                    }
                ]

            if decision["action_type"] == "apply_repo_edit":
                if budget.edit_cycles_used >= budget.max_edit_cycles:
                    _log_controller_status("Edit-cycle budget exhausted before applying a repo edit.")
                    state["codex_state"]["last_stop_reason"] = "edit cycle budget exhausted"
                    save_controller_state(state_path, _persist_budget(state, budget))
                    break
                _log_controller_status(
                    "Applying Codex-proposed repo edit touching: "
                    + ", ".join(decision["apply_repo_edit"]["touched_files"])
                )
                edit_result = apply_codex_repo_edit(
                    proposal=decision["apply_repo_edit"],
                    analysis_summary=decision["analysis_summary"],
                )
                pending_controller_edits = [edit_result]
                _append_limited_history(state["edit_history"], edit_result, limit=DEFAULT_DECISION_HISTORY_LIMIT)
                budget = _increment_loop_budget_counter(budget, "edit_cycles_used")
                save_controller_state(state_path, _persist_budget(state, budget))
                update_memory_with_codex_analysis_file(
                    memory_path,
                    decision=decision,
                    controller_edits=[edit_result],
                )
                stop_reason = _budget_stop_reason(budget, include_iterations=False)
                if stop_reason is not None:
                    _log_controller_status(f"Stopping optimizer loop after repo edit: {stop_reason}.")
                    state["codex_state"]["last_stop_reason"] = stop_reason
                    save_controller_state(state_path, _persist_budget(state, budget))
                    break
                followup_context = build_codex_context_bundle(
                    train_config=train_config,
                    memory_text=memory_path.read_text(encoding="utf-8") if memory_path.exists() else memory_text,
                    state=state,
                    budget=budget,
                    pending_controller_edits=pending_controller_edits,
                    memory_mode=memory_mode,
                    session_policy={
                        "session_id": state.get("codex_state", {}).get("session_id"),
                        "reuse_session": memory_mode != "controller-only"
                        and bool(state.get("codex_state", {}).get("session_id")),
                        "session_turns": state.get("codex_state", {}).get("session_turns", 0),
                        "reset_reason": None,
                    },
                )
                decision, decision_call_count, decision_result = request_codex_loop_decision(
                    context_bundle=followup_context,
                    codex_model=codex_model,
                    codex_timeout_seconds=codex_timeout_seconds,
                    session_id=state.get("codex_state", {}).get("session_id"),
                )
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=decision_call_count,
                )
                _update_codex_session_state(
                    state=state,
                    result=decision_result,
                    memory_mode=memory_mode,
                    codex_model=codex_model,
                    session_policy=session_policy,
                )
                _record_codex_decision(state, decision=decision, budget=budget)
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=0
                    if dry_run
                    else _refresh_codex_memory_summary(
                        state=state,
                        decision=decision,
                        budget=budget,
                        codex_model=codex_model,
                        codex_timeout_seconds=codex_timeout_seconds,
                        memory_mode=memory_mode,
                    ),
                )
                save_controller_state(state_path, _persist_budget(state, budget))
                _log_controller_status(
                    f"Codex follow-up decision after repo edit: {decision['action_type']} | "
                    f"{decision['analysis_summary']}"
                )
                if decision["action_type"] not in {"run_experiment", "stop"}:
                    state["codex_state"]["last_stop_reason"] = (
                        "Codex must return `run_experiment` or `stop` after an edit cycle."
                    )
                    save_controller_state(state_path, _persist_budget(state, budget))
                    break

            if decision["action_type"] == "stop":
                stop_reason = decision["stop"]["reason"] or decision["analysis_summary"]
                _log_controller_status(f"Codex requested stop: {stop_reason}")
                state["codex_state"]["last_stop_reason"] = stop_reason
                update_memory_with_codex_analysis_file(memory_path, decision=decision, controller_edits=pending_controller_edits)
                save_controller_state(state_path, _persist_budget(state, budget))
                break

            if decision["action_type"] != "run_experiment":
                raise RuntimeError(f"Unsupported Codex action after inspection/edit flow: {decision['action_type']}")

            plan = build_experiment_plan_from_codex_decision(
                train_config=train_config,
                decision=decision,
                state=state,
                memory_text=memory_text,
                stage_step_override=stage_step_override,
            )
            _log_controller_status(
                f"Executing experiment plan for {plan.experiment_name} -> step {plan.target_step}."
            )
            _print_plan(plan)
            try:
                record = run_experiment_stage(
                    plan=plan,
                    train_config_path=train_config_path,
                    memory_path=memory_path,
                    state=state,
                    eval_episode_index=eval_episode_index,
                    eval_start_frame=eval_start_frame,
                    reference_frame_offset=reference_frame_offset,
                    reference_video=reference_video,
                    allow_policy_self_edits=False,
                    extra_controller_edits=pending_controller_edits,
                    codex_analysis={
                        "analysis_summary": decision["analysis_summary"],
                        "reasoning": decision["reasoning"],
                        "action_type": decision["action_type"],
                        "next_work_note": decision.get("next_work_note", ""),
                    },
                )
            except Exception as exc:
                budget = _increment_loop_budget_counter(budget, "failed_runs_used")
                _log_controller_status(
                    f"Experiment stage failed for {plan.experiment_name}: {exc}"
                )
                state["codex_state"]["last_run_failure"] = {
                    "timestamp": _utc_timestamp(),
                    "experiment_name": plan.experiment_name,
                    "error": str(exc),
                }
                save_controller_state(state_path, _persist_budget(state, budget))
                if budget.failed_runs_used >= budget.max_failed_runs:
                    raise
                pending_controller_edits = []
                continue

            visual_review_call_count = _attach_codex_visual_review_to_record(
                record=record,
                state=state,
                budget=budget,
                codex_model=codex_model,
                codex_timeout_seconds=codex_timeout_seconds,
                memory_mode=memory_mode,
                session_policy=session_policy,
            )
            if visual_review_call_count:
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=visual_review_call_count,
                )

            update_memory_file(memory_path, record)
            state = append_stage_record(state, record)
            budget = _increment_loop_budget_counter(budget, "real_runs_used")
            state = _persist_budget(state, budget)
            save_controller_state(state_path, state)
            records.append(record)
            pending_controller_edits = []
            _log_controller_status(
                f"Recorded completed stage for {record['experiment_name']} at step {record['target_step']}."
            )
            if _consume_stop_after_stage_request(state_path):
                stop_reason = (
                    f"stop requested via {_stop_after_stage_request_path(state_path).name} "
                    "after stage finalization"
                )
                _log_controller_status(f"Stopping optimizer loop: {stop_reason}.")
                state["codex_state"]["last_stop_reason"] = stop_reason
                save_controller_state(state_path, state)
                break

    return records


def select_experiment_plan(
    *,
    train_config: TrainScriptConfig,
    memory_text: str,
    state: dict[str, Any],
    stage_step_override: int | None = None,
) -> ExperimentPlan:
    """Choose the next conservative experiment from memory plus controller state."""
    hints = extract_memory_hints(memory_text, train_config=train_config)
    stage_step = _resolve_stage_step(train_config=train_config, hints=hints, stage_step_override=stage_step_override)
    latest_recommendation = state.get("latest_recommendation")
    if isinstance(latest_recommendation, dict) and _recommendation_matches_hints(latest_recommendation, hints):
        return _plan_from_recommendation(
            train_config=train_config,
            recommendation=latest_recommendation,
            hints=hints,
        )
    return _plan_from_memory_hints(
        train_config=train_config,
        hints=hints,
        stage_step=stage_step,
        history=state.get("history", []),
    )


def extract_memory_hints(memory_text: str, *, train_config: TrainScriptConfig) -> MemoryHints:
    """Extract conservative planning hints from the optimization markdown."""
    _, sections, _ = parse_markdown_sections(memory_text)
    next_work = sections.get("Next Work", "")
    combined = next_work
    overrides: dict[str, Any] = {}
    reasoning: list[str] = []

    direct_change_pattern = re.compile(
        r"change only\s+`(?P<key>[a-z_]+)`\s+from\s+`(?P<old>[^`]+)`\s+to\s+`(?P<new>[^`]+)`",
        re.IGNORECASE,
    )
    for match in direct_change_pattern.finditer(combined):
        key = match.group("key")
        if not hasattr(train_config, key):
            continue
        overrides[key] = _coerce_like(getattr(train_config, key), match.group("new"))
        reasoning.append(
            f"Next Work explicitly says to change only {key} from {match.group('old')} to {match.group('new')}."
        )

    for code_span in re.findall(r"`([^`]+)`", combined):
        for assignment in re.finditer(r"([a-z_]+)\s*=\s*([A-Za-z0-9_.+-]+)", code_span):
            key, raw_value = assignment.groups()
            if not hasattr(train_config, key):
                continue
            overrides[key] = _coerce_like(getattr(train_config, key), raw_value)
            reasoning.append(f"Next Work pins {key}={raw_value}.")

        resolution_match = re.search(r"\b(?P<width>\d+)x(?P<height>\d+)\b", code_span)
        if resolution_match is not None:
            width = int(resolution_match.group("width"))
            height = int(resolution_match.group("height"))
            overrides["frame_width"] = width
            overrides["frame_height"] = height
            reasoning.append(f"Next Work pins resolution to {width}x{height}.")

        episode_match = re.search(r"\bepisode\s+(?P<episode>\d+)\b", code_span, re.IGNORECASE)
        if episode_match is not None:
            episode_index = int(episode_match.group("episode"))
            overrides["episodes"] = (episode_index,)
            reasoning.append(f"Next Work pins training to episode {episode_index}.")

    lowered = combined.lower()
    if (
        "action-conditioning" in lowered or "action conditioning" in lowered
    ) and "conditioning_mode" not in overrides:
        overrides["conditioning_mode"] = "action"
        reasoning.append("Next Work restores action conditioning.")

    stage_step = None
    stage_match = re.search(r"every\s+`?(?P<step>\d+)`?\s+steps", combined, re.IGNORECASE)
    if stage_match is not None:
        stage_step = int(stage_match.group("step"))
        reasoning.append(f"Next Work evaluates checkpoints every {stage_step} steps.")

    locked_keys: set[str] = set()
    if "do not change lr" in lowered or "do not change learning rate" in lowered:
        locked_keys.add("lr")
        reasoning.append("Next Work freezes the learning rate for now.")
    if "frame count" in lowered:
        locked_keys.update({"context_len", "horizon_len", "frame_width", "frame_height"})
        reasoning.append("Next Work freezes frame-count and resolution choices for now.")
    if "dataset scope" in lowered:
        locked_keys.update({"repo_id", "episodes", "subset_size", "video_key"})
        reasoning.append("Next Work freezes dataset scope for now.")

    train_from_scratch = "train from scratch" in lowered
    if train_from_scratch:
        reasoning.append("Next Work explicitly says to train the next branch from scratch.")

    return MemoryHints(
        overrides=overrides,
        locked_keys=tuple(sorted(locked_keys)),
        stage_step=stage_step,
        train_from_scratch=train_from_scratch,
        reasoning=tuple(reasoning),
    )


def summarize_metrics_rows(
    *,
    metrics_rows: list[dict[str, Any]],
    previous_step: int,
    target_step: int,
    previous_stage_mean_loss: float | None,
) -> MetricsSummary:
    """Summarize the recorded metrics for one controller-managed stage."""
    stage_rows = [
        row
        for row in metrics_rows
        if previous_step < int(row["step"]) <= target_step
    ]
    if not stage_rows:
        raise ValueError(
            f"No metrics rows found for steps {previous_step + 1}..{target_step}."
        )

    losses = [float(row["loss"]) for row in stage_rows]
    trailing_window = losses[-min(len(losses), 10):]
    stage_mean_loss = sum(losses) / len(losses)
    trailing_mean_loss = sum(trailing_window) / len(trailing_window)
    last_loss = losses[-1]
    best_loss = min(losses)
    if previous_stage_mean_loss is None:
        relative_stage_improvement = None
    elif previous_stage_mean_loss <= 0.0:
        relative_stage_improvement = math.inf
    else:
        relative_stage_improvement = (
            previous_stage_mean_loss - stage_mean_loss
        ) / previous_stage_mean_loss

    return MetricsSummary(
        previous_step=previous_step,
        target_step=target_step,
        last_loss=last_loss,
        best_loss=best_loss,
        stage_mean_loss=stage_mean_loss,
        trailing_mean_loss=trailing_mean_loss,
        stage_row_count=len(stage_rows),
        relative_stage_improvement=relative_stage_improvement,
    )


def _normalize_controller_state(state: dict[str, Any]) -> dict[str, Any]:
    """Backfill controller-state sections needed by both deterministic and Codex loops."""
    normalized = dict(state)
    normalized.setdefault("history", [])
    normalized.setdefault("latest_recommendation", None)
    normalized.setdefault("latest_record", None)
    normalized.setdefault("codex_state", {})
    normalized.setdefault("budget", {})
    normalized.setdefault("decision_history", [])
    normalized.setdefault("inspection_history", [])
    normalized.setdefault("edit_history", [])
    normalized.setdefault("context_history", [])
    normalized.setdefault("codex_memory_summary", {})
    normalized.setdefault("retrieved_context_cache", {"code": {}, "artifacts": {}})
    normalized.setdefault("comparison_baselines", {})
    return normalized


def _normalize_codex_memory_mode(memory_mode: str) -> str:
    """Normalize and validate the configured Codex memory mode."""
    normalized = memory_mode.strip().lower()
    if normalized not in {"hybrid", "session-only", "controller-only"}:
        raise ValueError(
            f"Unsupported codex_memory {memory_mode!r}; expected 'hybrid', 'session-only', or 'controller-only'."
        )
    return normalized


def _initialize_codex_memory_state(
    *,
    state: dict[str, Any],
    memory_mode: str,
    codex_model: str | None,
    explicit_session_id: str | None,
) -> None:
    """Seed durable Codex memory metadata for one autonomous invocation."""
    codex_state = dict(state.get("codex_state", {}))
    codex_state.setdefault("memory_mode", memory_mode)
    codex_state.setdefault("session_turns", 0)
    codex_state.setdefault("session_cwd", str(REPO_ROOT))
    codex_state.setdefault("session_model", codex_model)
    if explicit_session_id is not None:
        codex_state["session_id"] = explicit_session_id
        codex_state["last_session_reset_reason"] = "explicit_session_override"
    state["codex_state"] = codex_state


def _resolve_codex_session_policy(
    *,
    state: dict[str, Any],
    memory_mode: str,
    codex_model: str | None,
    explicit_session_id: str | None,
    max_session_turns: int,
    max_session_age_minutes: int,
) -> dict[str, Any]:
    """Decide whether the next Codex turn should reuse or reset session continuity."""
    codex_state = dict(state.get("codex_state", {}))
    previous_session_id = explicit_session_id or codex_state.get("session_id")
    policy = {
        "memory_mode": memory_mode,
        "session_id": None,
        "reuse_session": False,
        "reset_reason": None,
        "session_turns": int(codex_state.get("session_turns", 0)),
    }
    if memory_mode == "controller-only":
        if previous_session_id:
            policy["reset_reason"] = "controller_only_mode"
        return policy
    if explicit_session_id:
        metadata = load_codex_session_metadata(explicit_session_id)
        if metadata is None:
            policy["reset_reason"] = f"missing_explicit_session:{explicit_session_id}"
            return policy
        policy["session_id"] = explicit_session_id
        policy["reuse_session"] = True
        return policy
    if not isinstance(previous_session_id, str) or not previous_session_id.strip():
        policy["reset_reason"] = "no_persisted_session"
        return policy
    metadata = load_codex_session_metadata(previous_session_id)
    if metadata is None:
        policy["reset_reason"] = f"missing_session:{previous_session_id}"
        return policy
    if metadata.cwd not in {None, str(REPO_ROOT)}:
        policy["reset_reason"] = "workspace_changed"
        return policy
    if codex_state.get("session_model") != codex_model:
        policy["reset_reason"] = "model_changed"
        return policy
    if codex_state.get("memory_mode") != memory_mode:
        policy["reset_reason"] = "memory_mode_changed"
        return policy
    if max_session_turns > 0 and int(codex_state.get("session_turns", 0)) >= max_session_turns:
        policy["reset_reason"] = "session_turn_limit"
        return policy
    session_started_at = _parse_optional_iso_datetime(codex_state.get("session_started_at"))
    if session_started_at is not None:
        elapsed_minutes = (datetime.now(timezone.utc) - session_started_at).total_seconds() / 60.0
        if elapsed_minutes >= max_session_age_minutes:
            policy["reset_reason"] = "session_age_limit"
            return policy
    policy["session_id"] = previous_session_id
    policy["reuse_session"] = True
    return policy


def _parse_optional_iso_datetime(raw_value: Any) -> datetime | None:
    """Parse a persisted optional ISO timestamp."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stop_after_stage_request_path(state_path: str | Path) -> Path:
    """Return the filesystem path used to request a safe stop after stage finalization."""
    return Path(state_path).resolve().parent / STOP_AFTER_STAGE_REQUEST_FILENAME


def _consume_stop_after_stage_request(state_path: str | Path) -> bool:
    """Consume a pending safe-stop request after the current stage is fully persisted."""
    request_path = _stop_after_stage_request_path(state_path)
    if not request_path.exists():
        return False
    request_path.unlink()
    return True


def _build_loop_budget(
    *,
    existing_budget: dict[str, Any],
    iterations: int,
    max_real_runs: int | None,
    max_codex_calls: int | None,
    max_failed_runs: int,
    max_edit_cycles: int,
    max_wall_clock_minutes: int | None,
) -> LoopBudget:
    """Resolve autonomous-loop limits for one autonomous invocation."""
    resolved_max_real_runs = iterations if max_real_runs is None else max_real_runs
    resolved_max_codex_calls = (
        max(DEFAULT_MAX_CODEX_CALLS_MULTIPLIER * iterations, 4)
        if max_codex_calls is None
        else max_codex_calls
    )
    return LoopBudget(
        max_iterations=iterations,
        max_real_runs=resolved_max_real_runs,
        max_codex_calls=resolved_max_codex_calls,
        max_failed_runs=max_failed_runs,
        max_edit_cycles=max_edit_cycles,
        max_wall_clock_minutes=max_wall_clock_minutes,
        iterations_used=0,
        real_runs_used=0,
        codex_calls_used=0,
        failed_runs_used=0,
        edit_cycles_used=0,
        started_at=_utc_timestamp(),
    )


def loop_budget_to_dict(budget: LoopBudget) -> dict[str, Any]:
    """Convert a loop-budget dataclass to a JSON-friendly mapping."""
    return asdict(budget)


def _persist_budget(state: dict[str, Any], budget: LoopBudget) -> dict[str, Any]:
    """Persist the current budget counters back into controller state."""
    updated = _normalize_controller_state(state)
    updated["budget"] = loop_budget_to_dict(budget)
    return updated


def _increment_loop_budget_counter(budget: LoopBudget, field_name: str, *, count: int = 1) -> LoopBudget:
    """Return a new budget with one usage counter incremented."""
    payload = loop_budget_to_dict(budget)
    payload[field_name] = int(payload[field_name]) + count
    return LoopBudget(**payload)


def _budget_stop_reason(budget: LoopBudget, *, include_iterations: bool = True) -> str | None:
    """Return the first hard-stop reason for the current autonomous-loop budget."""
    if include_iterations and budget.iterations_used >= budget.max_iterations:
        return f"max_iterations={budget.max_iterations} exhausted"
    if budget.real_runs_used >= budget.max_real_runs:
        return f"max_real_runs={budget.max_real_runs} exhausted"
    if budget.codex_calls_used >= budget.max_codex_calls:
        return f"max_codex_calls={budget.max_codex_calls} exhausted"
    if budget.failed_runs_used >= budget.max_failed_runs:
        return f"max_failed_runs={budget.max_failed_runs} exhausted"
    if budget.edit_cycles_used >= budget.max_edit_cycles and budget.max_edit_cycles == 0:
        return "max_edit_cycles=0 exhausted"
    if budget.max_wall_clock_minutes is not None:
        started_at = datetime.fromisoformat(budget.started_at)
        elapsed_minutes = (datetime.now(timezone.utc) - started_at).total_seconds() / 60.0
        if elapsed_minutes >= budget.max_wall_clock_minutes:
            return f"max_wall_clock_minutes={budget.max_wall_clock_minutes} exhausted"
    return None


def build_codex_context_bundle(
    *,
    train_config: TrainScriptConfig,
    memory_text: str,
    state: dict[str, Any],
    budget: LoopBudget,
    pending_controller_edits: list[dict[str, Any]],
    memory_mode: str = DEFAULT_CODEX_MEMORY_MODE,
    session_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a compact structured context bundle for one Codex planning turn."""
    _, sections, _ = parse_markdown_sections(memory_text)
    history = list(state.get("history", []))
    latest_record = history[-1] if history else None
    latest_experiment = None if latest_record is None else latest_record.get("experiment_name")
    artifacts = _summarize_latest_artifacts(latest_record)
    codex_state = dict(state.get("codex_state", {}))
    reuse_session = bool(session_policy and session_policy.get("reuse_session"))
    recent_runs = _summarize_recent_history(history, compact=reuse_session)
    recent_decisions = _summarize_decision_history(
        state.get("decision_history", []),
        compact=reuse_session,
    )
    recent_edits = _summarize_edit_history(
        state.get("edit_history", []),
        compact=reuse_session,
    )
    context_bundle = {
        "context_mode": "continuation" if reuse_session else "full",
        "memory_mode": memory_mode,
        "session_memory": {
            "session_id": None if session_policy is None else session_policy.get("session_id"),
            "reuse_session": reuse_session,
            "session_turns": codex_state.get("session_turns", 0),
            "last_session_reset_reason": codex_state.get("last_session_reset_reason"),
        },
        "goal": sections.get("Goal", "").strip(),
        "training_goal": sections.get("Training Goal", "").strip(),
        "current_signal": _compact_text_block(sections.get("Current Signal", ""), max_lines=3),
        "next_work": _compact_text_block(sections.get("Next Work", ""), max_lines=8),
        "durable_memory_summary": _summarize_durable_memory(state),
        "base_train_config": {
            "repo_id": train_config.repo_id,
            "video_key": train_config.video_key,
            "frame_width": train_config.frame_width,
            "frame_height": train_config.frame_height,
            "conditioning_mode": train_config.conditioning_mode,
            "trainable_backbone": train_config.trainable_backbone,
            "lora_rank": train_config.lora_rank,
            "lr": train_config.lr,
            "batch_size": train_config.batch_size,
            "context_len": train_config.context_len,
            "horizon_len": train_config.horizon_len,
            "k": train_config.k,
        },
        "budget": loop_budget_to_dict(budget),
        "latest_experiment": latest_experiment,
        "latest_codex_visual_review": _summarize_visual_review_for_prompt(
            None if latest_record is None else latest_record.get("codex_visual_review")
        ),
        "latest_run_summary": _summarize_latest_run_for_prompt(latest_record),
        "comparison_context": _summarize_comparison_context(
            record=latest_record,
            state=state,
        ),
        "latest_recommendation": _summarize_latest_recommendation(
            state.get("latest_recommendation"),
            compact=reuse_session,
        ),
        "recent_runs": recent_runs,
        "recent_decisions": recent_decisions,
        "recent_edits": recent_edits,
        "recent_failures": state.get("codex_state", {}).get("last_run_failure"),
        "pending_controller_edits": pending_controller_edits,
        "latest_artifacts": artifacts,
        "codex_memory_summary": state.get("codex_memory_summary", {}),
        "retrieved_context_cache": _summarize_retrieved_context_cache(state.get("retrieved_context_cache", {})),
        "workspace_hints": _build_workspace_hints(
            latest_artifacts=artifacts,
            state=state,
        ),
    }
    if reuse_session:
        context_bundle["continuation_note"] = (
            "This is a continuation inside the same Codex session. Reuse prior file/artifact context from session memory; "
            "only newly attached inspection payloads are fresh in this turn."
        )
    else:
        context_bundle["stable_findings_summary"] = _compact_text_block(sections.get("Stable Findings", ""), max_lines=6)
        context_bundle["codex_analysis_summary"] = _compact_text_block(sections.get("Codex Analysis", ""), max_lines=4)
        context_bundle["controller_edits_summary"] = _compact_text_block(sections.get("Controller Edits", ""), max_lines=4)
        context_bundle["codex_visual_reviews_summary"] = _compact_text_block(
            sections.get(CODEX_VISUAL_REVIEW_SECTION, ""),
            max_lines=4,
        )
    return context_bundle


def _summarize_retrieved_context_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Summarize retrieval-cache state without inlining full excerpts or attachments."""
    code_cache = cache.get("code", {}) if isinstance(cache, dict) else {}
    artifact_cache = cache.get("artifacts", {}) if isinstance(cache, dict) else {}
    return {
        "code_paths": sorted(code_cache.keys())[-10:],
        "artifact_paths": sorted(artifact_cache.keys())[-10:],
    }


def _build_workspace_hints(*, latest_artifacts: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """List high-value workspace paths Codex can inspect on demand."""
    hints = {
        "memory_path": str(DEFAULT_MEMORY_PATH.relative_to(REPO_ROOT)),
        "state_path": str(DEFAULT_STATE_PATH.relative_to(REPO_ROOT)),
        "likely_code_paths": [
            "scripts/train/world_model.py",
            "src/world_model/training/chunkwise_training.py",
            "src/world_model/training/flow_matching.py",
            "src/world_model/optimization/controller.py",
        ],
        "cached_code_paths": _summarize_retrieved_context_cache(state.get("retrieved_context_cache", {})).get(
            "code_paths",
            [],
        ),
    }
    if latest_artifacts.get("evaluation_dir"):
        hints["latest_evaluation_dir"] = str(latest_artifacts["evaluation_dir"])
    if latest_artifacts.get("comparison_video"):
        hints["latest_comparison_video"] = str(latest_artifacts["comparison_video"])
    return hints


def _compact_text_block(text: str, *, max_lines: int) -> str:
    """Trim a multiline text block for compact session-continuation prompts."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["... [truncated]"])


def _summarize_durable_memory(state: dict[str, Any]) -> str:
    """Return one short durable-memory string for fresh and resumed prompts."""
    summary = state.get("codex_memory_summary", {})
    if not isinstance(summary, dict):
        return ""
    summary_text = str(summary.get("summary", "")).strip()
    if summary_text:
        return summary_text
    key_findings = [str(item).strip() for item in summary.get("key_findings", []) if str(item).strip()]
    return " ".join(key_findings[:2]).strip()


def _summarize_visual_review_for_prompt(review: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the latest visual-review fields that matter for planning."""
    if not isinstance(review, dict):
        return None
    return {
        "verdict": review.get("verdict"),
        "summary": review.get("summary"),
        "most_likely_hypothesis": review.get("most_likely_hypothesis"),
        "next_test_rationale": review.get("next_test_rationale"),
        "recommended_action": review.get("recommended_action"),
    }


def _summarize_latest_run_for_prompt(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a compact latest-run summary for the planning prompt."""
    if not isinstance(record, dict):
        return None
    metrics = record.get("metrics", {})
    plausibility = record.get("plausibility", {})
    sweep = record.get("sweep", {})
    review = record.get("codex_visual_review", {})
    return {
        "experiment_name": record.get("experiment_name"),
        "target_step": record.get("target_step"),
        "stage_mean_loss": metrics.get("stage_mean_loss"),
        "relative_stage_improvement": metrics.get("relative_stage_improvement"),
        "plausible": plausibility.get("plausible"),
        "temporal_delta_ratio": plausibility.get("temporal_delta_ratio"),
        "sweep_status": sweep.get("status"),
        "codex_visual_review_verdict": review.get("verdict"),
        "codex_visual_review_summary": review.get("summary"),
        "codex_visual_review_most_likely_hypothesis": review.get("most_likely_hypothesis"),
    }


def _summarize_comparison_context(
    *,
    record: dict[str, Any] | None,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a compact lineage/baseline summary for one stage."""
    if not isinstance(record, dict):
        return None
    experiment_name = str(record.get("experiment_name", "")).strip()
    baseline = {}
    if experiment_name:
        baseline = dict(state.get("comparison_baselines", {}).get(experiment_name, {}))
    baseline_stage_step = record.get("baseline_stage_step", baseline.get("baseline_stage_step"))
    return {
        "parent_stage_step": record.get("parent_stage_step"),
        "stage_kind": record.get("stage_kind"),
        "baseline_stage_step": baseline_stage_step,
        "baseline_locked": bool(
            baseline_stage_step is not None and baseline_stage_step != record.get("target_step")
        ),
        "config_delta_keys": list(record.get("config_delta_from_parent", [])),
    }


def _summarize_recent_history(history: list[dict[str, Any]], *, compact: bool = False) -> list[dict[str, Any]]:
    """Keep only the most recent stage records and their high-signal fields."""
    items: list[dict[str, Any]] = []
    limit = DEFAULT_CONTINUATION_HISTORY_LIMIT if compact else DEFAULT_HISTORY_SUMMARY_LIMIT
    for record in history[-limit:]:
        metrics = record.get("metrics", {})
        plausibility = record.get("plausibility", {})
        codex_visual_review = record.get("codex_visual_review", {})
        item = {
            "timestamp": record.get("timestamp"),
            "experiment_name": record.get("experiment_name"),
            "target_step": record.get("target_step"),
            "last_loss": metrics.get("last_loss"),
            "stage_mean_loss": metrics.get("stage_mean_loss"),
            "relative_stage_improvement": metrics.get("relative_stage_improvement"),
            "plausible": plausibility.get("plausible"),
            "temporal_delta_ratio": plausibility.get("temporal_delta_ratio"),
            "video_flags": plausibility.get("video_flags", []),
            "codex_visual_review_verdict": codex_visual_review.get("verdict"),
            "codex_visual_review_summary": codex_visual_review.get("summary"),
            "codex_visual_review_observations": codex_visual_review.get("observations", []),
            "codex_visual_review_most_likely_hypothesis": codex_visual_review.get("most_likely_hypothesis"),
            "codex_visual_review_next_test_rationale": codex_visual_review.get("next_test_rationale"),
            "learning_summary": record.get("learning_summary"),
        }
        if compact:
            item = {
                "timestamp": item["timestamp"],
                "experiment_name": item["experiment_name"],
                "target_step": item["target_step"],
                "stage_mean_loss": item["stage_mean_loss"],
                "relative_stage_improvement": item["relative_stage_improvement"],
                "plausible": item["plausible"],
                "temporal_delta_ratio": item["temporal_delta_ratio"],
                "codex_visual_review_verdict": item["codex_visual_review_verdict"],
                "codex_visual_review_summary": item["codex_visual_review_summary"],
                "codex_visual_review_most_likely_hypothesis": item["codex_visual_review_most_likely_hypothesis"],
                "codex_visual_review_next_test_rationale": item["codex_visual_review_next_test_rationale"],
            }
        items.append(item)
    return items


def _summarize_decision_history(history: list[dict[str, Any]], *, compact: bool = False) -> list[dict[str, Any]]:
    """Keep only the latest Codex decisions needed for the next turn."""
    limit = DEFAULT_CONTINUATION_DECISION_LIMIT if compact else 5
    items = list(history)[-limit:]
    if not compact:
        return items
    return [
        {
            "timestamp": item.get("timestamp"),
            "action_type": item.get("action_type"),
            "analysis_summary": item.get("analysis_summary"),
        }
        for item in items
    ]


def _summarize_edit_history(history: list[dict[str, Any]], *, compact: bool = False) -> list[dict[str, Any]]:
    """Keep only the latest repo-edit outcomes relevant to planning."""
    limit = DEFAULT_CONTINUATION_EDIT_LIMIT if compact else 5
    items = list(history)[-limit:]
    if not compact:
        return items
    return [
        {
            "timestamp": item.get("timestamp"),
            "edit_id": item.get("edit_id"),
            "applied": item.get("applied"),
            "suspected_root_cause": item.get("suspected_root_cause"),
            "error": item.get("error"),
        }
        for item in items
    ]


def _summarize_latest_recommendation(
    recommendation: dict[str, Any] | None,
    *,
    compact: bool = False,
) -> dict[str, Any] | None:
    """Keep the next-run recommendation concise for resumed Codex turns."""
    if not isinstance(recommendation, dict):
        return recommendation
    if not compact:
        return recommendation
    return {
        "experiment_name": recommendation.get("experiment_name"),
        "current_step": recommendation.get("current_step"),
        "target_step": recommendation.get("target_step"),
        "stage_step": recommendation.get("stage_step"),
        "resume_from": recommendation.get("resume_from"),
        "summary": recommendation.get("summary"),
    }


def _summarize_latest_artifacts(record: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize inspectable artifacts from the latest completed run record."""
    if record is None:
        return {}
    sweep = record.get("sweep", {})
    visual_review = record.get("visual_review", {})
    artifacts: dict[str, Any] = {
        "evaluation_dir": record.get("evaluation_dir"),
        "reference_video": record.get("reference_video"),
    }
    if isinstance(sweep, dict):
        artifacts["generated_video"] = sweep.get("output_path")
        artifacts["comparison_video"] = sweep.get("comparison_output_path")
        artifacts["sweep_status"] = sweep.get("status")
    if isinstance(visual_review, dict):
        artifacts["focus_points"] = visual_review.get("focus_points", [])
    return artifacts


def request_codex_loop_decision(
    *,
    context_bundle: dict[str, Any],
    codex_model: str | None,
    codex_timeout_seconds: int = DEFAULT_CODEX_EXEC_TIMEOUT_SECONDS,
    session_id: str | None = None,
    inspection_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int, CodexExecutionResult]:
    """Ask Codex for the next autonomous-loop action as structured JSON."""
    images = None if inspection_context is None else [Path(item) for item in inspection_context.get("image_inputs", [])]
    response_schema = _loop_decision_schema()
    base_prompt, prompt_debug = _build_codex_decision_prompt(
        context_bundle=context_bundle,
        inspection_context=inspection_context,
    )
    last_error: Exception | None = None
    for attempt in range(1, 3):
        prompt = base_prompt
        debug_metadata = dict(prompt_debug)
        debug_metadata["prompt_chars"] = len(prompt)
        debug_metadata["retry_attempt"] = attempt
        if last_error is not None:
            prompt += (
                "\n\nThe previous response did not validate against the required schema. "
                "Fix the structured output and return a corrected JSON object only. "
                f"Error: {last_error}\n"
                "Use exactly these top-level keys: action_type, analysis_summary, reasoning, next_work_note, "
                "run_experiment, inspect_artifact, apply_repo_edit, stop."
            )
            debug_metadata["prompt_chars"] = len(prompt)
        result = run_codex_exec(
            prompt=prompt,
            schema=response_schema,
            model=codex_model,
            images=images,
            cwd=REPO_ROOT,
            timeout_seconds=codex_timeout_seconds,
            session_id=session_id,
            debug_metadata=debug_metadata,
        )
        try:
            return _validate_loop_decision_payload(result.payload), attempt, result
        except Exception as exc:  # pragma: no cover - retry path exercised in tests
            last_error = exc
            session_id = result.session_id
    assert last_error is not None
    raise last_error


def request_codex_visual_review(
    *,
    record: dict[str, Any],
    codex_model: str | None,
    codex_timeout_seconds: int,
    session_id: str | None = None,
) -> tuple[dict[str, Any], CodexExecutionResult]:
    """Ask Codex to review the latest comparison video and return a structured verdict."""
    comparison_video = Path(str(record["visual_review"]["comparison_video"]))
    inspection_dir = REPO_ROOT / "runs" / "training_optimizer" / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = _build_video_contact_sheet(comparison_video, inspection_dir=inspection_dir)
    review_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "verdict",
            "summary",
            "observations",
            "hypotheses",
            "most_likely_hypothesis",
            "uncertainties",
            "next_test_rationale",
            "focus_points_reviewed",
            "recommended_action",
        ],
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "summary": {"type": "string"},
            "observations": {"type": "array", "items": {"type": "string"}},
            "hypotheses": {"type": "array", "items": {"type": "string"}},
            "most_likely_hypothesis": {"type": "string"},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "next_test_rationale": {"type": "string"},
            "focus_points_reviewed": {"type": "array", "items": {"type": "string"}},
            "recommended_action": {"type": "string"},
        },
    }
    prompt_payload = {
        "experiment_name": record["experiment_name"],
        "target_step": record["target_step"],
        "comparison_layout": record["visual_review"]["comparison_layout"],
        "focus_points": record["visual_review"]["focus_points"],
        "visual_review_summary": record["visual_review"]["summary"],
        "plausibility": record["plausibility"],
        "metrics": {
            "last_loss": record["metrics"]["last_loss"],
            "stage_mean_loss": record["metrics"]["stage_mean_loss"],
            "relative_stage_improvement": record["metrics"].get("relative_stage_improvement"),
        },
    }
    prompt = "\n".join(
        [
            "You are reviewing a side-by-side rollout comparison image derived from the latest comparison video.",
            "The left side is the target/reference and the right side is the generated rollout.",
            "Your job is not just to describe artifacts. Your job is to reason from the video toward the most likely failure cause and the most informative next test.",
            "Follow this process:",
            "1. Start from concrete visual observations only. Describe what is visibly happening in the generated rollout relative to the reference.",
            "2. Separate observations from hypotheses. Do not present a causal explanation as if it were directly observed.",
            "3. Form a small number of plausible hypotheses that could explain the observations.",
            "4. Judge which hypothesis is best supported by the visual evidence alone and which important uncertainties remain.",
            "5. Recommend the next bounded test that would reduce uncertainty the most efficiently. Prefer the next test that most reduces uncertainty about the failure, not the one that sounds most powerful.",
            "6. If the video appears visually acceptable, still explain what evidence supports that conclusion and what uncertainty remains.",
            "Use these failure classes only as anchors, not as a rigid checklist:",
            "- motion path mismatch",
            "- collapse to a common/default pose or scene prior",
            "- temporal drift or late-rollout degradation",
            "- ghosting / instability on moving parts",
            "- contact or kinematic inconsistency",
            "- brightness/color artifacts only if they are actually relevant",
            "Important:",
            "- Start from a concrete observation, then form hypotheses.",
            '- Example: "the arm falls toward the bottom-right corner" is an observation.',
            '- "dataset prior dominates conditioning" is a hypothesis.',
            "- Prefer local, falsifiable next tests over broad expensive retraining.",
            "- A plausibility pass does not outweigh a clear visual motion/control failure.",
            "Return one JSON object only.",
            "",
            "Context JSON:",
            json.dumps(prompt_payload, indent=2, sort_keys=True),
        ]
    )
    result = run_codex_exec(
        prompt=prompt,
        schema=review_schema,
        model=codex_model,
        images=[contact_sheet_path],
        cwd=REPO_ROOT,
        timeout_seconds=codex_timeout_seconds,
        session_id=session_id,
    )
    payload = result.payload
    return {
        "timestamp": _utc_timestamp(),
        "verdict": str(payload.get("verdict", "")).strip().lower(),
        "summary": str(payload.get("summary", "")).strip(),
        "observations": [str(item).strip() for item in payload.get("observations", []) if str(item).strip()],
        "hypotheses": [str(item).strip() for item in payload.get("hypotheses", []) if str(item).strip()],
        "most_likely_hypothesis": str(payload.get("most_likely_hypothesis", "")).strip(),
        "uncertainties": [str(item).strip() for item in payload.get("uncertainties", []) if str(item).strip()],
        "next_test_rationale": str(payload.get("next_test_rationale", "")).strip(),
        "focus_points_reviewed": [str(item).strip() for item in payload.get("focus_points_reviewed", []) if str(item).strip()],
        "recommended_action": str(payload.get("recommended_action", "")).strip(),
        "comparison_video": str(comparison_video),
        "contact_sheet": str(contact_sheet_path),
    }, result


def _build_codex_decision_prompt(
    *,
    context_bundle: dict[str, Any],
    inspection_context: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Render a session-first planning prompt and report its compaction metadata."""
    context_mode = str(context_bundle.get("context_mode", "full"))
    char_budget = (
        DEFAULT_CODEX_CONTINUATION_PROMPT_CHAR_BUDGET
        if context_mode == "continuation"
        else DEFAULT_CODEX_FRESH_PROMPT_CHAR_BUDGET
    )
    prompt_prefix = "\n".join(
        [
            "You are the autonomous experiment planner for this repository.",
            "Human instructions under `next_work` are the highest-priority steering signal.",
            "Choose exactly one action: run_experiment, inspect_artifact, apply_repo_edit, or stop.",
            "Return one JSON object only with these top-level keys: action_type, analysis_summary, reasoning, next_work_note, run_experiment, inspect_artifact, apply_repo_edit, stop.",
            "Always fill all four action payload objects. Use empty/default values for inactive actions.",
            "For run_experiment.overrides, return an array of objects with `key` and `value` fields.",
            "Do not assume you can launch arbitrary long-running commands yourself; the controller owns experiment execution.",
            "Use apply_repo_edit only when repo logic itself appears wrong. Use inspect_artifact when you need more evidence.",
            "Use the latest Codex visual review as primary evidence about whether the branch is actually improving.",
            "Prefer the next test that most reduces uncertainty about the failure, not the one that sounds most powerful.",
            "Start from a concrete observation, then form hypotheses.",
            'Example: "the arm falls toward the bottom-right corner" is an observation.',
            '"dataset prior dominates conditioning" is a hypothesis.',
            "If the latest Codex visual review verdict is `fail`, treat that as stronger evidence than an automated plausibility pass.",
            "Prefer local causes before global ones: train/infer mismatch, data/window bias, conditioning weakness, optimization issues, then horizon/architecture changes.",
            "Use longer horizon only when short-term action following is already correct and the failure mainly appears over longer temporal structure.",
            "Use the listed workspace files as expandable context. Open them only if needed; do not assume every listed file must be reread on this turn.",
        ]
    )
    prompt_sections = _build_codex_prompt_sections(
        context_bundle=context_bundle,
        inspection_context=inspection_context,
    )
    prompt, dropped_sections = _apply_prompt_budget(
        prompt_prefix=prompt_prefix,
        sections=prompt_sections,
        char_budget=char_budget,
    )
    return prompt, {
        "context_mode": context_mode,
        "prompt_char_budget": char_budget,
        "prompt_compaction_mode": "budgeted_sections",
        "dropped_sections": dropped_sections,
        "inspection_context_included": inspection_context is not None,
    }


def _build_codex_prompt_sections(
    *,
    context_bundle: dict[str, Any],
    inspection_context: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Assemble prompt sections in priority order for deterministic compaction."""
    latest_visual_review = context_bundle.get("latest_codex_visual_review")
    latest_run_summary = context_bundle.get("latest_run_summary")
    comparison_context = context_bundle.get("comparison_context")
    sections = [
        {
            "name": "objective",
            "required": True,
            "body": _render_prompt_section(
                "Current Objective",
                {
                    "goal": context_bundle.get("goal"),
                    "training_goal": context_bundle.get("training_goal"),
                    "current_signal": context_bundle.get("current_signal"),
                    "next_work": context_bundle.get("next_work"),
                    "durable_memory_summary": context_bundle.get("durable_memory_summary"),
                },
            ),
        },
        {
            "name": "latest_run",
            "required": latest_run_summary is not None,
            "body": _render_prompt_section("Latest Run", latest_run_summary),
        },
        {
            "name": "latest_visual_review",
            "required": latest_visual_review is not None,
            "body": _render_prompt_section("Latest Codex Visual Review", latest_visual_review),
        },
        {
            "name": "comparison_context",
            "required": comparison_context is not None,
            "body": _render_prompt_section("Comparison Context", comparison_context),
        },
        {
            "name": "workspace_hints",
            "required": True,
            "body": _render_prompt_section("Workspace Hints", context_bundle.get("workspace_hints")),
        },
        {
            "name": "latest_recommendation",
            "required": False,
            "body": _render_prompt_section("Latest Recommendation", context_bundle.get("latest_recommendation")),
        },
        {
            "name": "recent_decisions",
            "required": False,
            "body": _render_prompt_section("Recent Decisions", context_bundle.get("recent_decisions")),
        },
        {
            "name": "recent_edits",
            "required": False,
            "body": _render_prompt_section("Recent Edits", context_bundle.get("recent_edits")),
        },
        {
            "name": "pending_controller_edits",
            "required": False,
            "body": _render_prompt_section(
                "Pending Controller Edits",
                context_bundle.get("pending_controller_edits"),
            ),
        },
        {
            "name": "inspection_context",
            "required": inspection_context is not None,
            "body": ""
            if inspection_context is None
            else _render_prompt_section(
                "Additional Inspection Context",
                inspection_context.get("payload", {}),
            ),
        },
    ]
    return sections


def _render_prompt_section(title: str, payload: Any) -> str:
    """Render one compact prompt section from structured data."""
    if payload in (None, "", [], {}):
        return ""
    if isinstance(payload, str):
        body = payload.strip()
    else:
        body = json.dumps(payload, indent=2, sort_keys=True)
    if not body:
        return ""
    return f"\n\n## {title}\n{body}"


def _apply_prompt_budget(
    *,
    prompt_prefix: str,
    sections: list[dict[str, Any]],
    char_budget: int,
) -> tuple[str, list[str]]:
    """Drop low-priority sections until the prompt fits the configured budget."""
    kept_sections = list(sections)
    dropped_sections: list[str] = []

    def render(current_sections: list[dict[str, Any]]) -> str:
        bodies = [section["body"] for section in current_sections if section.get("body")]
        return prompt_prefix + "".join(bodies)

    prompt = render(kept_sections)
    for section_name in (
        "recent_decisions",
        "recent_edits",
        "pending_controller_edits",
        "latest_recommendation",
        "inspection_context",
    ):
        if len(prompt) <= char_budget:
            break
        for index, section in enumerate(kept_sections):
            if section["name"] != section_name or section.get("required"):
                continue
            dropped_sections.append(section_name)
            del kept_sections[index]
            prompt = render(kept_sections)
            break
    return prompt, dropped_sections


def _loop_decision_schema() -> dict[str, Any]:
    """Return the JSON schema used for Codex planning decisions."""
    run_experiment_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "continue_latest",
            "train_from_scratch",
            "stage_step",
            "overrides",
        ],
        "properties": {
            "continue_latest": {"type": "boolean"},
            "train_from_scratch": {"type": "boolean"},
            "stage_step": {"type": ["integer", "null"]},
            "overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["key", "value"],
                    "properties": {
                        "key": {"type": "string"},
                        "value": {
                            "type": ["string", "number", "boolean", "array", "null"],
                            "items": {"type": ["string", "number", "boolean", "null"]},
                        },
                    },
                },
            },
        },
    }
    inspect_artifact_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "artifact_paths",
            "code_paths",
            "questions",
        ],
        "properties": {
            "artifact_paths": {"type": "array", "items": {"type": "string"}},
            "code_paths": {"type": "array", "items": {"type": "string"}},
            "questions": {"type": "array", "items": {"type": "string"}},
        },
    }
    apply_repo_edit_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "suspected_root_cause",
            "evidence",
            "intended_behavior_change",
            "touched_files",
            "validation_commands",
            "smoke_test_commands",
            "unified_diff",
        ],
        "properties": {
            "suspected_root_cause": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "intended_behavior_change": {"type": "string"},
            "touched_files": {"type": "array", "items": {"type": "string"}},
            "validation_commands": {"type": "array", "items": {"type": "string"}},
            "smoke_test_commands": {"type": "array", "items": {"type": "string"}},
            "unified_diff": {"type": "string"},
        },
    }
    stop_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason"],
        "properties": {
            "reason": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action_type",
            "analysis_summary",
            "reasoning",
            "next_work_note",
            "run_experiment",
            "inspect_artifact",
            "apply_repo_edit",
            "stop",
        ],
        "properties": {
            "action_type": {
                "type": "string",
                "enum": ["run_experiment", "inspect_artifact", "apply_repo_edit", "stop"],
            },
            "analysis_summary": {"type": "string"},
            "reasoning": {"type": "array", "items": {"type": "string"}},
            "next_work_note": {"type": "string"},
            "run_experiment": run_experiment_schema,
            "inspect_artifact": inspect_artifact_schema,
            "apply_repo_edit": apply_repo_edit_schema,
            "stop": stop_schema,
        },
    }


def _validate_loop_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Codex decision payload returned by the CLI."""
    if not isinstance(payload, dict):
        raise ValueError("Codex decision payload must be a JSON object.")
    action_type = str(payload.get("action_type", ""))
    if action_type not in {"run_experiment", "inspect_artifact", "apply_repo_edit", "stop"}:
        raise ValueError(f"Unsupported Codex action_type: {action_type!r}")
    analysis_summary = payload.get("analysis_summary")
    if not isinstance(analysis_summary, str) or not analysis_summary.strip():
        raise ValueError("Codex decision requires a non-empty `analysis_summary`.")
    reasoning = payload.get("reasoning", [])
    if not isinstance(reasoning, list) or not all(isinstance(item, str) for item in reasoning):
        raise ValueError("Codex decision `reasoning` must be a list of strings.")
    normalized = {
        "action_type": action_type,
        "analysis_summary": analysis_summary.strip(),
        "reasoning": [item.strip() for item in reasoning if item.strip()],
        "next_work_note": str(payload.get("next_work_note", "")).strip(),
        "run_experiment": {
            "continue_latest": False,
            "train_from_scratch": False,
            "stage_step": None,
            "overrides": {},
        },
        "inspect_artifact": {
            "artifact_paths": [],
            "code_paths": [],
            "questions": [],
        },
        "apply_repo_edit": {
            "suspected_root_cause": "",
            "evidence": [],
            "intended_behavior_change": "",
            "touched_files": [],
            "validation_commands": [],
            "smoke_test_commands": [],
            "unified_diff": "",
        },
        "stop": {
            "reason": "",
        },
    }
    if action_type == "run_experiment":
        section = payload.get("run_experiment", {})
        if not isinstance(section, dict):
            raise ValueError("Codex `run_experiment` section must be an object.")
        raw_stage_step = section.get("stage_step")
        normalized_stage_step: int | None = None
        if raw_stage_step is not None:
            if isinstance(raw_stage_step, bool) or not isinstance(raw_stage_step, int):
                raise ValueError("Codex `run_experiment.stage_step` must be an integer when provided.")
            if raw_stage_step > 0:
                normalized_stage_step = raw_stage_step
        overrides = section.get("overrides", {})
        if not isinstance(overrides, list):
            raise ValueError("Codex `run_experiment.overrides` must be a list of `{key, value}` entries.")
        override_mapping: dict[str, Any] = {}
        for item in overrides:
            if not isinstance(item, dict):
                raise ValueError("Codex `run_experiment.overrides` entries must be objects.")
            key = item.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Codex `run_experiment.overrides` entries require a non-empty string `key`.")
            override_mapping[key] = item.get("value")
        normalized["run_experiment"] = {
            "continue_latest": bool(section.get("continue_latest", False)),
            "train_from_scratch": bool(section.get("train_from_scratch", False)),
            "stage_step": normalized_stage_step,
            "overrides": override_mapping,
        }
    elif action_type == "inspect_artifact":
        section = payload.get("inspect_artifact", {})
        if not isinstance(section, dict):
            raise ValueError("Codex `inspect_artifact` section must be an object.")
        normalized["inspect_artifact"] = {
            "artifact_paths": [str(item) for item in section.get("artifact_paths", [])],
            "code_paths": [str(item) for item in section.get("code_paths", [])],
            "questions": [str(item) for item in section.get("questions", [])],
        }
    elif action_type == "apply_repo_edit":
        section = payload.get("apply_repo_edit", {})
        if not isinstance(section, dict):
            raise ValueError("Codex `apply_repo_edit` section must be an object.")
        normalized["apply_repo_edit"] = {
            "suspected_root_cause": str(section.get("suspected_root_cause", "")).strip(),
            "evidence": [str(item) for item in section.get("evidence", [])],
            "intended_behavior_change": str(section.get("intended_behavior_change", "")).strip(),
            "touched_files": [str(item) for item in section.get("touched_files", [])],
            "validation_commands": [str(item) for item in section.get("validation_commands", [])],
            "smoke_test_commands": [str(item) for item in section.get("smoke_test_commands", [])],
            "unified_diff": str(section.get("unified_diff", "")),
        }
        if not normalized["apply_repo_edit"]["touched_files"]:
            raise ValueError("Codex repo edits must include at least one touched file.")
        if not normalized["apply_repo_edit"]["validation_commands"]:
            raise ValueError("Codex repo edits must include validation commands.")
        if not normalized["apply_repo_edit"]["smoke_test_commands"]:
            raise ValueError("Codex repo edits must include smoke-test commands.")
        if not normalized["apply_repo_edit"]["unified_diff"].strip():
            raise ValueError("Codex repo edits must include a unified diff.")
    else:
        section = payload.get("stop", {})
        if not isinstance(section, dict):
            raise ValueError("Codex `stop` section must be an object.")
        normalized["stop"] = {
            "reason": str(section.get("reason", "")).strip(),
        }
    return normalized


def _record_codex_decision(state: dict[str, Any], *, decision: dict[str, Any], budget: LoopBudget) -> None:
    """Record the latest Codex decision in controller state."""
    codex_state = dict(state.get("codex_state", {}))
    codex_state.update(
        {
            "last_action_type": decision["action_type"],
            "last_analysis_summary": decision["analysis_summary"],
            "last_reasoning": list(decision["reasoning"]),
            "budget": loop_budget_to_dict(budget),
            "timestamp": _utc_timestamp(),
        }
    )
    state["codex_state"] = codex_state
    _append_limited_history(
        state["decision_history"],
        {
            "timestamp": _utc_timestamp(),
            "action_type": decision["action_type"],
            "analysis_summary": decision["analysis_summary"],
            "reasoning": decision["reasoning"],
            "session_id": state.get("codex_state", {}).get("session_id"),
        },
        limit=DEFAULT_DECISION_HISTORY_LIMIT,
    )


def _update_codex_session_state(
    *,
    state: dict[str, Any],
    result: CodexExecutionResult,
    memory_mode: str,
    codex_model: str | None,
    session_policy: dict[str, Any],
) -> None:
    """Persist the active Codex session metadata after one successful call."""
    codex_state = dict(state.get("codex_state", {}))
    timestamp = _utc_timestamp()
    if result.session_reset_reason is not None:
        codex_state["last_session_reset_reason"] = result.session_reset_reason
    elif session_policy.get("reset_reason") is not None and not result.session_reused:
        codex_state["last_session_reset_reason"] = session_policy["reset_reason"]
    codex_state["memory_mode"] = memory_mode
    codex_state["session_id"] = result.session_id
    codex_state["session_status"] = "resumed" if result.session_reused else "active"
    codex_state["session_cwd"] = str(REPO_ROOT)
    codex_state["session_model"] = codex_model
    codex_state["last_successful_turn_at"] = timestamp
    codex_state["session_turns"] = int(codex_state.get("session_turns", 0)) + 1
    if not result.session_reused or codex_state.get("session_started_at") is None:
        codex_state["session_started_at"] = timestamp
        if result.session_reset_reason is not None:
            codex_state["session_turns"] = 1
    state["codex_state"] = codex_state


def _clear_codex_session_state(state: dict[str, Any], *, reason: str) -> None:
    """Clear persisted session continuity when controller-only memory is requested."""
    codex_state = dict(state.get("codex_state", {}))
    codex_state["session_id"] = None
    codex_state["session_status"] = "disabled"
    codex_state["session_turns"] = 0
    codex_state["last_session_reset_reason"] = reason
    state["codex_state"] = codex_state


def _should_refresh_codex_memory_summary(
    *,
    state: dict[str, Any],
    decision: dict[str, Any],
    memory_mode: str,
) -> bool:
    """Decide whether the durable Codex memory summary should be refreshed now."""
    if memory_mode not in {"hybrid", "session-only"}:
        return False
    if not state.get("codex_memory_summary"):
        return True
    session_turns = int(state.get("codex_state", {}).get("session_turns", 0))
    if decision["action_type"] == "apply_repo_edit":
        return True
    if decision["action_type"] == "inspect_artifact":
        return session_turns % (DEFAULT_CODEX_MEMORY_SUMMARIZE_EVERY_TURNS * 2) == 0
    if decision["action_type"] == "run_experiment":
        return True
    if decision["action_type"] == "stop":
        return False
    return session_turns % DEFAULT_CODEX_MEMORY_SUMMARIZE_EVERY_TURNS == 0


def _refresh_codex_memory_summary(
    *,
    state: dict[str, Any],
    decision: dict[str, Any],
    budget: LoopBudget,
    codex_model: str | None,
    codex_timeout_seconds: int,
    memory_mode: str,
) -> int:
    """Refresh the durable summary used to seed future fresh Codex sessions."""
    if not _should_refresh_codex_memory_summary(state=state, decision=decision, memory_mode=memory_mode):
        return 0
    if budget.codex_calls_used >= budget.max_codex_calls:
        return 0
    session_id = state.get("codex_state", {}).get("session_id")
    if not isinstance(session_id, str) or not session_id:
        state["codex_memory_summary"] = _build_fallback_codex_memory_summary(state=state, decision=decision)
        return 0
    prompt = _build_codex_memory_summary_prompt(state=state, decision=decision, budget=budget)
    summary_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "key_findings", "open_questions", "relevant_paths"],
        "properties": {
            "summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "relevant_paths": {"type": "array", "items": {"type": "string"}},
        },
    }
    result = run_codex_exec(
        prompt=prompt,
        schema=summary_schema,
        model=codex_model,
        cwd=REPO_ROOT,
        timeout_seconds=codex_timeout_seconds,
        session_id=session_id,
    )
    payload = result.payload
    state["codex_memory_summary"] = {
        "summary": str(payload.get("summary", "")).strip(),
        "key_findings": [str(item) for item in payload.get("key_findings", [])],
        "open_questions": [str(item) for item in payload.get("open_questions", [])],
        "relevant_paths": [str(item) for item in payload.get("relevant_paths", [])],
        "timestamp": _utc_timestamp(),
        "session_id": result.session_id,
    }
    _update_codex_session_state(
        state=state,
        result=result,
        memory_mode=memory_mode,
        codex_model=codex_model,
        session_policy={"reset_reason": None},
    )
    return 1


def _build_codex_memory_summary_prompt(
    *,
    state: dict[str, Any],
    decision: dict[str, Any],
    budget: LoopBudget,
) -> str:
    """Build the prompt used to distill durable memory from the live Codex session."""
    summary_schema = {
        "summary": "short paragraph",
        "key_findings": ["durable finding"],
        "open_questions": ["open question"],
        "relevant_paths": ["repo/path.py"],
    }
    prompt_payload = {
        "decision": {
            "action_type": decision.get("action_type"),
            "analysis_summary": decision.get("analysis_summary"),
            "reasoning": decision.get("reasoning", []),
        },
        "existing_summary": state.get("codex_memory_summary", {}),
        "recent_decisions": _summarize_decision_history(
            list(state.get("decision_history", [])),
            compact=True,
        ),
        "recent_edits": _summarize_edit_history(
            list(state.get("edit_history", [])),
            compact=True,
        ),
        "recent_runs": _summarize_recent_history(
            list(state.get("history", [])),
            compact=True,
        ),
        "budget": loop_budget_to_dict(budget),
    }
    return "\n".join(
        [
            "Summarize the durable optimizer memory for future fresh Codex sessions.",
            "Capture only stable conclusions, current blocker, unresolved questions, and the most relevant repo paths.",
            "Return one JSON object only with this shape:",
            json.dumps(summary_schema, indent=2, sort_keys=True),
            "",
            "Context JSON:",
            json.dumps(prompt_payload, indent=2, sort_keys=True),
        ]
    )


def _build_fallback_codex_memory_summary(
    *,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic fallback summary when no live Codex session is available."""
    return {
        "summary": decision.get("analysis_summary", ""),
        "key_findings": list(decision.get("reasoning", []))[:4],
        "open_questions": [],
        "relevant_paths": list(state.get("retrieved_context_cache", {}).get("code", {}).keys())[-5:],
        "timestamp": _utc_timestamp(),
        "session_id": state.get("codex_state", {}).get("session_id"),
    }


def _append_limited_history(history: list[dict[str, Any]], item: dict[str, Any], *, limit: int) -> None:
    """Append one history item and trim the list in place."""
    history.append(item)
    if len(history) > limit:
        del history[:-limit]


def prepare_codex_inspection_context(
    *,
    request: dict[str, Any],
    state: dict[str, Any],
    session_id: str | None = None,
    memory_mode: str = DEFAULT_CODEX_MEMORY_MODE,
) -> dict[str, Any]:
    """Collect artifact and code excerpts for one targeted Codex inspection round."""
    latest_record = state.get("history", [])[-1] if state.get("history") else None
    inspection_dir = REPO_ROOT / "runs" / "training_optimizer" / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    cache = state.setdefault("retrieved_context_cache", {"code": {}, "artifacts": {}})
    code_cache = cache.setdefault("code", {})
    artifact_cache = cache.setdefault("artifacts", {})
    image_inputs: list[str] = []
    artifact_summaries: list[dict[str, Any]] = []
    reused_artifact_paths = 0
    for raw_path in request.get("artifact_paths", []):
        resolved = _resolve_repo_relative_path(raw_path)
        if not resolved.exists():
            artifact_summaries.append({"path": raw_path, "missing": True})
            continue
        artifact_summary: dict[str, Any] = {"path": _display_path(resolved)}
        suffix = resolved.suffix.lower()
        cache_key = _display_path(resolved)
        fingerprint = _build_path_fingerprint(resolved)
        cached_item = artifact_cache.get(cache_key, {})
        already_shared = (
            memory_mode in {"hybrid", "session-only"}
            and isinstance(session_id, str)
            and cached_item.get("session_id") == session_id
            and cached_item.get("fingerprint") == fingerprint
        )
        if suffix in {".png", ".jpg", ".jpeg"}:
            if already_shared:
                artifact_summary["already_shared_in_session"] = True
                reused_artifact_paths += 1
            else:
                image_inputs.append(str(resolved))
        elif suffix in {".mp4", ".mov"}:
            contact_sheet_path = _build_video_contact_sheet(resolved, inspection_dir=inspection_dir)
            artifact_summary["contact_sheet"] = str(contact_sheet_path)
            if already_shared:
                artifact_summary["already_shared_in_session"] = True
                reused_artifact_paths += 1
            else:
                image_inputs.append(str(contact_sheet_path))
        elif suffix in {".json", ".jsonl", ".md", ".txt", ".py", ".yaml", ".yml", ".sh"}:
            if already_shared:
                artifact_summary["already_shared_in_session"] = True
                reused_artifact_paths += 1
            else:
                artifact_summary["excerpt"] = _read_file_excerpt(resolved)
        artifact_cache[cache_key] = {
            "fingerprint": fingerprint,
            "session_id": session_id,
            "timestamp": _utc_timestamp(),
        }
        artifact_summaries.append(artifact_summary)

    code_snippets: list[dict[str, Any]] = []
    reused_code_paths = 0
    for raw_path in request.get("code_paths", []):
        resolved = _resolve_repo_relative_path(raw_path)
        if not resolved.exists():
            continue
        excerpt = _read_file_excerpt(resolved)
        excerpt_hash = _short_hash(excerpt)
        cache_key = _display_path(resolved)
        cached_item = code_cache.get(cache_key, {})
        already_shared = (
            memory_mode in {"hybrid", "session-only"}
            and isinstance(session_id, str)
            and cached_item.get("session_id") == session_id
            and cached_item.get("excerpt_hash") == excerpt_hash
        )
        snippet = {
            "path": cache_key,
            "fingerprint": _build_path_fingerprint(resolved),
        }
        if already_shared:
            snippet["already_shared_in_session"] = True
            reused_code_paths += 1
        else:
            snippet["excerpt"] = excerpt
        code_cache[cache_key] = {
            "excerpt_hash": excerpt_hash,
            "fingerprint": snippet["fingerprint"],
            "session_id": session_id,
            "timestamp": _utc_timestamp(),
        }
        code_snippets.append(snippet)
    payload = {
        "latest_record": None if latest_record is None else {
            "experiment_name": latest_record.get("experiment_name"),
            "target_step": latest_record.get("target_step"),
            "learning_summary": latest_record.get("learning_summary"),
        },
        "artifact_summaries": artifact_summaries,
        "code_snippets": code_snippets,
        "questions": list(request.get("questions", [])),
    }
    return {
        "payload": payload,
        "image_inputs": image_inputs,
        "summary": {
            "timestamp": _utc_timestamp(),
            "artifact_paths": list(request.get("artifact_paths", [])),
            "code_paths": list(request.get("code_paths", [])),
            "image_inputs": image_inputs,
            "reused_code_paths": reused_code_paths,
            "reused_artifact_paths": reused_artifact_paths,
        },
    }


def _build_path_fingerprint(path: Path) -> str:
    """Build a stable fingerprint for one local file path."""
    stat_result = path.stat()
    return f"{stat_result.st_size}:{stat_result.st_mtime_ns}"


def _build_video_contact_sheet(video_path: Path, *, inspection_dir: Path, frame_count: int = 4) -> Path:
    """Render a horizontal contact sheet PNG from a local video path."""
    reader = iio.get_reader(video_path)
    try:
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"Video contains no frames: {video_path}")
    indices = np.linspace(0, len(frames) - 1, min(frame_count, len(frames)), dtype=int)
    selected = [np.asarray(frames[index]) for index in indices]
    contact_sheet = np.concatenate(selected, axis=1)
    output_path = inspection_dir / f"{video_path.stem}_{_short_hash(str(video_path))}_contact_sheet.png"
    iio.imwrite(output_path, contact_sheet)
    return output_path


def _read_file_excerpt(path: Path, *, max_lines: int = 160) -> str:
    """Read a text excerpt with a hard line cap for prompt compaction."""
    lines = path.read_text(encoding="utf-8").splitlines()
    excerpt = lines[:max_lines]
    suffix = "" if len(lines) <= max_lines else "\n... [truncated]"
    return "\n".join(excerpt) + suffix


def build_experiment_plan_from_codex_decision(
    *,
    train_config: TrainScriptConfig,
    decision: dict[str, Any],
    state: dict[str, Any],
    memory_text: str,
    stage_step_override: int | None,
) -> ExperimentPlan:
    """Build an executable experiment plan from a Codex `run_experiment` decision."""
    if decision["action_type"] != "run_experiment":
        raise ValueError("Codex plan building requires a `run_experiment` decision.")
    run_spec = decision["run_experiment"]
    overrides = _coerce_codex_overrides(
        train_config=train_config,
        overrides=run_spec.get("overrides", {}),
    )
    hints = extract_memory_hints(memory_text, train_config=train_config)
    decision_stage_step = run_spec.get("stage_step")
    if isinstance(decision_stage_step, int) and decision_stage_step <= 0:
        decision_stage_step = None
    stage_step = _resolve_stage_step(
        train_config=train_config,
        hints=hints,
        stage_step_override=stage_step_override if stage_step_override is not None else decision_stage_step,
    )
    if run_spec.get("continue_latest") and isinstance(state.get("latest_recommendation"), dict) and not overrides and not run_spec.get("train_from_scratch"):
        return _plan_from_recommendation(
            train_config=train_config,
            recommendation=state["latest_recommendation"],
            hints=hints,
        )
    return _fresh_plan_from_overrides(
        train_config=train_config,
        overrides=overrides,
        stage_step=stage_step,
        reasoning=tuple(decision["reasoning"]) or (decision["analysis_summary"],),
        train_from_scratch=bool(run_spec.get("train_from_scratch", False)),
        history=state.get("history", []),
    )


def _coerce_codex_overrides(
    *,
    train_config: TrainScriptConfig,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Coerce Codex-provided override values to the config field shape."""
    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(train_config, key):
            continue
        default_value = getattr(train_config, key)
        if isinstance(default_value, tuple):
            if isinstance(value, list):
                resolved[key] = tuple(value)
            else:
                resolved[key] = (value,)
            continue
        resolved[key] = value
    return resolved


def apply_codex_repo_edit(
    *,
    proposal: dict[str, Any],
    analysis_summary: str,
) -> dict[str, Any]:
    """Apply one Codex-proposed repo edit and validate it before keeping the change."""
    touched_paths = [_normalize_edit_path(raw_path) for raw_path in proposal["touched_files"]]
    for path in touched_paths:
        if not _is_path_allowed_for_autonomous_edit(path):
            raise ValueError(f"Codex edit path is not allowed: {path}")
    diff_candidates = _build_unified_diff_candidates(proposal["unified_diff"])
    if not diff_candidates:
        raise ValueError("Codex repo edits must include a recognizable unified diff.")
    diff_paths = _extract_unified_diff_paths(diff_candidates[0])
    declared_paths = {_display_path(path) for path in touched_paths}
    if diff_paths != declared_paths:
        raise ValueError(
            "Codex repo edit diff paths do not match declared touched files: "
            f"declared={sorted(declared_paths)} diff={sorted(diff_paths)}"
        )
    snapshots = _snapshot_repo_files(touched_paths)
    edit_id = f"codex_repo_edit_{_short_hash(diff_candidates[0])}"
    repair_attempted = False
    try:
        repair_attempted = _apply_unified_diff_with_repair(diff_candidates)
        _run_validation_commands(proposal["validation_commands"])
        _run_validation_commands(proposal["smoke_test_commands"])
    except Exception as exc:
        _restore_repo_files(snapshots)
        return {
            "edit_id": edit_id,
            "target_file": _display_path(touched_paths[0]),
            "touched_files": [_display_path(path) for path in touched_paths],
            "applied": False,
            "summary": f"Codex repo edit failed validation: {analysis_summary}",
            "reason": proposal["suspected_root_cause"],
            "error": str(exc),
            "validation_commands": proposal["validation_commands"],
            "smoke_test_commands": proposal["smoke_test_commands"],
            "repair_attempted": repair_attempted,
        }
    return {
        "edit_id": edit_id,
        "target_file": _display_path(touched_paths[0]),
        "touched_files": [_display_path(path) for path in touched_paths],
        "applied": True,
        "summary": f"Codex repo edit applied: {analysis_summary}",
        "reason": proposal["suspected_root_cause"],
        "validation_commands": proposal["validation_commands"],
        "smoke_test_commands": proposal["smoke_test_commands"],
        "repair_attempted": repair_attempted,
    }


def _normalize_edit_path(raw_path: str) -> Path:
    """Resolve one edit target relative to the repository root."""
    path = _resolve_repo_relative_path(raw_path)
    if path.exists():
        return path
    resolved = (REPO_ROOT / raw_path).resolve()
    if REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
        raise ValueError(f"Edit path escapes repo root: {raw_path}")
    return resolved


def _is_path_allowed_for_autonomous_edit(path: Path) -> bool:
    """Restrict autonomous edits to repo-tracked source-like files under the repo root."""
    if REPO_ROOT not in path.parents and path != REPO_ROOT:
        return False
    disallowed_roots = (REPO_ROOT / ".venv", REPO_ROOT / "runs")
    if any(root == path or root in path.parents for root in disallowed_roots):
        return False
    if path.is_dir():
        return False
    return _is_repo_tracked_path(path)


def _is_repo_tracked_path(path: Path) -> bool:
    """Return whether a path is tracked by git in the current repository."""
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", _display_path(path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return completed.returncode == 0


def _extract_unified_diff_paths(unified_diff: str) -> set[str]:
    """Collect repo-relative file paths referenced by a unified diff."""
    paths: set[str] = set()
    for line in unified_diff.splitlines():
        if line.startswith("+++ b/"):
            paths.add(line.removeprefix("+++ b/").strip())
            continue
        if line.startswith("diff --git "):
            match = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
            if match is not None:
                paths.add(match.group(2).strip())
    return {path for path in paths if path and path != "/dev/null"}


def _build_unified_diff_candidates(raw_diff: str) -> list[str]:
    """Build cleaned unified-diff candidates from raw Codex output."""
    stripped = raw_diff.strip()
    if not stripped:
        return []
    candidates: list[str] = []
    for candidate in (
        _normalize_unified_diff_text(_extract_fenced_diff_block(stripped) or ""),
        _normalize_unified_diff_text(_extract_patch_region(stripped) or ""),
        _normalize_unified_diff_text(stripped),
    ):
        if not candidate or not _extract_unified_diff_paths(candidate):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _extract_fenced_diff_block(raw_text: str) -> str | None:
    """Extract a fenced diff or patch block from prose-wrapped Codex output."""
    patterns = (
        r"```diff\s*(.*?)```",
        r"```patch\s*(.*?)```",
        r"```(?:[A-Za-z0-9_-]+)?\s*(diff --git .*?)```",
        r"```(?:[A-Za-z0-9_-]+)?\s*(--- .*?)```",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.DOTALL)
        if match is not None:
            return match.group(1).strip()
    return None


def _extract_patch_region(raw_text: str) -> str | None:
    """Extract the likely patch region from prose wrapped around a diff."""
    lines = raw_text.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            start_index = index
            break
    if start_index is None:
        return None
    patch_lines: list[str] = []
    for line in lines[start_index:]:
        if _line_looks_like_patch_content(line):
            patch_lines.append(line)
            continue
        if patch_lines:
            break
    return "\n".join(patch_lines).strip()


def _normalize_unified_diff_text(raw_text: str) -> str:
    """Normalize a unified diff candidate before passing it to `git apply`."""
    stripped = raw_text.strip()
    return stripped + ("\n" if stripped else "")


def _line_looks_like_patch_content(line: str) -> bool:
    """Return whether one line plausibly belongs to a unified diff."""
    return (
        line.startswith("diff --git ")
        or line.startswith("index ")
        or line.startswith("--- ")
        or line.startswith("+++ ")
        or line.startswith("@@")
        or line.startswith("new file mode ")
        or line.startswith("deleted file mode ")
        or line.startswith("old mode ")
        or line.startswith("new mode ")
        or line.startswith("similarity index ")
        or line.startswith("rename from ")
        or line.startswith("rename to ")
        or line.startswith("Binary files ")
        or line.startswith("\\ No newline at end of file")
        or line.startswith("+")
        or line.startswith("-")
        or line.startswith(" ")
    )


def _snapshot_repo_files(paths: list[Path]) -> dict[str, bytes | None]:
    """Snapshot touched repo files so a failed edit can be restored exactly."""
    snapshot: dict[str, bytes | None] = {}
    for path in paths:
        snapshot[str(path)] = path.read_bytes() if path.exists() else None
    return snapshot


def _restore_repo_files(snapshot: dict[str, bytes | None]) -> None:
    """Restore files from an in-memory snapshot after failed validation."""
    for raw_path, payload in snapshot.items():
        path = Path(raw_path)
        if payload is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _apply_unified_diff(unified_diff: str) -> None:
    """Apply a unified diff against the current repo with `git apply`."""
    completed = subprocess.run(
        ["git", "apply", "--recount", "--whitespace=nowarn", "-"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        input=unified_diff,
        capture_output=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git apply failed: stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )


def _apply_unified_diff_with_repair(diff_candidates: list[str]) -> bool:
    """Apply a cleaned Codex diff and report whether fallback repair was needed."""
    last_error: Exception | None = None
    attempted_candidates = diff_candidates[:2]
    for candidate in attempted_candidates:
        try:
            _apply_unified_diff(candidate)
            return candidate != attempted_candidates[0]
        except Exception as exc:
            last_error = exc
    for candidate in attempted_candidates:
        try:
            _apply_text_unified_diff_fallback(candidate)
            return True
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class _ParsedUnifiedHunk:
    """Represent one unified-diff hunk with old/new line payloads."""

    old_start: int
    old_count: int
    old_lines: tuple[str, ...]
    new_lines: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedUnifiedFilePatch:
    """Represent one file patch extracted from a unified diff."""

    path: str
    hunks: tuple[_ParsedUnifiedHunk, ...]


def _apply_text_unified_diff_fallback(unified_diff: str) -> None:
    """Apply simple text hunks by matching old content instead of trusting line numbers."""
    file_patches = _parse_unified_diff_for_fallback(unified_diff)
    if not file_patches:
        raise RuntimeError("No fallback-applicable file patches found.")
    for file_patch in file_patches:
        path = _resolve_repo_relative_path(file_patch.path)
        if not path.exists():
            raise RuntimeError(f"Fallback patch target does not exist: {file_patch.path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        for hunk in file_patch.hunks:
            lines = _apply_hunk_to_lines(lines, hunk)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _parse_unified_diff_for_fallback(unified_diff: str) -> tuple[_ParsedUnifiedFilePatch, ...]:
    """Parse a unified diff into per-file hunks for fallback application."""
    lines = unified_diff.splitlines()
    file_patches: list[_ParsedUnifiedFilePatch] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise RuntimeError("Unified diff fallback parser found malformed file header.")
        new_path = lines[index + 1].removeprefix("+++ ").strip()
        path = _normalize_edit_diff_path(new_path)
        index += 2
        hunks: list[_ParsedUnifiedHunk] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index].startswith("@@"):
                index += 1
                continue
            header = lines[index]
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
            if match is None:
                raise RuntimeError(f"Unsupported unified diff hunk header: {header}")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            index += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("--- "):
                payload_line = lines[index]
                if payload_line.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not payload_line:
                    raise RuntimeError("Fallback unified diff parser encountered an empty payload line.")
                prefix = payload_line[0]
                content = payload_line[1:]
                if prefix == " ":
                    old_lines.append(content)
                    new_lines.append(content)
                elif prefix == "-":
                    old_lines.append(content)
                elif prefix == "+":
                    new_lines.append(content)
                else:
                    raise RuntimeError(f"Unsupported unified diff payload line: {payload_line}")
                index += 1
            hunks.append(
                _ParsedUnifiedHunk(
                    old_start=old_start,
                    old_count=old_count,
                    old_lines=tuple(old_lines),
                    new_lines=tuple(new_lines),
                )
            )
        file_patches.append(_ParsedUnifiedFilePatch(path=path, hunks=tuple(hunks)))
    return tuple(file_patches)


def _normalize_edit_diff_path(raw_path: str) -> str:
    """Normalize one diff header path into a repo-relative file path."""
    path = raw_path.strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    if not path or path == "/dev/null":
        raise RuntimeError(f"Fallback patch path is not a supported repo file: {raw_path}")
    return path


def _apply_hunk_to_lines(lines: list[str], hunk: _ParsedUnifiedHunk) -> list[str]:
    """Apply one parsed hunk to a list of text lines using content matching."""
    old_lines = list(hunk.old_lines)
    new_lines = list(hunk.new_lines)
    if not old_lines:
        insert_at = max(0, min(len(lines), hunk.old_start - 1))
        return lines[:insert_at] + new_lines + lines[insert_at:]
    match_index = _find_best_hunk_match(lines, old_lines, hint_index=max(0, hunk.old_start - 1))
    if match_index is None:
        raise RuntimeError(
            "Fallback patch application could not match hunk old content in target file "
            f"(old_start={hunk.old_start}, old_count={hunk.old_count})."
        )
    return lines[:match_index] + new_lines + lines[match_index + len(old_lines) :]


def _find_best_hunk_match(lines: list[str], needle: list[str], *, hint_index: int) -> int | None:
    """Find the best content match for one hunk, preferring positions near the header hint."""
    matches: list[int] = []
    max_index = len(lines) - len(needle) + 1
    for index in range(max(0, max_index)):
        if lines[index : index + len(needle)] == needle:
            matches.append(index)
    if not matches:
        return None
    return min(matches, key=lambda index: abs(index - hint_index))


def _run_validation_commands(commands: list[str]) -> None:
    """Run repo-edit validation commands inside the repo virtualenv."""
    for command in commands:
        _run_shell_in_venv(command)


def _run_shell_in_venv(command: str) -> None:
    """Run one shell command inside the repo virtualenv."""
    completed = subprocess.run(
        ["bash", "-lc", f"source .venv/bin/activate && {command}"],
        cwd=REPO_ROOT,
        check=False,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Validation command failed: {command}\nstdout={completed.stdout.strip()}\nstderr={completed.stderr.strip()}"
        )


def _resolve_repo_relative_path(raw_path: str | Path) -> Path:
    """Resolve a repo-relative or absolute path and keep it inside the repo root."""
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
    if REPO_ROOT not in resolved.parents and resolved != REPO_ROOT:
        raise ValueError(f"Path escapes repo root: {raw_path}")
    return resolved


def update_memory_with_codex_analysis_file(
    memory_path: str | Path,
    *,
    decision: dict[str, Any],
    controller_edits: list[dict[str, Any]] | None = None,
) -> None:
    """Write the latest Codex analysis and optional edit results into markdown memory."""
    path = Path(memory_path)
    updated = update_memory_with_codex_analysis(
        path.read_text(encoding="utf-8") if path.exists() else "",
        decision=decision,
        controller_edits=controller_edits or [],
    )
    path.write_text(updated, encoding="utf-8")


def update_memory_with_codex_analysis(
    memory_text: str,
    *,
    decision: dict[str, Any],
    controller_edits: list[dict[str, Any]],
) -> str:
    """Update markdown memory with one Codex analysis entry and any edit results."""
    preamble, sections, order = parse_markdown_sections(memory_text)
    timestamp = _utc_timestamp()
    analysis_bullet = _build_codex_analysis_bullet(timestamp=timestamp, analysis=decision)
    sections["Codex Analysis"] = _prepend_controller_bullet(
        sections.get("Codex Analysis", ""),
        analysis_bullet,
    )
    if controller_edits:
        entries = "\n\n".join(
            _format_controller_edit_entry(timestamp=timestamp, edit=edit)
            for edit in controller_edits
        )
        sections["Controller Edits"] = _append_section_entry(
            sections.get("Controller Edits", ""),
            entries,
            leading_blank_line=True,
        )
    final_order = list(order)
    for heading in ("Codex Analysis", "Controller Edits"):
        if heading not in final_order:
            final_order.append(heading)
    return render_markdown_sections(preamble, sections, final_order)


@contextmanager
def _controller_loop_lock(state_path: Path):
    """Prevent concurrent autonomous loops from mutating the same optimizer state."""
    lock_path = state_path.with_suffix(".lock")
    lock_payload = {"pid": os.getpid(), "timestamp": _utc_timestamp()}
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(existing.get("pid", 0))
        timestamp = str(existing.get("timestamp", ""))
        if pid and _pid_is_running(pid):
            raise RuntimeError(f"Another optimizer loop is active (pid={pid}, timestamp={timestamp}).")
        if pid and not _pid_is_running(pid):
            lock_path.unlink()
        elif timestamp:
            lock_age = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
            if lock_age < DEFAULT_LOOP_LOCK_TIMEOUT_SECONDS:
                raise RuntimeError(f"Optimizer loop lock already exists: {lock_path}")
            lock_path.unlink()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        yield
    finally:
        if lock_path.exists():
            lock_path.unlink()


def _pid_is_running(pid: int) -> bool:
    """Return whether a local process identifier is currently alive."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _short_hash(text: str) -> str:
    """Return a short stable hash for filenames and edit identifiers."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _log_controller_status(message: str) -> None:
    """Emit a flushed status line for local controller activity."""
    print(f"[training-optimizer] {message}", flush=True)


def _attach_codex_visual_review_to_record(
    *,
    record: dict[str, Any],
    state: dict[str, Any],
    budget: LoopBudget,
    codex_model: str | None,
    codex_timeout_seconds: int,
    memory_mode: str,
    session_policy: dict[str, Any],
) -> int:
    """Run Codex visual review for one completed stage record and attach it in place."""
    if isinstance(record.get("codex_visual_review"), dict):
        return 0
    if budget.codex_calls_used >= budget.max_codex_calls:
        return 0
    session_id = session_policy["session_id"] if session_policy.get("reuse_session") else None
    review, result = request_codex_visual_review(
        record=record,
        codex_model=codex_model,
        codex_timeout_seconds=codex_timeout_seconds,
        session_id=session_id,
    )
    record["codex_visual_review"] = review
    stage_record_path = Path(str(record["output_dir"])) / f"controller_stage_{int(record['target_step']):07d}.json"
    if stage_record_path.exists():
        stage_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _update_codex_session_state(
        state=state,
        result=result,
        memory_mode=memory_mode,
        codex_model=codex_model,
        session_policy=session_policy,
    )
    _log_controller_status(
        f"Codex visual review for {record['experiment_name']} step {record['target_step']}: "
        f"{review['verdict']} | {review['summary']}"
    )
    return 1


def _ensure_latest_codex_visual_review(
    *,
    state: dict[str, Any],
    memory_path: Path,
    budget: LoopBudget,
    codex_model: str | None,
    codex_timeout_seconds: int,
    memory_mode: str,
    session_policy: dict[str, Any],
) -> int:
    """Backfill a missing Codex visual review for the latest completed stage before planning."""
    pending = _latest_stage_requires_codex_visual_review(state)
    if pending is None:
        return 0
    history = list(state.get("history", []))
    if not history:
        return 0
    latest_record = history[-1]
    call_count = _attach_codex_visual_review_to_record(
        record=latest_record,
        state=state,
        budget=budget,
        codex_model=codex_model,
        codex_timeout_seconds=codex_timeout_seconds,
        memory_mode=memory_mode,
        session_policy=session_policy,
    )
    if call_count:
        update_memory_file(memory_path, latest_record)
        updated = append_stage_record(state, latest_record)
        state.clear()
        state.update(updated)
    return call_count


def update_memory_markdown(memory_text: str, *, record: dict[str, Any]) -> str:
    """Append concise controller findings and the next recommendation to markdown memory."""
    preamble, sections, order = parse_markdown_sections(memory_text)
    timestamp = record["timestamp"]
    metrics = record["metrics"]
    plausibility = record["plausibility"]
    recommendation = record["next_recommendation"]
    controller_edits = record.get("controller_edits", [])
    codex_analysis = record.get("codex_analysis")
    codex_visual_review = record.get("codex_visual_review")
    current_signal_bullet = (
        f"{CONTROLLER_BULLET_PREFIX}{timestamp}] "
        f"`{record['experiment_name']}` reached step `{record['target_step']}` with "
        f"last loss `{metrics['last_loss']:.6f}` and stage mean loss `{metrics['stage_mean_loss']:.6f}`; "
        f"plausibility is `{'PASS' if plausibility['plausible'] else 'FAIL'}` "
        f"(mean MAE `{plausibility['mean_frame_mae_rgb_0_255']:.3f}`, "
        f"temporal delta ratio `{plausibility['temporal_delta_ratio']:.3f}`)."
    )
    next_work_bullet = (
        f"{CONTROLLER_BULLET_PREFIX}{timestamp}] "
        f"Next experiment: {recommendation['summary']}"
    )
    training_run_entry = _format_training_run_entry(record)

    sections["Current Signal"] = _append_section_entry(
        sections.get("Current Signal", ""),
        current_signal_bullet,
    )
    sections["Next Work"] = _prepend_controller_bullet(
        sections.get("Next Work", ""),
        next_work_bullet,
    )
    sections["Training runs"] = _append_section_entry(
        sections.get("Training runs", ""),
        training_run_entry,
        leading_blank_line=True,
    )
    if isinstance(codex_visual_review, dict):
        sections[CODEX_VISUAL_REVIEW_SECTION] = _append_section_entry(
            sections.get(CODEX_VISUAL_REVIEW_SECTION, ""),
            _build_codex_visual_review_bullet(record),
        )
    if isinstance(codex_analysis, dict):
        sections["Codex Analysis"] = _prepend_controller_bullet(
            sections.get("Codex Analysis", ""),
            _build_codex_analysis_bullet(timestamp=timestamp, analysis=codex_analysis),
        )
    if controller_edits:
        controller_edit_entries = "\n\n".join(
            _format_controller_edit_entry(timestamp=timestamp, edit=edit)
            for edit in controller_edits
        )
        sections["Controller Edits"] = _append_section_entry(
            sections.get("Controller Edits", ""),
            controller_edit_entries,
            leading_blank_line=True,
        )

    final_order = list(order)
    for heading in ("Current Signal", "Next Work", "Training runs", CODEX_VISUAL_REVIEW_SECTION, "Codex Analysis", "Controller Edits"):
        if heading not in final_order:
            final_order.append(heading)
    return render_markdown_sections(preamble, sections, final_order)


def _extract_stage_step_from_checkpoint_path(checkpoint_path: str | Path | None) -> int | None:
    """Parse one `step_XXXXXXX.pt` checkpoint path into its integer step."""
    if checkpoint_path is None:
        return None
    match = re.search(r"step_(\d+)\.pt$", str(checkpoint_path))
    if match is None:
        return None
    return int(match.group(1))


def _find_record_for_experiment_step(
    history: list[dict[str, Any]],
    *,
    experiment_name: str,
    target_step: int | None,
) -> dict[str, Any] | None:
    """Find one exact stage record for an experiment and target step."""
    if target_step is None or target_step <= 0:
        return None
    for item in history:
        if item.get("experiment_name") != experiment_name:
            continue
        if int(item.get("target_step", 0)) == target_step:
            return item
    return None


def _config_delta_from_parent(
    *,
    parent_record: dict[str, Any] | None,
    resolved_config: dict[str, Any],
) -> list[str]:
    """List only the resolved-config keys that changed from the parent stage."""
    if not isinstance(parent_record, dict):
        return []
    parent_plan = parent_record.get("plan", {})
    parent_resolved = parent_plan.get("resolved_config", {})
    if not isinstance(parent_resolved, dict):
        return []
    delta_keys: list[str] = []
    for key in sorted(set(parent_resolved) | set(resolved_config)):
        if not _values_match(_json_ready(parent_resolved.get(key)), _json_ready(resolved_config.get(key))):
            delta_keys.append(key)
    return delta_keys


def _classify_stage_kind(*, parent_stage_step: int | None, config_delta_keys: list[str]) -> str:
    """Classify one stage as a fresh start, continuation, or diagnostic branch."""
    if parent_stage_step is None:
        return "fresh_start"
    if any(key in DIAGNOSTIC_CONFIG_KEYS for key in config_delta_keys):
        return "diagnostic"
    return "continuation"


def _comparison_video_from_record(record: dict[str, Any] | None) -> str | None:
    """Return the preferred comparison-video path stored on one record."""
    if not isinstance(record, dict):
        return None
    visual_review = record.get("visual_review", {})
    if isinstance(visual_review, dict) and visual_review.get("comparison_video"):
        return str(visual_review["comparison_video"])
    sweep = record.get("sweep", {})
    if isinstance(sweep, dict) and sweep.get("comparison_output_path"):
        return str(sweep["comparison_output_path"])
    return None


def _resolve_stage_baseline(
    *,
    state: dict[str, Any],
    experiment_name: str,
    stage_kind: str,
    target_step: int,
    comparison_video: Path,
) -> tuple[int | None, str | None]:
    """Resolve the pinned baseline for one stage without auto-promoting diagnostics."""
    baselines = state.get("comparison_baselines", {})
    existing = baselines.get(experiment_name, {}) if isinstance(baselines, dict) else {}
    existing_step = existing.get("baseline_stage_step")
    existing_video = existing.get("baseline_comparison_video")
    if existing_step is not None and existing_video:
        return int(existing_step), str(existing_video)

    history = list(state.get("history", []))
    prior_non_diagnostic = [
        item
        for item in history
        if item.get("experiment_name") == experiment_name and item.get("stage_kind") != "diagnostic"
    ]
    if prior_non_diagnostic:
        baseline_record = sorted(prior_non_diagnostic, key=lambda item: int(item.get("target_step", 0)))[-1]
        return int(baseline_record.get("target_step", 0)), _comparison_video_from_record(baseline_record)
    if stage_kind != "diagnostic":
        return target_step, str(comparison_video)
    return None, None


def _build_stage_comparison_metadata(
    *,
    plan: ExperimentPlan,
    state: dict[str, Any],
    comparison_video: Path,
) -> dict[str, Any]:
    """Build compact lineage and baseline metadata for one completed stage."""
    history = list(state.get("history", []))
    parent_stage_step = _extract_stage_step_from_checkpoint_path(plan.resume_from)
    parent_record = _find_record_for_experiment_step(
        history,
        experiment_name=plan.experiment_name,
        target_step=parent_stage_step,
    )
    config_delta_keys = _config_delta_from_parent(
        parent_record=parent_record,
        resolved_config=plan.resolved_config,
    )
    stage_kind = _classify_stage_kind(
        parent_stage_step=parent_stage_step,
        config_delta_keys=config_delta_keys,
    )
    baseline_stage_step, baseline_comparison_video = _resolve_stage_baseline(
        state=state,
        experiment_name=plan.experiment_name,
        stage_kind=stage_kind,
        target_step=plan.target_step,
        comparison_video=comparison_video,
    )
    return {
        "parent_stage_step": parent_stage_step,
        "parent_checkpoint_path": None if plan.resume_from is None else str(plan.resume_from),
        "config_delta_from_parent": config_delta_keys,
        "stage_kind": stage_kind,
        "baseline_stage_step": baseline_stage_step,
        "baseline_comparison_video": baseline_comparison_video,
    }


def run_experiment_stage(
    *,
    plan: ExperimentPlan,
    train_config_path: Path,
    memory_path: Path,
    state: dict[str, Any],
    eval_episode_index: int,
    eval_start_frame: int,
    reference_frame_offset: int | None,
    reference_video: Path | None,
    allow_policy_self_edits: bool = True,
    extra_controller_edits: list[dict[str, Any]] | None = None,
    codex_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one controller stage, then analyze and persist the result."""
    output_dir = plan.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_record_path = output_dir / f"controller_stage_{plan.target_step:07d}.json"
    if stage_record_path.exists():
        return json.loads(stage_record_path.read_text(encoding="utf-8"))

    checkpoint_path = output_dir / "checkpoints" / f"step_{plan.target_step:07d}.pt"
    evaluation_dir = (
        REPO_ROOT
        / "runs"
        / "training_optimizer"
        / "eval"
        / f"{plan.experiment_name}_step_{plan.target_step:07d}"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    train_command = build_train_command(plan=plan, train_config_path=train_config_path)
    if not _stage_is_valid(output_dir=output_dir, target_step=plan.target_step):
        _run_command(train_command)

    validate_training_stage(output_dir, expected_step=plan.target_step)

    sweep_command = build_sweep_command(
        plan=plan,
        checkpoint_path=checkpoint_path,
        evaluation_dir=evaluation_dir,
        eval_episode_index=eval_episode_index,
        eval_start_frame=eval_start_frame,
    )
    if not list(evaluation_dir.glob("*_summary.json")):
        _run_command(sweep_command)

    resolved_config = plan.resolved_config
    reference_path = ensure_reference_video(
        repo_id=str(resolved_config["repo_id"]),
        video_key=str(resolved_config["video_key"]),
        episode_index=eval_episode_index,
        frame_offset=eval_start_frame if reference_frame_offset is None else reference_frame_offset,
        num_frames=int(resolved_config["context_len"]) + int(resolved_config["horizon_len"]),
        reference_video=reference_video,
    )
    sweep_item = _load_sweep_summary_item(evaluation_dir)
    generated_video = Path(str(sweep_item["output_path"]))
    comparison_video = Path(str(sweep_item["comparison_output_path"]))
    plausibility_path = evaluation_dir / "plausibility_report.json"
    plausibility_command = build_plausibility_command(
        reference_video=reference_path,
        generated_video=generated_video,
        output_json=plausibility_path,
    )
    if not plausibility_path.exists():
        _run_command(plausibility_command)

    metrics_rows = load_metrics_rows(output_dir / "metrics.jsonl")
    previous_record = _latest_record_for_experiment(
        state.get("history", []),
        experiment_name=plan.experiment_name,
        target_step_lt=plan.target_step,
    )
    previous_stage_mean_loss = None if previous_record is None else float(previous_record["metrics"]["stage_mean_loss"])
    metrics = summarize_metrics_rows(
        metrics_rows=metrics_rows,
        previous_step=plan.current_step,
        target_step=plan.target_step,
        previous_stage_mean_loss=previous_stage_mean_loss,
    )
    plausibility = _load_plausibility_summary(plausibility_path)
    score = compute_stage_score(metrics=metrics, plausibility=plausibility, sweep_item=sweep_item)
    memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
    controller_edits = list(extra_controller_edits or [])
    if allow_policy_self_edits:
        controller_edits.extend(
            maybe_apply_controller_edits(
                history=state.get("history", []),
                experiment_name=plan.experiment_name,
                metrics=metrics,
                plausibility=plausibility,
                sweep_item=sweep_item,
                memory_text=memory_text,
            )
        )
    learning_summary = build_learning_summary(
        plan=plan,
        metrics=metrics,
        plausibility=plausibility,
        sweep_item=sweep_item,
    )
    visual_review = build_visual_review_summary(
        experiment_name=plan.experiment_name,
        comparison_video=comparison_video,
        generated_video=generated_video,
    )
    comparison_metadata = _build_stage_comparison_metadata(
        plan=plan,
        state=state,
        comparison_video=comparison_video,
    )
    base_train_config = load_train_config(train_config_path)
    memory_hints = extract_memory_hints(
        memory_text,
        train_config=base_train_config,
    )
    next_plan = recommend_next_experiment(
        base_train_config=base_train_config,
        current_plan=plan,
        metrics=metrics,
        plausibility=plausibility,
        sweep_item=sweep_item,
        memory_hints=memory_hints,
    )

    record = {
        "timestamp": _utc_timestamp(),
        "experiment_name": plan.experiment_name,
        "output_dir": str(output_dir),
        "target_step": plan.target_step,
        "current_step": plan.current_step,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_dir": str(evaluation_dir),
        "reference_video": str(reference_path),
        "commands": {
            "train": train_command,
            "sweep": sweep_command,
            "plausibility": plausibility_command,
        },
        "plan": experiment_plan_to_dict(plan),
        "metrics": metrics_summary_to_dict(metrics),
        "plausibility": plausibility_summary_to_dict(plausibility),
        "sweep": sweep_item,
        "score": score,
        "learning_summary": learning_summary,
        "controller_policy": dict(CONTROLLER_POLICY),
        "controller_edits": controller_edits,
        "visual_review": visual_review,
        "next_recommendation": experiment_plan_to_recommendation_dict(next_plan),
        **comparison_metadata,
    }
    if codex_analysis is not None:
        record["codex_analysis"] = dict(codex_analysis)
    stage_record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(learning_summary)
    for edit in controller_edits:
        print(f"Controller edit: {edit['summary']}")
    print(f"Visual review: {visual_review['summary']}")
    print(f"Next experiment: {record['next_recommendation']['summary']}")
    return record


def parse_markdown_sections(text: str) -> tuple[str, dict[str, str], list[str]]:
    """Split markdown into preamble plus top-level `##` sections."""
    preamble_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current_heading: str | None = None

    for line in text.splitlines():
        heading_match = re.match(r"^##\s+(?P<heading>.+?)\s*$", line)
        if heading_match is not None:
            current_heading = heading_match.group("heading")
            order.append(current_heading)
            sections[current_heading] = []
            continue
        if current_heading is None:
            preamble_lines.append(line)
        else:
            sections[current_heading].append(line)

    return (
        "\n".join(preamble_lines).strip("\n"),
        {heading: "\n".join(lines).strip("\n") for heading, lines in sections.items()},
        order,
    )


def render_markdown_sections(preamble: str, sections: dict[str, str], order: list[str]) -> str:
    """Render markdown from a preamble plus ordered top-level sections."""
    parts: list[str] = []
    if preamble:
        parts.append(preamble.strip("\n"))
    for heading in order:
        body = sections.get(heading, "").strip("\n")
        parts.append(f"## {heading}")
        if body:
            parts.append(body)
    return "\n\n".join(parts).rstrip() + "\n"


def load_controller_state(path: str | Path) -> dict[str, Any]:
    """Load the controller state JSON, returning an empty state when absent."""
    state_path = Path(path)
    if not state_path.exists():
        return _normalize_controller_state({})
    return _normalize_controller_state(json.loads(state_path.read_text(encoding="utf-8")))


def save_controller_state(path: str | Path, state: dict[str, Any]) -> None:
    """Persist controller state JSON to disk."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(_normalize_controller_state(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_stage_record(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Append or replace one stage record in controller state."""
    updated = _normalize_controller_state(state)
    history = list(updated.get("history", []))
    history = [
        item
        for item in history
        if not (
            item.get("experiment_name") == record.get("experiment_name")
            and int(item.get("target_step", -1)) == int(record.get("target_step", -2))
        )
    ]
    history.append(record)
    updated["history"] = history
    baselines = dict(updated.get("comparison_baselines", {}))
    experiment_name = str(record.get("experiment_name", "")).strip()
    if experiment_name and experiment_name not in baselines:
        baseline_stage_step = record.get("baseline_stage_step")
        baseline_comparison_video = record.get("baseline_comparison_video")
        if baseline_stage_step is not None and baseline_comparison_video:
            baselines[experiment_name] = {
                "baseline_stage_step": int(baseline_stage_step),
                "baseline_comparison_video": str(baseline_comparison_video),
            }
    updated["comparison_baselines"] = baselines
    updated["latest_recommendation"] = record["next_recommendation"]
    updated["latest_record"] = {
        "experiment_name": record["experiment_name"],
        "target_step": record["target_step"],
        "score": record["score"],
    }
    return updated


def _latest_stage_requires_codex_visual_review(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the latest stage info when Codex has not yet reviewed the comparison video."""
    history = list(state.get("history", []))
    if not history:
        return None
    latest_record = history[-1]
    experiment_name = str(latest_record.get("experiment_name", ""))
    target_step = int(latest_record.get("target_step", 0))
    if not experiment_name or target_step <= 0:
        return None
    if isinstance(latest_record.get("codex_visual_review"), dict):
        return None
    return {
        "experiment_name": experiment_name,
        "target_step": target_step,
    }


def _latest_stage_codex_visual_gate(
    *,
    memory_text: str,
    state: dict[str, Any],
) -> str | None:
    """Return the current Codex-visual-review gate message for the latest completed stage."""
    pending = _latest_stage_requires_codex_visual_review(state)
    if pending is not None:
        return (
            "Codex visual review required for "
            f"{pending['experiment_name']} step {pending['target_step']}"
        )
    return None


def find_codex_visual_review(
    memory_text: str,
    *,
    experiment_name: str,
    target_step: int,
) -> dict[str, Any] | None:
    """Find the latest Codex visual-review verdict for one experiment stage from markdown."""
    _, sections, _ = parse_markdown_sections(memory_text)
    reviews_text = sections.get(CODEX_VISUAL_REVIEW_SECTION, "")
    pattern = re.compile(
        r"^- \[controller (?P<timestamp>[^\]]+)\]\s+(?P<experiment>\S+)\s+step\s+(?P<step>\d+):\s*"
        r"(?P<verdict>pass|fail)\s*\|\s*(?P<summary>.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    latest_match: dict[str, Any] | None = None
    for match in pattern.finditer(reviews_text):
        if match.group("experiment") != experiment_name:
            continue
        if int(match.group("step")) != target_step:
            continue
        latest_match = {
            "timestamp": match.group("timestamp").strip(),
            "experiment_name": match.group("experiment"),
            "target_step": int(match.group("step")),
            "verdict": match.group("verdict").lower(),
            "summary": match.group("summary").strip(),
        }
    return latest_match


def update_memory_file(memory_path: str | Path, record: dict[str, Any]) -> None:
    """Load, update, and rewrite the human-readable optimization markdown."""
    path = Path(memory_path)
    updated = update_memory_markdown(
        path.read_text(encoding="utf-8") if path.exists() else "",
        record=record,
    )
    path.write_text(updated, encoding="utf-8")


def build_train_command(*, plan: ExperimentPlan, train_config_path: Path) -> list[str]:
    """Build the canonical training command for one staged experiment."""
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "world_model.py"),
        "--config",
        str(train_config_path),
        "--output-dir",
        str(plan.output_dir),
        "--max-steps",
        str(plan.target_step),
    ]
    if plan.resume_from is not None:
        command.extend(["--resume-from", str(plan.resume_from)])
    command.extend(_overrides_to_cli_args(plan.overrides))
    return command


def build_sweep_command(
    *,
    plan: ExperimentPlan,
    checkpoint_path: Path,
    evaluation_dir: Path,
    eval_episode_index: int,
    eval_start_frame: int,
) -> list[str]:
    """Build the canonical checkpoint sweep command for one validated stage."""
    cfg = plan.resolved_config
    frame_width = int(cfg["frame_width"])
    frame_height = int(cfg["frame_height"])
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            "Controller evaluation requires a concrete frame_width/frame_height. "
            "Pass an explicit resolution in the train config or memory notes."
        )
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check" / "sweep_local_repo_resolutions.py"),
        "--mode",
        "checkpoint",
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(evaluation_dir),
        "--repo-id",
        str(cfg["repo_id"]),
        "--episode-index",
        str(eval_episode_index),
        "--start-frame",
        str(eval_start_frame),
        "--video-key",
        str(cfg["video_key"]),
        "--context-len",
        str(cfg["context_len"]),
        "--horizon-len",
        str(cfg["horizon_len"]),
        "--k",
        str(cfg["k"]),
        "--resolutions",
        f"{frame_width}x{frame_height}",
    ]


def build_plausibility_command(
    *,
    reference_video: Path,
    generated_video: Path,
    output_json: Path,
) -> list[str]:
    """Build the canonical plausibility-check command for a generated video."""
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check" / "check_generated_video_plausibility.py"),
        "--reference-video",
        str(reference_video),
        "--generated-video",
        str(generated_video),
        "--resize-reference",
        "--output-json",
        str(output_json),
    ]


def ensure_reference_video(
    *,
    repo_id: str,
    video_key: str,
    episode_index: int,
    frame_offset: int,
    num_frames: int,
    reference_video: Path | None,
) -> Path:
    """Resolve or create the reference preview clip used by plausibility checks."""
    if reference_video is not None:
        if not reference_video.exists():
            raise FileNotFoundError(f"Reference video not found: {reference_video}")
        return reference_video

    if "aloha" not in repo_id.lower():
        raise ValueError(
            "Auto-generated reference previews are only implemented for ALOHA datasets. "
            "Pass --reference-video for other datasets."
        )

    preview_dir = (
        REPO_ROOT
        / "runs"
        / "training_optimizer"
        / "reference"
        / f"{_slugify(repo_id.split('/')[-1])}_ep{episode_index}_start{frame_offset}_frames{num_frames}"
    )
    preview_path = preview_dir / "preview.mp4"
    if preview_path.exists():
        return preview_path

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check" / "preview_aloha_sequence.py"),
        "--repo-id",
        repo_id,
        "--episode-index",
        str(episode_index),
        "--frame-offset",
        str(frame_offset),
        "--video-key",
        video_key,
        "--num-frames",
        str(num_frames),
        "--output-dir",
        str(preview_dir),
    ]
    _run_command(command)
    if not preview_path.exists():
        raise FileNotFoundError(f"Reference preview was not created at {preview_path}")
    return preview_path


def recommend_next_experiment(
    *,
    base_train_config: TrainScriptConfig,
    current_plan: ExperimentPlan,
    metrics: MetricsSummary,
    plausibility: PlausibilitySummary,
    sweep_item: dict[str, Any],
    memory_hints: MemoryHints,
) -> ExperimentPlan:
    """Recommend the next conservative experiment after analyzing one stage."""
    if str(sweep_item.get("status")) != "ok" or not plausibility.plausible:
        fallback_overrides, fallback_reason = _build_fallback_overrides(
            base_train_config=base_train_config,
            current_plan=current_plan,
            memory_hints=memory_hints,
        )
        if fallback_overrides is not None:
            return _fresh_plan_from_overrides(
                train_config=base_train_config,
                overrides=fallback_overrides,
                stage_step=current_plan.stage_step,
                reasoning=(fallback_reason,),
                train_from_scratch=True,
                history=[],
            )

    improvement_threshold = max(
        float(CONTROLLER_POLICY["improvement_threshold_floor"]),
        float(base_train_config.auto_stop_min_relative_improvement),
    )
    if metrics.relative_stage_improvement is None or metrics.relative_stage_improvement >= improvement_threshold:
        next_target_step = current_plan.target_step + current_plan.stage_step
        next_resolved_config = dict(current_plan.resolved_config)
        next_resume = current_plan.output_dir / "checkpoints" / f"step_{current_plan.target_step:07d}.pt"
        summary_reason = (
            "continue the same branch to "
            f"`step {next_target_step}` because the checkpoint passed the plausibility gate"
        )
        if metrics.relative_stage_improvement is not None:
            summary_reason += (
                " and the stage mean loss improved by "
                f"`{metrics.relative_stage_improvement * 100.0:.2f}%`."
            )
        else:
            summary_reason += "."
        return ExperimentPlan(
            experiment_name=current_plan.experiment_name,
            output_dir=current_plan.output_dir,
            overrides=dict(current_plan.overrides),
            resolved_config=next_resolved_config,
            current_step=current_plan.target_step,
            target_step=next_target_step,
            stage_step=current_plan.stage_step,
            resume_from=next_resume,
            reasoning=(summary_reason,),
        )

    fallback_overrides, fallback_reason = _build_fallback_overrides(
        base_train_config=base_train_config,
        current_plan=current_plan,
        memory_hints=memory_hints,
    )
    if fallback_overrides is not None:
        return _fresh_plan_from_overrides(
            train_config=base_train_config,
            overrides=fallback_overrides,
            stage_step=current_plan.stage_step,
            reasoning=(fallback_reason,),
            train_from_scratch=True,
            history=[],
        )

    next_target_step = current_plan.target_step + current_plan.stage_step
    return ExperimentPlan(
        experiment_name=current_plan.experiment_name,
        output_dir=current_plan.output_dir,
        overrides=dict(current_plan.overrides),
        resolved_config=dict(current_plan.resolved_config),
        current_step=current_plan.target_step,
        target_step=next_target_step,
        stage_step=current_plan.stage_step,
        resume_from=current_plan.output_dir / "checkpoints" / f"step_{current_plan.target_step:07d}.pt",
        reasoning=(
            "continue the same branch one more stage because no lower-risk fallback was available.",
        ),
    )


def compute_stage_score(
    *,
    metrics: MetricsSummary,
    plausibility: PlausibilitySummary,
    sweep_item: dict[str, Any],
) -> float:
    """Compute a coarse score that favors plausible checkpoints with lower loss."""
    score = 0.0
    if str(sweep_item.get("status")) == "ok":
        score += 1.0
    if plausibility.plausible:
        score += 2.0
    score += max(0.0, 1.0 - min(metrics.last_loss, 1.0))
    score -= plausibility.mean_frame_mae_rgb_0_255 / 30.0
    score -= max(0.0, plausibility.temporal_delta_ratio - 2.0)
    if metrics.relative_stage_improvement is not None and math.isfinite(metrics.relative_stage_improvement):
        score += max(-1.0, min(1.0, metrics.relative_stage_improvement))
    return float(score)


def build_learning_summary(
    *,
    plan: ExperimentPlan,
    metrics: MetricsSummary,
    plausibility: PlausibilitySummary,
    sweep_item: dict[str, Any],
) -> str:
    """Render a compact one-line summary of what this stage taught us."""
    verdict = "PASS" if plausibility.plausible else "FAIL"
    status = str(sweep_item.get("status"))
    improvement = ""
    if metrics.relative_stage_improvement is not None and math.isfinite(metrics.relative_stage_improvement):
        improvement = f", relative improvement={metrics.relative_stage_improvement * 100.0:.2f}%"
    return (
        f"Stage summary for {plan.experiment_name}: step={plan.target_step}, "
        f"last_loss={metrics.last_loss:.6f}, stage_mean_loss={metrics.stage_mean_loss:.6f}"
        f"{improvement}, sweep_status={status}, plausibility={verdict}, "
        f"mean_frame_mae={plausibility.mean_frame_mae_rgb_0_255:.3f}, "
        f"temporal_delta_ratio={plausibility.temporal_delta_ratio:.3f}."
    )


def maybe_apply_controller_edits(
    *,
    history: list[dict[str, Any]],
    experiment_name: str,
    metrics: MetricsSummary,
    plausibility: PlausibilitySummary,
    sweep_item: dict[str, Any],
    memory_text: str,
) -> list[dict[str, Any]]:
    """Apply one or more bounded self-edits when the controller process should change."""
    proposals = recommend_controller_policy_edits(
        history=history,
        experiment_name=experiment_name,
        metrics=metrics,
        plausibility=plausibility,
        sweep_item=sweep_item,
        memory_text=memory_text,
    )
    if not proposals:
        return []
    return apply_controller_policy_edits(
        proposals=proposals,
        controller_source_path=CONTROLLER_SOURCE_PATH,
    )


def recommend_controller_policy_edits(
    *,
    history: list[dict[str, Any]],
    experiment_name: str,
    metrics: MetricsSummary,
    plausibility: PlausibilitySummary,
    sweep_item: dict[str, Any],
    memory_text: str,
) -> tuple[ControllerPolicyEditProposal, ...]:
    """Choose small controller-policy edits from repeated stage outcomes and notes."""
    proposals: list[ControllerPolicyEditProposal] = []
    improvement_floor = float(CONTROLLER_POLICY["improvement_threshold_floor"])
    fallback_lr_scale = float(CONTROLLER_POLICY["fallback_lr_scale"])
    recent = _recent_experiment_records(history, experiment_name=experiment_name)

    recent_low_improvement = _count_trailing_matching_records(
        recent,
        predicate=lambda item: _record_has_low_improvement(
            item,
            threshold=improvement_floor,
        ),
    )
    current_low_improvement = (
        str(sweep_item.get("status")) == "ok"
        and plausibility.plausible
        and metrics.relative_stage_improvement is not None
        and metrics.relative_stage_improvement < improvement_floor
    )
    if current_low_improvement and (recent_low_improvement + 1) >= 2 and improvement_floor < 0.05:
        new_floor = round(min(0.05, improvement_floor + 0.01), 3)
        proposals.append(
            ControllerPolicyEditProposal(
                key="improvement_threshold_floor",
                old_value=improvement_floor,
                new_value=new_floor,
                reason=(
                    f"Two consecutive plausible stages on `{experiment_name}` improved by less than "
                    f"`{improvement_floor * 100.0:.2f}%`, so the controller should pivot away from weak branches earlier."
                ),
            )
        )

    recent_temporal_failures = _count_trailing_matching_records(
        recent,
        predicate=_record_has_temporal_instability_failure,
    )
    current_temporal_failure = (
        str(sweep_item.get("status")) != "ok"
        or (not plausibility.plausible and "temporal_instability" in plausibility.video_flags)
    )
    if current_temporal_failure and (recent_temporal_failures + 1) >= 2 and fallback_lr_scale > 0.25:
        new_scale = round(max(0.25, fallback_lr_scale * 0.5), 3)
        proposals.append(
            ControllerPolicyEditProposal(
                key="fallback_lr_scale",
                old_value=fallback_lr_scale,
                new_value=new_scale,
                reason=(
                    f"Repeated temporal-instability failures on `{experiment_name}` suggest the controller should "
                    "use a more conservative learning-rate fallback when architecture changes are exhausted."
                ),
            )
        )

    if _memory_requests_motion_priority(memory_text):
        proposals.extend(_motion_priority_policy_edits())

    deduped: dict[str, ControllerPolicyEditProposal] = {}
    for proposal in proposals:
        deduped[proposal.key] = proposal
    return tuple(deduped.values())


def apply_controller_policy_edits(
    *,
    proposals: tuple[ControllerPolicyEditProposal, ...],
    controller_source_path: Path,
) -> list[dict[str, Any]]:
    """Rewrite the controller's bounded policy block and return auditable edit records."""
    if not proposals:
        return []

    updated_policy = dict(CONTROLLER_POLICY)
    for proposal in proposals:
        updated_policy[proposal.key] = proposal.new_value

    try:
        source_text = controller_source_path.read_text(encoding="utf-8")
        updated_text = _replace_controller_policy_block(source_text, new_policy=updated_policy)
        controller_source_path.write_text(updated_text, encoding="utf-8")
    except Exception as exc:
        return [
            {
                "edit_id": f"controller_policy.{proposal.key}",
                "target_file": _display_path(controller_source_path),
                "applied": False,
                "old_value": proposal.old_value,
                "new_value": proposal.new_value,
                "summary": (
                    f"failed to update controller policy `{proposal.key}` from `{proposal.old_value}` "
                    f"to `{proposal.new_value}`: {exc}"
                ),
                "reason": proposal.reason,
                "error": str(exc),
            }
            for proposal in proposals
        ]

    CONTROLLER_POLICY.clear()
    CONTROLLER_POLICY.update(updated_policy)
    return [
        {
            "edit_id": f"controller_policy.{proposal.key}",
            "target_file": _display_path(controller_source_path),
            "applied": True,
            "old_value": proposal.old_value,
            "new_value": proposal.new_value,
            "summary": (
                f"updated controller policy `{proposal.key}` from `{proposal.old_value}` "
                f"to `{proposal.new_value}`."
            ),
            "reason": proposal.reason,
        }
        for proposal in proposals
    ]


def build_visual_review_summary(
    *,
    experiment_name: str,
    comparison_video: Path,
    generated_video: Path,
) -> dict[str, Any]:
    """Build a reusable manual-review guide for the comparison video artifacts."""
    comparison_display = _display_path(comparison_video)
    generated_display = _display_path(generated_video)
    frame_pattern = Path("/tmp") / f"{_slugify(experiment_name)}_%03d.png"
    focus_points, focus_note = _resolve_visual_review_focus()
    summary = (
        f"Inspect `{comparison_display}` with `ffplay -loop 0 {comparison_display}`. "
        "The left side is the target/reference and the right side is the generated rollout. "
        "Compare the reference and generated motion first, identify the most salient concrete differences, "
        "form likely causes only after those observations, and use the result to choose the next highest-information bounded test. "
        f"Use {focus_note.lower()} only as anchor failure classes, not as a rigid checklist. "
        f"If the clip is too short to judge in real time, extract frames with "
        f"`ffmpeg -i {comparison_display} {frame_pattern}`."
    )
    return {
        "summary": summary,
        "comparison_video": str(comparison_video),
        "generated_video": str(generated_video),
        "comparison_layout": "left=reference,right=generated",
        "focus_points": list(focus_points),
        "focus_mode": str(CONTROLLER_POLICY["visual_review_focus"]),
        "ffplay_command": ["ffplay", "-loop", "0", comparison_display],
        "ffmpeg_extract_command": ["ffmpeg", "-i", comparison_display, str(frame_pattern)],
    }


def _resolve_visual_review_focus() -> tuple[tuple[str, ...], str]:
    """Return the active visual-review checklist encoded in controller policy."""
    if str(CONTROLLER_POLICY.get("visual_review_focus", "generic")) == "motion":
        return MOTION_VISUAL_REVIEW_FOCUS_POINTS, MOTION_VISUAL_REVIEW_FOCUS_NOTE
    return GENERIC_VISUAL_REVIEW_FOCUS_POINTS, GENERIC_VISUAL_REVIEW_FOCUS_NOTE


def _motion_priority_policy_edits() -> tuple[ControllerPolicyEditProposal, ...]:
    """Promote motion-focused visual review when the notes say color is no longer the blocker."""
    if str(CONTROLLER_POLICY.get("visual_review_focus", "generic")) == "motion":
        return ()
    return (
        ControllerPolicyEditProposal(
            key="visual_review_focus",
            old_value=CONTROLLER_POLICY.get("visual_review_focus", "generic"),
            new_value="motion",
            reason=(
                "The current notes say motion/control fidelity is the bottleneck, so future run reviews should "
                "prioritize arm pose, tool path, and contact dynamics over generic image artifacts."
            ),
        ),
    )


def metrics_summary_to_dict(summary: MetricsSummary) -> dict[str, Any]:
    """Convert a metrics summary dataclass into a JSON-friendly mapping."""
    return asdict(summary)


def plausibility_summary_to_dict(summary: PlausibilitySummary) -> dict[str, Any]:
    """Convert a plausibility summary dataclass into a JSON-friendly mapping."""
    return {
        "plausible": summary.plausible,
        "mean_frame_mae_rgb_0_255": summary.mean_frame_mae_rgb_0_255,
        "temporal_delta_ratio": summary.temporal_delta_ratio,
        "num_failing_frames": summary.num_failing_frames,
        "video_flags": list(summary.video_flags),
    }


def experiment_plan_to_dict(plan: ExperimentPlan) -> dict[str, Any]:
    """Convert an experiment plan into a JSON-friendly mapping."""
    return {
        "experiment_name": plan.experiment_name,
        "output_dir": str(plan.output_dir),
        "overrides": _json_ready(plan.overrides),
        "resolved_config": _json_ready(plan.resolved_config),
        "current_step": plan.current_step,
        "target_step": plan.target_step,
        "stage_step": plan.stage_step,
        "resume_from": None if plan.resume_from is None else str(plan.resume_from),
        "reasoning": list(plan.reasoning),
    }


def experiment_plan_to_recommendation_dict(plan: ExperimentPlan) -> dict[str, Any]:
    """Convert an experiment plan into a compact recommendation payload."""
    resolved = experiment_plan_to_dict(plan)
    resolved["summary"] = " ".join(plan.reasoning)
    return resolved


def _plan_from_memory_hints(
    *,
    train_config: TrainScriptConfig,
    hints: MemoryHints,
    stage_step: int,
    history: list[dict[str, Any]],
) -> ExperimentPlan:
    """Build a new staged plan from the extracted memory hints."""
    return _fresh_plan_from_overrides(
        train_config=train_config,
        overrides=hints.overrides,
        stage_step=stage_step,
        reasoning=hints.reasoning,
        train_from_scratch=hints.train_from_scratch,
        history=history,
    )


def _fresh_plan_from_overrides(
    *,
    train_config: TrainScriptConfig,
    overrides: dict[str, Any],
    stage_step: int,
    reasoning: tuple[str, ...],
    train_from_scratch: bool,
    history: list[dict[str, Any]],
) -> ExperimentPlan:
    """Build a fresh plan from explicit overrides and current run progress."""
    resolved_config = _resolve_config_dict(train_config, overrides)
    experiment_name = build_experiment_name(train_config=train_config, resolved_config=resolved_config)
    output_dir = REPO_ROOT / "runs" / experiment_name
    if train_from_scratch and not _history_contains_experiment(history, experiment_name):
        output_dir = _unique_output_dir(output_dir)
        experiment_name = output_dir.name
    progress = inspect_run_progress(output_dir)
    target_step = progress.current_step + stage_step
    resume_from = progress.checkpoint_path
    resolved_config["output_dir"] = str(output_dir)
    return ExperimentPlan(
        experiment_name=experiment_name,
        output_dir=output_dir,
        overrides=_diff_config_overrides(train_config, resolved_config),
        resolved_config=resolved_config,
        current_step=progress.current_step,
        target_step=target_step,
        stage_step=stage_step,
        resume_from=resume_from,
        reasoning=reasoning or ("start from the highest-priority memory hint.",),
    )


def _plan_from_recommendation(
    *,
    train_config: TrainScriptConfig,
    recommendation: dict[str, Any],
    hints: MemoryHints,
) -> ExperimentPlan:
    """Rehydrate a persisted recommendation into the next executable plan."""
    output_dir = Path(str(recommendation["output_dir"]))
    progress = inspect_run_progress(output_dir)
    stage_step = int(recommendation["stage_step"])
    target_step = int(recommendation["target_step"])
    if progress.current_step >= target_step:
        target_step = progress.current_step + stage_step
    resolved_config = _resolve_config_dict(train_config, recommendation.get("overrides", {}))
    resolved_config["output_dir"] = str(output_dir)
    return ExperimentPlan(
        experiment_name=str(recommendation["experiment_name"]),
        output_dir=output_dir,
        overrides=_diff_config_overrides(train_config, resolved_config),
        resolved_config=resolved_config,
        current_step=progress.current_step,
        target_step=target_step,
        stage_step=stage_step,
        resume_from=progress.checkpoint_path,
        reasoning=tuple(recommendation.get("reasoning", hints.reasoning or ("follow the persisted controller recommendation.",))),
    )


def build_experiment_name(*, train_config: TrainScriptConfig, resolved_config: dict[str, Any]) -> str:
    """Derive a stable experiment name from the active runtime choices."""
    repo_slug = _slugify(str(resolved_config["repo_id"]).split("/")[-1])
    episodes = resolved_config.get("episodes", ())
    if episodes:
        scope = "ep" + "-".join(str(int(item)) for item in episodes)
    else:
        scope = "full"
    frame_width = int(resolved_config.get("frame_width", 0))
    frame_height = int(resolved_config.get("frame_height", 0))
    resolution = "native" if frame_width <= 0 or frame_height <= 0 else f"{frame_width}x{frame_height}"
    backbone = str(resolved_config["trainable_backbone"])
    if backbone == "lora":
        backbone = f"lora{int(resolved_config['lora_rank'])}"
    conditioning = str(resolved_config["conditioning_mode"])
    pieces = [
        "optimizer",
        repo_slug,
        scope,
        resolution,
        backbone,
        conditioning,
    ]
    for key in ("lr", "batch_size", "context_len", "horizon_len"):
        if resolved_config.get(key) != getattr(train_config, key):
            pieces.append(_slugify(f"{key}{resolved_config[key]}"))
    return "_".join(piece for piece in pieces if piece)


def inspect_run_progress(output_dir: str | Path) -> RunProgress:
    """Inspect the resumable state inside an output directory."""
    run_dir = Path(output_dir)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    latest_checkpoint_step = 0
    latest_checkpoint_path: Path | None = None
    if checkpoint_dir.exists():
        for checkpoint_path in checkpoint_dir.glob("step_*.pt"):
            match = re.search(r"step_(\d+)\.pt$", checkpoint_path.name)
            if match is None:
                continue
            step = int(match.group(1))
            if step > latest_checkpoint_step:
                latest_checkpoint_step = step
                latest_checkpoint_path = checkpoint_path

    latest_metric_step = 0
    if metrics_path.exists():
        rows = load_metrics_rows(metrics_path)
        if rows:
            latest_metric_step = int(rows[-1]["step"])

    if latest_checkpoint_step > 0:
        # Resume only from a persisted checkpoint. Metrics may run ahead if a job is interrupted
        # between log writes and checkpoint emission, so prefer the checkpoint as the durable step.
        current_step = latest_checkpoint_step
    else:
        current_step = latest_metric_step
    return RunProgress(
        current_step=current_step,
        checkpoint_path=latest_checkpoint_path,
        metrics_path=metrics_path if metrics_path.exists() else None,
    )


def _resolve_stage_step(
    *,
    train_config: TrainScriptConfig,
    hints: MemoryHints,
    stage_step_override: int | None,
) -> int:
    """Choose the checkpoint spacing for the next optimization stage."""
    if stage_step_override is not None:
        if stage_step_override <= 0:
            raise ValueError(f"stage_step_override must be > 0, got {stage_step_override}")
        return stage_step_override
    if hints.stage_step is not None and hints.stage_step > 0:
        return hints.stage_step
    if train_config.checkpoint_early_every > 0:
        return int(train_config.checkpoint_early_every)
    if train_config.checkpoint_every > 0:
        return int(train_config.checkpoint_every)
    return 100


def _recommendation_matches_hints(recommendation: dict[str, Any], hints: MemoryHints) -> bool:
    """Keep following a persisted recommendation when it still respects markdown hints."""
    recommendation_overrides = recommendation.get("overrides", {})
    recommendation_resolved = recommendation.get("resolved_config", {})
    for key, value in hints.overrides.items():
        if key in recommendation_overrides and _values_match(recommendation_overrides[key], value):
            continue
        if key in recommendation_resolved and _values_match(recommendation_resolved[key], value):
            continue
        if _values_match(recommendation.get(key), value):
            continue
        return False
    return True


def _resolve_config_dict(train_config: TrainScriptConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve a training config plus overrides into a plain dict."""
    payload = asdict(train_config)
    payload.update(overrides)
    return payload


def _diff_config_overrides(train_config: TrainScriptConfig, resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Keep only runtime values that differ from the base training config."""
    defaults = asdict(train_config)
    return {
        key: value
        for key, value in resolved_config.items()
        if key in defaults and defaults[key] != value
    }


def _build_fallback_overrides(
    *,
    base_train_config: TrainScriptConfig,
    current_plan: ExperimentPlan,
    memory_hints: MemoryHints,
) -> tuple[dict[str, Any] | None, str | None]:
    """Choose one lower-risk fallback when the current branch stops teaching us."""
    current_cfg = dict(current_plan.resolved_config)
    locked = set(memory_hints.locked_keys)

    if current_cfg.get("conditioning_mode") == "action":
        fallback = dict(current_plan.overrides)
        fallback["conditioning_mode"] = "none"
        return (
            fallback,
            "start a fresh `conditioning_mode=none` control run because the action-conditioned branch was the newest changed factor and did not clear the evaluation gate.",
        )

    if current_cfg.get("trainable_backbone") == "lora" and "trainable_backbone" not in locked:
        fallback = dict(current_plan.overrides)
        fallback["trainable_backbone"] = "head"
        return (
            fallback,
            "start a fresh `trainable_backbone=head` run because the LoRA branch did not deliver a stable evaluation gain.",
        )

    if "lr" not in locked:
        fallback = dict(current_plan.overrides)
        fallback["lr"] = float(current_cfg.get("lr", base_train_config.lr)) * float(CONTROLLER_POLICY["fallback_lr_scale"])
        return (
            fallback,
            "start a fresh lower-learning-rate run because the current branch stalled without a clearer architectural fallback.",
        )

    return None, None


def _recent_experiment_records(history: list[dict[str, Any]], *, experiment_name: str) -> list[dict[str, Any]]:
    """Return earlier controller records for one experiment in target-step order."""
    matches = [item for item in history if item.get("experiment_name") == experiment_name]
    return sorted(matches, key=lambda item: int(item.get("target_step", 0)))


def _count_trailing_matching_records(
    records: list[dict[str, Any]],
    *,
    predicate: Any,
) -> int:
    """Count matching records from the end of an already ordered record list."""
    count = 0
    for item in reversed(records):
        if not predicate(item):
            break
        count += 1
    return count


def _record_has_low_improvement(record: dict[str, Any], *, threshold: float) -> bool:
    """Check whether one stored record was plausible but improved less than the threshold."""
    sweep = record.get("sweep", {})
    plausibility = record.get("plausibility", {})
    metrics = record.get("metrics", {})
    improvement = metrics.get("relative_stage_improvement")
    if str(sweep.get("status")) != "ok" or not bool(plausibility.get("plausible")):
        return False
    if improvement is None:
        return False
    return float(improvement) < threshold


def _record_has_temporal_instability_failure(record: dict[str, Any]) -> bool:
    """Check whether one stored record failed due to temporal instability."""
    sweep = record.get("sweep", {})
    plausibility = record.get("plausibility", {})
    if str(sweep.get("status")) != "ok":
        return True
    if bool(plausibility.get("plausible")):
        return False
    return "temporal_instability" in {str(flag) for flag in plausibility.get("video_flags", [])}


def _memory_requests_motion_priority(memory_text: str) -> bool:
    """Detect when the markdown notes say motion fidelity is the current review bottleneck."""
    _, sections, _ = parse_markdown_sections(memory_text)
    combined = "\n".join(
        section
        for section in (
            sections.get("Current Signal", ""),
            sections.get("Next Work", ""),
        )
        if section
    ).lower()
    keywords = (
        "arm",
        "trajectory",
        "tool path",
        "contact dynamics",
        "motion fidelity",
        "falls down",
    )
    return any(keyword in combined for keyword in keywords)


def _replace_controller_policy_block(source_text: str, *, new_policy: dict[str, Any]) -> str:
    """Replace the self-editable controller policy block in the source text."""
    pattern = re.compile(
        r"# controller-self-edit: policy begin\n.*?# controller-self-edit: policy end",
        re.DOTALL,
    )
    replacement = _render_controller_policy_block(new_policy)
    updated_text, count = pattern.subn(replacement, source_text, count=1)
    if count != 1:
        raise ValueError("controller self-edit policy block was not found exactly once")
    return updated_text


def _render_controller_policy_block(policy: dict[str, Any]) -> str:
    """Render the bounded self-edit policy block as valid Python source."""
    return (
        "# controller-self-edit: policy begin\n"
        f"CONTROLLER_POLICY = {json.dumps(policy, indent=4, sort_keys=True)}\n"
        "# controller-self-edit: policy end"
    )


def _load_sweep_summary_item(evaluation_dir: Path) -> dict[str, Any]:
    """Load the single-item summary emitted by the checkpoint sweep helper."""
    summary_paths = sorted(evaluation_dir.glob("*_summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No sweep summary JSON found under {evaluation_dir}")
    payload = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"Sweep summary is empty: {summary_paths[0]}")
        item = payload[0]
    else:
        item = payload
    if not isinstance(item, dict):
        raise ValueError(f"Unexpected sweep summary payload in {summary_paths[0]}")
    return item


def _load_plausibility_summary(path: Path) -> PlausibilitySummary:
    """Load the compact summary emitted by the plausibility checker."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Plausibility report missing summary: {path}")
    return PlausibilitySummary(
        plausible=bool(summary["plausible"]),
        mean_frame_mae_rgb_0_255=float(summary["mean_frame_mae_rgb_0_255"]),
        temporal_delta_ratio=float(summary["temporal_delta_ratio"]),
        num_failing_frames=int(summary["num_failing_frames"]),
        video_flags=tuple(str(flag) for flag in summary.get("video_flags", [])),
    )


def _run_command(command: list[str]) -> None:
    """Run one subprocess in the repo root and fail with the full command on error."""
    rendered = format_command(command)
    print(f"Running command:\n{rendered}\n")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {rendered}")


def _overrides_to_cli_args(overrides: dict[str, Any]) -> list[str]:
    """Convert config overrides into world-model CLI flags."""
    args: list[str] = []
    bool_flags = {
        "disable_amp": ("--disable-amp", "--enable-amp"),
        "gradient_checkpointing": ("--gradient-checkpointing", "--no-gradient-checkpointing"),
        "load_pretrained_backbone": ("--load-pretrained-backbone", "--no-load-pretrained-backbone"),
        "overfit_one_batch": ("--overfit-one-batch", "--no-overfit-one-batch"),
    }
    for key, value in overrides.items():
        if key in {"output_dir", "max_steps", "resume_from"}:
            continue
        if key in bool_flags:
            true_flag, false_flag = bool_flags[key]
            args.append(true_flag if bool(value) else false_flag)
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, (tuple, list)):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        args.extend([flag, str(value)])
    return args


def _coerce_like(default_value: Any, raw_value: str) -> Any:
    """Coerce a string into the same broad type family as a default config value."""
    if isinstance(default_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, tuple):
        item_default = default_value[0] if default_value else 0
        return (_coerce_like(item_default, raw_value),)
    return raw_value


def _append_section_entry(section_text: str, entry: str, *, leading_blank_line: bool = False) -> str:
    """Append one entry to a markdown section body."""
    stripped = section_text.strip("\n")
    if not stripped:
        return entry
    if entry in stripped:
        return stripped
    separator = "\n\n" if leading_blank_line else "\n"
    return stripped + separator + entry


def _prepend_controller_bullet(section_text: str, bullet: str) -> str:
    """Replace older controller bullets with a new one at the top of the section."""
    lines = [
        line
        for line in section_text.splitlines()
        if not line.startswith(CONTROLLER_BULLET_PREFIX)
    ]
    remainder = "\n".join(lines).strip("\n")
    if remainder:
        return bullet + "\n" + remainder
    return bullet


def _format_training_run_entry(record: dict[str, Any]) -> str:
    """Render one controller-managed training log block for markdown memory."""
    metrics = record["metrics"]
    plausibility = record["plausibility"]
    recommendation = record["next_recommendation"]
    controller_edits = record.get("controller_edits", [])
    codex_analysis = record.get("codex_analysis")
    codex_visual_review = record.get("codex_visual_review")
    visual_review = record.get("visual_review")
    lines = [
        f"### [controller {record['timestamp']}] {record['experiment_name']} step {record['target_step']}",
        f"- `output_dir`: `{record['output_dir']}`",
        f"- `checkpoint`: `{record['checkpoint_path']}`",
        f"- `last_loss`: `{metrics['last_loss']:.6f}`",
        f"- `stage_mean_loss`: `{metrics['stage_mean_loss']:.6f}`",
        f"- `plausibility`: `{'PASS' if plausibility['plausible'] else 'FAIL'}` "
        f"(mean MAE `{plausibility['mean_frame_mae_rgb_0_255']:.3f}`, "
        f"temporal delta ratio `{plausibility['temporal_delta_ratio']:.3f}`)",
        f"- `learning`: {record['learning_summary']}",
        f"- `next`: {recommendation['summary']}",
        f"- `train_command`: `{format_command(record['commands']['train'])}`",
        f"- `sweep_command`: `{format_command(record['commands']['sweep'])}`",
        f"- `plausibility_command`: `{format_command(record['commands']['plausibility'])}`",
    ]
    comparison_summary = _format_record_comparison_summary(record)
    if comparison_summary:
        lines.append(f"- `comparison_context`: {comparison_summary}")
    if isinstance(codex_analysis, dict):
        lines.extend(
            [
                f"- `codex_action`: `{codex_analysis.get('action_type', 'unknown')}`",
                f"- `codex_analysis`: {codex_analysis.get('analysis_summary', '')}",
            ]
        )
    for edit in controller_edits:
        lines.append(f"- `controller_edit`: {edit['summary']}")
    if isinstance(visual_review, dict):
        lines.extend(
            [
                f"- `comparison_video`: `{visual_review['comparison_video']}`",
                f"- `generated_video`: `{visual_review['generated_video']}`",
                f"- `visual_review`: {visual_review['summary']}",
                f"- `ffplay_command`: `{format_command(visual_review['ffplay_command'])}`",
                f"- `ffmpeg_extract_command`: `{format_command(visual_review['ffmpeg_extract_command'])}`",
            ]
        )
    if isinstance(codex_visual_review, dict):
        lines.append(
            f"- `codex_visual_review`: `{codex_visual_review['verdict'].upper()}` | "
            f"{codex_visual_review['summary']}"
        )
    return "\n".join(lines)


def _format_record_comparison_summary(record: dict[str, Any]) -> str:
    """Render one short lineage/baseline summary for markdown memory."""
    parts: list[str] = []
    parent_stage_step = record.get("parent_stage_step")
    stage_kind = record.get("stage_kind")
    baseline_stage_step = record.get("baseline_stage_step")
    config_delta_keys = list(record.get("config_delta_from_parent", []))
    if parent_stage_step is not None:
        parts.append(f"parent_step={parent_stage_step}")
    if stage_kind:
        parts.append(f"stage_kind={stage_kind}")
    if baseline_stage_step is not None:
        baseline_locked = baseline_stage_step != record.get("target_step")
        parts.append(f"baseline_step={baseline_stage_step}")
        parts.append(f"baseline_locked={str(baseline_locked).lower()}")
    if config_delta_keys:
        parts.append(f"config_delta_keys={','.join(config_delta_keys)}")
    return " ".join(parts)


def _build_codex_visual_review_bullet(record: dict[str, Any]) -> str:
    """Build one controller bullet summarizing Codex's visual review for a completed stage."""
    review = record["codex_visual_review"]
    return (
        f"{CONTROLLER_BULLET_PREFIX}{record['timestamp']}] "
        f"{record['experiment_name']} step {record['target_step']}: "
        f"{review['verdict']} | {review['summary']}"
    )


def _format_controller_edit_entry(*, timestamp: str, edit: dict[str, Any]) -> str:
    """Render one controller self-edit entry for markdown memory."""
    lines = [
        f"### [controller {timestamp}] {edit['edit_id']}",
        f"- `target_file`: `{edit['target_file']}`",
        f"- `status`: `{'applied' if edit['applied'] else 'failed'}`",
        f"- `summary`: {edit['summary']}",
        f"- `reason`: {edit['reason']}",
    ]
    if "old_value" in edit and "new_value" in edit:
        lines.append(f"- `change`: `{edit['old_value']}` -> `{edit['new_value']}`")
    if edit.get("touched_files"):
        touched_files = ", ".join(f"`{path}`" for path in edit["touched_files"])
        lines.append(f"- `touched_files`: {touched_files}")
    if edit.get("validation_commands"):
        lines.append(f"- `validation_commands`: `{'; '.join(edit['validation_commands'])}`")
    if edit.get("smoke_test_commands"):
        lines.append(f"- `smoke_test_commands`: `{'; '.join(edit['smoke_test_commands'])}`")
    if edit.get("error"):
        lines.append(f"- `error`: `{edit['error']}`")
    return "\n".join(lines)


def _build_codex_analysis_bullet(*, timestamp: str, analysis: dict[str, Any]) -> str:
    """Render one compact Codex-analysis bullet for markdown memory."""
    next_work_note = str(analysis.get("next_work_note", "")).strip()
    suffix = "" if not next_work_note else f" Next note: {next_work_note}"
    return (
        f"{CONTROLLER_BULLET_PREFIX}{timestamp}] "
        f"Codex chose `{analysis['action_type']}`: {analysis['analysis_summary']}{suffix}"
    )


def _latest_record_for_experiment(
    history: list[dict[str, Any]],
    *,
    experiment_name: str,
    target_step_lt: int,
) -> dict[str, Any] | None:
    """Return the latest earlier stage record for the same experiment."""
    matches = [
        item
        for item in history
        if item.get("experiment_name") == experiment_name
        and int(item.get("target_step", 0)) < target_step_lt
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda item: int(item["target_step"]))[-1]


def _history_contains_experiment(history: list[dict[str, Any]], experiment_name: str) -> bool:
    """Check whether controller state already knows about one experiment name."""
    return any(item.get("experiment_name") == experiment_name for item in history)


def _unique_output_dir(path: Path) -> Path:
    """Choose a unique run directory when a fresh start must avoid an existing branch."""
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_restart{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find a unique output directory near {path}")


def _slugify(text: str) -> str:
    """Normalize free-form text into a filesystem-safe experiment slug."""
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return collapsed.strip("_").lower()


def _json_ready(value: Any) -> Any:
    """Convert Paths and tuples recursively into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _display_path(path: Path) -> str:
    """Prefer repo-relative paths when rendering human-facing file references."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _values_match(left: Any, right: Any) -> bool:
    """Compare persisted JSON values against in-memory config values."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp without sub-second noise."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stage_is_valid(*, output_dir: Path, target_step: int) -> bool:
    """Check whether a staged checkpoint already validates cleanly."""
    try:
        validate_training_stage(output_dir, expected_step=target_step)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _print_plan(plan: ExperimentPlan) -> None:
    """Print the current controller plan before any long-running work starts."""
    print(
        f"Controller plan: experiment={plan.experiment_name} current_step={plan.current_step} "
        f"target_step={plan.target_step}"
    )
    for reason in plan.reasoning:
        print(f"  reason: {reason}")


def format_command(command: list[str]) -> str:
    """Render one subprocess argument list as a shell-safe string."""
    return shlex.join(command)
