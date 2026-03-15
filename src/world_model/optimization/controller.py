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
    ensure_codex_chatgpt_login,
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
DEFAULT_MAX_INSPECTION_ROUNDS = 2
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
    ensure_codex_chatgpt_login()
    train_config = load_train_config(train_config_path)
    state = load_controller_state(state_path)
    state = _normalize_controller_state(state)
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

    records: list[dict[str, Any]] = []
    pending_controller_edits: list[dict[str, Any]] = []

    with _controller_loop_lock(state_path):
        while True:
            stop_reason = _budget_stop_reason(budget)
            if stop_reason is not None:
                state["codex_state"]["last_stop_reason"] = stop_reason
                save_controller_state(state_path, state)
                break

            budget = _increment_loop_budget_counter(budget, "iterations_used")
            memory_text = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
            context_bundle = build_codex_context_bundle(
                train_config=train_config,
                memory_text=memory_text,
                state=state,
                budget=budget,
                pending_controller_edits=pending_controller_edits,
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

            decision, decision_call_count = request_codex_loop_decision(
                context_bundle=context_bundle,
                codex_model=codex_model,
            )
            budget = _increment_loop_budget_counter(
                budget,
                "codex_calls_used",
                count=decision_call_count,
            )
            _record_codex_decision(state, decision=decision, budget=budget)
            save_controller_state(state_path, _persist_budget(state, budget))

            inspection_context: dict[str, Any] | None = None
            inspection_rounds = 0
            stop_after_inspection = False
            while decision["action_type"] == "inspect_artifact":
                if inspection_rounds >= DEFAULT_MAX_INSPECTION_ROUNDS:
                    state["codex_state"]["last_stop_reason"] = (
                        f"max_inspection_rounds={DEFAULT_MAX_INSPECTION_ROUNDS} exhausted"
                    )
                    update_memory_with_codex_analysis_file(memory_path, decision=decision)
                    save_controller_state(state_path, _persist_budget(state, budget))
                    stop_after_inspection = True
                    break
                stop_reason = _budget_stop_reason(budget, include_iterations=False)
                if stop_reason is not None:
                    state["codex_state"]["last_stop_reason"] = stop_reason
                    update_memory_with_codex_analysis_file(memory_path, decision=decision)
                    save_controller_state(state_path, _persist_budget(state, budget))
                    stop_after_inspection = True
                    break
                inspection_context = prepare_codex_inspection_context(
                    request=decision["inspect_artifact"],
                    state=state,
                )
                _append_limited_history(
                    state["inspection_history"],
                    inspection_context["summary"],
                    limit=DEFAULT_DECISION_HISTORY_LIMIT,
                )
                decision, decision_call_count = request_codex_loop_decision(
                    context_bundle=context_bundle,
                    codex_model=codex_model,
                    inspection_context=inspection_context,
                )
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=decision_call_count,
                )
                _record_codex_decision(state, decision=decision, budget=budget)
                save_controller_state(state_path, _persist_budget(state, budget))
                inspection_rounds += 1
            if stop_after_inspection:
                break

            if dry_run:
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
                    state["codex_state"]["last_stop_reason"] = "edit cycle budget exhausted"
                    save_controller_state(state_path, _persist_budget(state, budget))
                    break
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
                    state["codex_state"]["last_stop_reason"] = stop_reason
                    save_controller_state(state_path, _persist_budget(state, budget))
                    break
                followup_context = build_codex_context_bundle(
                    train_config=train_config,
                    memory_text=memory_path.read_text(encoding="utf-8") if memory_path.exists() else memory_text,
                    state=state,
                    budget=budget,
                    pending_controller_edits=pending_controller_edits,
                )
                decision, decision_call_count = request_codex_loop_decision(
                    context_bundle=followup_context,
                    codex_model=codex_model,
                )
                budget = _increment_loop_budget_counter(
                    budget,
                    "codex_calls_used",
                    count=decision_call_count,
                )
                _record_codex_decision(state, decision=decision, budget=budget)
                save_controller_state(state_path, _persist_budget(state, budget))
                if decision["action_type"] not in {"run_experiment", "stop"}:
                    state["codex_state"]["last_stop_reason"] = (
                        "Codex must return `run_experiment` or `stop` after an edit cycle."
                    )
                    save_controller_state(state_path, _persist_budget(state, budget))
                    break

            if decision["action_type"] == "stop":
                state["codex_state"]["last_stop_reason"] = decision["stop"]["reason"]
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

            update_memory_file(memory_path, record)
            state = append_stage_record(state, record)
            budget = _increment_loop_budget_counter(budget, "real_runs_used")
            state = _persist_budget(state, budget)
            save_controller_state(state_path, state)
            records.append(record)
            pending_controller_edits = []

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
    return normalized


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
) -> dict[str, Any]:
    """Assemble a compact structured context bundle for one Codex planning turn."""
    _, sections, _ = parse_markdown_sections(memory_text)
    history = list(state.get("history", []))
    latest_record = history[-1] if history else None
    latest_experiment = None if latest_record is None else latest_record.get("experiment_name")
    artifacts = _summarize_latest_artifacts(latest_record)
    return {
        "goal": sections.get("Goal", "").strip(),
        "training_goal": sections.get("Training Goal", "").strip(),
        "stable_findings": sections.get("Stable Findings", "").strip(),
        "current_signal": sections.get("Current Signal", "").strip(),
        "next_work": sections.get("Next Work", "").strip(),
        "codex_analysis": sections.get("Codex Analysis", "").strip(),
        "controller_edits": sections.get("Controller Edits", "").strip(),
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
        "latest_recommendation": state.get("latest_recommendation"),
        "recent_runs": _summarize_recent_history(history),
        "recent_decisions": list(state.get("decision_history", []))[-5:],
        "recent_edits": list(state.get("edit_history", []))[-5:],
        "recent_failures": state.get("codex_state", {}).get("last_run_failure"),
        "pending_controller_edits": pending_controller_edits,
        "latest_artifacts": artifacts,
        "available_code_paths": [
            "scripts/train/world_model.py",
            "src/world_model/training/chunkwise_training.py",
            "src/world_model/training/flow_matching.py",
            "src/world_model/optimization/controller.py",
        ],
    }


def _summarize_recent_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the most recent stage records and their high-signal fields."""
    items: list[dict[str, Any]] = []
    for record in history[-DEFAULT_HISTORY_SUMMARY_LIMIT:]:
        metrics = record.get("metrics", {})
        plausibility = record.get("plausibility", {})
        items.append(
            {
                "timestamp": record.get("timestamp"),
                "experiment_name": record.get("experiment_name"),
                "target_step": record.get("target_step"),
                "last_loss": metrics.get("last_loss"),
                "stage_mean_loss": metrics.get("stage_mean_loss"),
                "relative_stage_improvement": metrics.get("relative_stage_improvement"),
                "plausible": plausibility.get("plausible"),
                "temporal_delta_ratio": plausibility.get("temporal_delta_ratio"),
                "video_flags": plausibility.get("video_flags", []),
                "learning_summary": record.get("learning_summary"),
            }
        )
    return items


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
    inspection_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Ask Codex for the next autonomous-loop action as structured JSON."""
    images = None if inspection_context is None else [Path(item) for item in inspection_context.get("image_inputs", [])]
    base_prompt = _build_codex_decision_prompt(
        context_bundle=context_bundle,
        inspection_context=inspection_context,
    )
    last_error: Exception | None = None
    for attempt in range(1, 3):
        prompt = base_prompt
        if last_error is not None:
            prompt += (
                "\n\nThe previous response did not validate against the required schema. "
                f"Fix the structured output and return a corrected JSON object only. Error: {last_error}"
            )
        result = run_codex_exec(
            prompt=prompt,
            schema=_loop_decision_schema(),
            model=codex_model,
            images=images,
            cwd=REPO_ROOT,
        )
        try:
            return _validate_loop_decision_payload(result.payload), attempt
        except Exception as exc:  # pragma: no cover - retry path exercised in tests
            last_error = exc
    assert last_error is not None
    raise last_error


def _build_codex_decision_prompt(
    *,
    context_bundle: dict[str, Any],
    inspection_context: dict[str, Any] | None,
) -> str:
    """Render the autonomous-loop planning prompt passed into Codex."""
    prompt_parts = [
        "You are the autonomous experiment planner for this repository.",
        "Human instructions under `next_work` are the highest-priority steering signal.",
        "Choose exactly one action: run_experiment, inspect_artifact, apply_repo_edit, or stop.",
        "Always return all four action payload objects. Fill the inactive ones with empty/default values, not null.",
        "For run_experiment.overrides, return an array of objects with `key` and `value` fields.",
        "Do not assume you can launch arbitrary long-running commands yourself; the controller owns experiment execution.",
        "Use apply_repo_edit only when the repo logic itself appears wrong. Provide a unified diff, touched files, evidence, and validation commands.",
        "If you need visual or code evidence before deciding, use inspect_artifact.",
        "Return only data that matches the provided JSON schema.",
        "",
        "Context JSON:",
        json.dumps(context_bundle, indent=2, sort_keys=True),
    ]
    if inspection_context is not None:
        prompt_parts.extend(
            [
                "",
                "Additional inspection context JSON:",
                json.dumps(inspection_context.get("payload", {}), indent=2, sort_keys=True),
            ]
        )
    return "\n".join(prompt_parts)


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
            "stage_step": section.get("stage_step"),
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
        },
        limit=DEFAULT_DECISION_HISTORY_LIMIT,
    )


def _append_limited_history(history: list[dict[str, Any]], item: dict[str, Any], *, limit: int) -> None:
    """Append one history item and trim the list in place."""
    history.append(item)
    if len(history) > limit:
        del history[:-limit]


def prepare_codex_inspection_context(
    *,
    request: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Collect artifact and code excerpts for one targeted Codex inspection round."""
    latest_record = state.get("history", [])[-1] if state.get("history") else None
    inspection_dir = REPO_ROOT / "runs" / "training_optimizer" / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    image_inputs: list[str] = []
    artifact_summaries: list[dict[str, Any]] = []
    for raw_path in request.get("artifact_paths", []):
        resolved = _resolve_repo_relative_path(raw_path)
        if not resolved.exists():
            artifact_summaries.append({"path": raw_path, "missing": True})
            continue
        artifact_summary: dict[str, Any] = {"path": _display_path(resolved)}
        suffix = resolved.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            image_inputs.append(str(resolved))
        elif suffix in {".mp4", ".mov"}:
            contact_sheet_path = _build_video_contact_sheet(resolved, inspection_dir=inspection_dir)
            image_inputs.append(str(contact_sheet_path))
            artifact_summary["contact_sheet"] = str(contact_sheet_path)
        elif suffix in {".json", ".jsonl", ".md", ".txt", ".py", ".yaml", ".yml", ".sh"}:
            artifact_summary["excerpt"] = _read_file_excerpt(resolved)
        artifact_summaries.append(artifact_summary)

    code_snippets = [
        {
            "path": _display_path(_resolve_repo_relative_path(raw_path)),
            "excerpt": _read_file_excerpt(_resolve_repo_relative_path(raw_path)),
        }
        for raw_path in request.get("code_paths", [])
        if _resolve_repo_relative_path(raw_path).exists()
    ]
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
        },
    }


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
    stage_step = _resolve_stage_step(
        train_config=train_config,
        hints=hints,
        stage_step_override=stage_step_override if stage_step_override is not None else run_spec.get("stage_step"),
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
    diff_paths = _extract_unified_diff_paths(proposal["unified_diff"])
    declared_paths = {_display_path(path) for path in touched_paths}
    if diff_paths != declared_paths:
        raise ValueError(
            "Codex repo edit diff paths do not match declared touched files: "
            f"declared={sorted(declared_paths)} diff={sorted(diff_paths)}"
        )
    snapshots = _snapshot_repo_files(touched_paths)
    edit_id = f"codex_repo_edit_{_short_hash(proposal['unified_diff'])}"
    try:
        _apply_unified_diff(proposal["unified_diff"])
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


def update_memory_markdown(memory_text: str, *, record: dict[str, Any]) -> str:
    """Append concise controller findings and the next recommendation to markdown memory."""
    preamble, sections, order = parse_markdown_sections(memory_text)
    timestamp = record["timestamp"]
    metrics = record["metrics"]
    plausibility = record["plausibility"]
    recommendation = record["next_recommendation"]
    controller_edits = record.get("controller_edits", [])
    codex_analysis = record.get("codex_analysis")
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
    for heading in ("Current Signal", "Next Work", "Training runs", "Codex Analysis", "Controller Edits"):
        if heading not in final_order:
            final_order.append(heading)
    return render_markdown_sections(preamble, sections, final_order)


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
    updated["latest_recommendation"] = record["next_recommendation"]
    updated["latest_record"] = {
        "experiment_name": record["experiment_name"],
        "target_step": record["target_step"],
        "score": record["score"],
    }
    return updated


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
        f"{focus_note} "
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

    if latest_checkpoint_step and latest_metric_step and latest_checkpoint_step != latest_metric_step:
        raise ValueError(
            f"Run directory is inconsistent: latest checkpoint step {latest_checkpoint_step} "
            f"does not match latest metrics step {latest_metric_step} in {run_dir}."
        )

    current_step = max(latest_checkpoint_step, latest_metric_step)
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
    return "\n".join(lines)


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
