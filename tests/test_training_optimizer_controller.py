"""Tests for the staged training-optimization controller helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import world_model.optimization.controller as controller_module
from world_model.config import TrainScriptConfig
from world_model.optimization.controller import (
    LoopBudget,
    append_stage_record,
    apply_codex_repo_edit,
    build_visual_review_summary,
    build_codex_context_bundle,
    extract_memory_hints,
    maybe_apply_controller_edits,
    parse_markdown_sections,
    request_codex_loop_decision,
    select_experiment_plan,
    summarize_metrics_rows,
    update_memory_markdown,
)


def test_extract_memory_hints_reads_next_work_contract() -> None:
    """Parse the current optimization-note style into explicit controller hints."""
    train_config = TrainScriptConfig()
    memory_text = """
## Next Work

- The next training run should restore the intended action-conditioning setup while keeping the stable recipe fixed: `episode 0 + 320x240 + trainable_backbone=lora + lora_rank=8`.
- Do not change LR, frame count, or dataset scope yet. Change only `conditioning_mode` from `none` to `action`, train from scratch, and evaluate checkpoints every `100` steps with the corrected evaluator.
"""

    hints = extract_memory_hints(memory_text, train_config=train_config)

    assert hints.overrides["episodes"] == (0,)
    assert hints.overrides["frame_width"] == 320
    assert hints.overrides["frame_height"] == 240
    assert hints.overrides["trainable_backbone"] == "lora"
    assert hints.overrides["lora_rank"] == 8
    assert hints.overrides["conditioning_mode"] == "action"
    assert hints.stage_step == 100
    assert hints.train_from_scratch is True
    assert set(hints.locked_keys) >= {
        "lr",
        "context_len",
        "horizon_len",
        "frame_width",
        "frame_height",
        "repo_id",
        "episodes",
        "subset_size",
        "video_key",
    }


def test_select_experiment_plan_uses_memory_hints_for_first_stage(monkeypatch, tmp_path: Path) -> None:
    """Build the first staged action-conditioning plan from markdown memory."""
    train_config = TrainScriptConfig(
        repo_id="lerobot/aloha_static_fork_pick_up",
        video_key="observation.images.cam_high",
        frame_width=320,
        frame_height=240,
        trainable_backbone="lora",
        conditioning_mode="none",
        lora_rank=8,
        checkpoint_early_every=100,
    )
    memory_text = """
## Next Work

- Keep the stable recipe fixed: `episode 0 + 320x240 + trainable_backbone=lora + lora_rank=8`.
- Change only `conditioning_mode` from `none` to `action`, train from scratch, and evaluate checkpoints every `100` steps with the corrected evaluator.
"""

    monkeypatch.setattr(
        "world_model.optimization.controller.REPO_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        "world_model.optimization.controller.inspect_run_progress",
        lambda output_dir: __import__(
            "world_model.optimization.controller",
            fromlist=["RunProgress"],
        ).RunProgress(current_step=0, checkpoint_path=None, metrics_path=None),
    )

    plan = select_experiment_plan(
        train_config=train_config,
        memory_text=memory_text,
        state={"history": [], "latest_recommendation": None},
    )

    assert plan.experiment_name == "optimizer_aloha_static_fork_pick_up_ep0_320x240_lora8_action"
    assert plan.current_step == 0
    assert plan.target_step == 100
    assert plan.resume_from is None
    assert plan.overrides["episodes"] == (0,)
    assert plan.overrides["conditioning_mode"] == "action"


def test_select_experiment_plan_prefers_compatible_latest_recommendation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Follow a persisted continuation recommendation when it still matches the notes."""
    train_config = TrainScriptConfig(
        repo_id="lerobot/aloha_static_fork_pick_up",
        video_key="observation.images.cam_high",
        frame_width=320,
        frame_height=240,
        trainable_backbone="lora",
        conditioning_mode="none",
        lora_rank=8,
        checkpoint_early_every=100,
    )
    memory_text = """
## Next Work

- Keep the stable recipe fixed: `episode 0 + 320x240 + trainable_backbone=lora + lora_rank=8`.
- Change only `conditioning_mode` from `none` to `action`, train from scratch, and evaluate checkpoints every `100` steps with the corrected evaluator.
"""
    checkpoint_path = tmp_path / "runs" / "optimizer_aloha_static_fork_pick_up_ep0_320x240_lora8_action" / "checkpoints" / "step_0000100.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "world_model.optimization.controller.inspect_run_progress",
        lambda output_dir: __import__(
            "world_model.optimization.controller",
            fromlist=["RunProgress"],
        ).RunProgress(current_step=100, checkpoint_path=checkpoint_path, metrics_path=None),
    )

    state = {
        "history": [],
        "latest_recommendation": {
            "experiment_name": "optimizer_aloha_static_fork_pick_up_ep0_320x240_lora8_action",
            "output_dir": str(checkpoint_path.parents[1]),
            "overrides": {
                "episodes": [0],
                "frame_width": 320,
                "frame_height": 240,
                "trainable_backbone": "lora",
                "lora_rank": 8,
                "conditioning_mode": "action",
            },
            "resolved_config": {
                "episodes": [0],
                "frame_width": 320,
                "frame_height": 240,
                "trainable_backbone": "lora",
                "lora_rank": 8,
                "conditioning_mode": "action",
            },
            "current_step": 100,
            "target_step": 200,
            "stage_step": 100,
            "resume_from": str(checkpoint_path),
            "reasoning": ["continue the same branch to `step 200` because the checkpoint passed the plausibility gate."],
            "summary": "continue the same branch to `step 200` because the checkpoint passed the plausibility gate.",
        },
    }

    plan = select_experiment_plan(
        train_config=train_config,
        memory_text=memory_text,
        state=state,
    )

    assert plan.current_step == 100
    assert plan.target_step == 200
    assert plan.resume_from == checkpoint_path
    assert plan.overrides["conditioning_mode"] == "action"


def test_summarize_metrics_rows_reports_relative_stage_improvement() -> None:
    """Summarize one stage and compare it against the previous stage mean."""
    rows = [
        {"step": 1, "loss": 0.8},
        {"step": 2, "loss": 0.6},
        {"step": 3, "loss": 0.4},
        {"step": 4, "loss": 0.2},
    ]

    summary = summarize_metrics_rows(
        metrics_rows=rows,
        previous_step=2,
        target_step=4,
        previous_stage_mean_loss=0.7,
    )

    assert summary.last_loss == 0.2
    assert summary.best_loss == 0.2
    assert summary.stage_row_count == 2
    assert abs(summary.stage_mean_loss - 0.3) < 1e-9
    assert abs(summary.relative_stage_improvement - ((0.7 - 0.3) / 0.7)) < 1e-9


def test_update_memory_markdown_replaces_old_controller_next_work_bullet() -> None:
    """Keep one current controller recommendation while appending persistent findings."""
    memory_text = """
## Current Signal

- A stable human-written finding.

## Next Work

- [controller 2026-03-14T00:00:00+00:00] Next experiment: stale recommendation.
- A human next-work bullet.

## Training runs

- Existing human run note.
"""
    record = {
        "timestamp": "2026-03-15T12:00:00+00:00",
        "experiment_name": "optimizer_example",
        "output_dir": "runs/optimizer_example",
        "target_step": 100,
        "checkpoint_path": "runs/optimizer_example/checkpoints/step_0000100.pt",
        "learning_summary": "Stage summary for optimizer_example: step=100, last_loss=0.120000, stage_mean_loss=0.150000, sweep_status=ok, plausibility=PASS, mean_frame_mae=5.200, temporal_delta_ratio=1.800.",
        "commands": {
            "train": ["python", "scripts/train/world_model.py"],
            "sweep": ["python", "scripts/check/sweep_local_repo_resolutions.py"],
            "plausibility": ["python", "scripts/check/check_generated_video_plausibility.py"],
        },
        "metrics": {
            "last_loss": 0.12,
            "stage_mean_loss": 0.15,
        },
        "plausibility": {
            "plausible": True,
            "mean_frame_mae_rgb_0_255": 5.2,
            "temporal_delta_ratio": 1.8,
        },
        "visual_review": {
            "summary": "Inspect `runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4` with `ffplay -loop 0 runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4`.",
            "comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4",
            "generated_video": "runs/training_optimizer/eval/optimizer_example_step_0000100/generated.mp4",
            "ffplay_command": [
                "ffplay",
                "-loop",
                "0",
                "runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4",
            ],
            "ffmpeg_extract_command": [
                "ffmpeg",
                "-i",
                "runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4",
                "/tmp/optimizer_example_%03d.png",
            ],
        },
        "controller_edits": [
            {
                "edit_id": "controller_policy.visual_review_focus",
                "target_file": "src/world_model/optimization/controller.py",
                "applied": True,
                "old_value": "generic",
                "new_value": "motion",
                "summary": "updated controller policy `visual_review_focus` from `generic` to `motion`.",
                "reason": "Motion fidelity is now the main bottleneck.",
            }
        ],
        "codex_analysis": {
            "action_type": "run_experiment",
            "analysis_summary": "The action-conditioned branch is still the highest-priority run.",
            "reasoning": ["The latest plausible failure is in arm-motion fidelity, not color."],
            "next_work_note": "Keep the action-conditioned LoRA recipe fixed while comparing early checkpoints.",
        },
        "next_recommendation": {
            "summary": "continue the same branch to `step 200` because the checkpoint passed the plausibility gate.",
        },
    }

    updated = update_memory_markdown(memory_text, record=record)
    _, sections, _ = parse_markdown_sections(updated)

    assert sections["Current Signal"].count("[controller 2026-03-15T12:00:00+00:00]") == 1
    assert "stale recommendation" not in sections["Next Work"]
    assert sections["Next Work"].count("[controller 2026-03-15T12:00:00+00:00]") == 1
    assert "continue the same branch to `step 200`" in sections["Next Work"]
    assert "### [controller 2026-03-15T12:00:00+00:00] optimizer_example step 100" in sections["Training runs"]
    assert "`comparison_video`: `runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4`" in sections["Training runs"]
    assert "`ffplay_command`: `ffplay -loop 0 runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4`" in sections["Training runs"]
    assert "Codex chose `run_experiment`" in sections["Codex Analysis"]
    assert "Keep the action-conditioned LoRA recipe fixed" in sections["Codex Analysis"]
    assert "### [controller 2026-03-15T12:00:00+00:00] controller_policy.visual_review_focus" in sections["Controller Edits"]
    assert "`status`: `applied`" in sections["Controller Edits"]


def test_build_visual_review_summary_uses_repo_relative_commands(monkeypatch, tmp_path: Path) -> None:
    """Render a reusable comparison-video review block with local inspection commands."""
    monkeypatch.setattr(
        "world_model.optimization.controller.REPO_ROOT",
        tmp_path,
    )
    comparison_video = tmp_path / "runs" / "training_optimizer" / "eval" / "example_comparison.mp4"
    generated_video = tmp_path / "runs" / "training_optimizer" / "eval" / "example.mp4"

    summary = build_visual_review_summary(
        experiment_name="optimizer_example",
        comparison_video=comparison_video,
        generated_video=generated_video,
    )

    assert summary["comparison_video"] == str(comparison_video)
    assert summary["generated_video"] == str(generated_video)
    assert summary["ffplay_command"] == [
        "ffplay",
        "-loop",
        "0",
        "runs/training_optimizer/eval/example_comparison.mp4",
    ]
    assert summary["ffmpeg_extract_command"] == [
        "ffmpeg",
        "-i",
        "runs/training_optimizer/eval/example_comparison.mp4",
        "/tmp/optimizer_example_%03d.png",
    ]
    assert summary["focus_mode"] == "generic"
    assert "left side is the target/reference" in summary["summary"]


def test_maybe_apply_controller_edits_rewrites_policy_block(monkeypatch, tmp_path: Path) -> None:
    """Apply a bounded self-edit to the controller policy and record the result."""
    controller_source = tmp_path / "controller.py"
    controller_source.write_text(
        """
# controller-self-edit: policy begin
CONTROLLER_POLICY = {
    "fallback_lr_scale": 0.5,
    "improvement_threshold_floor": 0.02,
    "visual_review_focus": "generic"
}
# controller-self-edit: policy end
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "world_model.optimization.controller.CONTROLLER_SOURCE_PATH",
        controller_source,
    )
    monkeypatch.setattr(
        "world_model.optimization.controller.CONTROLLER_POLICY",
        {
            "fallback_lr_scale": 0.5,
            "improvement_threshold_floor": 0.02,
            "visual_review_focus": "generic",
        },
    )

    edits = maybe_apply_controller_edits(
        history=[],
        experiment_name="optimizer_example",
        metrics=controller_module.MetricsSummary(
            previous_step=0,
            target_step=100,
            last_loss=0.12,
            best_loss=0.11,
            stage_mean_loss=0.13,
            trailing_mean_loss=0.13,
            stage_row_count=10,
            relative_stage_improvement=0.03,
        ),
        plausibility=controller_module.PlausibilitySummary(
            plausible=True,
            mean_frame_mae_rgb_0_255=5.0,
            temporal_delta_ratio=1.2,
            num_failing_frames=0,
            video_flags=(),
        ),
        sweep_item={"status": "ok"},
        memory_text="""
## Current Signal

- Colors stay plausible, but the robot arm falls down instead of tracking the target motion.

## Next Work

- Prioritize arm pose, tool path, and contact dynamics over color.
""",
    )

    updated_source = controller_source.read_text(encoding="utf-8")
    assert len(edits) == 1
    assert edits[0]["applied"] is True
    assert edits[0]["edit_id"] == "controller_policy.visual_review_focus"
    assert '"visual_review_focus": "motion"' in updated_source
    assert controller_module.CONTROLLER_POLICY["visual_review_focus"] == "motion"


def test_append_stage_record_preserves_codex_state_sections() -> None:
    """Keep autonomous-loop state alongside the appended stage history."""
    state = {
        "history": [],
        "latest_recommendation": None,
        "codex_state": {"last_action_type": "inspect_artifact"},
        "budget": {"iterations_used": 1},
        "decision_history": [{"action_type": "inspect_artifact"}],
        "inspection_history": [{"artifact_paths": ["runs/example.mp4"]}],
        "edit_history": [{"edit_id": "codex_repo_edit_123"}],
        "context_history": [{"latest_experiment": "optimizer_example"}],
    }
    record = {
        "experiment_name": "optimizer_example",
        "target_step": 100,
        "score": 0.75,
        "next_recommendation": {"summary": "continue to `step 200`"},
    }

    updated = append_stage_record(state, record)

    assert updated["history"] == [record]
    assert updated["latest_recommendation"] == {"summary": "continue to `step 200`"}
    assert updated["codex_state"]["last_action_type"] == "inspect_artifact"
    assert updated["budget"]["iterations_used"] == 1
    assert updated["decision_history"] == [{"action_type": "inspect_artifact"}]
    assert updated["inspection_history"] == [{"artifact_paths": ["runs/example.mp4"]}]


def test_build_codex_context_bundle_compacts_recent_history() -> None:
    """Build a compact Codex context bundle from markdown and recent state."""
    state = {
        "history": [
            {
                "timestamp": f"2026-03-15T12:0{index}:00+00:00",
                "experiment_name": f"optimizer_example_{index}",
                "target_step": (index + 1) * 100,
                "metrics": {
                    "last_loss": 0.2 - 0.01 * index,
                    "stage_mean_loss": 0.25 - 0.01 * index,
                    "relative_stage_improvement": 0.05,
                },
                "plausibility": {
                    "plausible": True,
                    "temporal_delta_ratio": 1.1,
                    "video_flags": [],
                },
                "learning_summary": f"summary {index}",
                "sweep": {
                    "status": "ok",
                    "output_path": f"runs/generated_{index}.mp4",
                    "comparison_output_path": f"runs/comparison_{index}.mp4",
                },
                "visual_review": {"focus_points": ["blur", "ghosting"]},
            }
            for index in range(4)
        ],
        "latest_recommendation": {"summary": "continue"},
        "decision_history": [{"action_type": "inspect_artifact"}],
        "edit_history": [{"edit_id": "codex_repo_edit_123"}],
        "codex_state": {"last_run_failure": None},
    }
    budget = LoopBudget(
        max_iterations=2,
        max_real_runs=2,
        max_codex_calls=4,
        max_failed_runs=2,
        max_edit_cycles=1,
        max_wall_clock_minutes=None,
        iterations_used=1,
        real_runs_used=0,
        codex_calls_used=1,
        failed_runs_used=0,
        edit_cycles_used=0,
        started_at="2026-03-15T12:00:00+00:00",
    )
    memory_text = """
## Goal

Keep the optimizer focused on stable action-conditioning.

## Current Signal

- Motion fidelity is still the bottleneck.

## Next Work

- Keep the stable action-conditioned recipe fixed.
"""

    bundle = build_codex_context_bundle(
        train_config=TrainScriptConfig(),
        memory_text=memory_text,
        state=state,
        budget=budget,
        pending_controller_edits=[],
    )

    assert bundle["goal"] == "Keep the optimizer focused on stable action-conditioning."
    assert bundle["current_signal"] == "- Motion fidelity is still the bottleneck."
    assert len(bundle["recent_runs"]) == 3
    assert bundle["recent_runs"][-1]["experiment_name"] == "optimizer_example_3"
    assert bundle["latest_artifacts"]["comparison_video"] == "runs/comparison_3.mp4"


def test_request_codex_loop_decision_retries_on_invalid_payload(monkeypatch) -> None:
    """Retry once when Codex returns malformed structured output."""
    prompts: list[str] = []
    payloads = [
        {"action_type": "run_experiment", "analysis_summary": "", "reasoning": [], "next_work_note": ""},
        {
            "action_type": "stop",
            "analysis_summary": "Budget exhausted for now.",
            "reasoning": ["The current budget should stop the loop."],
            "next_work_note": "Resume after reviewing the last checkpoint.",
            "stop": {"reason": "budget exhausted"},
        },
    ]

    def fake_run_codex_exec(*, prompt: str, **_: object) -> SimpleNamespace:
        prompts.append(prompt)
        return SimpleNamespace(payload=payloads[len(prompts) - 1])

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    decision, attempt_count = request_codex_loop_decision(
        context_bundle={"goal": "test"},
        codex_model=None,
    )

    assert decision["action_type"] == "stop"
    assert attempt_count == 2
    assert len(prompts) == 2
    assert "did not validate against the required schema" in prompts[1]


def test_loop_decision_schema_requires_all_branch_keys() -> None:
    """Keep the Codex output schema compatible with strict structured-output validation."""
    schema = controller_module._loop_decision_schema()

    assert set(schema["required"]) == {
        "action_type",
        "analysis_summary",
        "reasoning",
        "next_work_note",
        "run_experiment",
        "inspect_artifact",
        "apply_repo_edit",
        "stop",
    }
    inspect_artifact = schema["properties"]["inspect_artifact"]
    assert inspect_artifact["type"] == "object"
    assert inspect_artifact["required"] == ["artifact_paths", "code_paths", "questions"]
    run_experiment = schema["properties"]["run_experiment"]
    assert run_experiment["required"] == [
        "continue_latest",
        "train_from_scratch",
        "stage_step",
        "overrides",
    ]
    override_item = run_experiment["properties"]["overrides"]["items"]
    assert override_item["required"] == ["key", "value"]


def test_apply_codex_repo_edit_restores_files_after_failed_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Rollback touched files when a Codex repo edit fails fast validation."""
    source_path = tmp_path / "src" / "example.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("VALUE = 'old'\n", encoding="utf-8")
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "_is_path_allowed_for_autonomous_edit", lambda path: True)

    def fake_apply_unified_diff(unified_diff: str) -> None:
        assert "src/example.py" in unified_diff
        source_path.write_text("VALUE = 'new'\n", encoding="utf-8")

    validation_calls: list[list[str]] = []

    def fake_run_validation_commands(commands: list[str]) -> None:
        validation_calls.append(list(commands))
        if commands == ["python smoke.py"]:
            raise RuntimeError("smoke failed")

    monkeypatch.setattr(controller_module, "_apply_unified_diff", fake_apply_unified_diff)
    monkeypatch.setattr(controller_module, "_run_validation_commands", fake_run_validation_commands)

    result = apply_codex_repo_edit(
        proposal={
            "suspected_root_cause": "The trainer uses the wrong target tensor.",
            "evidence": ["Loss plateaus immediately."],
            "intended_behavior_change": "Use the corrected target tensor.",
            "touched_files": ["src/example.py"],
            "validation_commands": ["pytest tests/test_example.py"],
            "smoke_test_commands": ["python smoke.py"],
            "unified_diff": (
                "diff --git a/src/example.py b/src/example.py\n"
                "--- a/src/example.py\n"
                "+++ b/src/example.py\n"
                "@@\n"
                "-VALUE = 'old'\n"
                "+VALUE = 'new'\n"
            ),
        },
        analysis_summary="The trainer target should be corrected before another real run.",
    )

    assert result["applied"] is False
    assert source_path.read_text(encoding="utf-8") == "VALUE = 'old'\n"
    assert validation_calls == [["pytest tests/test_example.py"], ["python smoke.py"]]


def test_codex_dry_run_loop_persists_budget_and_decision(monkeypatch, tmp_path: Path) -> None:
    """Persist one dry-run Codex decision without launching any experiment commands."""
    memory_path = tmp_path / "training_optimizer.md"
    memory_path.write_text(
        """
## Goal

Stay within budget.

## Next Work

- Keep the stable recipe fixed.
""".lstrip(),
        encoding="utf-8",
    )
    state_path = tmp_path / "runs" / "training_optimizer" / "controller_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: tmp_path / "codex")
    monkeypatch.setattr(controller_module, "load_train_config", lambda path: TrainScriptConfig())
    monkeypatch.setattr(
        controller_module,
        "request_codex_loop_decision",
        lambda **_: (
            {
                "action_type": "stop",
                "analysis_summary": "Dry-run stop after one planning turn.",
                "reasoning": ["No real run should launch in dry-run mode."],
                "next_work_note": "Keep the stable recipe fixed.",
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
                "stop": {"reason": "dry run"},
            },
            1,
        ),
    )

    records = controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        state_path=state_path,
        planner="codex",
        iterations=1,
        dry_run=True,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert records[0]["dry_run"] is True
    assert records[0]["decision"]["action_type"] == "stop"
    assert state["budget"]["iterations_used"] == 1
    assert state["budget"]["codex_calls_used"] == 1
    assert state["decision_history"][-1]["action_type"] == "stop"
