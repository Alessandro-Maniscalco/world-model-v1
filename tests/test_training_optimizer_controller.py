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
    find_codex_visual_review,
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


def test_inspect_run_progress_prefers_latest_checkpoint_when_metrics_run_ahead(tmp_path: Path) -> None:
    """Resume from the last real checkpoint when an interrupted run logged extra metric rows."""
    run_dir = tmp_path / "runs" / "optimizer_demo"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "step_0000826.pt"
    checkpoint_path.write_text("placeholder", encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                json.dumps({"step": 826, "loss": 0.1}),
                json.dumps({"step": 911, "loss": 0.08}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress = controller_module.inspect_run_progress(run_dir)

    assert progress.current_step == 826
    assert progress.checkpoint_path == checkpoint_path
    assert progress.metrics_path == metrics_path


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
        "codex_visual_review": {
            "timestamp": "2026-03-15T12:00:30+00:00",
            "verdict": "fail",
            "summary": "The generated arm path still drifts late in the rollout.",
            "observations": ["The right arm falls off the target trajectory in the last frames."],
            "hypotheses": ["Conditioning may be weaker than the scene prior."],
            "most_likely_hypothesis": "Conditioning may be weaker than the scene prior.",
            "uncertainties": ["Need to inspect whether training windows overrepresent the resting pose."],
            "next_test_rationale": "Inspect the action-conditioning contract before another continuation run.",
            "focus_points_reviewed": ["motion path vs target", "ghosting on moving objects"],
            "recommended_action": "Do not auto-continue without addressing motion fidelity.",
            "comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0000100/comparison.mp4",
            "contact_sheet": "runs/training_optimizer/inspection/optimizer_example_contact_sheet.png",
        },
        "parent_stage_step": 80,
        "parent_checkpoint_path": "runs/optimizer_example/checkpoints/step_0000080.pt",
        "config_delta_from_parent": ["overfit_one_batch"],
        "stage_kind": "diagnostic",
        "baseline_stage_step": 80,
        "baseline_comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0000080/comparison.mp4",
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
    assert "`codex_visual_review`: `FAIL` | The generated arm path still drifts late in the rollout." in sections["Training runs"]
    assert (
        "`comparison_context`: parent_step=80 stage_kind=diagnostic baseline_step=80 "
        "baseline_locked=true config_delta_keys=overfit_one_batch"
    ) in sections["Training runs"]
    assert "Codex chose `run_experiment`" in sections["Codex Analysis"]
    assert "Keep the action-conditioned LoRA recipe fixed" in sections["Codex Analysis"]
    assert "optimizer_example step 100: fail | The generated arm path still drifts late in the rollout." in sections["Codex Visual Reviews"]
    assert "### [controller 2026-03-15T12:00:00+00:00] controller_policy.visual_review_focus" in sections["Controller Edits"]
    assert "`status`: `applied`" in sections["Controller Edits"]


def test_find_codex_visual_review_reads_latest_matching_entry() -> None:
    """Read the latest Codex visual pass/fail review for one stage from markdown."""
    memory_text = """
## Codex Visual Reviews

- [controller 2026-03-16T05:40:00+00:00] optimizer_demo step 400: pass | coarse motion looks acceptable
- [controller 2026-03-16T05:50:00+00:00] optimizer_demo step 826: fail | right arm jumbles late in rollout
- [controller 2026-03-16T05:55:00+00:00] optimizer_demo step 826: fail | final frames still warp the arm
"""

    verdict = find_codex_visual_review(
        memory_text,
        experiment_name="optimizer_demo",
        target_step=826,
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["summary"] == "final frames still warp the arm"


def test_latest_stage_requires_codex_visual_review_until_present() -> None:
    """Block follow-up planning until the newest stage has a Codex visual review."""
    state = {
        "history": [
            {
                "experiment_name": "optimizer_demo",
                "target_step": 826,
            }
        ]
    }

    pending = controller_module._latest_stage_requires_codex_visual_review(state)
    state["history"][0]["codex_visual_review"] = {"verdict": "fail", "summary": "arm jumbles"}
    cleared = controller_module._latest_stage_requires_codex_visual_review(state)

    assert pending == {"experiment_name": "optimizer_demo", "target_step": 826}
    assert cleared is None


def test_latest_stage_codex_visual_gate_does_not_block_on_fail_verdict() -> None:
    """Allow the loop to continue planning after a Codex visual fail verdict is present."""
    state = {
        "history": [
            {
                "experiment_name": "optimizer_demo",
                "target_step": 826,
                "codex_visual_review": {
                    "verdict": "fail",
                    "summary": "arm jumbles",
                },
            }
        ]
    }

    stop_reason = controller_module._latest_stage_codex_visual_gate(
        memory_text=(
            "## Codex Visual Reviews\n\n"
            "- [controller 2026-03-16T05:55:00+00:00] optimizer_demo step 826: fail | arm jumbles\n"
        ),
        state=state,
    )
    assert stop_reason is None


def test_latest_stage_codex_visual_gate_allows_fail_without_override() -> None:
    """A Codex visual fail should feed the next decision, not require a Next Work override."""
    state = {
        "history": [
            {
                "experiment_name": "optimizer_demo",
                "target_step": 826,
                "codex_visual_review": {
                    "verdict": "fail",
                    "summary": "arm jumbles",
                },
            }
        ]
    }

    cleared = controller_module._latest_stage_codex_visual_gate(
        memory_text=(
            "## Next Work\n\n"
            "- Keep the stable recipe fixed while investigating the failure.\n\n"
            "## Codex Visual Reviews\n\n"
            "- [controller 2026-03-16T05:55:00+00:00] optimizer_demo step 826: fail | arm jumbles\n"
        ),
        state=state,
    )

    assert cleared is None


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
        "baseline_stage_step": 100,
        "baseline_comparison_video": "runs/comparison_100.mp4",
    }

    updated = append_stage_record(state, record)

    assert updated["history"] == [record]
    assert updated["latest_recommendation"] == {"summary": "continue to `step 200`"}
    assert updated["comparison_baselines"]["optimizer_example"] == {
        "baseline_stage_step": 100,
        "baseline_comparison_video": "runs/comparison_100.mp4",
    }
    assert updated["codex_state"]["last_action_type"] == "inspect_artifact"
    assert updated["budget"]["iterations_used"] == 1
    assert updated["decision_history"] == [{"action_type": "inspect_artifact"}]
    assert updated["inspection_history"] == [{"artifact_paths": ["runs/example.mp4"]}]


def test_build_stage_comparison_metadata_marks_clean_continuation() -> None:
    """Record parent lineage and keep the pinned non-diagnostic baseline for a continuation."""
    state = {
        "history": [
            {
                "experiment_name": "optimizer_example",
                "target_step": 400,
                "stage_kind": "continuation",
                "plan": {
                    "resolved_config": {
                        "conditioning_mode": "action",
                        "frame_height": 240,
                        "frame_width": 320,
                        "overfit_one_batch": False,
                    }
                },
                "visual_review": {
                    "comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0000400/comparison.mp4"
                },
            }
        ],
        "comparison_baselines": {
            "optimizer_example": {
                "baseline_stage_step": 400,
                "baseline_comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0000400/comparison.mp4",
            }
        },
    }
    plan = controller_module.ExperimentPlan(
        experiment_name="optimizer_example",
        output_dir=Path("runs/optimizer_example"),
        overrides={},
        resolved_config={
            "conditioning_mode": "action",
            "frame_height": 240,
            "frame_width": 320,
            "overfit_one_batch": False,
        },
        current_step=400,
        target_step=800,
        stage_step=400,
        resume_from=Path("runs/optimizer_example/checkpoints/step_0000400.pt"),
        reasoning=("continue",),
    )

    metadata = controller_module._build_stage_comparison_metadata(
        plan=plan,
        state=state,
        comparison_video=Path("runs/training_optimizer/eval/optimizer_example_step_0000800/comparison.mp4"),
    )

    assert metadata["parent_stage_step"] == 400
    assert metadata["parent_checkpoint_path"] == "runs/optimizer_example/checkpoints/step_0000400.pt"
    assert metadata["config_delta_from_parent"] == []
    assert metadata["stage_kind"] == "continuation"
    assert metadata["baseline_stage_step"] == 400
    assert metadata["baseline_comparison_video"] == "runs/training_optimizer/eval/optimizer_example_step_0000400/comparison.mp4"


def test_build_stage_comparison_metadata_marks_diagnostic_without_promoting_baseline() -> None:
    """Keep the pinned baseline when a resumed stage changes diagnostic knobs."""
    state = {
        "history": [
            {
                "experiment_name": "optimizer_example",
                "target_step": 2452,
                "stage_kind": "continuation",
                "plan": {
                    "resolved_config": {
                        "conditioning_mode": "action",
                        "frame_height": 240,
                        "frame_width": 320,
                        "overfit_one_batch": False,
                    }
                },
                "visual_review": {
                    "comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0002452/comparison.mp4"
                },
            }
        ],
        "comparison_baselines": {
            "optimizer_example": {
                "baseline_stage_step": 2452,
                "baseline_comparison_video": "runs/training_optimizer/eval/optimizer_example_step_0002452/comparison.mp4",
            }
        },
    }
    plan = controller_module.ExperimentPlan(
        experiment_name="optimizer_example",
        output_dir=Path("runs/optimizer_example"),
        overrides={"overfit_one_batch": True},
        resolved_config={
            "conditioning_mode": "action",
            "frame_height": 240,
            "frame_width": 320,
            "overfit_one_batch": True,
        },
        current_step=2452,
        target_step=2852,
        stage_step=400,
        resume_from=Path("runs/optimizer_example/checkpoints/step_0002452.pt"),
        reasoning=("diagnose",),
    )

    metadata = controller_module._build_stage_comparison_metadata(
        plan=plan,
        state=state,
        comparison_video=Path("runs/training_optimizer/eval/optimizer_example_step_0002852/comparison.mp4"),
    )

    assert metadata["parent_stage_step"] == 2452
    assert metadata["stage_kind"] == "diagnostic"
    assert metadata["config_delta_from_parent"] == ["overfit_one_batch"]
    assert metadata["baseline_stage_step"] == 2452
    assert metadata["baseline_comparison_video"] == "runs/training_optimizer/eval/optimizer_example_step_0002452/comparison.mp4"


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
                "parent_stage_step": None if index == 0 else index * 100,
                "stage_kind": "continuation",
                "baseline_stage_step": 100,
                "config_delta_from_parent": [],
                "visual_review": {"focus_points": ["blur", "ghosting"]},
            }
            for index in range(4)
        ],
        "latest_recommendation": {"summary": "continue"},
        "decision_history": [{"action_type": "inspect_artifact"}],
        "edit_history": [{"edit_id": "codex_repo_edit_123"}],
        "codex_state": {"last_run_failure": None},
        "comparison_baselines": {
            "optimizer_example_3": {
                "baseline_stage_step": 100,
                "baseline_comparison_video": "runs/comparison_0.mp4",
            }
        },
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
    assert bundle["workspace_hints"]["memory_path"] == "docs/training_optimizer.md"
    assert "src/world_model/training/flow_matching.py" in bundle["workspace_hints"]["likely_code_paths"]
    assert len(bundle["recent_runs"]) == 3
    assert bundle["recent_runs"][-1]["experiment_name"] == "optimizer_example_3"
    assert bundle["latest_artifacts"]["comparison_video"] == "runs/comparison_3.mp4"
    assert bundle["comparison_context"] == {
        "parent_stage_step": 300,
        "stage_kind": "continuation",
        "baseline_stage_step": 100,
        "baseline_locked": True,
        "config_delta_keys": [],
    }
    assert "codex_visual_review_verdict" in bundle["recent_runs"][-1]


def test_build_codex_context_bundle_compacts_for_active_session() -> None:
    """Trim broad repeated sections when a live Codex session is being continued."""
    state = {
        "history": [
            {
                "timestamp": "2026-03-15T12:00:00+00:00",
                "experiment_name": "optimizer_example",
                "target_step": 2452,
                "metrics": {
                    "last_loss": 0.12,
                    "stage_mean_loss": 0.2,
                    "relative_stage_improvement": 0.08,
                },
                "plausibility": {
                    "plausible": True,
                    "temporal_delta_ratio": 2.1,
                    "video_flags": [],
                },
                "codex_visual_review": {
                    "verdict": "fail",
                    "summary": "The arm falls toward a common resting pose.",
                    "observations": ["The arm drifts down-right late in the rollout."],
                    "most_likely_hypothesis": "A frequent-pose prior may dominate conditioning.",
                    "next_test_rationale": "Inspect data/window composition before retraining.",
                },
                "parent_stage_step": 826,
                "stage_kind": "diagnostic",
                "baseline_stage_step": 2452,
                "config_delta_from_parent": ["overfit_one_batch"],
            }
        ],
        "latest_recommendation": {
            "experiment_name": "optimizer_example",
            "current_step": 2452,
            "target_step": 2852,
            "stage_step": 400,
            "resume_from": "runs/optimizer_example/checkpoints/step_0002452.pt",
            "summary": "Continue only if the next test still supports the branch.",
            "resolved_config": {"lr": 1e-4},
        },
        "latest_record": None,
        "codex_state": {"session_turns": 2, "last_session_reset_reason": None},
        "decision_history": [
            {
                "timestamp": "2026-03-15T11:55:00+00:00",
                "action_type": "inspect_artifact",
                "analysis_summary": "Inspect the window composition.",
                "reasoning": ["Visual failure is stronger than the plausibility pass."],
                "session_id": "session-123",
            }
        ],
        "edit_history": [
            {
                "timestamp": "2026-03-15T11:58:00+00:00",
                "edit_id": "edit-1",
                "applied": False,
                "suspected_root_cause": "conditioning mismatch",
                "error": "patch failed",
            }
        ],
        "retrieved_context_cache": {
            "code": {"src/world_model/training/flow_matching.py": {}},
            "artifacts": {"runs/sweep_local/demo.mp4": {}},
        },
        "codex_memory_summary": {"summary": "The active blocker is motion fidelity."},
        "comparison_baselines": {
            "optimizer_example": {
                "baseline_stage_step": 2452,
                "baseline_comparison_video": "runs/comparison_2452.mp4",
            }
        },
    }
    budget = LoopBudget(
        max_iterations=1,
        max_real_runs=1,
        max_codex_calls=4,
        max_failed_runs=3,
        max_edit_cycles=1,
        max_wall_clock_minutes=None,
        iterations_used=0,
        real_runs_used=0,
        codex_calls_used=0,
        failed_runs_used=0,
        edit_cycles_used=0,
        started_at="2026-03-15T12:00:00+00:00",
    )
    memory_text = """
## Stable Findings

- Stable finding A.
- Stable finding B.

## Codex Analysis

- Prior analysis.

## Controller Edits

- Prior edit.

## Next Work

- Continue the same branch.
"""

    bundle = build_codex_context_bundle(
        train_config=TrainScriptConfig(),
        memory_text=memory_text,
        state=state,
        budget=budget,
        pending_controller_edits=[],
        memory_mode="hybrid",
        session_policy={"session_id": "session-123", "reuse_session": True, "session_turns": 2, "reset_reason": None},
    )

    assert bundle["context_mode"] == "continuation"
    assert "codex_visual_reviews" not in bundle
    assert "stable_findings" not in bundle
    assert len(bundle["recent_runs"]) == 1
    assert bundle["recent_runs"][0]["experiment_name"] == "optimizer_example"
    assert "codex_visual_review_observations" not in bundle["recent_runs"][0]
    assert bundle["comparison_context"] == {
        "parent_stage_step": 826,
        "stage_kind": "diagnostic",
        "baseline_stage_step": 2452,
        "baseline_locked": False,
        "config_delta_keys": ["overfit_one_batch"],
    }
    assert bundle["durable_memory_summary"] == "The active blocker is motion fidelity."
    assert bundle["workspace_hints"]["memory_path"] == "docs/training_optimizer.md"
    assert bundle["latest_recommendation"] == {
        "experiment_name": "optimizer_example",
        "current_step": 2452,
        "target_step": 2852,
        "stage_step": 400,
        "resume_from": "runs/optimizer_example/checkpoints/step_0002452.pt",
        "summary": "Continue only if the next test still supports the branch.",
    }
    assert bundle["recent_decisions"] == [
        {
            "timestamp": "2026-03-15T11:55:00+00:00",
            "action_type": "inspect_artifact",
            "analysis_summary": "Inspect the window composition.",
        }
    ]
    assert bundle["recent_edits"] == [
        {
            "timestamp": "2026-03-15T11:58:00+00:00",
            "edit_id": "edit-1",
            "applied": False,
            "suspected_root_cause": "conditioning mismatch",
            "error": "patch failed",
        }
    ]
    assert bundle["codex_memory_summary"]["summary"] == "The active blocker is motion fidelity."
    assert bundle["retrieved_context_cache"]["code_paths"] == ["src/world_model/training/flow_matching.py"]


def test_resolve_codex_session_policy_allows_unbounded_turn_reuse(monkeypatch) -> None:
    """Keep reusing the same Codex session when the turn limit is disabled."""
    monkeypatch.setattr(
        controller_module,
        "load_codex_session_metadata",
        lambda session_id: SimpleNamespace(cwd=str(controller_module.REPO_ROOT)) if session_id == "session-123" else None,
    )

    policy = controller_module._resolve_codex_session_policy(
        state={
            "codex_state": {
                "session_id": "session-123",
                "session_turns": 20,
                "session_model": None,
                "memory_mode": "hybrid",
                "session_started_at": controller_module._utc_timestamp(),
            }
        },
        memory_mode="hybrid",
        codex_model=None,
        explicit_session_id=None,
        max_session_turns=0,
        max_session_age_minutes=180,
    )

    assert policy["reuse_session"] is True
    assert policy["session_id"] == "session-123"
    assert policy["reset_reason"] is None


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
        return SimpleNamespace(
            payload=payloads[len(prompts) - 1],
            session_id="session-123",
            session_reused=False,
            session_reset_reason=None,
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    decision, attempt_count, result = request_codex_loop_decision(
        context_bundle={"goal": "test"},
        codex_model=None,
    )

    assert decision["action_type"] == "stop"
    assert attempt_count == 2
    assert result.session_id == "session-123"
    assert len(prompts) == 2
    assert "did not validate against the required schema" in prompts[1]


def test_request_codex_loop_decision_prompt_includes_uncertainty_reduction_guidance(monkeypatch) -> None:
    """Encode the observation-first reasoning heuristics into the planner prompt."""
    seen: dict[str, str] = {}

    def fake_run_codex_exec(*, prompt: str, **_: object) -> SimpleNamespace:
        seen["prompt"] = prompt
        return SimpleNamespace(
            payload={
                "action_type": "stop",
                "analysis_summary": "Stop after prompt inspection.",
                "reasoning": ["Prompt content verified."],
                "next_work_note": "Resume later.",
                "run_experiment": {
                    "continue_latest": False,
                    "train_from_scratch": False,
                    "stage_step": None,
                    "overrides": [],
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
                "stop": {"reason": "done"},
            },
            session_id="session-123",
            session_reused=False,
            session_reset_reason=None,
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.request_codex_loop_decision(
        context_bundle={
            "goal": "test",
            "workspace_hints": {
                "memory_path": "docs/training_optimizer.md",
                "state_path": "runs/training_optimizer/controller_state.json",
                "likely_code_paths": ["src/world_model/training/flow_matching.py"],
            },
        },
        codex_model=None,
    )

    assert "Prefer the next test that most reduces uncertainty about the failure" in seen["prompt"]
    assert "Start from a concrete observation, then form hypotheses." in seen["prompt"]
    assert '"dataset prior dominates conditioning" is a hypothesis.' in seen["prompt"]
    assert "Required JSON schema for the final response" not in seen["prompt"]
    assert "Context JSON:" not in seen["prompt"]
    assert "## Current Objective" in seen["prompt"]
    assert "## Workspace Hints" in seen["prompt"]


def test_request_codex_loop_decision_prompt_uses_workspace_hints_and_budget(monkeypatch) -> None:
    """Render a section-based planner prompt under the continuation budget."""
    seen: dict[str, str] = {}

    def fake_run_codex_exec(*, prompt: str, debug_metadata: dict[str, object], **_: object) -> SimpleNamespace:
        seen["prompt"] = prompt
        seen["debug_metadata"] = json.dumps(debug_metadata, sort_keys=True)
        return SimpleNamespace(
            payload={
                "action_type": "stop",
                "analysis_summary": "Stop after prompt inspection.",
                "reasoning": ["Prompt content verified."],
                "next_work_note": "Resume later.",
                "run_experiment": {
                    "continue_latest": False,
                    "train_from_scratch": False,
                    "stage_step": None,
                    "overrides": [],
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
                "stop": {"reason": "done"},
            },
            session_id="session-123",
            session_reused=True,
            session_reset_reason=None,
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    controller_module.request_codex_loop_decision(
        context_bundle={
            "context_mode": "continuation",
            "goal": "Keep motion faithful.",
            "current_signal": "- The arm drifts down-right.",
            "next_work": "- Prefer the next test that reduces uncertainty.",
            "durable_memory_summary": "The current blocker is motion fidelity.",
            "latest_run_summary": {
                "experiment_name": "optimizer_demo",
                "target_step": 2452,
                "codex_visual_review_verdict": "fail",
            },
            "latest_codex_visual_review": {
                "verdict": "fail",
                "summary": "The arm falls toward a common resting pose.",
                "most_likely_hypothesis": "A frequent-pose prior may dominate conditioning.",
                "next_test_rationale": "Inspect window composition before retraining.",
            },
            "comparison_context": {
                "parent_stage_step": 2452,
                "stage_kind": "diagnostic",
                "baseline_stage_step": 2452,
                "baseline_locked": True,
                "config_delta_keys": ["overfit_one_batch"],
            },
            "workspace_hints": {
                "memory_path": "docs/training_optimizer.md",
                "state_path": "runs/training_optimizer/controller_state.json",
                "latest_evaluation_dir": "runs/training_optimizer/eval/demo",
                "likely_code_paths": ["src/world_model/training/flow_matching.py"],
            },
            "recent_decisions": [{"action_type": "inspect_artifact", "analysis_summary": "Inspect first."}],
            "recent_edits": [{"edit_id": "edit-1", "applied": False}],
            "pending_controller_edits": [{"edit_id": "edit-2", "applied": False}],
            "latest_recommendation": {"summary": "Continue only if visual evidence improves."},
        },
        codex_model=None,
        session_id="session-123",
    )

    assert "docs/training_optimizer.md" in seen["prompt"]
    assert "runs/training_optimizer/controller_state.json" in seen["prompt"]
    assert "## Comparison Context" in seen["prompt"]
    assert '"config_delta_keys": [' in seen["prompt"]
    assert len(seen["prompt"]) <= controller_module.DEFAULT_CODEX_CONTINUATION_PROMPT_CHAR_BUDGET
    assert '"prompt_compaction_mode": "budgeted_sections"' in seen["debug_metadata"]


def test_request_codex_visual_review_uses_observation_first_prompt_and_schema(monkeypatch, tmp_path: Path) -> None:
    """Request richer observation-first visual review output from Codex."""
    comparison_video = tmp_path / "comparison.mp4"
    comparison_video.write_bytes(b"video")
    seen: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "_build_video_contact_sheet", lambda *_, **__: tmp_path / "contact.png")

    def fake_run_codex_exec(*, prompt: str, schema: dict[str, object], **_: object) -> SimpleNamespace:
        seen["prompt"] = prompt
        seen["schema"] = schema
        return SimpleNamespace(
            payload={
                "verdict": "fail",
                "summary": "The arm falls toward a common resting pose.",
                "observations": ["The arm drops toward the bottom-right late in the rollout."],
                "hypotheses": ["A frequent-pose prior may dominate conditioning."],
                "most_likely_hypothesis": "A frequent-pose prior may dominate conditioning.",
                "uncertainties": ["Need to inspect training-window motion density."],
                "next_test_rationale": "Inspect window composition before trying a longer horizon.",
                "focus_points_reviewed": ["motion path mismatch", "collapse to a common/default pose or scene prior"],
                "recommended_action": "Inspect data/window composition.",
            },
            session_id="session-123",
            session_reused=False,
            session_reset_reason=None,
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    review, _ = controller_module.request_codex_visual_review(
        record={
            "experiment_name": "optimizer_demo",
            "target_step": 2452,
            "visual_review": {
                "comparison_video": str(comparison_video),
                "comparison_layout": "left=reference,right=generated",
                "focus_points": ["motion path mismatch"],
                "summary": "review summary",
            },
            "plausibility": {"plausible": True},
            "metrics": {"last_loss": 0.1, "stage_mean_loss": 0.2, "relative_stage_improvement": 0.3},
        },
        codex_model=None,
        codex_timeout_seconds=30,
    )

    schema = seen["schema"]
    assert "Start from concrete visual observations only." in seen["prompt"]
    assert "Prefer the next test that most reduces uncertainty about the failure" in seen["prompt"]
    assert schema["required"] == [
        "verdict",
        "summary",
        "observations",
        "hypotheses",
        "most_likely_hypothesis",
        "uncertainties",
        "next_test_rationale",
        "focus_points_reviewed",
        "recommended_action",
    ]
    assert review["most_likely_hypothesis"] == "A frequent-pose prior may dominate conditioning."
    assert review["next_test_rationale"] == "Inspect window composition before trying a longer horizon."


def test_request_codex_loop_decision_prefers_inspection_after_visual_fail(monkeypatch) -> None:
    """A visual fail should steer the planner toward inspection instead of blind continuation."""
    def fake_run_codex_exec(*, prompt: str, **_: object) -> SimpleNamespace:
        assert "Latest Codex Visual Review" in prompt
        assert '"verdict": "fail"' in prompt
        return SimpleNamespace(
            payload={
                "action_type": "inspect_artifact",
                "analysis_summary": "Inspect the action-conditioning contract first.",
                "reasoning": ["The visual fail is stronger evidence than the plausibility pass."],
                "next_work_note": "Inspect before retraining.",
                "run_experiment": {
                    "continue_latest": False,
                    "train_from_scratch": False,
                    "stage_step": None,
                    "overrides": [],
                },
                "inspect_artifact": {
                    "artifact_paths": ["runs/comparison.mp4"],
                    "code_paths": ["src/world_model/training/flow_matching.py"],
                    "questions": ["Is conditioning weaker than the scene prior?"],
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
                "stop": {"reason": ""},
            },
            session_id="session-123",
            session_reused=False,
            session_reset_reason=None,
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    decision, attempt_count, _ = controller_module.request_codex_loop_decision(
        context_bundle={
            "goal": "test",
            "recent_runs": [
                {
                    "experiment_name": "optimizer_demo",
                    "codex_visual_review_verdict": "fail",
                    "codex_visual_review_summary": "The arm falls toward a common resting pose.",
                    "codex_visual_review_observations": ["The arm drops toward the bottom-right."],
                    "codex_visual_review_most_likely_hypothesis": "A frequent-pose prior may dominate conditioning.",
                    "codex_visual_review_next_test_rationale": "Inspect window composition before trying a longer horizon.",
                }
            ],
            "latest_codex_visual_review": {
                "verdict": "fail",
                "summary": "The arm falls toward a common resting pose.",
            },
        },
        codex_model=None,
    )

    assert attempt_count == 1
    assert decision["action_type"] == "inspect_artifact"


def test_request_codex_loop_decision_forwards_timeout(monkeypatch) -> None:
    """Pass the configured Codex timeout through to the CLI wrapper."""
    seen: dict[str, object] = {}

    def fake_run_codex_exec(*, timeout_seconds: int, **_: object) -> SimpleNamespace:
        seen["timeout_seconds"] = timeout_seconds
        return SimpleNamespace(
            payload={
                "action_type": "stop",
                "analysis_summary": "Stop after timeout plumbing check.",
                "reasoning": ["The timeout value reached the Codex runner."],
                "next_work_note": "Resume later.",
                "run_experiment": {
                    "continue_latest": False,
                    "train_from_scratch": False,
                    "stage_step": None,
                    "overrides": [],
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
                "stop": {"reason": "done"},
            },
            session_id="session-123",
            session_reused=False,
            session_reset_reason=None,
        )

    monkeypatch.setattr(controller_module, "run_codex_exec", fake_run_codex_exec)

    decision, attempt_count, _ = request_codex_loop_decision(
        context_bundle={"goal": "test"},
        codex_model=None,
        codex_timeout_seconds=42,
    )

    assert decision["action_type"] == "stop"
    assert attempt_count == 1
    assert seen["timeout_seconds"] == 42


def test_build_codex_memory_summary_prompt_uses_compact_context() -> None:
    """Keep durable-memory refresh prompts smaller than the full planning context."""
    state = {
        "codex_memory_summary": {"summary": "Current blocker is motion fidelity."},
        "decision_history": [
            {
                "timestamp": "2026-03-15T11:55:00+00:00",
                "action_type": "inspect_artifact",
                "analysis_summary": "Inspect the window composition.",
                "reasoning": ["The visual fail is stronger than the plausibility pass."],
            }
        ],
        "edit_history": [
            {
                "timestamp": "2026-03-15T11:58:00+00:00",
                "edit_id": "edit-1",
                "applied": False,
                "suspected_root_cause": "conditioning mismatch",
                "error": "patch failed",
            }
        ],
        "history": [
            {
                "timestamp": "2026-03-15T12:00:00+00:00",
                "experiment_name": "optimizer_example",
                "target_step": 2452,
                "metrics": {
                    "last_loss": 0.12,
                    "stage_mean_loss": 0.2,
                    "relative_stage_improvement": 0.08,
                },
                "plausibility": {
                    "plausible": True,
                    "temporal_delta_ratio": 2.1,
                    "video_flags": [],
                },
                "codex_visual_review": {
                    "verdict": "fail",
                    "summary": "The arm falls toward a common resting pose.",
                    "observations": ["The arm drifts down-right late in the rollout."],
                    "most_likely_hypothesis": "A frequent-pose prior may dominate conditioning.",
                    "next_test_rationale": "Inspect data/window composition before retraining.",
                },
            }
        ],
    }
    budget = LoopBudget(
        max_iterations=12,
        max_real_runs=6,
        max_codex_calls=30,
        max_failed_runs=6,
        max_edit_cycles=4,
        max_wall_clock_minutes=None,
        iterations_used=1,
        real_runs_used=1,
        codex_calls_used=3,
        failed_runs_used=0,
        edit_cycles_used=0,
        started_at="2026-03-15T12:00:00+00:00",
    )

    prompt = controller_module._build_codex_memory_summary_prompt(
        state=state,
        decision={
            "action_type": "inspect_artifact",
            "analysis_summary": "Inspect the window composition.",
            "reasoning": ["The visual fail is stronger than the plausibility pass."],
        },
        budget=budget,
    )

    assert "existing_summary" in prompt
    assert "codex_visual_review_observations" not in prompt
    assert '"experiment_name": "optimizer_example"' in prompt


def test_should_refresh_codex_memory_summary_is_less_eager_for_inspection() -> None:
    """Avoid refreshing durable memory after every inspection-only turn."""
    state = {
        "codex_state": {"session_turns": 2},
        "codex_memory_summary": {"summary": "Current blocker is motion fidelity."},
    }

    should_refresh = controller_module._should_refresh_codex_memory_summary(
        state=state,
        decision={"action_type": "inspect_artifact"},
        memory_mode="hybrid",
    )

    assert should_refresh is False


def test_prepare_codex_inspection_context_reuses_cached_paths_within_session(tmp_path: Path) -> None:
    """Avoid reattaching unchanged excerpts and artifacts in the same Codex session."""
    code_path = tmp_path / "src" / "example.py"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text("VALUE = 1\n", encoding="utf-8")
    artifact_path = tmp_path / "runs" / "metrics.jsonl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text('{"step": 1, "loss": 0.5}\n', encoding="utf-8")

    state = {
        "history": [],
        "retrieved_context_cache": {
                "code": {
                    "src/example.py": {
                        "excerpt_hash": controller_module._short_hash("VALUE = 1"),
                        "fingerprint": controller_module._build_path_fingerprint(code_path),
                        "session_id": "session-123",
                    }
                },
            "artifacts": {
                "runs/metrics.jsonl": {
                    "fingerprint": controller_module._build_path_fingerprint(artifact_path),
                    "session_id": "session-123",
                }
            },
        },
    }

    controller_module.REPO_ROOT = tmp_path
    try:
        context = controller_module.prepare_codex_inspection_context(
            request={
                "artifact_paths": ["runs/metrics.jsonl"],
                "code_paths": ["src/example.py"],
                "questions": [],
            },
            state=state,
            session_id="session-123",
            memory_mode="hybrid",
        )
    finally:
        controller_module.REPO_ROOT = Path(__file__).resolve().parents[1]

    assert context["image_inputs"] == []
    assert context["summary"]["reused_code_paths"] == 1
    assert context["summary"]["reused_artifact_paths"] == 1
    assert context["payload"]["code_snippets"][0]["already_shared_in_session"] is True


def test_validate_loop_decision_payload_normalizes_non_positive_stage_step() -> None:
    """Treat a non-positive Codex stage step as unspecified instead of crashing later."""
    payload = controller_module._validate_loop_decision_payload(
        {
            "action_type": "run_experiment",
            "analysis_summary": "Continue the current branch.",
            "reasoning": ["The same recipe should continue."],
            "next_work_note": "Run one more stage.",
            "run_experiment": {
                "continue_latest": True,
                "train_from_scratch": False,
                "stage_step": 0,
                "overrides": [],
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
            "stop": {"reason": ""},
        }
    )

    assert payload["run_experiment"]["stage_step"] is None


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


def test_apply_codex_repo_edit_strips_prose_around_diff(monkeypatch, tmp_path: Path) -> None:
    """Recover a valid unified diff when Codex wraps it in prose."""
    source_path = tmp_path / "src" / "example.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("VALUE = 'old'\n", encoding="utf-8")
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "_is_path_allowed_for_autonomous_edit", lambda path: True)
    monkeypatch.setattr(controller_module, "_run_validation_commands", lambda commands: None)

    seen: dict[str, str] = {}

    def fake_apply_unified_diff(unified_diff: str) -> None:
        seen["diff"] = unified_diff
        source_path.write_text("VALUE = 'new'\n", encoding="utf-8")

    monkeypatch.setattr(controller_module, "_apply_unified_diff", fake_apply_unified_diff)

    result = apply_codex_repo_edit(
        proposal={
            "suspected_root_cause": "The trainer uses the wrong target tensor.",
            "evidence": ["Loss plateaus immediately."],
            "intended_behavior_change": "Use the corrected target tensor.",
            "touched_files": ["src/example.py"],
            "validation_commands": ["pytest tests/test_example.py"],
            "smoke_test_commands": ["python smoke.py"],
            "unified_diff": (
                "I found the fix below.\n\n"
                "diff --git a/src/example.py b/src/example.py\n"
                "--- a/src/example.py\n"
                "+++ b/src/example.py\n"
                "@@\n"
                "-VALUE = 'old'\n"
                "+VALUE = 'new'\n"
                "\nPlease apply this patch.\n"
            ),
        },
        analysis_summary="Apply the corrected target tensor fix.",
    )

    assert result["applied"] is True
    assert seen["diff"].startswith("diff --git a/src/example.py b/src/example.py\n")
    assert "Please apply this patch." not in seen["diff"]


def test_apply_codex_repo_edit_recovers_fenced_diff_block(monkeypatch, tmp_path: Path) -> None:
    """Extract and apply a fenced diff block from Codex output."""
    source_path = tmp_path / "src" / "example.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("VALUE = 'old'\n", encoding="utf-8")
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "_is_path_allowed_for_autonomous_edit", lambda path: True)
    monkeypatch.setattr(controller_module, "_run_validation_commands", lambda commands: None)

    seen: dict[str, str] = {}

    def fake_apply_unified_diff(unified_diff: str) -> None:
        seen["diff"] = unified_diff
        source_path.write_text("VALUE = 'new'\n", encoding="utf-8")

    monkeypatch.setattr(controller_module, "_apply_unified_diff", fake_apply_unified_diff)

    result = apply_codex_repo_edit(
        proposal={
            "suspected_root_cause": "The trainer uses the wrong target tensor.",
            "evidence": ["Loss plateaus immediately."],
            "intended_behavior_change": "Use the corrected target tensor.",
            "touched_files": ["src/example.py"],
            "validation_commands": ["pytest tests/test_example.py"],
            "smoke_test_commands": ["python smoke.py"],
            "unified_diff": (
                "Here is the patch:\n"
                "```diff\n"
                "diff --git a/src/example.py b/src/example.py\n"
                "--- a/src/example.py\n"
                "+++ b/src/example.py\n"
                "@@\n"
                "-VALUE = 'old'\n"
                "+VALUE = 'new'\n"
                "```\n"
            ),
        },
        analysis_summary="Apply the corrected target tensor fix.",
    )

    assert result["applied"] is True
    assert seen["diff"].startswith("diff --git a/src/example.py b/src/example.py\n")


def test_apply_codex_repo_edit_allows_one_repair_retry(monkeypatch, tmp_path: Path) -> None:
    """Retry once with a cleaned alternate diff when the first apply attempt fails."""
    source_path = tmp_path / "src" / "example.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("VALUE = 'old'\n", encoding="utf-8")
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "_is_path_allowed_for_autonomous_edit", lambda path: True)
    monkeypatch.setattr(controller_module, "_run_validation_commands", lambda commands: None)

    apply_attempts: list[str] = []

    def fake_apply_unified_diff(unified_diff: str) -> None:
        apply_attempts.append(unified_diff)
        if len(apply_attempts) == 1:
            raise RuntimeError("git apply failed: patch with only garbage at line 5")
        source_path.write_text("VALUE = 'new'\n", encoding="utf-8")

    monkeypatch.setattr(controller_module, "_apply_unified_diff", fake_apply_unified_diff)

    result = apply_codex_repo_edit(
        proposal={
            "suspected_root_cause": "The trainer uses the wrong target tensor.",
            "evidence": ["Loss plateaus immediately."],
            "intended_behavior_change": "Use the corrected target tensor.",
            "touched_files": ["src/example.py"],
            "validation_commands": ["pytest tests/test_example.py"],
            "smoke_test_commands": ["python smoke.py"],
            "unified_diff": (
                "Please use this change.\n"
                "```diff\n"
                "diff --git a/src/example.py b/src/example.py\n"
                "--- a/src/example.py\n"
                "+++ b/src/example.py\n"
                "@@\n"
                "-VALUE = 'old'\n"
                "+VALUE = 'new'\n"
                "```\n"
            ),
        },
        analysis_summary="Apply the corrected target tensor fix.",
    )

    assert result["applied"] is True
    assert result["repair_attempted"] is True
    assert len(apply_attempts) == 2


def test_apply_codex_repo_edit_repairs_stale_hunk_headers(monkeypatch, tmp_path: Path) -> None:
    """Apply a stale-hunk Codex patch by matching current file content instead of line numbers."""
    flow_path = tmp_path / "src" / "world_model" / "training" / "flow_matching.py"
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    flow_path.write_text(
        (
            '"""Temporary flow-matching fixture."""\n\n'
            "def _chunkwise_teacher_forcing_video_loss(model, noisy_suffix, observed_video, action_tokens, timestep, attn_mask, observed_mask):\n"
            "    for start, end in ((0, 4), (4, 8)):\n"
            "        pred_suffix = model(\n"
            "            noisy_future_video=noisy_suffix,\n"
            "            observed_video=observed_video,\n"
            "            action_tokens=action_tokens[:, start:],\n"
            "            timestep_t=timestep,\n"
            "            block_causal_attention_mask=attn_mask,\n"
            "            observed_mask=observed_mask,\n"
            "            control_hidden_states_scale=None,\n"
            "        )\n"
            "    return pred_suffix\n"
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "tests" / "test_flow_matching.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        (
            '"""Temporary flow-matching tests."""\n\n'
            "from world_model.training.flow_matching import (\n"
            "    make_noisy_and_target,\n"
            "    normalized_t_to_scheduler_timestep,\n"
            "    sample_t,\n"
            "    w,\n"
            ")\n\n"
            "\n"
            "def test_sample_t_shape_and_bounds():\n"
            "    assert True\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(controller_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(controller_module, "_is_path_allowed_for_autonomous_edit", lambda path: True)
    monkeypatch.setattr(controller_module, "_run_validation_commands", lambda commands: None)

    def fake_apply_unified_diff(unified_diff: str) -> None:
        raise RuntimeError(
            "git apply failed: stdout= stderr=error: patch failed: src/world_model/training/flow_matching.py:320\n"
            "error: src/world_model/training/flow_matching.py: patch does not apply"
        )

    monkeypatch.setattr(controller_module, "_apply_unified_diff", fake_apply_unified_diff)

    result = apply_codex_repo_edit(
        proposal={
            "suspected_root_cause": "Chunkwise teacher forcing exposes future action tokens.",
            "evidence": ["Training uses `action_tokens[:, start:]` while inference slices to the active chunk."],
            "intended_behavior_change": "Slice action tokens to the active chunk during teacher forcing.",
            "touched_files": [
                "src/world_model/training/flow_matching.py",
                "tests/test_flow_matching.py",
            ],
            "validation_commands": ["pytest tests/test_flow_matching.py -k current_chunk_action_window"],
            "smoke_test_commands": ["pytest tests/test_flow_matching.py"],
            "unified_diff": (
                "--- a/src/world_model/training/flow_matching.py\n"
                "+++ b/src/world_model/training/flow_matching.py\n"
                "@@ -294,9 +294,9 @@ def _chunkwise_teacher_forcing_video_loss(\n"
                "         pred_suffix = model(\n"
                "             noisy_future_video=noisy_suffix,\n"
                "             observed_video=observed_video,\n"
                "-            action_tokens=action_tokens[:, start:],\n"
                "+            action_tokens=action_tokens[:, start:end],\n"
                "             timestep_t=timestep,\n"
                "             block_causal_attention_mask=attn_mask,\n"
                "             observed_mask=observed_mask,\n"
                "             control_hidden_states_scale=None,\n"
                "         )\n"
                "--- a/tests/test_flow_matching.py\n"
                "+++ b/tests/test_flow_matching.py\n"
                "@@ -3,10 +3,17 @@\n"
                " from world_model.training.flow_matching import (\n"
                "+    chunkwise_teacher_forcing_loss,\n"
                "     make_noisy_and_target,\n"
                "     normalized_t_to_scheduler_timestep,\n"
                "     sample_t,\n"
                "     w,\n"
                " )\n"
                "+\n"
                "+\n"
                "+def test_chunkwise_teacher_forcing_uses_current_chunk_action_window():\n"
                "+    assert chunkwise_teacher_forcing_loss is not None\n"
                " \n"
                " \n"
                " def test_sample_t_shape_and_bounds():\n"
                "     assert True\n"
            ),
        },
        analysis_summary="Apply the active-chunk conditioning fix.",
    )

    assert result["applied"] is True
    assert result["repair_attempted"] is True
    assert "action_tokens=action_tokens[:, start:end]" in flow_path.read_text(encoding="utf-8")
    assert "chunkwise_teacher_forcing_loss" in test_path.read_text(encoding="utf-8")


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
                SimpleNamespace(
                    session_id="session-dry-run",
                    session_reused=False,
                    session_reset_reason=None,
                ),
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


def test_codex_loop_consumes_safe_stop_request_after_completed_stage(monkeypatch, tmp_path: Path) -> None:
    """Stop only after a completed stage has been validated and persisted."""
    memory_path = tmp_path / "training_optimizer.md"
    memory_path.write_text(
        """
## Goal

Run one safe stage.

## Next Work

- Continue the stable branch.
""".lstrip(),
        encoding="utf-8",
    )
    state_path = tmp_path / "runs" / "training_optimizer" / "controller_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    stop_request_path = state_path.parent / controller_module.STOP_AFTER_STAGE_REQUEST_FILENAME
    stop_request_path.write_text("stop after current stage\n", encoding="utf-8")
    train_config_path = tmp_path / "train.yaml"
    train_config_path.write_text("repo_id: demo\n", encoding="utf-8")

    monkeypatch.setattr(controller_module, "ensure_codex_chatgpt_login", lambda: tmp_path / "codex")
    monkeypatch.setattr(controller_module, "load_train_config", lambda path: TrainScriptConfig())
    monkeypatch.setattr(controller_module, "_refresh_codex_memory_summary", lambda **_: 0)
    monkeypatch.setattr(controller_module, "_attach_codex_visual_review_to_record", lambda **_: 0)
    monkeypatch.setattr(
        controller_module,
        "request_codex_loop_decision",
        lambda **_: (
            {
                "action_type": "run_experiment",
                "analysis_summary": "Run one stage then stop safely.",
                "reasoning": ["The stop request should be honored after the stage is recorded."],
                "next_work_note": "Resume later from the completed checkpoint.",
                "run_experiment": {
                    "continue_latest": True,
                    "train_from_scratch": False,
                    "stage_step": 400,
                    "overrides": [],
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
                "stop": {"reason": ""},
            },
            1,
            SimpleNamespace(
                session_id="session-safe-stop",
                session_reused=False,
                session_reset_reason=None,
            ),
        ),
    )
    monkeypatch.setattr(
        controller_module,
        "build_experiment_plan_from_codex_decision",
        lambda **_: controller_module.ExperimentPlan(
            experiment_name="optimizer_safe_stop_demo",
            output_dir=tmp_path / "runs" / "optimizer_safe_stop_demo",
            overrides={},
            resolved_config={"output_dir": str(tmp_path / "runs" / "optimizer_safe_stop_demo")},
            current_step=0,
            target_step=400,
            stage_step=400,
            resume_from=None,
            reasoning=("run one stage",),
        ),
    )
    monkeypatch.setattr(
        controller_module,
        "run_experiment_stage",
        lambda **_: {
            "timestamp": "2026-03-16T12:00:00+00:00",
            "experiment_name": "optimizer_safe_stop_demo",
            "output_dir": str(tmp_path / "runs" / "optimizer_safe_stop_demo"),
            "target_step": 400,
            "checkpoint_path": str(tmp_path / "runs" / "optimizer_safe_stop_demo" / "checkpoints" / "step_0000400.pt"),
            "learning_summary": "safe stop stage summary",
            "commands": {
                "train": ["python", "scripts/train/world_model.py"],
                "sweep": ["python", "scripts/check/sweep_local_repo_resolutions.py"],
                "plausibility": ["python", "scripts/check/check_generated_video_plausibility.py"],
            },
            "metrics": {"last_loss": 0.1, "stage_mean_loss": 0.2},
            "plausibility": {"plausible": True, "mean_frame_mae_rgb_0_255": 1.0, "temporal_delta_ratio": 1.0},
            "score": 1.23,
            "visual_review": {
                "summary": "review",
                "comparison_video": "runs/comparison.mp4",
                "generated_video": "runs/generated.mp4",
                "ffplay_command": ["ffplay", "runs/comparison.mp4"],
                "ffmpeg_extract_command": ["ffmpeg", "-i", "runs/comparison.mp4", "/tmp/frame_%03d.png"],
            },
            "next_recommendation": {
                "experiment_name": "optimizer_safe_stop_demo",
                "output_dir": str(tmp_path / "runs" / "optimizer_safe_stop_demo"),
                "overrides": {},
                "resolved_config": {"output_dir": str(tmp_path / "runs" / "optimizer_safe_stop_demo")},
                "current_step": 400,
                "target_step": 800,
                "stage_step": 400,
                "resume_from": str(tmp_path / "runs" / "optimizer_safe_stop_demo" / "checkpoints" / "step_0000400.pt"),
                "reasoning": ["continue later"],
                "summary": "continue later",
            },
        },
    )

    records = controller_module.run_training_optimization_loop(
        train_config_path=train_config_path,
        memory_path=memory_path,
        state_path=state_path,
        planner="codex",
        iterations=3,
        max_real_runs=3,
        max_codex_calls=6,
        max_failed_runs=2,
        max_edit_cycles=1,
    )

    saved_state = controller_module.load_controller_state(state_path)

    assert len(records) == 1
    assert stop_request_path.exists() is False
    assert saved_state["codex_state"]["last_stop_reason"] == (
        f"stop requested via {controller_module.STOP_AFTER_STAGE_REQUEST_FILENAME} after stage finalization"
    )
    assert saved_state["latest_record"]["experiment_name"] == "optimizer_safe_stop_demo"
