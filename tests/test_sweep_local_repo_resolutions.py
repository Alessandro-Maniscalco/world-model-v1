"""Tests for the local-repo resolution sweep smoke-check script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_script_module():
    """Load the local-repo resolution sweep script from its file path."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "check" / "sweep_local_repo_resolutions.py"
    spec = importlib.util.spec_from_file_location("sweep_local_repo_resolutions", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script_module()


def test_checkpoint_single_resolution_uses_run_stem_and_sweep_local_root() -> None:
    """Checkpoint sweeps should default to a run-named MP4 under runs/sweep_local."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="checkpoint",
        output_dir=None,
        checkpoint_path=Path(
            "runs/test_full_multi_320x240_lora8_none/checkpoints/step_0000800.pt"
        ),
        label="320x240",
        resolution_count=1,
    )

    assert output_path == Path(
        "runs/sweep_local/test_full_multi_320x240_lora8_none_step_0000800.mp4"
    )
    assert comparison_path == Path(
        "runs/sweep_local/test_full_multi_320x240_lora8_none_step_0000800_comparison.mp4"
    )
    assert summary_path == Path(
        "runs/sweep_local/test_full_multi_320x240_lora8_none_step_0000800_summary.json"
    )


def test_checkpoint_multi_resolution_adds_label_suffix() -> None:
    """Checkpoint sweeps should suffix the resolution when multiple sizes are requested."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="checkpoint",
        output_dir=Path("custom_out"),
        checkpoint_path=Path("runs/example_run/checkpoints/step_0000100.pt"),
        label="384x288",
        resolution_count=2,
    )

    assert output_path == Path("custom_out/example_run_step_0000100_384x288.mp4")
    assert comparison_path == Path("custom_out/example_run_step_0000100_384x288_comparison.mp4")
    assert summary_path == Path("custom_out/example_run_step_0000100_summary.json")


def test_base_mode_keeps_resolution_named_outputs() -> None:
    """Base sweeps should keep resolution-named outputs under the shared sweep root."""
    output_path, comparison_path, summary_path = script._resolve_output_artifacts(
        mode="base",
        output_dir=None,
        checkpoint_path=None,
        label="512x384",
        resolution_count=1,
    )

    assert output_path == Path("runs/sweep_local/512x384.mp4")
    assert comparison_path == Path(
        "runs/sweep_local/512x384_comparison.mp4"
    )
    assert summary_path == Path("runs/sweep_local/summary.json")


def test_checkpoint_mode_exports_generated_rollout_and_comparison_order(monkeypatch, tmp_path) -> None:
    """Checkpoint sweeps should export the generated rollout and keep it on the comparison right."""
    target_rollout = torch.full((1, 2, 3, 4, 5), 11.0)
    pred_rollout = torch.full((1, 2, 3, 4, 5), 22.0)
    export_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        script,
        "_load_checkpoint_runtime_config",
        lambda path: (
            {},
            SimpleNamespace(
                frame_width=320,
                frame_height=240,
                trainable_backbone="full",
                conditioning_mode="none",
                context_len=9,
                horizon_len=8,
            ),
        ),
    )
    monkeypatch.setattr(script, "_resolve_device", lambda device_name: torch.device("cpu"))
    monkeypatch.setattr(script, "_select_runtime_dtype", lambda device: torch.float32)
    monkeypatch.setattr(
        script,
        "_load_checkpoint_clip",
        lambda **kwargs: (torch.zeros(1, 17, 3, 240, 320), torch.zeros(17, 1)),
    )
    monkeypatch.setattr(
        script,
        "_run_checkpoint_world_model",
        lambda **kwargs: (target_rollout, pred_rollout),
    )
    monkeypatch.setattr(
        script,
        "_tensor_video_to_frames",
        lambda video_btchw: [float(video_btchw[0, 0, 0, 0, 0].item())],
    )

    side_by_side_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_build_side_by_side_video(*, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Record comparison inputs and return a sentinel tensor."""
        side_by_side_calls.append((left.clone(), right.clone()))
        return torch.full_like(right, 33.0)

    def _fake_export_video(*, video_frames: list[object], output_video_path: str, fps: int) -> None:
        """Capture export arguments without writing MP4 files."""
        export_calls.append(
            {
                "video_frames": video_frames,
                "output_video_path": output_video_path,
                "fps": fps,
            }
        )

    monkeypatch.setattr(script, "_build_side_by_side_video", _fake_build_side_by_side_video)
    monkeypatch.setattr(script, "_export_video", _fake_export_video)

    script._run_one_checkpoint_resolution(
        mode="checkpoint",
        config_path=Path("configs/train/aloha_fork_pick_up.yaml"),
        checkpoint_path=Path("runs/example/checkpoints/step_0000100.pt"),
        width=320,
        height=240,
        output_path=tmp_path / "generated.mp4",
        comparison_path=tmp_path / "comparison.mp4",
        repo_id="repo",
        episode_index=0,
        start_frame=0,
        video_key="observation.images.cam_high",
        context_len=9,
        horizon_len=8,
        k=1,
        integration_steps=10,
        fps=10,
        seed=0,
        single_chunk_rollout=True,
        device_name="cpu",
        action_source="auto",
        prompt="",
        negative_prompt="",
        guidance_scale=1.0,
        max_sequence_length=128,
    )

    assert len(export_calls) == 2
    assert export_calls[0]["video_frames"] == [22.0]
    assert export_calls[0]["output_video_path"] == str(tmp_path / "generated.mp4")
    assert len(side_by_side_calls) == 1
    assert torch.equal(side_by_side_calls[0][0], target_rollout)
    assert torch.equal(side_by_side_calls[0][1], pred_rollout)
    assert export_calls[1]["video_frames"] == [33.0]
    assert export_calls[1]["output_video_path"] == str(tmp_path / "comparison.mp4")
