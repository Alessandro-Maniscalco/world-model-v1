"""Tests for shared inference artifact helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from world_model.config import InferScriptConfig
from world_model.eval import artifacts


def test_select_runtime_dtype_returns_float32_on_cpu() -> None:
    """Keep CPU inference on float32 regardless of AMP preference."""
    dtype = artifacts.select_runtime_dtype(
        device=torch.device("cpu"),
        disable_amp=False,
    )

    assert dtype == torch.float32


def test_to_zero_one_converts_minus_one_to_one_inputs() -> None:
    """Convert normalized `[-1, 1]` videos into `[0, 1]` visualization space."""
    video = torch.tensor([[[[[-1.0]], [[1.0]], [[0.0]]]]], dtype=torch.float32)

    converted = artifacts.to_zero_one(video)

    assert torch.allclose(
        converted,
        torch.tensor([[[[[0.0]], [[1.0]], [[0.5]]]]], dtype=torch.float32),
    )


def test_resample_video_time_uses_nearest_step_selection() -> None:
    """Resample the time axis without changing the surrounding tensor layout."""
    video = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1, 1)

    resampled = artifacts.resample_video_time(video, 3)

    assert tuple(resampled.shape) == (1, 3, 1, 1, 1)
    assert torch.equal(resampled[:, :, 0, 0, 0], torch.tensor([[0.0, 2.0, 4.0]]))


def test_build_frame_report_tracks_raw_latent_and_decoded_counts() -> None:
    """Report the raw, latent, and decoded horizons with one shared helper."""
    cfg = InferScriptConfig(
        context_len=9,
        horizon_len=8,
        num_vis_frames=4,
    )
    prepared = type(
        "Prepared",
        (),
        {
            "total_latent_steps": 6,
            "context_latent_steps": 2,
            "horizon_latent_steps": 4,
        },
    )()

    report = artifacts.build_frame_report(
        cfg=cfg,
        prepared=prepared,
        source_video=torch.zeros(1, 17, 3, 8, 8),
        raw_future=torch.zeros(1, 8, 3, 8, 8),
        raw_future_aligned=torch.zeros(1, 5, 3, 8, 8),
        pred_video=torch.zeros(1, 5, 3, 8, 8),
        target_video=torch.zeros(1, 5, 3, 8, 8),
    )

    assert report["requested_context_frames"] == 9
    assert report["latent_total_steps"] == 6
    assert report["latent_future_steps"] == 4
    assert report["decoded_generated_future_frames"] == 5
    assert report["visualized_frames"] == 4


def test_save_json_report_writes_sorted_json(tmp_path: Path) -> None:
    """Persist artifact reports with stable sorted keys for diffs and inspection."""
    output_path = tmp_path / "report.json"

    artifacts.save_json_report({"b": 2, "a": 1}, output_path)

    assert output_path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
