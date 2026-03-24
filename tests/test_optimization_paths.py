"""Tests for shared training-optimizer path helpers."""

from __future__ import annotations

from pathlib import Path

from world_model.optimization import paths as optimization_paths


def test_derive_state_path_for_default_memory_uses_default_state_name(tmp_path: Path) -> None:
    """Keep the canonical investigation-state filename for the default memory path."""
    derived = optimization_paths.derive_state_path_for_memory_path(
        "docs/investigation.md",
        repo_root=tmp_path,
    )

    assert derived == tmp_path / "runs" / "training_optimizer" / "investigation_controller_state.json"


def test_derive_state_path_for_custom_memory_uses_matching_stem(tmp_path: Path) -> None:
    """Map custom investigation memory files onto isolated controller-state files."""
    derived = optimization_paths.derive_state_path_for_memory_path(
        "docs/investigation1.md",
        repo_root=tmp_path,
    )

    assert derived == tmp_path / "runs" / "training_optimizer" / "investigation1_controller_state.json"


def test_display_repo_path_relativizes_paths_inside_repo(tmp_path: Path) -> None:
    """Render repo-contained absolute paths relative to the chosen repo root."""
    report_path = tmp_path / "runs" / "training_optimizer" / "stop.md"

    assert (
        optimization_paths.display_repo_path(report_path, repo_root=tmp_path)
        == "runs/training_optimizer/stop.md"
    )
