"""Tests for the episode-0 ALOHA download-and-train helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from huggingface_hub.errors import OfflineModeIsEnabled

from world_model.training import aloha_episode0


def test_prefetch_episode_uses_requested_episode_subset(monkeypatch) -> None:
    """Restrict prefetching to the requested LeRobot episode."""
    calls: dict[str, object] = {}

    class FakeDataset:
        """Capture constructor kwargs without touching the network."""

        def __init__(self, repo_id: str, **kwargs) -> None:
            calls["repo_id"] = repo_id
            calls["kwargs"] = kwargs

    monkeypatch.setattr(aloha_episode0, "_load_lerobot_dataset_class", lambda: FakeDataset)

    aloha_episode0.prefetch_episode(
        repo_id="lerobot/aloha_static_fork_pick_up",
        episode=0,
        video_key="observation.images.cam_high",
    )

    assert calls["repo_id"] == "lerobot/aloha_static_fork_pick_up"
    assert calls["kwargs"] == {
        "episodes": [0],
        "delta_timestamps": {"observation.images.cam_high": [0.0]},
        "video_backend": "pyav",
    }


def test_prefetch_episode_fails_cleanly_when_offline_and_not_cached(monkeypatch) -> None:
    """Explain that the first fetch must run online when the cache is missing."""

    class MissingOfflineDataset:
        """Raise the same offline exception the hub emits on a missing cache."""

        def __init__(self, repo_id: str, **kwargs) -> None:
            del repo_id, kwargs
            raise OfflineModeIsEnabled("offline")

    monkeypatch.setattr(aloha_episode0, "_load_lerobot_dataset_class", lambda: MissingOfflineDataset)

    with pytest.raises(RuntimeError, match="first fetch must run without HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE"):
        aloha_episode0.prefetch_episode(
            repo_id="lerobot/aloha_static_fork_pick_up",
            episode=0,
            env={"HF_HUB_OFFLINE": "1"},
        )


def test_workflow_allows_cached_rerun_in_offline_mode(monkeypatch) -> None:
    """Proceed to training launch when the requested episode already resolves from cache."""
    launched: list[tuple[list[str], dict[str, str] | None]] = []

    class CachedDataset:
        """Succeed immediately to simulate a cache hit."""

        def __init__(self, repo_id: str, **kwargs) -> None:
            del repo_id, kwargs

    def fake_launcher(command, *, env=None) -> None:
        """Capture the launch request instead of starting training."""
        launched.append((list(command), None if env is None else dict(env)))

    monkeypatch.setattr(aloha_episode0, "_load_lerobot_dataset_class", lambda: CachedDataset)

    command = aloha_episode0.run_aloha_episode0_workflow(
        env={"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        launcher=fake_launcher,
    )

    assert command == launched[0][0]
    assert launched[0][1] == {
        "HF_HUB_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def test_build_training_command_targets_world_model_smoke_defaults() -> None:
    """Launch the existing smoke config through the main world-model entrypoint."""
    command = aloha_episode0.build_training_command()

    assert command[0]
    assert Path(command[1]) == aloha_episode0.WORLD_MODEL_SCRIPT_PATH
    assert command[2:4] == ["--config", str(aloha_episode0.DEFAULT_CONFIG_PATH)]
    assert command[4:6] == ["--repo-id", aloha_episode0.DEFAULT_REPO_ID]
    assert command[6:8] == ["--episodes", str(aloha_episode0.DEFAULT_EPISODE)]
