"""Tests for dataset collate and dataloader builder helpers."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from world_model.data.dataset import (
    build_lerobot_dataloader,
    collate_tensor_dict,
    resolve_lerobot_episode_ids,
    split_train_validation_episode_ids,
)


def test_collate_tensor_dict_stacks_tensors_and_lists_other_fields() -> None:
    batch = [
        {"x": torch.tensor([1, 2]), "label": "a"},
        {"x": torch.tensor([3, 4]), "label": "b"},
    ]
    out = collate_tensor_dict(batch)
    assert torch.equal(out["x"], torch.tensor([[1, 2], [3, 4]]))
    assert out["label"] == ["a", "b"]


def test_build_lerobot_dataloader_uses_expected_dataset_arguments(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakeDataset:
        def __init__(
            self,
            repo_id: str,
            delta_timestamps: dict[str, list[float]],
            video_backend: str,
            episodes: list[int] | None = None,
            download_videos: bool = True,
        ) -> None:
            calls["repo_id"] = repo_id
            calls["delta_timestamps"] = delta_timestamps
            calls["video_backend"] = video_backend
            calls["episodes"] = episodes
            calls["download_videos"] = download_videos
            self._samples = [
                {"x": torch.tensor([i], dtype=torch.float32), "meta": f"m{i}"}
                for i in range(6)
            ]

        def __len__(self) -> int:
            return len(self._samples)

        def __getitem__(self, idx: int) -> dict[str, object]:
            return self._samples[idx]

    lerobot_pkg = types.ModuleType("lerobot")
    datasets_pkg = types.ModuleType("lerobot.datasets")
    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeDataset  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "lerobot", lerobot_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_mod)

    loader = build_lerobot_dataloader(
        repo_id="repo/x",
        episodes=(0, 3),
        video_key="observation.images.image",
        context_len=5,
        horizon_len=4,
        dt=0.1,
        batch_size=2,
        subset_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    assert list(loader.dataset.indices) == [0, 1, 2, 3]  # type: ignore[union-attr]
    first = next(iter(loader))
    assert first["x"].shape == (2, 1)
    assert len(first["meta"]) == 2
    assert calls["repo_id"] == "repo/x"
    assert calls["video_backend"] == "pyav"
    assert calls["episodes"] == [0, 3]
    assert "observation.images.image" in calls["delta_timestamps"]  # type: ignore[operator]
    assert "action" in calls["delta_timestamps"]  # type: ignore[operator]


def test_build_lerobot_dataloader_spreads_shuffled_training_subsets(monkeypatch) -> None:
    """Spread shuffled training subsets across the dataset instead of taking a contiguous head slice."""
    class _FakeDataset:
        def __init__(
            self,
            repo_id: str,
            delta_timestamps: dict[str, list[float]],
            video_backend: str,
            episodes: list[int] | None = None,
            download_videos: bool = True,
        ) -> None:
            del repo_id, delta_timestamps, video_backend, episodes, download_videos
            self._samples = [{"x": torch.tensor([i], dtype=torch.float32)} for i in range(6)]

        def __len__(self) -> int:
            return len(self._samples)

        def __getitem__(self, idx: int) -> dict[str, object]:
            return self._samples[idx]

    lerobot_pkg = types.ModuleType("lerobot")
    datasets_pkg = types.ModuleType("lerobot.datasets")
    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeDataset  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "lerobot", lerobot_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_mod)

    loader = build_lerobot_dataloader(
        repo_id="repo/x",
        video_key="observation.images.image",
        context_len=5,
        horizon_len=4,
        dt=0.1,
        batch_size=2,
        subset_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    assert list(loader.dataset.indices) == [0, 1, 3, 5]  # type: ignore[union-attr]


def test_resolve_lerobot_episode_ids_reads_sorted_metadata(monkeypatch) -> None:
    """Read sorted episode ids from LeRobot metadata without downloading videos."""
    class _FakeEpisodes:
        column_names = ["episode_index"]

        def __getitem__(self, key: str) -> list[int]:
            assert key == "episode_index"
            return [4, 1, 3, 1]

    class _FakeMeta:
        episodes = _FakeEpisodes()

    class _FakeDataset:
        def __init__(
            self,
            repo_id: str,
            video_backend: str,
            download_videos: bool,
            episodes: list[int] | None = None,
        ) -> None:
            assert repo_id == "repo/x"
            assert episodes is None
            assert video_backend == "pyav"
            assert download_videos is False
            self.meta = _FakeMeta()

    lerobot_pkg = types.ModuleType("lerobot")
    datasets_pkg = types.ModuleType("lerobot.datasets")
    dataset_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeDataset  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "lerobot", lerobot_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", dataset_mod)

    episode_ids = resolve_lerobot_episode_ids(repo_id="repo/x")

    assert episode_ids == [1, 3, 4]


def test_split_train_validation_episode_ids_uses_last_ratio_slice() -> None:
    """Reserve the last deterministic tail of episode ids for validation by default."""
    train_ids, validation_ids = split_train_validation_episode_ids(
        available_episode_ids=list(range(10)),
        validation_split_ratio=0.1,
    )

    assert train_ids == list(range(9))
    assert validation_ids == [9]


def test_split_train_validation_episode_ids_respects_explicit_validation_ids() -> None:
    """Exclude explicit validation episodes from the active candidate episode pool."""
    train_ids, validation_ids = split_train_validation_episode_ids(
        available_episode_ids=list(range(10)),
        requested_episode_ids=[2, 3, 4, 5],
        validation_episode_ids=[4, 5],
    )

    assert train_ids == [2, 3]
    assert validation_ids == [4, 5]


def test_split_train_validation_episode_ids_rejects_empty_train_side() -> None:
    """Fail fast when the requested split would leave no training episodes."""
    with pytest.raises(ValueError, match="leave at least one training episode|left no training episodes"):
        split_train_validation_episode_ids(
            available_episode_ids=[0],
            validation_split_ratio=0.1,
        )


def test_split_train_validation_episode_ids_rejects_unknown_validation_ids() -> None:
    """Fail fast when explicit validation episodes fall outside the candidate pool."""
    with pytest.raises(ValueError, match="Validation episodes must be drawn"):
        split_train_validation_episode_ids(
            available_episode_ids=[0, 1, 2],
            requested_episode_ids=[0, 1],
            validation_episode_ids=[2],
        )
