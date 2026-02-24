"""Tests for dataset collate and dataloader builder helpers."""

from __future__ import annotations

import sys
import types

import torch

from world_model.data.dataset import build_lerobot_dataloader, collate_tensor_dict


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
        def __init__(self, repo_id: str, delta_timestamps: dict[str, list[float]], video_backend: str) -> None:
            calls["repo_id"] = repo_id
            calls["delta_timestamps"] = delta_timestamps
            calls["video_backend"] = video_backend
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
        video_key="observation.images.image",
        context_len=3,
        horizon_len=2,
        dt=0.1,
        batch_size=2,
        subset_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    first = next(iter(loader))
    assert first["x"].shape == (2, 1)
    assert len(first["meta"]) == 2
    assert calls["repo_id"] == "repo/x"
    assert calls["video_backend"] == "pyav"
    assert "observation.images.image" in calls["delta_timestamps"]  # type: ignore[operator]
