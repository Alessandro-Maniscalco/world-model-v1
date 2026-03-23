"""Dataset-loading helpers used by training and evaluation scripts."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Subset

from world_model.data.temporal import build_frame_deltas


def collate_tensor_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate list-of-dicts by stacking tensor values and listing non-tensors."""
    if not batch:
        raise ValueError("batch must contain at least one sample")

    out: dict[str, Any] = {}
    for key in batch[0]:
        first = batch[0][key]
        if torch.is_tensor(first):
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)
        else:
            out[key] = [sample[key] for sample in batch]
    return out


def resolve_lerobot_episode_ids(
    *,
    repo_id: str,
    video_backend: str = "pyav",
) -> list[int]:
    """Return sorted episode ids available in a LeRobot dataset's metadata."""
    LeRobotDataset = _load_lerobot_dataset_class()
    dataset = LeRobotDataset(
        repo_id,
        video_backend=video_backend,
        download_videos=False,
    )
    meta = getattr(dataset, "meta", None)
    meta_episodes = getattr(meta, "episodes", None)
    if meta_episodes is None:
        raise ValueError(f"LeRobot dataset {repo_id!r} does not expose metadata episodes")
    column_names = getattr(meta_episodes, "column_names", ())
    if "episode_index" not in column_names:
        raise ValueError(f"LeRobot dataset {repo_id!r} metadata is missing 'episode_index'")

    episode_ids = sorted({int(value) for value in meta_episodes["episode_index"]})
    if not episode_ids:
        raise ValueError(f"LeRobot dataset {repo_id!r} exposes no episodes")
    return episode_ids


def split_train_validation_episode_ids(
    *,
    available_episode_ids: list[int] | tuple[int, ...],
    requested_episode_ids: list[int] | tuple[int, ...] | None = None,
    validation_episode_ids: list[int] | tuple[int, ...] | None = None,
    validation_split_ratio: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Split a candidate episode pool into deterministic train and validation ids."""
    available = sorted({int(episode_id) for episode_id in available_episode_ids})
    if not available:
        raise ValueError("available_episode_ids must not be empty")

    candidate = available
    if requested_episode_ids:
        requested = sorted({int(episode_id) for episode_id in requested_episode_ids})
        unknown_requested = sorted(set(requested) - set(available))
        if unknown_requested:
            raise ValueError(
                "Requested training episodes are unavailable: "
                f"{unknown_requested}"
            )
        candidate = requested

    if validation_episode_ids:
        validation = sorted({int(episode_id) for episode_id in validation_episode_ids})
        unknown_validation = sorted(set(validation) - set(candidate))
        if unknown_validation:
            raise ValueError(
                "Validation episodes must be drawn from the active candidate episode pool, got "
                f"{unknown_validation}."
            )
    else:
        if not 0.0 < validation_split_ratio < 1.0:
            raise ValueError(
                "validation_split_ratio must be strictly between 0 and 1, "
                f"got {validation_split_ratio}."
            )
        validation_count = max(1, int(math.ceil(len(candidate) * validation_split_ratio)))
        if validation_count >= len(candidate):
            raise ValueError(
                "Validation split must leave at least one training episode, got "
                f"{len(candidate)} candidate episodes and validation_count={validation_count}."
            )
        validation = candidate[-validation_count:]

    train = [episode_id for episode_id in candidate if episode_id not in set(validation)]
    if not train:
        raise ValueError("Validation split left no training episodes")
    if not validation:
        raise ValueError("Validation split left no validation episodes")
    return train, validation


def _select_subset_indices(*, dataset_size: int, subset_size: int, shuffle: bool) -> list[int]:
    """Choose deterministic subset indices, spreading shuffled training subsets across the dataset."""
    if dataset_size < 0:
        raise ValueError(f"dataset_size must be non-negative, got {dataset_size}")
    if subset_size < 0:
        raise ValueError(f"subset_size must be non-negative, got {subset_size}")
    if subset_size == 0 or dataset_size == 0:
        return []

    capped_subset = min(subset_size, dataset_size)
    if capped_subset == dataset_size or not shuffle or capped_subset == 1:
        return list(range(capped_subset))

    max_index = dataset_size - 1
    return [
        (offset * max_index) // (capped_subset - 1)
        for offset in range(capped_subset)
    ]


def build_lerobot_dataloader(
    *,
    repo_id: str,
    episodes: list[int] | tuple[int, ...] | None = None,
    video_key: str,
    context_len: int,
    horizon_len: int,
    dt: float,
    batch_size: int,
    subset_size: int = 0,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    video_backend: str = "pyav",
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = collate_tensor_dict,
) -> DataLoader:
    """Build a LeRobot dataloader configured for frame-time context+horizon windows."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if subset_size < 0:
        raise ValueError(f"subset_size must be non-negative, got {subset_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")

    LeRobotDataset = _load_lerobot_dataset_class()

    deltas = build_frame_deltas(context_len=context_len, horizon_len=horizon_len, dt=dt)
    dataset_kwargs: dict[str, Any] = {
        "delta_timestamps": {
            video_key: deltas,
            "action": deltas,
        },
        "video_backend": video_backend,
    }
    if episodes:
        dataset_kwargs["episodes"] = list(episodes)

    dataset = LeRobotDataset(repo_id, **dataset_kwargs)
    if subset_size > 0:
        dataset = Subset(
            dataset,
            _select_subset_indices(
                dataset_size=len(dataset),
                subset_size=subset_size,
                shuffle=shuffle,
            ),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def _load_lerobot_dataset_class():
    """Import and return the canonical LeRobot dataset class."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("lerobot is required to build the LIBERO dataloader") from exc
    return LeRobotDataset
