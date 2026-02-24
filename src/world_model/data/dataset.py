"""Dataset-loading helpers used by training and evaluation scripts."""

from __future__ import annotations

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


def build_lerobot_dataloader(
    *,
    repo_id: str,
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

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("lerobot is required to build the LIBERO dataloader") from exc

    deltas = build_frame_deltas(context_len=context_len, horizon_len=horizon_len, dt=dt)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={video_key: deltas},
        video_backend=video_backend,
    )
    if subset_size > 0:
        dataset = Subset(dataset, range(subset_size))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
