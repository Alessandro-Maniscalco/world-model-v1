"""Canonical training API for flow matching and optimizer orchestration."""

from world_model.training.chunkwise_training import (
    ChunkwiseStepMetrics,
    append_jsonl,
    save_checkpoint,
    train_chunkwise_batch,
)
from world_model.training.flow_matching import (
    ChunkwiseLossInfo,
    chunkwise_teacher_forcing_loss,
    make_noisy_and_target,
    sample_t,
    w,
)

__all__ = [
    "sample_t",
    "make_noisy_and_target",
    "w",
    "chunkwise_teacher_forcing_loss",
    "ChunkwiseLossInfo",
    "ChunkwiseStepMetrics",
    "train_chunkwise_batch",
    "save_checkpoint",
    "append_jsonl",
]
