from world_model.config import TrainConfig


def build_deltas(config: TrainConfig) -> list[float]:
    return config.deltas()


def build_delta_timestamps(config: TrainConfig) -> dict[str, list[float]]:
    return {config.video_key: build_deltas(config)}


def load_lerobot_dataset(config: TrainConfig, video_backend: str = "pyav"):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("LeRobotDataset is required to load data. Install lerobot dependencies.") from exc

    return LeRobotDataset(
        config.repo_id,
        delta_timestamps=build_delta_timestamps(config),
        video_backend=video_backend,
    )
