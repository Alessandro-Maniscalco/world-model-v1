from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    repo_id: str = "lerobot/libero"
    video_key: str = "observation.images.image"

    use_proprio: bool = True

    # Windowing at 10 Hz for LIBERO
    context_len: int = 8
    horizon_len: int = 8
    dt: float = 0.1

    # Overfit subset
    subset_indices: int = 8
    batch_size: int = 2
    num_steps: int = 300
    lr: float = 2e-3

    # Model size
    hidden: int = 2048

    # Output
    out_dir: str = "runs/overfit_test"
    seed: int = 0

    @property
    def total_window_len(self) -> int:
        return self.context_len + self.horizon_len

    def deltas(self) -> list[float]:
        return [-(self.total_window_len - 1 - i) * self.dt for i in range(self.total_window_len)]
