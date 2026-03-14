"""Episode-0 ALOHA cache bootstrap and smoke-training launcher."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from huggingface_hub.errors import OfflineModeIsEnabled


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPO_ID = "lerobot/aloha_static_fork_pick_up"
DEFAULT_EPISODE = 0
DEFAULT_VIDEO_KEY = "observation.images.cam_high"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train" / "aloha_fork_pick_up_smoke.yaml"
WORLD_MODEL_SCRIPT_PATH = REPO_ROOT / "scripts" / "train" / "world_model.py"


def _env_flag_enabled(value: str | None) -> bool:
    """Interpret common env-var falsey values consistently."""
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


def offline_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether Hugging Face access is effectively forced offline."""
    active_env = os.environ if env is None else env
    return _env_flag_enabled(active_env.get("HF_HUB_OFFLINE")) or _env_flag_enabled(
        active_env.get("TRANSFORMERS_OFFLINE")
    )


def _load_lerobot_dataset_class():
    """Import and return the LeRobot dataset class on demand."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError("lerobot is required for the ALOHA episode-0 helper.") from exc
    return LeRobotDataset


def prefetch_episode(
    *,
    repo_id: str,
    episode: int,
    video_key: str = DEFAULT_VIDEO_KEY,
    env: Mapping[str, str] | None = None,
) -> None:
    """Ensure the requested ALOHA episode is cached locally."""
    if episode < 0:
        raise ValueError(f"episode must be >= 0, got {episode}")

    dataset_class = _load_lerobot_dataset_class()
    try:
        dataset_class(
            repo_id,
            episodes=[episode],
            delta_timestamps={video_key: [0.0]},
            video_backend="pyav",
        )
    except OfflineModeIsEnabled as exc:
        if offline_mode_enabled(env):
            raise RuntimeError(
                "Episode 0 is not cached locally. The first fetch must run without "
                "HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE."
            ) from exc
        raise


def build_training_command(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    repo_id: str = DEFAULT_REPO_ID,
    episode: int = DEFAULT_EPISODE,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the world-model training command for the cached episode-0 recipe."""
    return [
        python_executable,
        str(WORLD_MODEL_SCRIPT_PATH),
        "--config",
        str(config_path),
        "--repo-id",
        repo_id,
        "--episodes",
        str(episode),
    ]


def format_command_for_print(command: Sequence[str]) -> str:
    """Render a shell-safe preview string for the training command."""
    return subprocess.list2cmdline(list(command))


def launch_training_command(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    """Run the smoke training command and surface subprocess failures."""
    print(f"Launching training: {format_command_for_print(command)}")
    subprocess.run(list(command), check=True, cwd=REPO_ROOT, env=None if env is None else dict(env))


def run_aloha_episode0_workflow(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    episode: int = DEFAULT_EPISODE,
    config_path: Path = DEFAULT_CONFIG_PATH,
    video_key: str = DEFAULT_VIDEO_KEY,
    env: Mapping[str, str] | None = None,
    python_executable: str = sys.executable,
    launcher=launch_training_command,
) -> list[str]:
    """Fetch the requested episode if needed, then launch smoke training."""
    resolved_env = os.environ if env is None else env
    prefetch_episode(repo_id=repo_id, episode=episode, video_key=video_key, env=resolved_env)
    command = build_training_command(
        config_path=config_path,
        repo_id=repo_id,
        episode=episode,
        python_executable=python_executable,
    )
    launcher(command, env=resolved_env)
    return command
