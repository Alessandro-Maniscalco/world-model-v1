"""Set up and smoke-run the world-model-eval checkpoint workflow."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


WORLD_MODEL_EVAL_REPO_URL = "https://github.com/world-model-eval/world-model-eval.git"
CHECKPOINT_FILE_ID = "1uiRP2BuavapMsyP9Cbr25mi_ymk9SEJb"
CHECKPOINT_FILE_NAME = "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt"
SD3_VAE_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
SD3_VAE_CONFIG_PATH = "vae/config.json"
CHECKPOINTS_TO_KWARGS: dict[str, dict[str, float | bool]] = {
    "bridge_v2_ckpt.pt": {"use_pixel_rope": True},
    CHECKPOINT_FILE_NAME: {"use_pixel_rope": False, "default_cfg": 3.0},
}


def _repo_root() -> Path:
    """Return the repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def default_repo_dir() -> Path:
    """Return the default checkout path for world-model-eval."""
    return _repo_root() / "runs" / "dreamdojo" / "world-model-eval"


def default_checkpoint_dir() -> Path:
    """Return the default directory used to store world-model checkpoints."""
    return _repo_root() / "runs" / "dreamdojo" / "checkpoints"


def default_output_path() -> Path:
    """Return the default smoke-rollout output video path."""
    return _repo_root() / "runs" / "dreamdojo" / "smoke_rollout.mp4"


def checkpoint_kwargs(checkpoint_path: Path) -> dict[str, float | bool]:
    """Return checkpoint-specific world-model construction kwargs."""
    return CHECKPOINTS_TO_KWARGS.get(checkpoint_path.name, {})


def to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a float RGB frame in [0, 1] to uint8 HWC."""
    clipped = np.clip(frame, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess command and raise if it fails."""
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"$ {printable}")
    subprocess.run(command, cwd=cwd, check=True)


def install_world_model_eval(repo_dir: Path) -> None:
    """Install world-model-eval in editable mode with a Python 3.13 fallback."""
    editable_install = [sys.executable, "-m", "pip", "install", "-e", str(repo_dir)]
    try:
        run_command(editable_install)
        return
    except subprocess.CalledProcessError:
        # world-model-eval pins backports.strenum==1.3.1, which is unavailable on Python 3.13.
        print("Editable install with dependencies failed; retrying with --no-deps.")
    run_command([*editable_install, "--no-deps"])
    ensure_runtime_dependencies()
    print("Installed world-model-eval with --no-deps. Install any missing runtime deps manually if needed.")


def ensure_runtime_dependencies() -> None:
    """Install runtime deps needed by world-model-eval package import if absent."""
    if importlib.util.find_spec("openai") is None:
        run_command([sys.executable, "-m", "pip", "install", "openai"])


def configure_huggingface_token(hf_token: str | None) -> None:
    """Configure Hugging Face token environment variables for this process."""
    if hf_token is None:
        return
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token


def _download_hf_file(*, repo_id: str, filename: str, token: str | None) -> str:
    """Download a file from Hugging Face Hub and return the local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


def ensure_sd3_vae_access(hf_token: str | None) -> None:
    """Verify the SD3 VAE repo is accessible before world model creation."""
    try:
        _download_hf_file(repo_id=SD3_VAE_MODEL_ID, filename=SD3_VAE_CONFIG_PATH, token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot access Hugging Face repo `{SD3_VAE_MODEL_ID}`. "
            "This model is gated; accept its license on Hugging Face, then run `hf auth login` "
            "or pass `--hf-token`."
        ) from exc


def setup_world_model_eval(
    *,
    repo_dir: Path,
    checkpoint_dir: Path,
    download_checkpoint: bool,
) -> Path:
    """Clone/install world-model-eval and optionally download the checkpoint."""
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        run_command(["git", "clone", "--depth", "1", WORLD_MODEL_EVAL_REPO_URL, str(repo_dir)])
    else:
        print(f"Using existing checkout: {repo_dir}")

    install_world_model_eval(repo_dir)

    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE_NAME
    if not download_checkpoint:
        print(
            "Checkpoint download skipped. Add --download-checkpoint to fetch it automatically "
            f"into {checkpoint_path}."
        )
        return checkpoint_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path

    if shutil.which("gdown") is None:
        raise RuntimeError(
            "gdown is not installed in this environment. Run `source .venv/bin/activate` and "
            "`pip install gdown`, or install from requirements.txt."
        )

    run_command(["gdown", CHECKPOINT_FILE_ID, "-O", str(checkpoint_path)])
    return checkpoint_path


def run_smoke_rollout(
    *,
    checkpoint_path: Path,
    start_image_path: Path,
    output_path: Path,
    steps: int,
    action_scale: float,
    fps: int,
    seed: int,
    cfg: float | None,
    hf_token: str | None,
) -> Path:
    """Run a random-action rollout against the world model checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not start_image_path.exists():
        raise FileNotFoundError(f"Start image not found: {start_image_path}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if action_scale <= 0:
        raise ValueError("action_scale must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by world-model-eval's WorldModel implementation.")

    configure_huggingface_token(hf_token)
    ensure_sd3_vae_access(hf_token)

    from world_model_eval.world_model import WorldModel

    start_image = Image.open(start_image_path).convert("RGB").resize((256, 256))
    start_frame = np.array(start_image)

    kwargs = checkpoint_kwargs(checkpoint_path)
    wm = WorldModel(str(checkpoint_path), **kwargs)
    if cfg is not None:
        wm.cfg = cfg

    frames: list[np.ndarray] = [start_frame]
    start_tensor = torch.from_numpy(start_frame).cuda().float() / 255.0
    wm.reset(start_tensor)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    for _ in range(steps):
        action = torch.randn((wm.model.action_dim,), generator=generator, device="cuda") * action_scale
        latest_frame = None
        for _, decoded in wm.generate_chunk(action):
            latest_frame = decoded[0, 0].detach().cpu().numpy()
        if latest_frame is None:
            raise RuntimeError("World model did not produce a frame for one of the rollout steps.")
        frames.append(to_uint8_frame(latest_frame))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved smoke rollout video: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for setup and smoke subcommands."""
    parser = argparse.ArgumentParser(
        description="Dreamdojo helper for setting up and smoke-running world-model-eval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Clone/install world-model-eval and optionally fetch checkpoint.")
    setup_parser.add_argument("--repo-dir", type=Path, default=default_repo_dir())
    setup_parser.add_argument("--checkpoint-dir", type=Path, default=default_checkpoint_dir())
    setup_parser.add_argument("--download-checkpoint", action="store_true")

    smoke_parser = subparsers.add_parser("smoke", help="Run a random-action world model rollout.")
    smoke_parser.add_argument("--checkpoint-path", type=Path, default=default_checkpoint_dir() / CHECKPOINT_FILE_NAME)
    smoke_parser.add_argument("--start-image", type=Path, required=True)
    smoke_parser.add_argument("--output-path", type=Path, default=default_output_path())
    smoke_parser.add_argument("--steps", type=int, default=8)
    smoke_parser.add_argument("--action-scale", type=float, default=0.35)
    smoke_parser.add_argument("--fps", type=int, default=8)
    smoke_parser.add_argument("--seed", type=int, default=0)
    smoke_parser.add_argument("--cfg", type=float, default=None)
    smoke_parser.add_argument("--hf-token", type=str, default=None)

    return parser


def _run_setup_from_args(args: argparse.Namespace) -> int:
    """Execute the setup subcommand."""
    setup_world_model_eval(
        repo_dir=args.repo_dir,
        checkpoint_dir=args.checkpoint_dir,
        download_checkpoint=args.download_checkpoint,
    )
    return 0


def _run_smoke_from_args(args: argparse.Namespace) -> int:
    """Execute the smoke subcommand."""
    run_smoke_rollout(
        checkpoint_path=args.checkpoint_path,
        start_image_path=args.start_image,
        output_path=args.output_path,
        steps=args.steps,
        action_scale=args.action_scale,
        fps=args.fps,
        seed=args.seed,
        cfg=args.cfg,
        hf_token=args.hf_token,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "setup":
        return _run_setup_from_args(args)
    return _run_smoke_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
