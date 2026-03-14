"""Run a Wan VAE roundtrip on the first DROID image by default.

This smoke-check saves the original frame and its VAE reconstruction as PNGs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    """Convert a frame or clip tensor into `BCTHW` with batch size 1."""
    if video.ndim == 3:
        if video.shape[0] == 3:
            video = video.unsqueeze(0)
        elif video.shape[-1] == 3:
            video = video.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unrecognized single-frame shape: {tuple(video.shape)}")

    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got {video.ndim}D: {tuple(video.shape)}")

    if video.shape[1] == 3:
        video_tchw = video
    elif video.shape[-1] == 3:
        video_tchw = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")

    return video_tchw.permute(1, 0, 2, 3).unsqueeze(0)


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """Normalize a uint8/float image tensor to the VAE range `[-1, 1]`."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.max().detach().cpu()) > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def denormalize_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert a `BCTHW` `[-1,1]` tensor into `THWC` uint8 frames."""
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)
    return x[0].permute(1, 2, 3, 0).contiguous().detach().cpu().numpy()


def _resample_time(video_bcthw: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Resample `BCTHW` video along time to match a target frame count."""
    if video_bcthw.ndim != 5:
        raise ValueError(f"Expected BCTHW video, got {tuple(video_bcthw.shape)}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")

    source_steps = int(video_bcthw.shape[2])
    if source_steps <= 0:
        raise ValueError("Cannot resample a video with zero frames")
    if source_steps == target_steps:
        return video_bcthw

    positions = ((torch.arange(target_steps, device=video_bcthw.device) + 0.5) * source_steps / target_steps) - 0.5
    indices = positions.round().long().clamp(0, source_steps - 1)
    return video_bcthw.index_select(dim=2, index=indices)


def _align_spatial(video_bcthw: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """Align `BCTHW` spatial size to the decoded output for metric comparison."""
    if video_bcthw.ndim != 5:
        raise ValueError(f"Expected BCTHW video, got {tuple(video_bcthw.shape)}")

    source_height = int(video_bcthw.shape[3])
    source_width = int(video_bcthw.shape[4])
    if source_height == target_height and source_width == target_width:
        return video_bcthw

    if source_height >= target_height and source_width >= target_width:
        top = (source_height - target_height) // 2
        left = (source_width - target_width) // 2
        return video_bcthw[:, :, :, top : top + target_height, left : left + target_width]

    batch, channels, timesteps = video_bcthw.shape[:3]
    resized = F.interpolate(
        video_bcthw.permute(0, 2, 1, 3, 4).reshape(batch * timesteps, channels, source_height, source_width),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, timesteps, channels, target_height, target_width).permute(0, 2, 1, 3, 4)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the DROID image roundtrip smoke-check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, default="lerobot/droid_1.0.1", help="LeRobot dataset repo id.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode to load from the DROID dataset.")
    parser.add_argument("--frame-index", type=int, default=0, help="Episode-local frame index to roundtrip.")
    parser.add_argument(
        "--video-key",
        type=str,
        default="observation.images.exterior_1_left",
        help="Camera key to load from the DROID sample.",
    )
    parser.add_argument(
        "--posterior-sample",
        action="store_true",
        help="Sample from the VAE posterior instead of using the deterministic mode/mean.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "assets" / "vae_roundtrip"),
        help="Directory for saved PNG artifacts.",
    )
    return parser


def _load_droid_frame(*, repo_id: str, episode_index: int, frame_index: int, video_key: str) -> torch.Tensor:
    """Load one DROID frame as `BCTHW` with batch size 1 and one timestep."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")

    dataset = LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
    if frame_index >= len(dataset):
        raise ValueError(
            f"Requested frame_index={frame_index} exceeds episode length {len(dataset)} for episode {episode_index}."
        )

    sample = dataset[frame_index]
    if video_key not in sample:
        available_keys = [key for key in sample if key.startswith("observation.images.")]
        raise KeyError(f"video_key={video_key!r} not found. Available image keys: {available_keys}")
    return to_bcthw(sample[video_key])


def _latent_dist_mode(latent_dist: object) -> torch.Tensor:
    """Read the deterministic latent from a diffusers posterior object."""
    mode = getattr(latent_dist, "mode", None)
    if callable(mode):
        return mode()
    mean = getattr(latent_dist, "mean", None)
    if torch.is_tensor(mean):
        return mean
    raise ValueError("latent_dist is missing both mode() and mean")


def _compute_metrics(orig: torch.Tensor, recon: torch.Tensor) -> dict[str, float]:
    """Compute simple reconstruction metrics on `[-1,1]` tensors."""
    diff = (orig.float() - recon.float()).detach()
    mse = float(torch.mean(diff.square()).cpu())
    mae = float(torch.mean(diff.abs()).cpu())
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((2.0**2) / mse))
    return {"mse": mse, "mae": mae, "psnr_db": psnr}


def _save_outputs(
    *,
    out_dir: Path,
    orig_frame: np.ndarray,
    recon_frame: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Persist original and roundtrip frames plus a text metrics summary."""
    original_path = out_dir / "original.png"
    roundtrip_path = out_dir / "vae_roundtrip.png"
    metrics_path = out_dir / "metrics.txt"

    iio.imwrite(original_path, orig_frame)
    iio.imwrite(roundtrip_path, recon_frame)
    metrics_path.write_text(
        "\n".join(
            [
                f"mse={metrics['mse']:.8f}",
                f"mae={metrics['mae']:.8f}",
                f"psnr_db={metrics['psnr_db']:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Saved:", original_path)
    print("Saved:", roundtrip_path)
    print("Saved:", metrics_path)


@torch.no_grad()
def main() -> None:
    """Encode/decode one DROID image and persist before/after PNGs."""
    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()

    source = _load_droid_frame(
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        frame_index=args.frame_index,
        video_key=args.video_key,
    )
    source_desc = (
        f"dataset {args.repo_id} episode {args.episode_index} "
        f"frame {args.frame_index} key {args.video_key}"
    )

    video = normalize_to_minus1_1(source).to(device)
    print("Source:", source_desc)
    print("Input video B,C,T,H,W:", tuple(video.shape), video.dtype)
    print("Input range:", float(video.min().cpu()), float(video.max().cpu()))

    enc = vae.encode(video)
    if args.posterior_sample:
        latents = enc.latent_dist.sample()
        print("Latent mode: posterior sample")
    else:
        latents = _latent_dist_mode(enc.latent_dist)
        print("Latent mode: deterministic mode/mean")
    print("Latents shape:", tuple(latents.shape), latents.dtype)
    print("Latents range:", float(latents.min().cpu()), float(latents.max().cpu()))
    print("Latents mean/std:", float(latents.mean().cpu()), float(latents.std().cpu()))

    dec = vae.decode(latents).sample
    print("Decoded video B,C,T,H,W:", tuple(dec.shape), dec.dtype)
    print("Decoded range:", float(dec.min().cpu()), float(dec.max().cpu()))
    print(f"Time compression: input_frames={video.shape[2]}, decoded_frames={dec.shape[2]}")

    aligned_video = _resample_time(video, target_steps=int(dec.shape[2]))
    aligned_video = _align_spatial(aligned_video, target_height=int(dec.shape[3]), target_width=int(dec.shape[4]))
    orig_frame = denormalize_to_uint8(video)[0]
    recon_frame = denormalize_to_uint8(dec)[0]
    metrics = _compute_metrics(aligned_video, dec)
    print("Metrics:", metrics)

    _save_outputs(
        out_dir=out_dir,
        orig_frame=orig_frame,
        recon_frame=recon_frame,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
