"""Shared structured latent-video batch preparation for training and inference."""

from __future__ import annotations

from typing import Any, Protocol

import imageio.v3 as iio
import numpy as np
import torch

from world_model.data.schema import PreparedPackedBatch
from world_model.data.temporal import (
    build_future_action_plan,
    latent_split_for_wan_frames,
    validate_wan_temporal_window,
)


class LatentEncoder(Protocol):
    """Protocol for latent encoders used in data preparation."""

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a video batch into `[B, C_lat, T_lat, H_lat, W_lat]` latents."""


_CONTROL_LATENT_TEMPLATE_CACHE: dict[tuple[int, tuple[int, ...], str, str], torch.Tensor] = {}


def preprocess_video_for_vae(
    video: torch.Tensor,
    *,
    frame_height: int = 0,
    frame_width: int = 0,
    spatial_multiple: int = 8,
) -> torch.Tensor:
    """Resize and center-crop `BTCHW` video to a VAE-safe spatial shape."""
    if video.ndim != 5:
        raise ValueError(f"Expected video [B,T,C,H,W], got {tuple(video.shape)}")
    if spatial_multiple <= 0:
        raise ValueError(f"spatial_multiple must be positive, got {spatial_multiple}")

    processed = video
    if frame_height > 0 and frame_width > 0:
        batch, steps, channels, height, width = processed.shape
        resize_input = processed.reshape(batch * steps, channels, height, width)
        if not resize_input.is_floating_point():
            resize_input = resize_input.float()
        processed = torch.nn.functional.interpolate(
            resize_input,
            size=(frame_height, frame_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, steps, channels, frame_height, frame_width)

    return _center_crop_video_to_multiple(processed, spatial_multiple=spatial_multiple)


def load_local_video_clip(video_path: str, *, start_frame: int, total_frames: int) -> torch.Tensor:
    """Load a contiguous local RGB clip as `BTCHW` with batch size 1."""
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    video = iio.imread(video_path)
    if video.ndim == 3:
        video = video[None, ...]
    if video.ndim != 4:
        raise ValueError(f"Expected video array [T,H,W,C], got shape {tuple(video.shape)}")

    num_frames = int(video.shape[0])
    end_frame = start_frame + total_frames
    if end_frame > num_frames:
        raise ValueError(
            f"Requested frames [{start_frame}:{end_frame}] exceed video length {num_frames}. "
            "Reduce start_frame or use smaller context/horizon."
        )

    clip = video[start_frame:end_frame]
    if clip.shape[-1] == 4:
        clip = clip[..., :3]
    if clip.shape[-1] != 3:
        raise ValueError(f"Expected RGB video with C=3, got shape {tuple(clip.shape)}")

    clip_t = torch.from_numpy(np.ascontiguousarray(clip))
    return clip_t.permute(0, 3, 1, 2).unsqueeze(0)


def _center_crop_video_to_multiple(video: torch.Tensor, *, spatial_multiple: int) -> torch.Tensor:
    """Center-crop `BTCHW` video so height and width are divisible by `spatial_multiple`."""
    height = int(video.shape[3])
    width = int(video.shape[4])
    target_height = height - (height % spatial_multiple)
    target_width = width - (width % spatial_multiple)

    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"Video spatial size {(height, width)} is too small for spatial_multiple={spatial_multiple}."
        )
    if target_height == height and target_width == width:
        return video

    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return video[:, :, :, top : top + target_height, left : left + target_width]


def prepare_packed_batch(
    *,
    batch: dict[str, Any],
    encoder: LatentEncoder,
    device: torch.device,
    video_key: str,
    context_len: int,
    horizon_len: int,
    frame_height: int = 0,
    frame_width: int = 0,
    allow_missing_action: bool = False,
) -> PreparedPackedBatch:
    """Prepare one model-ready batch from decoded observations and conditioning."""
    validate_wan_temporal_window(context_len=context_len, horizon_len=horizon_len)
    video = batch[video_key].to(device)
    expected_total_frames = context_len + horizon_len
    actual_total_frames = int(video.shape[1])
    if actual_total_frames != expected_total_frames:
        raise ValueError(
            f"Expected {video_key!r} to contain context_len + horizon_len = {expected_total_frames} frames, "
            f"got {actual_total_frames}."
        )
    action = batch.get("action")
    if action is None:
        if not allow_missing_action:
            raise KeyError("batch is missing required 'action' tensor")
        action = torch.zeros((video.shape[0], 1), dtype=torch.float32, device=device)
    action = action.to(device=device, dtype=torch.float32)
    video = preprocess_video_for_vae(video, frame_height=frame_height, frame_width=frame_width)

    control_black_latents = _get_constant_control_latents(
        encoder=encoder,
        video=video,
        cache_key_suffix="black",
        zero_to_one_value=0.0,
    )
    control_gray_latents = _get_constant_control_latents(
        encoder=encoder,
        video=video,
        cache_key_suffix="gray",
        zero_to_one_value=(128.0 / 255.0),
    )
    latents = encoder.encode(video)
    if latents.ndim != 5:
        raise ValueError(f"encoder must return [B,C,T,H,W], got {tuple(latents.shape)}")
    if control_black_latents.shape != latents.shape:
        raise ValueError(
            "control_black_latents must match encoded video shape, "
            f"got {tuple(control_black_latents.shape)} and {tuple(latents.shape)}"
        )
    if control_gray_latents.shape != latents.shape:
        raise ValueError(
            "control_gray_latents must match encoded video shape, "
            f"got {tuple(control_gray_latents.shape)} and {tuple(latents.shape)}"
        )

    total_latent_steps = int(latents.shape[2])
    context_latent_steps, horizon_latent_steps = latent_split_for_wan_frames(
        total_latent_steps=total_latent_steps,
        context_frames=context_len,
        horizon_frames=horizon_len,
    )
    target_steps = context_latent_steps + horizon_latent_steps
    z_window_video = latents[:, :, :target_steps]
    z_window_black = control_black_latents[:, :, :target_steps]
    z_window_gray = control_gray_latents[:, :, :target_steps]
    z_past_video = z_window_video[:, :, :context_latent_steps]
    z_future_video = z_window_video[:, :, context_latent_steps:context_latent_steps + horizon_latent_steps]
    a_plan = build_future_action_plan(
        action,
        context_frames=context_len,
        horizon_frames=horizon_len,
        horizon_latent_steps=horizon_latent_steps,
    )
    latent_shape = (int(latents.shape[1]), int(latents.shape[3]), int(latents.shape[4]))
    return PreparedPackedBatch(
        z_past_video=z_past_video,
        z_future_video=z_future_video,
        control_black_latents=z_window_black,
        control_gray_latents=z_window_gray,
        a_plan=a_plan,
        latent_shape=latent_shape,
        total_latent_steps=total_latent_steps,
        context_latent_steps=context_latent_steps,
        horizon_latent_steps=horizon_latent_steps,
    )


def _get_constant_control_latents(
    *,
    encoder: LatentEncoder,
    video: torch.Tensor,
    cache_key_suffix: str,
    zero_to_one_value: float,
) -> torch.Tensor:
    """Encode and cache constant-frame control latents for the active video shape."""
    if video.ndim != 5:
        raise ValueError(f"video must be [B,T,C,H,W], got {tuple(video.shape)}")

    cache_key = (
        id(encoder),
        tuple(int(dim) for dim in video.shape),
        cache_key_suffix,
        _control_video_range_key(video),
    )
    cached = _CONTROL_LATENT_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached.to(device=video.device)

    constant_video = _make_constant_video_like(video=video, zero_to_one_value=zero_to_one_value)
    latents = encoder.encode(constant_video)
    _CONTROL_LATENT_TEMPLATE_CACHE[cache_key] = latents.detach().cpu()
    return latents


def _make_constant_video_like(*, video: torch.Tensor, zero_to_one_value: float) -> torch.Tensor:
    """Build a constant RGB video matching the input layout and numeric range."""
    if not (0.0 <= zero_to_one_value <= 1.0):
        raise ValueError(f"zero_to_one_value must be in [0,1], got {zero_to_one_value}")

    if video.dtype == torch.uint8:
        fill_value = round(zero_to_one_value * 255.0)
        return torch.full_like(video, fill_value=fill_value)

    video_float = video.float()
    if video_float.numel() == 0:
        fill_value = zero_to_one_value
    else:
        min_value = float(video_float.min().detach().cpu().item())
        max_value = float(video_float.max().detach().cpu().item())
        if min_value >= -1.1 and max_value <= 1.1 and min_value < -0.1:
            fill_value = zero_to_one_value * 2.0 - 1.0
        elif min_value >= 0.0 and max_value <= 255.0 and max_value > 1.1:
            fill_value = zero_to_one_value * 255.0
        else:
            fill_value = zero_to_one_value
    return torch.full_like(video_float, fill_value=fill_value)


def _control_video_range_key(video: torch.Tensor) -> str:
    """Build a small cache key describing the active control-video numeric range."""
    if video.dtype == torch.uint8:
        return "uint8"
    if video.numel() == 0:
        return "empty"

    video_float = video.float()
    min_value = float(video_float.min().detach().cpu().item())
    max_value = float(video_float.max().detach().cpu().item())
    if min_value >= -1.1 and max_value <= 1.1 and min_value < -0.1:
        return "minus_one_to_one"
    if min_value >= 0.0 and max_value <= 255.0 and max_value > 1.1:
        return "zero_to_255"
    return "zero_to_one"
