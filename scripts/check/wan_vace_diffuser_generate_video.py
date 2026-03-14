"""Generate one Wan VACE video with the public diffusers pipeline and CPU offload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = "Guitarist playing guitar"
NEGATIVE_PROMPT = ""
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
REFERENCE_IMAGE_SOURCE = "aloha"  # "lerobot", "lerobot_first_last", "droid", "droid_first_last", "aloha", "aloha_first_last", or "url"

LEROBOT_REPO_ID = "lerobot/libero"
LEROBOT_SAMPLE_INDEX = 0
LEROBOT_EPISODE_INDEX: int | None = None
LEROBOT_FRAME_OFFSET = 0
CONDITION_START_FRAME_INDEX = 5
LEROBOT_VIDEO_KEY = "observation.images.image"
DROID_REPO_ID = "lerobot/droid_1.0.1"
DROID_EPISODE_INDEX = 0
DROID_FRAME_OFFSET = 0
DROID_CONDITION_START_FRAME_INDEX = 25
DROID_VIDEO_KEY = "observation.images.exterior_1_left"
ALOHA_REPO_ID = "lerobot/aloha_static_battery"
ALOHA_EPISODE_INDEX = 0
ALOHA_FRAME_OFFSET = 0
ALOHA_CONDITION_START_FRAME_INDEX = 20
ALOHA_VIDEO_KEY = "observation.images.cam_high"

NUM_CONDITION_FRAMES = 5
OUTPUT_PATH = Path("runs/check_wan_vace_diffuser/aloha_0frames_5cond.mp4")
HEIGHT = 480
WIDTH = 832
NUM_TOTAL_FRAMES = 9
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
FPS = 10
WAN_VACE_TEMPORAL_SCALE_FACTOR = 4


def _export_video(*, video_frames: list[object], output_video_path: str, fps: int) -> str:
    """Export generated frames to an mp4 with diffusers utilities."""
    from diffusers.utils import export_to_video

    return export_to_video(video_frames=video_frames, output_video_path=output_video_path, fps=fps)


def _compute_sparse_condition_ranges(*, condition_indices: list[int], num_frames: int) -> list[range]:
    """Expand sparse keyframes to the latent-time buckets that survive Wan VACE mask downsampling."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

    num_latent_frames = (num_frames + WAN_VACE_TEMPORAL_SCALE_FACTOR - 1) // WAN_VACE_TEMPORAL_SCALE_FACTOR
    if num_latent_frames <= 0:
        raise ValueError(
            "num_latent_frames must be positive after temporal compression, "
            f"got {num_latent_frames} for num_frames={num_frames}."
        )

    representative_indices = [
        int(
            torch.nn.functional.interpolate(
                torch.arange(num_frames, dtype=torch.float32).view(1, 1, num_frames, 1, 1),
                size=(num_latent_frames, 1, 1),
                mode="nearest-exact",
            )[0, 0, latent_index, 0, 0].item()
        )
        for latent_index in range(num_latent_frames)
    ]
    range_bounds = [0]
    for left_index, right_index in zip(representative_indices, representative_indices[1:]):
        range_bounds.append((left_index + right_index) // 2 + 1)
    range_bounds.append(num_frames)

    condition_ranges: list[range] = []
    used_latent_indices: set[int] = set()
    for condition_index in condition_indices:
        latent_index = min(
            range(num_latent_frames),
            key=lambda idx: abs(representative_indices[idx] - condition_index),
        )
        if latent_index in used_latent_indices:
            raise ValueError(
                "Multiple sparse condition indices map to the same Wan VACE latent-time bucket; "
                f"got condition_indices={condition_indices} and representative_indices={representative_indices}."
            )
        used_latent_indices.add(latent_index)
        condition_ranges.append(range(range_bounds[latent_index], range_bounds[latent_index + 1]))
    return condition_ranges


def build_video_and_mask(
    condition_images: list[Image.Image],
    *,
    condition_indices: list[int] | None = None,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Build a conditioning video and generation mask from sparse or dense frame constraints."""
    if not condition_images:
        raise ValueError("condition_images must contain at least one frame.")

    use_dense_prefix = condition_indices is None
    if use_dense_prefix:
        num_condition_frames = min(len(condition_images), NUM_TOTAL_FRAMES)
        condition_indices = list(range(num_condition_frames))
        condition_images = condition_images[:num_condition_frames]
    elif len(condition_images) != len(condition_indices):
        raise ValueError(
            "condition_images and condition_indices must have equal length, "
            f"got {len(condition_images)} and {len(condition_indices)}."
        )

    if len(set(condition_indices)) != len(condition_indices):
        raise ValueError("condition_indices must be unique.")
    if any(idx < 0 or idx >= NUM_TOTAL_FRAMES for idx in condition_indices):
        raise ValueError(
            f"condition_indices must be in [0, {NUM_TOTAL_FRAMES - 1}], got {condition_indices}."
        )

    conditioned_frames = [image.convert("RGB").resize((WIDTH, HEIGHT)) for image in condition_images]
    placeholder = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))
    keep_frame = Image.new("L", (WIDTH, HEIGHT), 0)
    generate_frame = Image.new("L", (WIDTH, HEIGHT), 255)

    video = [placeholder.copy() for _ in range(NUM_TOTAL_FRAMES)]
    mask = [generate_frame.copy() for _ in range(NUM_TOTAL_FRAMES)]
    if use_dense_prefix:
        frame_ranges = [range(index, index + 1) for index in range(len(conditioned_frames))]
    else:
        frame_ranges = _compute_sparse_condition_ranges(
            condition_indices=condition_indices,
            num_frames=NUM_TOTAL_FRAMES,
        )
    for conditioned_frame, frame_range in zip(conditioned_frames, frame_ranges):
        for frame_index in frame_range:
            video[frame_index] = conditioned_frame
            mask[frame_index] = keep_frame.copy()
    return video, mask


def _to_pil_rgb(frame: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
    """Convert CHW/HWC tensor-array/image data into an RGB PIL image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor/array, got shape={array.shape}.")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    elif array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Cannot infer channel dimension from shape={array.shape}.")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.max(array)) if array.size else 0.0
        min_value = float(np.min(array)) if array.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            array = (array * 255.0).round()
        array = np.clip(array, 0.0, 255.0).astype(np.uint8, copy=False)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)

    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def load_lerobot_images(
    *,
    repo_id: str = LEROBOT_REPO_ID,
    sample_index: int = LEROBOT_SAMPLE_INDEX,
    episode_index: int | None = LEROBOT_EPISODE_INDEX,
    frame_offset: int = LEROBOT_FRAME_OFFSET,
    condition_start_frame_index: int = CONDITION_START_FRAME_INDEX,
    video_key: str = LEROBOT_VIDEO_KEY,
    num_frames: int = NUM_CONDITION_FRAMES,
) -> list[Image.Image]:
    """Load sequential LeRobot camera frames and convert them to PIL images."""
    return load_lerobot_images_at_offsets(
        repo_id=repo_id,
        sample_index=sample_index,
        episode_index=episode_index,
        frame_offset=frame_offset,
        condition_start_frame_index=condition_start_frame_index,
        video_key=video_key,
        frame_offsets=list(range(num_frames)),
    )


def load_lerobot_images_at_offsets(
    *,
    repo_id: str = LEROBOT_REPO_ID,
    sample_index: int = LEROBOT_SAMPLE_INDEX,
    episode_index: int | None = LEROBOT_EPISODE_INDEX,
    frame_offset: int = LEROBOT_FRAME_OFFSET,
    condition_start_frame_index: int = CONDITION_START_FRAME_INDEX,
    video_key: str = LEROBOT_VIDEO_KEY,
    frame_offsets: list[int],
) -> list[Image.Image]:
    """Load selected LeRobot frame offsets and convert them to PIL images."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}.")
    if episode_index is not None and episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}.")
    if frame_offset < 0:
        raise ValueError(f"frame_offset must be >= 0, got {frame_offset}.")
    if condition_start_frame_index < 0:
        raise ValueError(
            f"condition_start_frame_index must be >= 0, got {condition_start_frame_index}."
        )
    if not frame_offsets:
        raise ValueError("frame_offsets must contain at least one offset.")
    if any(offset < 0 for offset in frame_offsets):
        raise ValueError(f"frame_offsets must be >= 0, got {frame_offsets}.")

    if episode_index is None:
        dataset = LeRobotDataset(repo_id, video_backend="pyav")
        start_index = sample_index + condition_start_frame_index
    else:
        # Episode-local loading keeps DROID use-cases fast and avoids full-repo shard fetches.
        dataset = LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
        start_index = frame_offset + condition_start_frame_index

    frame_indices = [start_index + offset for offset in frame_offsets]
    if max(frame_indices) >= len(dataset):
        raise ValueError(
            f"Requested conditioning indices {frame_indices} exceed dataset length {len(dataset)}."
        )

    images: list[Image.Image] = []
    for idx in frame_indices:
        sample = dataset[idx]
        if video_key not in sample:
            available_keys = [key for key in sample if key.startswith("observation.images.")]
            raise KeyError(
                f"video_key={video_key!r} not found in sample at index {idx}. "
                f"Available camera keys: {available_keys}"
            )
        images.append(_to_pil_rgb(sample[video_key]))
    return images


def load_droid_images() -> list[Image.Image]:
    """Load DROID conditioning images with DROID-specific defaults."""
    return load_lerobot_images(
        repo_id=DROID_REPO_ID,
        episode_index=DROID_EPISODE_INDEX,
        frame_offset=DROID_FRAME_OFFSET,
        condition_start_frame_index=DROID_CONDITION_START_FRAME_INDEX,
        video_key=DROID_VIDEO_KEY,
        num_frames=NUM_CONDITION_FRAMES,
    )


def load_droid_first_last_images() -> list[Image.Image]:
    """Load only the first and last DROID conditioning frames for the configured rollout length."""
    return load_lerobot_images_at_offsets(
        repo_id=DROID_REPO_ID,
        episode_index=DROID_EPISODE_INDEX,
        frame_offset=DROID_FRAME_OFFSET,
        condition_start_frame_index=DROID_CONDITION_START_FRAME_INDEX,
        video_key=DROID_VIDEO_KEY,
        frame_offsets=[0, NUM_TOTAL_FRAMES - 1],
    )


def load_aloha_images() -> list[Image.Image]:
    """Load ALOHA conditioning images with ALOHA-specific defaults."""
    return load_lerobot_images(
        repo_id=ALOHA_REPO_ID,
        episode_index=ALOHA_EPISODE_INDEX,
        frame_offset=ALOHA_FRAME_OFFSET,
        condition_start_frame_index=ALOHA_CONDITION_START_FRAME_INDEX,
        video_key=ALOHA_VIDEO_KEY,
        num_frames=NUM_CONDITION_FRAMES,
    )


def load_aloha_first_last_images() -> list[Image.Image]:
    """Load only the first and last ALOHA conditioning frames for the configured rollout length."""
    return load_lerobot_images_at_offsets(
        repo_id=ALOHA_REPO_ID,
        episode_index=ALOHA_EPISODE_INDEX,
        frame_offset=ALOHA_FRAME_OFFSET,
        condition_start_frame_index=ALOHA_CONDITION_START_FRAME_INDEX,
        video_key=ALOHA_VIDEO_KEY,
        frame_offsets=[0, NUM_TOTAL_FRAMES - 1],
    )


def load_lerobot_first_last_images() -> list[Image.Image]:
    """Load only the first and last conditioning frames for the configured rollout length."""
    return load_lerobot_images_at_offsets(frame_offsets=[0, NUM_TOTAL_FRAMES - 1])


def load_reference_images() -> tuple[list[Image.Image], list[int] | None]:
    """Load the configured reference image source(s) and optional sparse frame indices."""
    if REFERENCE_IMAGE_SOURCE == "lerobot":
        return load_lerobot_images(), None
    if REFERENCE_IMAGE_SOURCE == "lerobot_first_last":
        return load_lerobot_first_last_images(), [0, NUM_TOTAL_FRAMES - 1]
    if REFERENCE_IMAGE_SOURCE == "droid":
        return load_droid_images(), None
    if REFERENCE_IMAGE_SOURCE == "droid_first_last":
        return load_droid_first_last_images(), [0, NUM_TOTAL_FRAMES - 1]
    if REFERENCE_IMAGE_SOURCE == "aloha":
        return load_aloha_images(), None
    if REFERENCE_IMAGE_SOURCE == "aloha_first_last":
        return load_aloha_first_last_images(), [0, NUM_TOTAL_FRAMES - 1]
    if REFERENCE_IMAGE_SOURCE == "url":
        from diffusers.utils import load_image

        return [load_image(IMAGE)], None
    raise ValueError(
        "Unsupported REFERENCE_IMAGE_SOURCE="
        f"{REFERENCE_IMAGE_SOURCE!r}; use 'lerobot', 'lerobot_first_last', "
        "'droid', 'droid_first_last', 'aloha', 'aloha_first_last', or 'url'."
    )


def generate_video(output_path: Path = OUTPUT_PATH) -> Path:
    """Load Wan VACE, enable CPU offload, generate one video, and save it."""
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    condition_images, condition_indices = load_reference_images()
    video, mask = build_video_and_mask(condition_images, condition_indices=condition_indices)
    frames = pipe(
        prompt=PROMPT,
        video=video,
        mask=mask,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_TOTAL_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).frames[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _export_video(video_frames=list(frames), output_video_path=str(output_path), fps=FPS)
    return output_path


def main() -> None:
    """Run the minimal Wan VACE smoke check."""
    output_path = generate_video()
    print(f"Saved generated video: {output_path}")


if __name__ == "__main__":
    main()
