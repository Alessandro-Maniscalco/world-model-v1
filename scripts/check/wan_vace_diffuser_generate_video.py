"""Generate one Wan VACE video with the public diffusers pipeline and CPU offload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = ""
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
REFERENCE_IMAGE_SOURCE = "lerobot"  # "lerobot" or "url"
LEROBOT_REPO_ID = "lerobot/libero"
LEROBOT_SAMPLE_INDEX = 0
LEROBOT_VIDEO_KEY = "observation.images.image"
NUM_CONDITION_FRAMES = 19
OUTPUT_PATH = Path("runs/check_wan_vace_diffuser/generated_robot_pick_19frames_no_prompt_50steps.mp4")
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 25
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
FPS = 16


def _export_video(*, video_frames: list[object], output_video_path: str, fps: int) -> str:
    """Export generated frames to an mp4 with diffusers utilities."""
    from diffusers.utils import export_to_video

    return export_to_video(video_frames=video_frames, output_video_path=output_video_path, fps=fps)


def build_video_and_mask(condition_images: list[Image.Image]) -> tuple[list[Image.Image], list[Image.Image]]:
    """Build a conditioning video and generation mask from base image frames."""
    if not condition_images:
        raise ValueError("condition_images must contain at least one frame.")
    num_condition_frames = min(len(condition_images), NUM_FRAMES)
    conditioned_frames = [image.convert("RGB").resize((WIDTH, HEIGHT)) for image in condition_images[:num_condition_frames]]
    placeholder = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))
    keep_frame = Image.new("L", (WIDTH, HEIGHT), 0)
    generate_frame = Image.new("L", (WIDTH, HEIGHT), 255)
    num_generated_frames = NUM_FRAMES - num_condition_frames
    video = conditioned_frames + [placeholder.copy() for _ in range(num_generated_frames)]
    mask = [keep_frame.copy() for _ in range(num_condition_frames)] + [
        generate_frame.copy() for _ in range(num_generated_frames)
    ]
    return video, mask


def _to_pil_rgb(frame_tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to an RGB PIL image."""
    frame = frame_tensor.detach().cpu().clamp(0.0, 1.0)
    frame_hwc = frame.permute(1, 2, 0).numpy()
    frame_uint8 = np.ascontiguousarray((frame_hwc * 255.0).round().astype("uint8"))
    return Image.fromarray(frame_uint8, mode="RGB")


def load_lerobot_images(
    *,
    repo_id: str = LEROBOT_REPO_ID,
    sample_index: int = LEROBOT_SAMPLE_INDEX,
    video_key: str = LEROBOT_VIDEO_KEY,
    num_frames: int = NUM_CONDITION_FRAMES,
) -> list[Image.Image]:
    """Load sequential LeRobot camera frames and convert them to PIL images."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
    dataset = LeRobotDataset(repo_id, video_backend="pyav")
    images: list[Image.Image] = []
    for offset in range(num_frames):
        sample = dataset[sample_index + offset]
        images.append(_to_pil_rgb(sample[video_key]))
    return images


def load_reference_images() -> list[Image.Image]:
    """Load the configured reference image source(s)."""
    if REFERENCE_IMAGE_SOURCE == "lerobot":
        return load_lerobot_images()
    if REFERENCE_IMAGE_SOURCE == "url":
        from diffusers.utils import load_image

        return [load_image(IMAGE)]
    raise ValueError(f"Unsupported REFERENCE_IMAGE_SOURCE={REFERENCE_IMAGE_SOURCE!r}; use 'lerobot' or 'url'.")


def generate_video(output_path: Path = OUTPUT_PATH) -> Path:
    """Load Wan VACE, enable CPU offload, generate one video, and save it."""
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    condition_images = load_reference_images()
    video, mask = build_video_and_mask(condition_images)
    frames = pipe(
        prompt=PROMPT,
        video=video,
        mask=mask,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
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
