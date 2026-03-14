"""Run a Wan VACE DROID smoke check via the explicit WanVACEPipeline API."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers.pipelines.wan.pipeline_wan_vace import WanVACEPipeline

import wan_vace_diffuser_generate_video as base

MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = ""
NEGATIVE_PROMPT = ""
OUTPUT_PATH = Path("runs/check_wan_vace_pipeline/droid_start25_cond13_total17.mp4")
HEIGHT = 480
WIDTH = 832
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
FPS = 10


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the upstream VACE smoke-check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def _validate_defaults() -> None:
    """Guard against drifting away from the requested DROID smoke-check setup."""
    if base.REFERENCE_IMAGE_SOURCE != "droid":
        raise ValueError(
            "This smoke-check expects REFERENCE_IMAGE_SOURCE='droid' in "
            "wan_vace_diffuser_generate_video.py"
        )
    if base.DROID_CONDITION_START_FRAME_INDEX != 25:
        raise ValueError(
            "This smoke-check expects DROID_CONDITION_START_FRAME_INDEX=25 in "
            "wan_vace_diffuser_generate_video.py"
        )
    if base.NUM_CONDITION_FRAMES != 13:
        raise ValueError(
            "This smoke-check expects NUM_CONDITION_FRAMES=13 in "
            "wan_vace_diffuser_generate_video.py"
        )
    if base.NUM_TOTAL_FRAMES != 17:
        raise ValueError(
            "This smoke-check expects NUM_TOTAL_FRAMES=17 in "
            "wan_vace_diffuser_generate_video.py"
        )


def generate_video(
    *,
    prompt: str = PROMPT,
    negative_prompt: str = NEGATIVE_PROMPT,
    output_path: Path = OUTPUT_PATH,
    height: int = HEIGHT,
    width: int = WIDTH,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: float = GUIDANCE_SCALE,
    fps: int = FPS,
) -> Path:
    """Generate one DROID-conditioned video using the explicit WanVACEPipeline class."""
    _validate_defaults()
    pipe = WanVACEPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    condition_images, condition_indices = base.load_reference_images()
    video, mask = base.build_video_and_mask(condition_images, condition_indices=condition_indices)
    frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        video=video,
        mask=mask,
        height=height,
        width=width,
        num_frames=base.NUM_TOTAL_FRAMES,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).frames[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base._export_video(video_frames=list(frames), output_video_path=str(output_path), fps=fps)
    return output_path


def main() -> None:
    """Run the DROID VACE smoke-check and print the output path."""
    args = _parse_args()
    output_path = generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output_path,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
    )
    print(f"Saved generated video: {output_path}")


if __name__ == "__main__":
    main()
