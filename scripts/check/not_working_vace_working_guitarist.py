"""DON'T CHANGE THIS. Generate one Wan VACE video with the public diffusers pipeline and CPU offload."""


from __future__ import annotations


from pathlib import Path


import torch
from PIL import Image


MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = ""
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
OUTPUT_PATH = Path("runs/check_wan_vace_diffuser/guitar_working.mp4")
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 17
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
FPS = 16


def _export_video(*, video_frames: list[object], output_video_path: str, fps: int) -> str:
   """Export generated frames to an mp4 with diffusers utilities."""
   from diffusers.utils import export_to_video

   return export_to_video(video_frames=video_frames, output_video_path=output_video_path, fps=fps)


def build_video_and_mask(image: Image.Image) -> tuple[list[Image.Image], list[Image.Image]]:
   """Build a first-frame conditioning video and generation mask."""
   first_frame = image.convert("RGB").resize((WIDTH, HEIGHT))
   placeholder = Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))
   keep_frame = Image.new("L", (WIDTH, HEIGHT), 0)
   generate_frame = Image.new("L", (WIDTH, HEIGHT), 255)
   video = [first_frame] + [placeholder.copy() for _ in range(NUM_FRAMES - 1)]
   mask = [keep_frame] + [generate_frame.copy() for _ in range(NUM_FRAMES - 1)]
   return video, mask


def generate_video(output_path: Path = OUTPUT_PATH) -> Path:
   """Load Wan VACE, enable CPU offload, generate one video, and save it."""
   from diffusers import DiffusionPipeline
   from diffusers.utils import load_image

   pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
   pipe.enable_sequential_cpu_offload()

   image = load_image(IMAGE)
   video, mask = build_video_and_mask(image)
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
