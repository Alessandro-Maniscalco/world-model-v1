"""Export a local preview clip from the first DROID episode window.

Run:

python scripts/check/preview_droid_sequence.py \
    --frame-offset 25 \
    --num-frames 30 \
    --output-dir runs/check_droid_preview_30frames_25offset

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_model.data.droid_video import export_droid_preview_clip


LEROBOT_REPO_ID = "lerobot/droid_1.0.1"
LEROBOT_EPISODE_INDEX = 0
LEROBOT_FRAME_OFFSET = 25
LEROBOT_VIDEO_KEY = "observation.images.exterior_1_left"
NUM_PREVIEW_FRAMES = 13
OUTPUT_DIR = Path("runs/check_droid_preview")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DROID frame preview export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=LEROBOT_REPO_ID)
    parser.add_argument("--episode-index", type=int, default=LEROBOT_EPISODE_INDEX)
    parser.add_argument("--frame-offset", type=int, default=LEROBOT_FRAME_OFFSET)
    parser.add_argument("--video-key", default=LEROBOT_VIDEO_KEY)
    parser.add_argument("--num-frames", type=int, default=NUM_PREVIEW_FRAMES)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Preview FPS. Defaults to dataset FPS when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Load DROID frames and export ordered PNGs plus a local MP4 preview."""
    args = _parse_args()
    export = export_droid_preview_clip(
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        frame_offset=args.frame_offset,
        video_key=args.video_key,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        fps=args.fps,
    )
    print(f"Saved {len(export.frame_paths)} ordered frame images to: {args.output_dir / 'frames'}")
    print(f"Saved sequential preview video to: {export.video_path}")
    print(f"Dataset FPS: {export.dataset_fps:.3f}; preview FPS used: {export.preview_fps}")
    print(f"Local dataset storage: {export.storage_type}")
    print(
        "Settings: "
        f"repo_id={args.repo_id}, episode_index={args.episode_index}, "
        f"frame_offset={args.frame_offset}, video_key={args.video_key}, num_frames={args.num_frames}"
    )


if __name__ == "__main__":
    main()
