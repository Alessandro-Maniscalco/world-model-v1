"""Sweep the local repo inference path over resolutions.

Supports pretrained local weights or a saved checkpoint overlay.

python scripts/check/sweep_local_repo_resolutions.py \
  --mode base \
  --resolutions 320x240

python scripts/check/sweep_local_repo_resolutions.py \
  --mode checkpoint \
  --checkpoint runs/hour_test_action/checkpoints/step_0000800.pt \
  --resolutions 320x240
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import imageio.v2 as iio
import json
import numpy as np
import os
from pathlib import Path
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKLWan
from diffusers.pipelines.wan.pipeline_wan_vace import WanVACEPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_vace import prompt_clean
from transformers import AutoTokenizer, UMT5EncoderModel


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train" / "aloha_fork_pick_up.yaml"
from world_model.data.prepare import prepare_packed_batch, preprocess_video_for_vae
from world_model.data.temporal import WAN_FRAME_GROUP_SIZE
from world_model.eval import infer_future_videos_chunkwise
from world_model.latents import WanVAE
from world_model.config import load_train_config
from world_model.models.wan_vace_factory import _attach_lora_adapters, build_wan_vace_runtime_modules
from world_model.vendor.wan import WanVACETransformer3DModel


DEFAULT_SWEEP_OUTPUT_DIR = Path("runs/sweep_local")
DEFAULT_RESOLUTIONS = (
    "320x240",
    "384x288",
    "512x384",
)
DEFAULT_MODE = "base"
DEFAULT_CHECKPOINT_REPO_ID = "lerobot/aloha_static_fork_pick_up"
DEFAULT_CHECKPOINT_VIDEO_KEY = "observation.images.cam_high"
DEFAULT_CONTEXT_LEN = 9
DEFAULT_HORIZON_LEN = 8
DEFAULT_K = 1
DEFAULT_DEVICE = "auto"
DEFAULT_ACTION_SOURCE = "auto"
DEFAULT_FPS = 10
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_BASE_PROMPT = ""
DEFAULT_BASE_NEGATIVE_PROMPT = ""
DEFAULT_BASE_GUIDANCE_SCALE = 5.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_BASE_TOTAL_FRAMES = 9
DEFAULT_BASE_CONDITION_FRAMES = 5


def _parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the local-resolution sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "checkpoint"), default=DEFAULT_MODE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory override. Defaults to "
            f"{DEFAULT_SWEEP_OUTPUT_DIR} for both modes."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Train-style YAML config used to define the local runtime for --mode base.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="List of WIDTHxHEIGHT values to test, e.g. 832x480 320x240.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help="Flow-matching integration steps for local inference.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_BASE_PROMPT, help="Base-mode prompt text.")
    parser.add_argument("--negative-prompt", default=DEFAULT_BASE_NEGATIVE_PROMPT, help="Base-mode negative prompt.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_BASE_GUIDANCE_SCALE,
        help="Classifier-free guidance scale for base-mode prompt conditioning.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="Maximum tokenizer sequence length for base-mode prompt conditioning.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path for --mode checkpoint.")
    parser.add_argument("--repo-id", default=DEFAULT_CHECKPOINT_REPO_ID, help="Dataset repo used for checkpoint mode.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index used for checkpoint mode.")
    parser.add_argument("--start-frame", type=int, default=60, help="Episode-local start frame used for checkpoint mode.")
    parser.add_argument("--video-key", default=DEFAULT_CHECKPOINT_VIDEO_KEY, help="Camera key used for checkpoint mode.")
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--horizon-len", type=int, default=DEFAULT_HORIZON_LEN)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--action-source",
        choices=("auto", "sample", "sequence"),
        default=DEFAULT_ACTION_SOURCE,
        help="Checkpoint-mode action layout. 'sample' matches the current training loader, 'sequence' uses per-frame actions.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=DEFAULT_DEVICE,
        help="Execution device for checkpoint mode. Use cpu if your GPU is busy with training.",
    )
    parser.add_argument(
        "--single-chunk-rollout",
        action="store_true",
        help="Use one full future chunk in checkpoint mode.",
    )
    return parser.parse_args()


def _parse_resolution(spec: str) -> tuple[int, int]:
    """Parse one WIDTHxHEIGHT resolution string."""
    normalized = spec.lower().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WIDTHxHEIGHT, got {spec!r}")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {spec!r}")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(
            f"Resolution must be divisible by 16 for Wan VACE, got {width}x{height}."
        )
    return width, height


def _checkpoint_run_stem(checkpoint_path: Path) -> str:
    """Derive a stable output stem from the checkpoint run directory and step."""
    checkpoint_stem = checkpoint_path.stem
    if checkpoint_path.parent.name == "checkpoints" and checkpoint_path.parent.parent != checkpoint_path.parent:
        return f"{checkpoint_path.parent.parent.name}_{checkpoint_stem}"
    if checkpoint_path.parent.name:
        return f"{checkpoint_path.parent.name}_{checkpoint_stem}"
    return checkpoint_stem


def _resolve_output_root(*, mode: str, output_dir: Path | None) -> Path:
    """Choose the effective output root for the active sweep mode."""
    if output_dir is not None:
        return output_dir
    return DEFAULT_SWEEP_OUTPUT_DIR


def _resolve_output_artifacts(
    *,
    mode: str,
    output_dir: Path | None,
    checkpoint_path: Path | None,
    label: str,
    resolution_count: int,
) -> tuple[Path, Path, Path]:
    """Resolve video, comparison, and summary paths for one sweep item."""
    output_root = _resolve_output_root(mode=mode, output_dir=output_dir)
    if mode == "checkpoint" and checkpoint_path is not None:
        run_stem = _checkpoint_run_stem(checkpoint_path)
        video_stem = run_stem if resolution_count == 1 else f"{run_stem}_{label}"
        return (
            output_root / f"{video_stem}.mp4",
            output_root / f"{video_stem}_comparison.mp4",
            output_root / f"{run_stem}_summary.json",
        )

    return (
        output_root / f"{label}.mp4",
        output_root / f"{label}_comparison.mp4",
        output_root / "summary.json",
    )


def _load_checkpoint_runtime_config(checkpoint_path: Path) -> tuple[dict[str, object], SimpleNamespace]:
    """Load a training checkpoint and convert saved config payload into a runtime namespace."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dict.")
    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, dict):
        raise ValueError("Checkpoint missing extra_state.config metadata.")
    saved_cfg = extra_state.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("Checkpoint missing saved config metadata.")
    return checkpoint, SimpleNamespace(**saved_cfg)


def _load_base_runtime_config(config_path: Path) -> SimpleNamespace:
    """Load a local base-runtime config with pretrained weights and no fine-tune overlays."""
    train_cfg = load_train_config(config_path)
    return SimpleNamespace(
        **vars(
            replace(
                train_cfg,
                trainable_backbone="full",
                conditioning_mode="prompt",
                load_pretrained_backbone=True,
            )
        ),
        prompt=DEFAULT_BASE_PROMPT,
        negative_prompt=DEFAULT_BASE_NEGATIVE_PROMPT,
        guidance_scale=DEFAULT_BASE_GUIDANCE_SCALE,
        max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
        single_chunk_rollout=True,
    )


def _infer_checkpoint_action_dim(checkpoint: dict[str, object]) -> int | None:
    """Infer the checkpoint action encoder input width from its saved state dict."""
    action_state = checkpoint.get("action_encoder_state_dict")
    if not isinstance(action_state, dict):
        return None

    preferred_keys = ("net.0.weight", "net.0.bias", "net.1.weight")
    for key in preferred_keys:
        tensor = action_state.get(key)
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 1:
            return int(tensor.shape[0])
        if tensor.ndim >= 2:
            return int(tensor.shape[-1])

    for tensor in action_state.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 1:
            return int(tensor.shape[0])
        if tensor.ndim >= 2:
            return int(tensor.shape[-1])
    return None


def _select_runtime_dtype(*, device: torch.device) -> torch.dtype:
    """Choose the mixed-precision dtype for checkpoint-mode inference."""
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _resolve_device(*, device_name: str) -> torch.device:
    """Resolve the requested execution device for checkpoint mode."""
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_context(*, device: torch.device, dtype: torch.dtype):
    """Build a lightweight autocast context for CUDA inference."""
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_checkpoint_clip(
    *,
    repo_id: str,
    episode_index: int,
    start_frame: int,
    total_frames: int,
    video_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a contiguous dataset clip and its per-frame raw actions for checkpoint mode."""
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, episodes=[episode_index], video_backend="pyav")
    end_frame = start_frame + total_frames
    if end_frame > len(dataset):
        raise ValueError(
            f"Requested frames [{start_frame}:{end_frame}] exceed episode-local length {len(dataset)}."
        )

    frames: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    for frame_index in range(start_frame, end_frame):
        sample = dataset[frame_index]
        if video_key not in sample:
            available = [key for key in sample if key.startswith("observation.images.")]
            raise KeyError(
                f"video_key={video_key!r} not found in sample. Available camera keys: {available}"
            )
        frames.append(sample[video_key].to(dtype=torch.float32))
        action = sample.get("action")
        if action is None:
            raise KeyError("Checkpoint-mode dataset sample is missing 'action'.")
        actions.append(action.to(dtype=torch.float32))

    video = torch.stack(frames, dim=0).unsqueeze(0).to(device=device)
    action_seq = torch.stack(actions, dim=0).unsqueeze(0).to(device=device)
    return video, action_seq


def _select_action_tensor(
    *,
    action_seq: torch.Tensor,
    action_source: str,
    expected_action_dim: int | None,
) -> torch.Tensor:
    """Choose sample-wise or per-frame actions to match the checkpoint's action encoder."""
    if action_seq.ndim != 3:
        raise ValueError(f"action_seq must be [B,T,A], got {tuple(action_seq.shape)}")

    raw_action_dim = int(action_seq.shape[-1])
    sequence_action_dim = raw_action_dim * WAN_FRAME_GROUP_SIZE
    resolved_source = action_source
    if action_source == "auto":
        if expected_action_dim == raw_action_dim:
            resolved_source = "sample"
        elif expected_action_dim == sequence_action_dim:
            resolved_source = "sequence"
        else:
            resolved_source = "sample"

    if resolved_source == "sample":
        return action_seq[:, 0]
    if resolved_source == "sequence":
        return action_seq
    raise ValueError(f"Unsupported action_source={action_source!r}")


def _scheduler_local_files_only() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _load_prompt_encoder(runtime_cfg: SimpleNamespace) -> tuple[object, UMT5EncoderModel]:
    """Load the upstream tokenizer and UMT5 encoder for prompt-conditioned base mode."""
    tokenizer = AutoTokenizer.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder="tokenizer",
        local_files_only=_scheduler_local_files_only(),
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder="text_encoder",
        local_files_only=_scheduler_local_files_only(),
    )
    text_encoder.eval()
    return tokenizer, text_encoder


@torch.no_grad()
def _get_t5_prompt_embeds(
    *,
    prompt: list[str],
    tokenizer: object,
    text_encoder: UMT5EncoderModel,
    encoder_device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    """Mirror the diffusers Wan prompt-embedding path for local base-mode inference."""
    cleaned_prompt = [prompt_clean(text) for text in prompt]
    text_inputs = tokenizer(
        cleaned_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(encoder_device)
    attention_mask = text_inputs.attention_mask.to(encoder_device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids, attention_mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(device=output_device, dtype=dtype)
    trimmed = [hidden[:seq_len] for hidden, seq_len in zip(prompt_embeds, seq_lens)]
    return torch.stack(
        [torch.cat([hidden, hidden.new_zeros(max_sequence_length - hidden.size(0), hidden.size(1))]) for hidden in trimmed],
        dim=0,
    )


@torch.no_grad()
def _build_prompt_conditioning_tokens(
    *,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    tokenizer: object,
    text_encoder: UMT5EncoderModel,
    encoder_device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build prompt and optional negative prompt embeddings for local base-mode inference."""
    prompt_embeds = _get_t5_prompt_embeds(
        prompt=[prompt] * batch_size,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        encoder_device=encoder_device,
        output_device=output_device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
    )
    negative_prompt_embeds = None
    if guidance_scale > 1.0:
        negative_prompt_embeds = _get_t5_prompt_embeds(
            prompt=[negative_prompt] * batch_size,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            encoder_device=encoder_device,
            output_device=output_device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
    return prompt_embeds, negative_prompt_embeds


def _tensor_video_to_frames(video_btchw: torch.Tensor) -> list[object]:
    """Convert a BTCHW zero-to-one tensor into HWC uint8 frames for MP4 export."""
    video = video_btchw[0].detach().float().cpu().clamp(0.0, 1.0)
    frames: list[object] = []
    for frame in video:
        frame_hwc = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8", copy=False)
        frames.append(np.ascontiguousarray(frame_hwc))
    return frames


def _export_video(*, video_frames: list[object], output_video_path: str, fps: int) -> str:
    """Export generated frames to an mp4 with an explicit RGB-safe writer."""
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in video_frames:
            writer.append_data(np.ascontiguousarray(frame))
    return str(output_path)


def _build_side_by_side_video(*, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Concatenate two BTCHW videos horizontally after aligning frame counts."""
    target_steps = min(int(left.shape[1]), int(right.shape[1]))
    left_aligned = left[:, :target_steps].detach().cpu()
    right_aligned = right[:, :target_steps].detach().cpu()
    return torch.cat([left_aligned, right_aligned], dim=4)


def _normalize_video_for_export(video: torch.Tensor) -> torch.Tensor:
    """Normalize BTCHW video tensors into the zero-to-one range expected by MP4 export."""
    normalized = video.detach().float()
    if normalized.numel() == 0:
        return normalized
    if float(normalized.max().item()) > 1.0:
        normalized = normalized / 255.0
    return normalized.clamp(0.0, 1.0)


def _to_pil_rgb_frame(frame: torch.Tensor | np.ndarray) -> "Image.Image":
    """Convert one CHW/HWC tensor or array into an RGB PIL image."""
    from PIL import Image

    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim != 3:
        raise ValueError(f"Expected one RGB frame with 3 dims, got shape {tuple(array.shape)}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame with 3 channels, got shape {tuple(array.shape)}")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max()) if array.size else 0.0
        min_value = float(array.min()) if array.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).round().astype(np.uint8, copy=False)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def _build_dense_prefix_condition_lists(
    *,
    target_video: torch.Tensor,
    context_len: int,
) -> tuple[list["Image.Image"], list["Image.Image"]]:
    """Build PIL conditioning video and binary mask lists for a dense prefix."""
    from PIL import Image

    video_btchw = _normalize_video_for_export(target_video)
    total_frames = int(video_btchw.shape[1])
    if context_len <= 0 or context_len > total_frames:
        raise ValueError(f"context_len must be in [1,{total_frames}], got {context_len}")

    height = int(video_btchw.shape[3])
    width = int(video_btchw.shape[4])
    placeholder = Image.new("RGB", (width, height), (128, 128, 128))
    keep_frame = Image.new("L", (width, height), 0)
    generate_frame = Image.new("L", (width, height), 255)

    video_frames: list["Image.Image"] = []
    mask_frames: list["Image.Image"] = []
    for frame_index in range(total_frames):
        if frame_index < context_len:
            video_frames.append(_to_pil_rgb_frame(video_btchw[0, frame_index]))
            mask_frames.append(keep_frame.copy())
        else:
            video_frames.append(placeholder.copy())
            mask_frames.append(generate_frame.copy())
    return video_frames, mask_frames


def _load_local_pipeline(
    *,
    runtime_cfg: SimpleNamespace,
    runtime_dtype: torch.dtype,
    transformer: WanVACETransformer3DModel | None = None,
) -> WanVACEPipeline:
    """Load a Wan VACE pipeline using the local vendored transformer implementation."""
    tokenizer = AutoTokenizer.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder="tokenizer",
        local_files_only=_scheduler_local_files_only(),
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder="text_encoder",
        torch_dtype=runtime_dtype,
        local_files_only=_scheduler_local_files_only(),
    )
    if transformer is None:
        transformer = WanVACETransformer3DModel.from_pretrained(
            runtime_cfg.wan_vace_model_id,
            subfolder=runtime_cfg.wan_vace_subfolder or None,
            local_files_only=_scheduler_local_files_only(),
        )
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=runtime_dtype,
        local_files_only=_scheduler_local_files_only(),
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder="scheduler",
        local_files_only=_scheduler_local_files_only(),
    )
    pipe = WanVACEPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
    return pipe


def _load_checkpoint_transformer(
    *,
    runtime_cfg: SimpleNamespace,
    checkpoint: dict[str, object],
) -> WanVACETransformer3DModel:
    """Load checkpoint-backed transformer weights into the canonical pipeline backbone."""
    transformer = WanVACETransformer3DModel.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder=runtime_cfg.wan_vace_subfolder or None,
        local_files_only=_scheduler_local_files_only(),
    )
    if getattr(runtime_cfg, "trainable_backbone", "full") == "lora":
        _attach_lora_adapters(backbone=transformer, cfg=runtime_cfg)

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint missing model_state_dict")
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in model_state.items()
        if key.startswith("backbone.")
    }
    incompatible = transformer.load_state_dict(backbone_state, strict=False)
    missing = [key for key in incompatible.missing_keys if "lora_" not in key]
    unexpected = [key for key in incompatible.unexpected_keys if "control_" not in key]
    if missing or unexpected:
        raise ValueError(
            "Checkpoint backbone overlay mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    return transformer


@torch.no_grad()
def _run_local_pipeline(
    *,
    pipe: WanVACEPipeline,
    video_frames: list["Image.Image"],
    mask_frames: list["Image.Image"],
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    generator: torch.Generator | None,
    guidance_scale: float,
    max_sequence_length: int,
    conditioning_scale: float,
    prompt: str,
    progress_label: str | None = None,
) -> np.ndarray:
    """Run the canonical Wan VACE pipeline path with a shared no-conditioning prompt."""

    if num_frames % pipe.vae_scale_factor_temporal != 1:
        num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device
    batch_size = 1
    vae_dtype = pipe.vae.dtype
    transformer_dtype = pipe.transformer.dtype
    do_cfg = False

    conditioning_scale_tensor = torch.tensor(
        [conditioning_scale] * len(pipe.transformer.config.vace_layers),
        device=device,
        dtype=transformer_dtype,
    )

    encoded_prompt_embeds, encoded_negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    encoded_prompt_embeds = encoded_prompt_embeds.to(device=device, dtype=transformer_dtype)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    video_tensor, mask_tensor, reference_images = pipe.preprocess_conditions(
        video_frames,
        mask_frames,
        None,
        batch_size,
        height,
        width,
        num_frames,
        torch.float32,
        device,
    )
    num_reference_images = len(reference_images[0])

    conditioning_latents = pipe.prepare_video_latents(
        video_tensor,
        mask_tensor,
        reference_images,
        generator,
        device,
    )
    prepared_masks = pipe.prepare_masks(mask_tensor, reference_images, generator)
    conditioning_latents = torch.cat([conditioning_latents, prepared_masks], dim=1).to(transformer_dtype)

    latents = pipe.prepare_latents(
        batch_size,
        pipe.transformer.config.in_channels,
        height,
        width,
        num_frames + num_reference_images * pipe.vae_scale_factor_temporal,
        torch.float32,
        device,
        generator,
        None,
    )

    with pipe.progress_bar(total=len(timesteps)) as progress_bar:
        if progress_label:
            progress_bar.set_description(progress_label)
        for timestep in timesteps:
            pipe._current_timestep = timestep
            latent_model_input = latents.to(transformer_dtype)
            timestep_batch = timestep.expand(latents.shape[0])

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=encoded_prompt_embeds,
                control_hidden_states=conditioning_latents,
                control_hidden_states_scale=conditioning_scale_tensor,
                attention_kwargs=None,
                return_dict=False,
            )[0]

            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            progress_bar.update()

    pipe._current_timestep = None

    latents = latents[:, :, num_reference_images:].to(vae_dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="np")
    pipe.maybe_free_model_hooks()
    return np.ascontiguousarray(video[0])


def _make_constant_video_like(*, video: torch.Tensor, zero_to_one_value: float) -> torch.Tensor:
    """Build a constant video that matches the active numeric range of `video`."""
    if not (0.0 <= zero_to_one_value <= 1.0):
        raise ValueError(f"zero_to_one_value must be in [0,1], got {zero_to_one_value}")

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


def _build_dense_prefix_control_inputs(
    *,
    target_video: torch.Tensor,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a gray-filled control video with the first `context_len` frames kept."""
    batch_size, total_frames, _, height, width = target_video.shape
    if context_len <= 0 or context_len > total_frames:
        raise ValueError(f"context_len must be in [1,{total_frames}], got {context_len}")

    known_bt1hw = torch.zeros(
        (batch_size, total_frames, 1, height, width),
        device=target_video.device,
        dtype=target_video.dtype,
    )
    known_bt1hw[:, :context_len] = 1.0
    gray_video = _make_constant_video_like(video=target_video, zero_to_one_value=128.0 / 255.0)
    control_video = (target_video * known_bt1hw) + (gray_video * (1.0 - known_bt1hw))
    control_mask = (1.0 - known_bt1hw).permute(0, 2, 1, 3, 4)
    return control_video, control_mask


def _resize_mask_to_latent(mask_bt1hw: torch.Tensor, *, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Resize a raw-frame mask to latent time and space with nearest sampling."""
    target_frames, target_height, target_width = target_shape
    return torch.nn.functional.interpolate(
        mask_bt1hw,
        size=(target_frames, target_height, target_width),
        mode="nearest-exact",
    )


def _offload_vae_to_cpu(vae: WanVAE) -> None:
    """Move the wrapped diffusers VAE to CPU to release GPU memory between encode and decode."""
    vae.vae.to("cpu")


def _reload_vae_to_device(vae: WanVAE, *, device: torch.device, runtime_dtype: torch.dtype) -> None:
    """Move the wrapped diffusers VAE back to the active runtime device for decode."""
    vae.vae.to(device=device, dtype=runtime_dtype)


def _checkpoint_autocast_context(*, device: torch.device, runtime_dtype: torch.dtype):
    """Build the autocast context used for checkpoint-mode world-model inference."""
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=runtime_dtype)


@torch.no_grad()
def _decode_future_latents(
    *,
    vae: WanVAE,
    pred_future_video: torch.Tensor,
    target_future_video: torch.Tensor,
    device: torch.device,
    runtime_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode generated and target future latents while avoiding GPU VAE memory spikes."""
    decode_pred = pred_future_video
    decode_target = target_future_video
    if device.type == "cuda":
        vae.vae.to(device="cpu", dtype=torch.float32)
        decode_pred = pred_future_video.to("cpu")
        decode_target = target_future_video.to("cpu")
        torch.cuda.empty_cache()
    pred_video = vae.decode(decode_pred, output_layout="BTCHW", output_range="zero_to_one")
    target_video = vae.decode(decode_target, output_layout="BTCHW", output_range="zero_to_one")
    return pred_video, target_video


@torch.no_grad()
def _run_checkpoint_world_model(
    *,
    runtime_cfg: SimpleNamespace,
    checkpoint: dict[str, object],
    video: torch.Tensor,
    action_seq: torch.Tensor,
    video_key: str,
    width: int,
    height: int,
    k: int,
    integration_steps: int,
    single_chunk_rollout: bool,
    action_source: str,
    device: torch.device,
    runtime_dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run checkpoint-mode inference with the repo world-model path used during training eval."""
    vae = WanVAE.from_pretrained(device=device, deterministic=True, torch_dtype=runtime_dtype)
    expected_action_dim = _infer_checkpoint_action_dim(checkpoint)
    action_tensor = _select_action_tensor(
        action_seq=action_seq,
        action_source=action_source,
        expected_action_dim=expected_action_dim,
    )
    batch = {
        video_key: video,
        "action": action_tensor,
    }
    prepared = prepare_packed_batch(
        batch=batch,
        encoder=vae,
        device=device,
        video_key=video_key,
        context_len=int(getattr(runtime_cfg, "context_len", DEFAULT_CONTEXT_LEN)),
        horizon_len=int(getattr(runtime_cfg, "horizon_len", DEFAULT_HORIZON_LEN)),
        frame_height=height,
        frame_width=width,
    )
    model, action_encoder = build_wan_vace_runtime_modules(
        runtime_cfg,
        prepared,
        device=device,
        checkpoint=checkpoint,
    )
    if device.type == "cuda":
        model = model.to(device=device, dtype=runtime_dtype)
        action_encoder = action_encoder.to(device=device, dtype=runtime_dtype)
    model.eval()
    action_encoder.eval()

    with _checkpoint_autocast_context(device=device, runtime_dtype=runtime_dtype):
        cross_attention_tokens = action_encoder(prepared.a_plan)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        runtime_cfg.wan_vace_model_id,
        subfolder="scheduler",
        local_files_only=_scheduler_local_files_only(),
    )
    with _checkpoint_autocast_context(device=device, runtime_dtype=runtime_dtype):
        pred_future_video = infer_future_videos_chunkwise(
            model,
            z_past_video=prepared.z_past_video,
            future_steps=prepared.z_future_video.shape[2],
            cross_attention_tokens=cross_attention_tokens,
            k=k,
            integration_steps=integration_steps,
            negative_cross_attention_tokens=None,
            guidance_scale=1.0,
            chunk_conditioning=(getattr(runtime_cfg, "conditioning_mode", "action") in ("none", "action")),
            single_chunk_rollout=single_chunk_rollout,
            scheduler=scheduler,
            generator=generator,
        )
    pred_video, target_video = _decode_future_latents(
        vae=vae,
        pred_future_video=pred_future_video,
        target_future_video=prepared.z_future_video,
        device=device,
        runtime_dtype=runtime_dtype,
    )
    target_full_video = preprocess_video_for_vae(
        video.detach().cpu(),
        frame_height=height,
        frame_width=width,
    )
    return _build_rollout_video(
        target_full_video=target_full_video,
        pred_future_video=pred_video,
        context_len=int(getattr(runtime_cfg, "context_len", DEFAULT_CONTEXT_LEN)),
    )


def _sample_local_base_full_video(
    *,
    backbone: WanVACETransformer3DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    z_target: torch.Tensor,
    control_hidden_states: torch.Tensor,
    latent_mask: torch.Tensor,
    cross_attention_tokens: torch.Tensor,
    negative_cross_attention_tokens: torch.Tensor | None,
    guidance_scale: float,
    integration_steps: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Denoise a full latent clip while clamping observed prefix latents after every step."""
    latent_state = torch.randn(
        z_target.shape,
        device=z_target.device,
        dtype=z_target.dtype,
        generator=generator,
    )
    latent_state = (latent_mask * latent_state) + ((1.0 - latent_mask) * z_target)

    scheduler.set_timesteps(integration_steps, device=z_target.device)
    for timestep in scheduler.timesteps:
        timestep_t = timestep.expand(z_target.shape[0]).to(device=z_target.device, dtype=z_target.dtype)
        velocity = backbone(
            hidden_states=latent_state,
            timestep=timestep_t,
            encoder_hidden_states=cross_attention_tokens,
            control_hidden_states=control_hidden_states,
            control_hidden_states_scale=None,
            attention_mask=None,
            return_dict=True,
        ).sample
        if negative_cross_attention_tokens is not None:
            velocity_uncond = backbone(
                hidden_states=latent_state,
                timestep=timestep_t,
                encoder_hidden_states=negative_cross_attention_tokens,
                control_hidden_states=control_hidden_states,
                control_hidden_states_scale=None,
                attention_mask=None,
                return_dict=True,
            ).sample
            velocity = velocity_uncond + guidance_scale * (velocity - velocity_uncond)
        latent_state = scheduler.step(velocity, timestep, latent_state, generator=generator, return_dict=False)[0]
        latent_state = (latent_mask * latent_state) + ((1.0 - latent_mask) * z_target)

    return latent_state


def _build_rollout_video(
    *,
    target_full_video: torch.Tensor,
    pred_future_video: torch.Tensor,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build full target and generated rollout videos with a shared context prefix."""
    target_full = _normalize_video_for_export(target_full_video)
    pred_future = _normalize_video_for_export(pred_future_video)
    context_video = target_full[:, :context_len]
    target_future = target_full[:, context_len:]
    aligned_future_steps = min(int(target_future.shape[1]), int(pred_future.shape[1]))
    target_rollout = torch.cat([context_video, target_future[:, :aligned_future_steps]], dim=1)
    pred_rollout = torch.cat([context_video, pred_future[:, :aligned_future_steps]], dim=1)
    return target_rollout, pred_rollout


def _run_one_checkpoint_resolution(
    *,
    mode: str,
    config_path: Path,
    checkpoint_path: Path,
    width: int,
    height: int,
    output_path: Path,
    comparison_path: Path,
    repo_id: str,
    episode_index: int,
    start_frame: int,
    video_key: str,
    context_len: int,
    horizon_len: int,
    k: int,
    integration_steps: int,
    fps: int,
    seed: int,
    single_chunk_rollout: bool,
    device_name: str,
    action_source: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    max_sequence_length: int,
) -> dict[str, object]:
    """Run one local world-model generation at a specific resolution."""
    label = f"{width}x{height}"
    effective_inference_steps = max(integration_steps, 50)
    start_time = time.time()
    try:
        checkpoint: dict[str, object] | None = None
        runtime_notes: list[str] = []
        result_metadata: dict[str, object] = {
            "requested_num_inference_steps": integration_steps,
            "effective_num_inference_steps": effective_inference_steps,
        }
        if mode == "checkpoint":
            checkpoint, runtime_cfg = _load_checkpoint_runtime_config(checkpoint_path)
            train_width = int(getattr(runtime_cfg, "frame_width", width) or width)
            train_height = int(getattr(runtime_cfg, "frame_height", height) or height)
            result_metadata.update(
                checkpoint_train_resolution=f"{train_width}x{train_height}",
                checkpoint_trainable_backbone=str(getattr(runtime_cfg, "trainable_backbone", "unknown")),
                checkpoint_conditioning_mode=str(getattr(runtime_cfg, "conditioning_mode", "unknown")),
                checkpoint_path=str(checkpoint_path),
            )
            if (train_width, train_height) != (width, height):
                runtime_notes.append(
                    "Requested inference resolution does not match the checkpoint training resolution "
                    f"({width}x{height} vs {train_width}x{train_height})."
                )
            if "wrong_architecture" in checkpoint_path.name:
                runtime_notes.append(
                    "Checkpoint filename marks this artifact as wrong_architecture; treat failures as expected."
                )
        else:
            runtime_cfg = _load_base_runtime_config(config_path)
            runtime_cfg.prompt = prompt
            runtime_cfg.negative_prompt = negative_prompt
            runtime_cfg.guidance_scale = guidance_scale
            runtime_cfg.max_sequence_length = max_sequence_length
            runtime_cfg.single_chunk_rollout = True
        if mode == "base":
            runtime_notes.append(
                "Base mode uses the canonical Wan VACE pipeline with dense-prefix mask conditioning."
            )
        else:
            runtime_notes.append(
                "Checkpoint mode uses the repo's direct chunkwise world-model inference path to match training."
            )
        device = _resolve_device(device_name=device_name)
        runtime_dtype = _select_runtime_dtype(device=device)
        total_frames = (
            context_len + horizon_len
            if mode == "checkpoint"
            else DEFAULT_BASE_TOTAL_FRAMES
        )
        torch.manual_seed(seed)
        generator = torch.Generator(device=device.type) if device.type == "cuda" else torch.Generator()
        generator.manual_seed(seed)

        video, action_seq = _load_checkpoint_clip(
            repo_id=repo_id,
            episode_index=episode_index,
            start_frame=start_frame,
            total_frames=total_frames,
            video_key=video_key,
            device=device,
        )

        if mode == "base":
            target_full_video = preprocess_video_for_vae(
                video,
                frame_height=height,
                frame_width=width,
            )
            video_frames, mask_frames = _build_dense_prefix_condition_lists(
                target_video=target_full_video,
                context_len=min(DEFAULT_BASE_CONDITION_FRAMES, int(target_full_video.shape[1])),
            )
            pipe = _load_local_pipeline(runtime_cfg=runtime_cfg, runtime_dtype=runtime_dtype)
            frames = _run_local_pipeline(
                pipe=pipe,
                video_frames=video_frames,
                mask_frames=mask_frames,
                height=height,
                width=width,
                num_frames=DEFAULT_BASE_TOTAL_FRAMES,
                num_inference_steps=effective_inference_steps,
                generator=generator,
                guidance_scale=1.0,
                max_sequence_length=int(runtime_cfg.max_sequence_length),
                conditioning_scale=float(getattr(runtime_cfg, "control_scale", 1.0)),
                prompt="",
                progress_label=f"{label} steps",
            )
            pred_rollout = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).unsqueeze(0).float()
            pred_rollout = _normalize_video_for_export(pred_rollout)
            target_rollout = _normalize_video_for_export(target_full_video)
        else:
            target_rollout, pred_rollout = _run_checkpoint_world_model(
                runtime_cfg=runtime_cfg,
                checkpoint=checkpoint,
                video=video,
                action_seq=action_seq,
                video_key=video_key,
                width=width,
                height=height,
                k=k,
                integration_steps=effective_inference_steps,
                single_chunk_rollout=single_chunk_rollout,
                action_source=action_source,
                device=device,
                runtime_dtype=runtime_dtype,
                generator=generator,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _export_video(video_frames=_tensor_video_to_frames(pred_rollout), output_video_path=str(output_path), fps=fps)
        comparison_video = _build_side_by_side_video(left=target_rollout, right=pred_rollout)
        _export_video(
            video_frames=_tensor_video_to_frames(comparison_video),
            output_video_path=str(comparison_path),
            fps=fps,
        )
        return {
            "resolution": label,
            "status": "ok",
            "output_path": str(output_path),
            "comparison_output_path": str(comparison_path),
            "elapsed_s": time.time() - start_time,
            "mode": mode,
            "notes": runtime_notes,
            **result_metadata,
        }
    except Exception as exc:  # pragma: no cover - manual smoke script
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            error = (
                f"{type(exc).__name__}: {exc}. "
                "Stop other GPU jobs, choose a smaller resolution, or rerun with --device cpu."
            )
        else:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "resolution": label,
            "status": "error",
            "error": error,
            "elapsed_s": time.time() - start_time,
            "mode": mode,
            "notes": runtime_notes,
            **result_metadata,
        }


def _save_summary(*, summary_path: Path, results: list[dict[str, object]]) -> Path:
    """Persist the sweep results as JSON for quick review."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    """Run the requested local-resolution sweep in base or checkpoint mode."""
    args = _parse_args()
    parsed_resolutions = [_parse_resolution(spec) for spec in args.resolutions]
    results: list[dict[str, object]] = []
    summary_path: Path | None = None

    if args.mode == "checkpoint" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --mode checkpoint.")

    for width, height in parsed_resolutions:
        label = f"{width}x{height}"
        output_path, comparison_path, summary_path = _resolve_output_artifacts(
            mode=args.mode,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            label=label,
            resolution_count=len(parsed_resolutions),
        )
        effective_inference_steps = max(args.num_inference_steps, 50)
        print(
            "Running local "
            f"{args.mode} at {label} with "
            f"{args.num_inference_steps} requested integration steps "
            f"({effective_inference_steps} effective)..."
        )
        result = _run_one_checkpoint_resolution(
            mode=args.mode,
            config_path=args.config,
            checkpoint_path=args.checkpoint if args.checkpoint is not None else Path(""),
            width=width,
            height=height,
            output_path=output_path,
            comparison_path=comparison_path,
            repo_id=args.repo_id,
            episode_index=args.episode_index,
            start_frame=args.start_frame,
            video_key=args.video_key,
            context_len=args.context_len,
            horizon_len=args.horizon_len,
            k=args.k,
            integration_steps=args.num_inference_steps,
            fps=args.fps,
            seed=args.seed,
            single_chunk_rollout=args.single_chunk_rollout,
            device_name=args.device,
            action_source=args.action_source,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
        )
        results.append(result)
        if result["status"] == "ok":
            print(
                f"{label}: saved {result['output_path']} "
                f"(steps={result['effective_num_inference_steps']})"
            )
            print(f"{label}: saved {result['comparison_output_path']}")
        else:
            print(f"{label}: {result['error']}")

    if summary_path is None:
        summary_path = _resolve_output_root(mode=args.mode, output_dir=args.output_dir) / "summary.json"
    summary_path = _save_summary(summary_path=summary_path, results=results)
    print(f"Saved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
