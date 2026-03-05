"""Run Wan VACE world-model inference and export GT-vs-generated grids.

This entrypoint uses typed YAML-backed config plus CLI overrides and shared
batch preparation utilities.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import random
from pathlib import Path
import sys
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan_vace import prompt_clean
from transformers import AutoTokenizer, UMT5EncoderModel

# Ensure local `src/` package imports work when run as `python scripts/train/infer_world_model.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    # Prevent sibling `world_model.py` from shadowing the `world_model` package.
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
loaded_world_model = sys.modules.get("world_model")
if loaded_world_model is not None and not hasattr(loaded_world_model, "__path__"):
    # Drop incorrectly loaded module objects so package import can succeed.
    sys.modules.pop("world_model", None)

from world_model.config import InferScriptConfig, apply_namespace_overrides, load_infer_config
from world_model.data import build_lerobot_dataloader, prepare_packed_batch
from world_model.data.schema import PreparedPackedBatch
from world_model.eval import infer_future_videos_chunkwise
from world_model.latents import WanVAE
from world_model.models.wan_vace_conditioning import ActionTokenEncoder
from world_model.models.wan_vace_factory import build_wan_vace_runtime_modules
from world_model.models.wan_vace_world_model import WanVACEWorldModel


def _config_parser() -> argparse.ArgumentParser:
    """Create parser for bootstrap config argument."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path; defaults to configs/eval/infer_world_model.yaml.",
    )
    return parser


def _build_parser(defaults: InferScriptConfig) -> argparse.ArgumentParser:
    """Create full CLI parser using dataclass defaults."""
    parser = argparse.ArgumentParser(description=__doc__, parents=[_config_parser()])
    parser.add_argument(
        "--checkpoint",
        default=defaults.checkpoint,
        help="Optional local fine-tune checkpoint .pt to overlay on the pretrained Wan VACE transformer.",
    )
    parser.add_argument(
        "--video-path",
        default=defaults.video_path,
        help="Optional local video file. If set, dataset loading is skipped.",
    )
    parser.add_argument("--start-frame", type=int, default=defaults.start_frame)
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--video-key", default=defaults.video_key)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--context-len", type=int, default=defaults.context_len)
    parser.add_argument("--horizon-len", type=int, default=defaults.horizon_len)
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--subset-size", type=int, default=defaults.subset_size)
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--integration-steps", type=int, default=defaults.integration_steps)
    parser.add_argument("--num-vis-frames", type=int, default=defaults.num_vis_frames)
    parser.add_argument(
        "--load-pretrained-backbone",
        action="store_true",
        default=defaults.load_pretrained_backbone,
    )
    parser.add_argument("--no-load-pretrained-backbone", dest="load_pretrained_backbone", action="store_false")
    parser.add_argument("--wan-vace-model-id", default=defaults.wan_vace_model_id)
    parser.add_argument("--wan-vace-subfolder", default=defaults.wan_vace_subfolder)
    parser.add_argument("--wan-num-attention-heads", type=int, default=defaults.wan_num_attention_heads)
    parser.add_argument("--wan-attention-head-dim", type=int, default=defaults.wan_attention_head_dim)
    parser.add_argument("--wan-text-dim", type=int, default=defaults.wan_text_dim)
    parser.add_argument("--wan-freq-dim", type=int, default=defaults.wan_freq_dim)
    parser.add_argument("--wan-ffn-dim", type=int, default=defaults.wan_ffn_dim)
    parser.add_argument("--wan-num-layers", type=int, default=defaults.wan_num_layers)
    parser.add_argument("--vace-layers", type=int, nargs="+", default=list(defaults.vace_layers))
    parser.add_argument("--control-scale", type=float, default=defaults.control_scale)
    parser.add_argument("--mask-channels", type=int, default=defaults.mask_channels)
    parser.add_argument(
        "--conditioning-mode",
        choices=("action", "prompt"),
        default=defaults.conditioning_mode,
        help="Use action-plan tokens for scope-aligned evaluation or prompt tokens for upstream-style smoke tests.",
    )
    parser.add_argument(
        "--prompt",
        default=defaults.prompt,
        help="Prompt text used when --conditioning-mode prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=defaults.negative_prompt,
        help="Negative prompt used for classifier-free guidance when --conditioning-mode prompt.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=defaults.guidance_scale,
        help="Classifier-free guidance scale for prompt conditioning.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=defaults.max_sequence_length,
        help="Maximum tokenizer sequence length for prompt conditioning.",
    )
    parser.add_argument(
        "--single-chunk-rollout",
        action="store_true",
        default=defaults.single_chunk_rollout,
        help="Roll out the entire future window as one chunk while keeping the chunked inference path.",
    )
    parser.add_argument(
        "--multi-chunk-rollout",
        dest="single_chunk_rollout",
        action="store_false",
    )
    parser.add_argument(
        "--action-path",
        default=defaults.action_path,
        help="Optional .npy action tensor with shape [A], [T,A], or [1,T,A].",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=defaults.action_dim,
        help="Action dim when --action-path is not set. 0 infers from checkpoint.",
    )
    parser.add_argument(
        "--action-value",
        type=float,
        default=defaults.action_value,
        help="Fill value for synthetic actions when --action-path is not set.",
    )
    parser.add_argument("--disable-amp", action="store_true", default=defaults.disable_amp)
    parser.add_argument("--enable-amp", dest="disable_amp", action="store_false")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _load_args() -> InferScriptConfig:
    """Load YAML config and apply CLI overrides into final infer config."""
    config_args, _ = _config_parser().parse_known_args()
    defaults = load_infer_config(config_args.config)
    parser = _build_parser(defaults)
    args = parser.parse_args()
    return apply_namespace_overrides(defaults, args)


def _set_seed(seed: int) -> None:
    """Set Python and torch RNG seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_runtime_dtype(*, device: torch.device, disable_amp: bool) -> torch.dtype:
    """Choose an inference dtype that fits the active device and AMP setting."""
    if device.type != "cuda" or disable_amp:
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(*, device: torch.device, disable_amp: bool, dtype: torch.dtype):
    """Build the appropriate autocast context for inference on the active device."""
    if device.type != "cuda" or disable_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    """Load a checkpoint payload from disk."""
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict")
    return payload


def _infer_action_dim_from_checkpoint(ckpt: dict[str, object]) -> int:
    """Infer action feature size from saved action-encoder state dict."""
    state_dict = ckpt.get("action_encoder_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing action_encoder_state_dict")

    for key in ("net.0.weight", "net.1.weight"):
        value = state_dict.get(key)
        if torch.is_tensor(value):
            if key == "net.0.weight":
                return int(value.shape[0])
            return int(value.shape[1])

    raise ValueError("Unable to infer action_dim from checkpoint action_encoder_state_dict")


def _load_video_clip(video_path: str | Path, start_frame: int, total_frames: int) -> torch.Tensor:
    """Load a contiguous clip from a local video as `[1,T,C,H,W]` uint8 tensor."""
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")

    video = iio.imread(Path(video_path))
    if video.ndim == 3:
        video = video[None, ...]
    if video.ndim != 4:
        raise ValueError(f"Expected video array [T,H,W,C], got shape {tuple(video.shape)}")

    num_frames = int(video.shape[0])
    end_frame = start_frame + total_frames
    if end_frame > num_frames:
        raise ValueError(
            f"Requested frames [{start_frame}:{end_frame}] exceed video length {num_frames}. "
            "Reduce --start-frame or use smaller context/horizon."
        )

    clip = video[start_frame:end_frame]
    if clip.shape[-1] == 4:
        clip = clip[..., :3]
    if clip.shape[-1] != 3:
        raise ValueError(f"Expected RGB video with C=3, got shape {tuple(clip.shape)}")

    clip_np = np.ascontiguousarray(clip)
    clip_t = torch.from_numpy(clip_np)
    if clip_t.dtype != torch.uint8:
        clip_t = clip_t.float()
    clip_t = clip_t.permute(0, 3, 1, 2).unsqueeze(0)
    return clip_t


def _load_action_tensor(
    *,
    action_path: str,
    action_dim: int,
    action_value: float,
    device: torch.device,
) -> torch.Tensor:
    """Load action conditioning tensor as `[1,A]` or `[1,T,A]`."""
    if action_path:
        action_np = np.load(Path(action_path))
        action_t = torch.from_numpy(np.ascontiguousarray(action_np))
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        elif action_t.ndim == 2:
            action_t = action_t.unsqueeze(0)
        elif action_t.ndim == 3 and action_t.shape[0] == 1:
            pass
        else:
            raise ValueError(
                f"Unsupported action array shape {tuple(action_t.shape)}; expected [A], [T,A], or [1,T,A]."
            )
        return action_t.to(device=device, dtype=torch.float32)

    if action_dim <= 0:
        raise ValueError("action_dim must be positive when --action-path is not provided")
    return torch.full((1, action_dim), fill_value=action_value, dtype=torch.float32, device=device)


def _prepare_from_local_video(
    *,
    cfg: InferScriptConfig,
    ckpt: dict[str, object] | None,
    vae: WanVAE,
    device: torch.device,
) -> Any:
    """Build a prepared batch from a local video clip plus action conditioning."""
    total_frames = cfg.context_len + cfg.horizon_len
    video_btchw = _load_video_clip(cfg.video_path, cfg.start_frame, total_frames).to(device)

    if cfg.action_dim > 0:
        action_dim = cfg.action_dim
    elif ckpt is not None:
        action_dim = _infer_action_dim_from_checkpoint(ckpt)
    elif cfg.conditioning_mode == "prompt":
        action_dim = 1
    else:
        action_dim = 0
    action = _load_action_tensor(
        action_path=cfg.action_path,
        action_dim=action_dim,
        action_value=cfg.action_value,
        device=device,
    )
    batch = {
        cfg.video_key: video_btchw,
        "action": action,
    }
    return prepare_packed_batch(
        batch=batch,
        encoder=vae,
        device=device,
        video_key=cfg.video_key,
        context_len=cfg.context_len,
        horizon_len=cfg.horizon_len,
    )


def build_runtime_modules(
    *,
    cfg: InferScriptConfig,
    prepared: PreparedPackedBatch,
    device: torch.device,
    checkpoint: dict[str, object] | None,
) -> tuple[WanVACEWorldModel, ActionTokenEncoder]:
    """Build Wan VACE runtime modules and optionally overlay a local fine-tune checkpoint."""
    return build_wan_vace_runtime_modules(
        cfg,
        prepared,
        device=device,
        checkpoint=checkpoint,
    )


def _load_flow_match_scheduler(cfg: InferScriptConfig) -> FlowMatchEulerDiscreteScheduler:
    """Load the upstream Wan flow-matching scheduler config."""
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.wan_vace_model_id,
        subfolder="scheduler",
        local_files_only=_offline_mode_enabled(),
    )


def _load_prompt_encoder(cfg: InferScriptConfig) -> tuple[Any, UMT5EncoderModel]:
    """Load the upstream tokenizer and UMT5 text encoder for prompt conditioning."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.wan_vace_model_id,
        subfolder="tokenizer",
        local_files_only=_offline_mode_enabled(),
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.wan_vace_model_id,
        subfolder="text_encoder",
        local_files_only=_offline_mode_enabled(),
    )
    text_encoder.eval()
    return tokenizer, text_encoder


@torch.no_grad()
def build_prompt_conditioning_tokens(
    *,
    prompt: str,
    negative_prompt: str,
    batch_size: int,
    tokenizer: Any,
    text_encoder: UMT5EncoderModel,
    encoder_device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    guidance_scale: float,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode upstream-style prompt embeddings for Wan cross-attention."""
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


@torch.no_grad()
def _get_t5_prompt_embeds(
    *,
    prompt: list[str],
    tokenizer: Any,
    text_encoder: UMT5EncoderModel,
    encoder_device: torch.device,
    output_device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    """Mirror the diffusers Wan prompt-embedding path for inference-time prompts."""
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


def _offline_mode_enabled() -> bool:
    """Mirror Hugging Face offline env handling for local-cache-only loading."""
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _save_grid(
    *,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    output_path: Path,
    num_frames: int,
) -> None:
    """Save a two-row ground-truth vs generated frame grid to disk."""
    pred_frames = pred_video[0].detach().float().cpu()
    target_frames = target_video[0].detach().float().cpu()
    vis_frames = min(num_frames, pred_frames.shape[0], target_frames.shape[0])
    if vis_frames <= 0:
        raise ValueError("No frames available for visualization")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        torch.save(
            {
                "pred_video": pred_video.detach().cpu(),
                "target_video": target_video.detach().cpu(),
            },
            output_path.with_suffix(".pt"),
        )
        return

    frame_h = int(pred_frames.shape[2])
    frame_w = int(pred_frames.shape[3])
    margin = 140
    gap = 12
    canvas_w = margin + vis_frames * frame_w
    canvas_h = frame_h * 2 + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, frame_h // 2), "Ground-truth", fill=(30, 30, 30))
    draw.text((24, frame_h + gap + frame_h // 2), "Generated", fill=(30, 30, 30))

    for idx in range(vis_frames):
        gt = (target_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        pred = (pred_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        canvas.paste(Image.fromarray(gt), (margin + idx * frame_w, 0))
        canvas.paste(Image.fromarray(pred), (margin + idx * frame_w, frame_h + gap))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def main() -> None:
    """Run chunkwise autoregressive inference from pretrained Wan VACE weights."""
    cfg = _load_args()
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_dtype = _select_runtime_dtype(device=device, disable_amp=cfg.disable_amp)
    ckpt = _load_checkpoint(cfg.checkpoint, device=device) if cfg.checkpoint else None

    vae = WanVAE.from_pretrained(
        device=device,
        deterministic=True,
        torch_dtype=runtime_dtype,
    )
    if cfg.video_path:
        prepared = _prepare_from_local_video(
            cfg=cfg,
            ckpt=ckpt,
            vae=vae,
            device=device,
        )
    else:
        loader = build_lerobot_dataloader(
            repo_id=cfg.repo_id,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            dt=cfg.dt,
            batch_size=cfg.batch_size,
            subset_size=cfg.subset_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        batch = next(iter(loader))
        prepared = prepare_packed_batch(
            batch=batch,
            encoder=vae,
            device=device,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
        )

    model, action_encoder = build_runtime_modules(
        cfg=cfg,
        prepared=prepared,
        device=device,
        checkpoint=ckpt,
    )
    if device.type == "cuda" and not cfg.disable_amp:
        model = model.to(device=device, dtype=runtime_dtype)
        action_encoder = action_encoder.to(device=device, dtype=runtime_dtype)
    scheduler = _load_flow_match_scheduler(cfg)

    model.eval()
    action_encoder.eval()

    if cfg.conditioning_mode == "prompt":
        tokenizer, text_encoder = _load_prompt_encoder(cfg)
        backbone_dtype = next(model.backbone.parameters()).dtype
        prompt_encoder_device = torch.device("cpu")
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens, negative_cross_attention_tokens = build_prompt_conditioning_tokens(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                batch_size=prepared.z_past_video.shape[0],
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                encoder_device=prompt_encoder_device,
                output_device=device,
                dtype=backbone_dtype,
                guidance_scale=cfg.guidance_scale,
                max_sequence_length=cfg.max_sequence_length,
            )
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
            cross_attention_tokens = action_encoder(prepared.a_plan)
        negative_cross_attention_tokens = None

    with _autocast_context(device=device, disable_amp=cfg.disable_amp, dtype=runtime_dtype):
        pred_future_video = infer_future_videos_chunkwise(
            model,
            z_past_video=prepared.z_past_video,
            future_steps=prepared.z_future_video.shape[2],
            cross_attention_tokens=cross_attention_tokens,
            k=cfg.k,
            integration_steps=cfg.integration_steps,
            negative_cross_attention_tokens=negative_cross_attention_tokens,
            guidance_scale=cfg.guidance_scale,
            chunk_conditioning=(cfg.conditioning_mode == "action"),
            single_chunk_rollout=cfg.single_chunk_rollout,
            scheduler=scheduler,
        )

        pred_video = vae.decode(pred_future_video, output_layout="BTCHW", output_range="zero_to_one")
        target_video = vae.decode(prepared.z_future_video, output_layout="BTCHW", output_range="zero_to_one")

    grid_path = output_dir / "comparison_grid.png"
    _save_grid(
        pred_video=pred_video,
        target_video=target_video,
        output_path=grid_path,
        num_frames=cfg.num_vis_frames,
    )
    print(f"Saved comparison grid: {grid_path}")


if __name__ == "__main__":
    main()
