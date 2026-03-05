"""Run a real-batch Wan VACE forward-pass smoke test for OOM validation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import torch

# Ensure local `src/` package imports work when run as `python scripts/check/forward_real_batch.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from world_model.chunking import build_full_sequence_chunk_ids
from world_model.data import build_lerobot_dataloader, prepare_packed_batch
from world_model.latents import WanVAE
from world_model.masking import build_block_causal_mask
from world_model.models.wan_vace_conditioning import ActionTokenEncoder
from world_model.models.wan_vace_world_model import WanVACEWorldModel
from world_model.vendor.wan.transformer_wan_vace import WanVACETransformer3DModel


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the forward-pass smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--context-len", type=int, default=10, help="l in frame-time")
    parser.add_argument("--horizon-len", type=int, default=8, help="H in frame-time")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Wan inner dim; must divide by num-heads")
    parser.add_argument("--ffn-dim", type=int, default=0, help="Defaults to 4x hidden dim when set to 0")
    parser.add_argument("--mask-channels", type=int, default=64)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """Set Python and torch RNG seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_small_wan_vace_modules(
    *,
    action_dim: int,
    latent_channels: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    mask_channels: int,
    gradient_checkpointing: bool,
    device: torch.device,
) -> tuple[WanVACEWorldModel, ActionTokenEncoder]:
    """Build a small local Wan VACE stack without downloading pretrained weights."""
    if hidden_dim <= 0:
        raise ValueError(f"--hidden-dim must be positive, got {hidden_dim}")
    if num_heads <= 0:
        raise ValueError(f"--num-heads must be positive, got {num_heads}")
    if hidden_dim % num_heads != 0:
        raise ValueError(f"--hidden-dim {hidden_dim} must be divisible by --num-heads {num_heads}")
    if num_layers <= 0:
        raise ValueError(f"--num-layers must be positive, got {num_layers}")
    if mask_channels <= 0:
        raise ValueError(f"--mask-channels must be positive, got {mask_channels}")

    attention_head_dim = hidden_dim // num_heads
    ffn_dim = hidden_dim * 4
    control_channels = (2 * latent_channels) + mask_channels
    backbone = WanVACETransformer3DModel(
        in_channels=latent_channels,
        out_channels=latent_channels,
        num_attention_heads=num_heads,
        attention_head_dim=attention_head_dim,
        text_dim=hidden_dim,
        freq_dim=min(256, max(8, hidden_dim)),
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        vace_layers=list(range(num_layers)),
        vace_in_channels=control_channels,
    )
    if gradient_checkpointing:
        backbone.enable_gradient_checkpointing()

    model = WanVACEWorldModel(
        backbone=backbone,
        mask_channels=mask_channels,
    ).to(device)
    action_encoder = ActionTokenEncoder(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    return model, action_encoder


@torch.no_grad()
def main() -> None:
    """Run one forward pass and report shape/memory diagnostics."""
    args = _parse_args()
    _set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this OOM smoke check, but no CUDA device is available")
    device = torch.device("cuda")

    print(f"Device: {device}")
    print(
        f"Requested frame-time pipeline: l={args.context_len}, H={args.horizon_len}, "
        f"batch_size={args.batch_size}, heads={args.num_heads}"
    )
    if args.ffn_dim not in (0, args.hidden_dim * 4):
        raise ValueError(
            f"--ffn-dim is no longer configurable for this smoke check; expected 0 or {args.hidden_dim * 4}, "
            f"got {args.ffn_dim}"
        )

    loader = build_lerobot_dataloader(
        repo_id=args.repo_id,
        video_key=args.video_key,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        dt=args.dt,
        batch_size=args.batch_size,
        subset_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    batch = next(iter(loader))

    vae = WanVAE.from_pretrained(device=device, deterministic=True)
    prepared = prepare_packed_batch(
        batch=batch,
        encoder=vae,
        device=device,
        video_key=args.video_key,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
    )

    action_dim = prepared.a_plan.shape[-1]

    chunk_ids = build_full_sequence_chunk_ids(
        past_steps=prepared.z_past_video.shape[2],
        future_steps=prepared.z_future_video.shape[2],
        k=args.k,
        device=device,
    )
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    model, action_encoder = _build_small_wan_vace_modules(
        action_dim=action_dim,
        latent_channels=prepared.z_future_video.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mask_channels=args.mask_channels,
        gradient_checkpointing=args.gradient_checkpointing,
        device=device,
    )
    model.eval()
    action_encoder.eval()

    action_tokens = action_encoder(prepared.a_plan)

    timestep_t = torch.rand(prepared.z_future_video.shape[0], device=device)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.disable_amp):
            out = model(
                noisy_future_video=prepared.z_future_video,
                observed_video=prepared.z_past_video,
                action_tokens=action_tokens,
                timestep_t=timestep_t,
                block_causal_attention_mask=mask,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
            raise RuntimeError(
                f"OOM during forward pass (l={args.context_len}, H={args.horizon_len}). "
                f"peak_allocated={allocated:.2f}GB peak_reserved={reserved:.2f}GB"
            ) from exc
        raise

    allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

    print(
        "Latent tokens: "
        f"total={prepared.total_latent_steps}, "
        f"context={prepared.context_latent_steps}, "
        f"horizon={prepared.horizon_latent_steps}"
    )
    print(
        "Tensor shapes: "
        f"z_past_video={tuple(prepared.z_past_video.shape)} "
        f"z_future_video={tuple(prepared.z_future_video.shape)} "
        f"action_tokens={tuple(action_tokens.shape)} "
        f"out={tuple(out.shape)}"
    )
    print(f"CUDA peak memory: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    print("PASS: forward pass completed without OOM")


if __name__ == "__main__":
    main()
