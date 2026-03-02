"""Run a real-batch world-model forward-pass smoke test for OOM validation."""

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
from world_model.conditioning import ActionEncoder, ProprioEncoder
from world_model.data import build_lerobot_dataloader, prepare_packed_batch
from world_model.latents import WanVAE
from world_model.masking import build_block_causal_mask
from world_model.models.wan_dit_wrapper import WanDiTWrapper


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
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--disable-proprio", action="store_true")
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
        proprio_mode="last",
    )

    latent_dim = prepared.z_future.shape[-1]
    action_dim = prepared.a_plan.shape[-1]

    chunk_ids = build_full_sequence_chunk_ids(
        past_steps=prepared.z_past.shape[1],
        future_steps=prepared.z_future.shape[1],
        k=args.k,
        device=device,
    )
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    action_encoder = ActionEncoder(action_dim=action_dim, hidden_dim=args.hidden_dim, pool="mean").to(device)
    proprio_encoder = None
    if not args.disable_proprio and prepared.q_last is not None:
        proprio_encoder = ProprioEncoder(proprio_dim=prepared.q_last.shape[-1], hidden_dim=args.hidden_dim).to(device)

    action_conditioning = action_encoder(prepared.a_plan)
    proprio_conditioning = None if proprio_encoder is None else proprio_encoder(prepared.q_last)

    model = WanDiTWrapper(
        hidden_dim=args.hidden_dim,
        latent_dim=latent_dim,
        cond_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mixed_precision=not args.disable_amp,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    model.eval()

    timestep_t = torch.rand(prepared.z_future.shape[0], device=device)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        out = model(
            noisy_future_chunk=prepared.z_future,
            past_clean_chunks=prepared.z_past,
            action_conditioning=action_conditioning,
            timestep_t=timestep_t,
            block_causal_attention_mask=mask,
            proprio_conditioning=proprio_conditioning,
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
        f"z_past={tuple(prepared.z_past.shape)} "
        f"z_future={tuple(prepared.z_future.shape)} "
        f"out={tuple(out.shape)}"
    )
    print(f"CUDA peak memory: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    print("PASS: forward pass completed without OOM")


if __name__ == "__main__":
    main()
