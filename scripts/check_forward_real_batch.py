"""Run a real-batch world-model forward-pass smoke test for OOM validation.

Builds masks from chunk ids, applies action/proprio conditioning via AdaLN paths,
and checks a single WanDiTWrapper forward pass for l=10, H=8 by default.
"""

from __future__ import annotations

import argparse
import random

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from world_model.chunking import build_full_sequence_chunk_ids
from world_model.conditioning import ActionEncoder, ProprioEncoder
from world_model.data import flatten_latents_per_timestep, pack_world_model_batch
from world_model.eval.forward_pass import (
    build_frame_deltas,
    expand_to_latent_steps,
    latent_split_from_frame_ratio,
)
from world_model.latents import WanVAE
from world_model.masking import build_block_causal_mask
from world_model.models import WanDiTWrapper


def _parse_args() -> argparse.Namespace:
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
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_dict(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0].keys():
        first_value = batch[0][key]
        if torch.is_tensor(first_value):
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)
        else:
            out[key] = [sample[key] for sample in batch]
    return out


def _prepare_conditioning_sequences(
    *,
    action: torch.Tensor,
    proprio: torch.Tensor | None,
    total_latent_steps: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    action_seq = expand_to_latent_steps(action, total_latent_steps)
    proprio_seq = None if proprio is None else expand_to_latent_steps(proprio, total_latent_steps)
    return action_seq, proprio_seq


@torch.no_grad()
def main() -> None:
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

    deltas = build_frame_deltas(args.context_len, args.horizon_len, args.dt)
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps={args.video_key: deltas},
        video_backend="pyav",
    )

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(args.batch_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_dict,
        drop_last=True,
    )
    batch = next(iter(loader))

    vae = WanVAE.from_pretrained(device=device, deterministic=True)

    video = batch[args.video_key].to(device)
    action = batch["action"].to(device)
    proprio = batch.get("observation.state")
    if proprio is not None:
        proprio = proprio.to(device)

    latents = vae.encode(video)
    z_tokens = flatten_latents_per_timestep(latents)

    total_latent_steps = z_tokens.shape[1]
    t_ctx, t_hor = latent_split_from_frame_ratio(
        total_latent_steps,
        context_frames=args.context_len,
        horizon_frames=args.horizon_len,
    )

    action_seq, proprio_seq = _prepare_conditioning_sequences(
        action=action,
        proprio=proprio,
        total_latent_steps=t_ctx + t_hor,
    )

    packed = pack_world_model_batch(
        z_tokens=z_tokens,
        actions=action_seq,
        proprio=proprio_seq,
        context_len=t_ctx,
        horizon_len=t_hor,
    )

    z_past = packed.z_past
    z_future = packed.z_future
    a_plan = packed.a_plan
    q_last = packed.q_last

    latent_dim = z_future.shape[-1]
    action_dim = a_plan.shape[-1]

    chunk_ids = build_full_sequence_chunk_ids(
        past_steps=z_past.shape[1],
        future_steps=z_future.shape[1],
        k=args.k,
        device=device,
    )
    mask = build_block_causal_mask(chunk_ids, mask_format="additive")

    action_encoder = ActionEncoder(action_dim=action_dim, hidden_dim=args.hidden_dim, pool="mean").to(device)
    proprio_encoder = None
    if not args.disable_proprio and q_last is not None:
        proprio_encoder = ProprioEncoder(proprio_dim=q_last.shape[-1], hidden_dim=args.hidden_dim).to(device)

    action_conditioning = action_encoder(a_plan)
    proprio_conditioning = None if proprio_encoder is None else proprio_encoder(q_last)

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

    timestep_t = torch.rand(z_future.shape[0], device=device)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        out = model(
            noisy_future_chunk=z_future,
            past_clean_chunks=z_past,
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

    print(f"Latent tokens: total={total_latent_steps}, context={t_ctx}, horizon={t_hor}")
    print(f"Tensor shapes: z_past={tuple(z_past.shape)} z_future={tuple(z_future.shape)} out={tuple(out.shape)}")
    print(f"CUDA peak memory: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
    print("PASS: forward pass completed without OOM")


if __name__ == "__main__":
    main()
