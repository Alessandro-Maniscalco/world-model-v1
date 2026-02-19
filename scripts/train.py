# scripts/train.py
#
# Tiny overfit test
# Purpose: validate the full pipeline end to end on a tiny subset.
# If you cannot overfit a single batch or a few hundred timesteps, something is wrong:
# time alignment, preprocessing, shapes, normalization, caching, masking, or the model wiring.
#
# What this script trains
# A deliberately small "world model" that predicts future VAE latents from:
#   past latents, past actions, and optionally proprio
#
# It uses the Wan video VAE only to encode frames into latents (frozen).
# It does NOT implement diffusion or flow matching yet.
# This is intentional: the goal is to validate data and wiring, not final objective.

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from diffusers import AutoencoderKLWan
from world_model.config import TrainConfig
from world_model.data import (
    flatten_latents_per_timestep,
    load_lerobot_dataset,
    pack_world_model_batch,
)
from world_model.latents import encode_window_to_latents
from world_model.models import TinyLatentWorldModel
from world_model.training import mse_prediction_loss


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_first(batch):
    # LeRobotDataset returns dict-like samples. We will keep it simple.
    # DataLoader with batch_size>1 will return a list of dicts. This collate stacks tensors.
    out = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_lerobot_dataset(cfg, video_backend="pyav")

    # Build a tiny subset: first N indices
    subset_indices = cfg.batch_size
    subset = list(range(subset_indices))
    ds_small = torch.utils.data.Subset(ds, subset)

    loader = DataLoader(
        ds_small,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_first,
        drop_last=False,
    )

    # Load Wan video VAE
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Peek one batch to infer dims and latent layout
    batch0 = next(iter(loader))

    # Video window
    video_window = batch0[cfg.video_key]  # [B, T, ...]
    # Actions and proprio
    # In LIBERO v3, these keys exist:
    # action: shape [B, A]
    # observation.state: shape [B, Q]
    # When delta_timestamps are used for images, action and state remain at current time index by default.
    # For this overfit test, we condition on the current action repeated, or you can expand later.
    #
    # To keep it robust, we handle either [B, A] or [B, T, A] if you later window actions too.
    action = batch0["action"]
    proprio = batch0.get("observation.state", None)

    # Encode the video window to latents
    latents0 = encode_window_to_latents(vae, video_window, device)
    z_tokens0 = flatten_latents_per_timestep(latents0)

    # Split into context and horizon in latent time
    # Important: Wan VAE may have temporal compression.
    # For a pipeline overfit test, we will split in latent time after encoding.
    # This means your "context_len" and "horizon_len" are in input frames,
    # but the actual latent timesteps may be fewer.
    t_lat = z_tokens0.shape[1]
    # Use a simple split: first half as context, second half as horizon, in latent time.
    # This avoids hard failure if temporal compression changes T.
    t_ctx = max(1, t_lat // 2)
    t_hor = t_lat - t_ctx
    if t_hor < 1:
        t_ctx = t_lat - 1
        t_hor = 1

    z_dim = z_tokens0.shape[2]
    print("Latent tokens per timestep z_dim:", z_dim)
    print("Latent timesteps total:", t_lat, "context:", t_ctx, "horizon:", t_hor)

    # Build conditioning tensors for the model
    # For this minimal test, we create an action sequence of length t_ctx.
    # If action is [B, A], repeat it.
    if action.ndim == 2:
        a_dim = action.shape[1]
        a_past0 = action.unsqueeze(1).repeat(1, t_ctx, 1)
    elif action.ndim == 3:
        a_dim = action.shape[2]
        a_past0 = action[:, :t_ctx]
    else:
        raise ValueError(f"Unexpected action shape: {tuple(action.shape)}")

    if proprio is None:
        q_dim = 0
        q_last0 = None
        use_proprio = False
    else:
        if proprio.ndim == 2:
            q_dim = proprio.shape[1]
            q_last0 = proprio
        elif proprio.ndim == 3:
            q_dim = proprio.shape[2]
            q_last0 = proprio[:, t_ctx - 1]
        else:
            raise ValueError(f"Unexpected proprio shape: {tuple(proprio.shape)}")
        use_proprio = cfg.use_proprio

    model = TinyLatentWorldModel(
        z_dim=z_dim,
        a_dim=a_dim,
        q_dim=q_dim,
        context_len=t_ctx,
        horizon_len=t_hor,
        hidden=cfg.hidden,
        use_proprio=use_proprio,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    print("Starting tiny overfit training")
    model.train()

    step = 0
    pbar = tqdm(total=cfg.num_steps)
    best = float("inf")

    while step < cfg.num_steps:
        for batch in loader:
            video_window = batch[cfg.video_key]
            action = batch["action"]
            proprio = batch.get("observation.state", None)

            latents = encode_window_to_latents(vae, video_window, device)
            z_tokens = flatten_latents_per_timestep(latents)  # [B, T_lat, z_dim]

            t_lat = z_tokens.shape[1]
            t_ctx = max(1, t_lat // 2)
            t_hor = t_lat - t_ctx
            if t_hor < 1:
                t_ctx = t_lat - 1
                t_hor = 1

            if action.ndim == 2:
                action_seq = action.to(device).unsqueeze(1).repeat(1, t_ctx + t_hor, 1)
            else:
                action_seq = action[:, :t_ctx + t_hor].to(device)

            if cfg.use_proprio and proprio is not None and proprio.ndim == 2:
                proprio_seq = proprio.to(device).unsqueeze(1).repeat(1, t_ctx + t_hor, 1)
            elif cfg.use_proprio and proprio is not None:
                proprio_seq = proprio[:, :t_ctx + t_hor].to(device)
            else:
                proprio_seq = None

            packed = pack_world_model_batch(
                z_tokens=z_tokens[:, :t_ctx + t_hor].to(device),
                actions=action_seq,
                proprio=proprio_seq,
                context_len=t_ctx,
                horizon_len=t_hor,
            )

            pred = model(packed.z_past, packed.a_past, packed.q_last)
            loss = mse_prediction_loss(pred, packed.z_future.to(device))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            pbar.update(1)

            val = float(loss.detach().cpu())
            if val < best:
                best = val
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "best_loss": best,
                    },
                    out_dir / "best.pt",
                )

            if step % 25 == 0:
                pbar.set_postfix(loss=val, best=best)

            if step >= cfg.num_steps:
                break

    pbar.close()
    print("Done. Best loss:", best)
    print("Saved checkpoint:", out_dir / "best.pt")


if __name__ == "__main__":
    main()
