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

import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from diffusers import AutoencoderKLWan
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class TrainConfig:
    repo_id: str = "lerobot/libero"
    video_key: str = "observation.images.image"

    use_proprio: bool = True

    # Windowing at 10 Hz for LIBERO
    context_len: int = 8   # l
    horizon_len: int = 8   # H
    dt: float = 0.1

    # Overfit subset
    subset_indices: int = 8          # number of samples in tiny dataset
    batch_size: int = 2
    num_steps: int = 300
    lr: float = 2e-3

    # Model size
    hidden: int = 2048

    # Output
    out_dir: str = "runs/overfit_test"
    seed: int = 0


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        if float(x.max().cpu()) > 1.5:
            x = x / 255.0
    return x * 2.0 - 1.0


def to_bcthw(video: torch.Tensor) -> torch.Tensor:
    # Accept T,C,H,W or T,H,W,C and return B,C,T,H,W
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got {video.ndim}D {tuple(video.shape)}")
    if video.shape[1] == 3:
        tchw = video
    elif video.shape[-1] == 3:
        tchw = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(video.shape)}")
    return tchw.permute(1, 0, 2, 3).unsqueeze(0)


class TinyLatentWorldModel(nn.Module):
    """
    A minimal predictor to overfit quickly.
    It predicts future latents z_future from flattened conditioning:
    z_past, actions_past, and optionally proprio_last.

    This is not your final DiT or diffusion model.
    It is a pipeline validator.
    """

    def __init__(
        self,
        z_dim: int,
        a_dim: int,
        q_dim: int,
        context_len: int,
        horizon_len: int,
        hidden: int,
        use_proprio: bool,
    ):
        super().__init__()
        self.use_proprio = use_proprio
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.z_dim = z_dim

        cond_dim = (context_len * z_dim) + (context_len * a_dim)
        if use_proprio:
            cond_dim += q_dim

        out_dim = horizon_len * z_dim

        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_past_flat: torch.Tensor, a_past: torch.Tensor, q_last: torch.Tensor | None):
        # z_past_flat: [B, context_len*z_dim]
        # a_past: [B, context_len, a_dim]
        b = z_past_flat.shape[0]
        a_flat = a_past.reshape(b, -1)

        if self.use_proprio:
            if q_last is None:
                raise ValueError("use_proprio=True but q_last is None")
            x = torch.cat([z_past_flat, a_flat, q_last], dim=1)
        else:
            x = torch.cat([z_past_flat, a_flat], dim=1)

        y = self.net(x)
        return y.reshape(b, self.horizon_len, self.z_dim)


def build_deltas(context_len: int, horizon_len: int, dt: float):
    # We want a contiguous window of length context_len + horizon_len ending at t=0
    # Example with total_len=16: deltas = [-1.5, ..., -0.1, 0.0]
    total_len = context_len + horizon_len
    deltas = [-(total_len - 1 - i) * dt for i in range(total_len)]
    return deltas


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


@torch.no_grad()
def encode_window_to_latents(vae: AutoencoderKLWan, video_window_tchw: torch.Tensor, device: torch.device):
    """
    video_window_tchw: [B, T, C, H, W] or [B, T, H, W, C] is NOT accepted.
    We expect the DataLoader output for video key as [B, T, C, H, W] or [B, T, H, W, C].
    We convert to [B, C, T, H, W], normalize to [-1,1], then encode.

    Returns latents as [B, C_lat, T_lat, H_lat, W_lat] flattened per timestep later.
    """
    if video_window_tchw.ndim != 5:
        raise ValueError(f"Expected 5D batched video, got {video_window_tchw.ndim}D {tuple(video_window_tchw.shape)}")

    # Detect whether input is [B,T,C,H,W] or [B,T,H,W,C]
    if video_window_tchw.shape[2] == 3:
        btchw = video_window_tchw
        bcthw = btchw.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
    elif video_window_tchw.shape[-1] == 3:
        bthwc = video_window_tchw
        btchw = bthwc.permute(0, 1, 4, 2, 3)  # B,T,C,H,W
        bcthw = btchw.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Unrecognized batched video shape: {tuple(video_window_tchw.shape)}")

    bcthw = normalize_to_minus1_1(bcthw).to(device)

    enc = vae.encode(bcthw)
    latents = enc.latent_dist.mean  # deterministic for training stability
    return latents


def flatten_latents_per_timestep(latents: torch.Tensor):
    """
    latents: [B, C_lat, T_lat, H_lat, W_lat]
    Return z_tokens: [B, T_lat, z_dim] where z_dim = C_lat*H_lat*W_lat
    """
    b, c, t, h, w = latents.shape
    z = latents.permute(0, 2, 1, 3, 4).contiguous()  # B,T,C,H,W
    z = z.reshape(b, t, c * h * w)
    return z


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset window
    deltas = build_deltas(cfg.context_len, cfg.horizon_len, cfg.dt)

    ds = LeRobotDataset(
        cfg.repo_id,
        delta_timestamps={cfg.video_key: deltas},
        video_backend="pyav",
    )

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
    loss_fn = nn.MSELoss()

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

            z_past = z_tokens[:, :t_ctx]                  # [B, t_ctx, z_dim]
            z_future = z_tokens[:, t_ctx:t_ctx + t_hor]   # [B, t_hor, z_dim]

            if action.ndim == 2:
                a_past = action.to(device).unsqueeze(1).repeat(1, t_ctx, 1)
            else:
                a_past = action[:, :t_ctx].to(device)

            if cfg.use_proprio and proprio is not None:
                if proprio.ndim == 2:
                    q_last = proprio.to(device)
                else:
                    q_last = proprio[:, t_ctx - 1].to(device)
            else:
                q_last = None

            z_past_flat = z_past.reshape(z_past.shape[0], -1).to(device)
            z_future = z_future.to(device)

            pred = model(z_past_flat, a_past, q_last)
            loss = loss_fn(pred, z_future)

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
