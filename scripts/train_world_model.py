"""Train the world model with chunkwise teacher-forced flow matching.

Runs real-batch optimization, emits JSONL logs, and saves periodic checkpoints.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from world_model.chunking import build_k_plus_one_schedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
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
from world_model.train import append_jsonl, save_checkpoint, train_chunkwise_batch
from world_model.training import chunkwise_teacher_forcing_loss, make_noisy_and_target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/libero")
    parser.add_argument("--video-key", default="observation.images.image")
    parser.add_argument("--output-dir", default="runs/world_model_train")
    parser.add_argument("--context-len", type=int, default=10, help="frame-time context length (l)")
    parser.add_argument("--horizon-len", type=int, default=8, help="frame-time horizon length (H)")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--k", type=int, default=1, help="K in K+1 chunk schedule")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-mode", choices=["uniform", "snr", "clipped_snr"], default="uniform")
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--disable-proprio", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--subset-size", type=int, default=0, help="0 uses full dataset")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--overfit-eval-t", type=float, default=0.5)
    parser.add_argument("--save-overfit-artifacts", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        first = batch[0][key]
        if torch.is_tensor(first):
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)
        else:
            out[key] = [sample[key] for sample in batch]
    return out


def _prepare_packed_batch(
    *,
    batch: dict[str, Any],
    vae: WanVAE,
    device: torch.device,
    video_key: str,
    context_len: int,
    horizon_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int, int]]:
    video = batch[video_key].to(device)
    action = batch["action"].to(device)
    proprio = batch.get("observation.state")
    if proprio is not None:
        proprio = proprio.to(device)

    latents = vae.encode(video)
    z_tokens = flatten_latents_per_timestep(latents)

    total_latent_steps = z_tokens.shape[1]
    t_ctx, t_hor = latent_split_from_frame_ratio(
        total_latent_steps=total_latent_steps,
        context_frames=context_len,
        horizon_frames=horizon_len,
    )

    action_seq = expand_to_latent_steps(action, target_steps=t_ctx + t_hor)
    proprio_seq = None if proprio is None else expand_to_latent_steps(proprio, target_steps=t_ctx + t_hor)

    packed = pack_world_model_batch(
        z_tokens=z_tokens,
        actions=action_seq,
        proprio=proprio_seq,
        context_len=t_ctx,
        horizon_len=t_hor,
    )
    latent_shape = (latents.shape[1], latents.shape[3], latents.shape[4])
    return packed.z_past, packed.z_future, packed.a_plan, packed.q_last, latent_shape


@torch.no_grad()
def _predict_clean_future_tokens(
    *,
    model: nn.Module,
    z_past: torch.Tensor,
    z_future: torch.Tensor,
    action_conditioning: torch.Tensor,
    proprio_conditioning: torch.Tensor | None,
    k: int,
    eval_t: float,
) -> torch.Tensor:
    schedule = build_k_plus_one_schedule(future_steps=z_future.shape[1], k=k, device=z_future.device)
    pred_clean = torch.zeros_like(z_future)
    timestep = torch.full((z_future.shape[0],), fill_value=eval_t, device=z_future.device, dtype=z_future.dtype)

    for start, end in schedule.boundaries:
        chunk_len = end - start
        clean_chunk = z_future[:, start:end, :]
        noisy_chunk, _ = make_noisy_and_target(clean_chunk, timestep)

        noisy_suffix = z_future[:, start:, :].clone()
        noisy_suffix[:, :chunk_len, :] = noisy_chunk
        teacher_forced_context = torch.cat([z_past, z_future[:, :start, :]], dim=1)

        suffix_chunk_ids = schedule.chunk_ids[start:]
        full_chunk_ids = torch.cat(
            [
                torch.full(
                    (teacher_forced_context.shape[1],),
                    -1,
                    device=z_future.device,
                    dtype=torch.long,
                ),
                suffix_chunk_ids,
            ],
            dim=0,
        )
        mask = build_block_causal_mask(full_chunk_ids, mask_format="additive")

        pred_suffix = model(
            noisy_future_chunk=noisy_suffix,
            past_clean_chunks=teacher_forced_context,
            action_conditioning=action_conditioning,
            timestep_t=timestep,
            block_causal_attention_mask=mask,
            proprio_conditioning=proprio_conditioning,
        )
        pred_chunk = pred_suffix[:, :chunk_len, :]
        pred_clean_chunk = noisy_chunk + (1.0 - eval_t) * pred_chunk
        pred_clean[:, start:end, :] = pred_clean_chunk

    return pred_clean


def _tokens_to_latents(
    tokens: torch.Tensor,
    *,
    latent_shape: tuple[int, int, int],
) -> torch.Tensor:
    c_lat, h_lat, w_lat = latent_shape
    b, t, z = tokens.shape
    expected_z = c_lat * h_lat * w_lat
    if z != expected_z:
        raise ValueError(f"Token feature dim {z} does not match latent shape product {expected_z}")
    return tokens.reshape(b, t, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def _save_overfit_artifacts(
    *,
    vae: WanVAE,
    pred_clean_tokens: torch.Tensor,
    target_clean_tokens: torch.Tensor,
    latent_shape: tuple[int, int, int],
    output_dir: Path,
) -> None:
    pred_latents = _tokens_to_latents(pred_clean_tokens, latent_shape=latent_shape)
    target_latents = _tokens_to_latents(target_clean_tokens, latent_shape=latent_shape)
    pred_video = vae.decode(pred_latents, output_layout="BTCHW", output_range="zero_to_one")
    target_video = vae.decode(target_latents, output_layout="BTCHW", output_range="zero_to_one")

    pred_frames = pred_video[0].detach().cpu()
    target_frames = target_video[0].detach().cpu()

    try:
        from PIL import Image
    except ImportError:
        torch.save(
            {
                "pred_video": pred_video.detach().cpu(),
                "target_video": target_video.detach().cpu(),
            },
            output_dir / "overfit_preview.pt",
        )
        return

    preview_dir = output_dir / "overfit_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    max_frames = min(4, pred_frames.shape[0])
    for idx in range(max_frames):
        pred_img = (pred_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        tgt_img = (target_frames[idx].clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        side = torch.from_numpy(pred_img), torch.from_numpy(tgt_img)
        stacked = torch.cat(side, dim=1).numpy()
        Image.fromarray(stacked).save(preview_dir / f"frame_{idx:03d}.png")


@torch.no_grad()
def _evaluate_loss(
    *,
    model: nn.Module,
    action_encoder: nn.Module,
    proprio_encoder: nn.Module | None,
    z_past: torch.Tensor,
    z_future: torch.Tensor,
    a_plan: torch.Tensor,
    q_last: torch.Tensor | None,
    k: int,
    t_min: float,
    t_max: float,
    weight_mode: str,
) -> float:
    model.eval()
    action_encoder.eval()
    if proprio_encoder is not None:
        proprio_encoder.eval()

    action_conditioning = action_encoder(a_plan)
    proprio_conditioning = None if proprio_encoder is None else proprio_encoder(q_last)
    loss = chunkwise_teacher_forcing_loss(
        model,
        z_past=z_past,
        z_future=z_future,
        action_conditioning=action_conditioning,
        proprio_conditioning=proprio_conditioning,
        k=k,
        t_min=t_min,
        t_max=t_max,
        weight_mode=weight_mode,
    )
    return float(loss.detach().cpu().item())


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Training config: steps={args.max_steps} batch={args.batch_size} "
        f"k={args.k} l={args.context_len} H={args.horizon_len}"
    )

    deltas = build_frame_deltas(args.context_len, args.horizon_len, args.dt)
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps={args.video_key: deltas},
        video_backend="pyav",
    )
    if args.subset_size > 0:
        dataset = Subset(dataset, range(args.subset_size))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.overfit_one_batch,
        num_workers=args.num_workers,
        collate_fn=_collate_dict,
        drop_last=True,
    )

    vae = WanVAE.from_pretrained(device=device, deterministic=True)
    data_iter = iter(loader)
    first_batch = next(data_iter)
    z_past, z_future, a_plan, q_last, latent_shape = _prepare_packed_batch(
        batch=first_batch,
        vae=vae,
        device=device,
        video_key=args.video_key,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
    )

    latent_dim = z_future.shape[-1]
    action_dim = a_plan.shape[-1]
    model = WanDiTWrapper(
        hidden_dim=args.hidden_dim,
        latent_dim=latent_dim,
        cond_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mixed_precision=not args.disable_amp,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    action_encoder = ActionEncoder(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        pool="mean",
    ).to(device)

    proprio_encoder = None
    if not args.disable_proprio and q_last is not None:
        proprio_encoder = ProprioEncoder(
            proprio_dim=q_last.shape[-1],
            hidden_dim=args.hidden_dim,
            enabled=True,
        ).to(device)

    parameter_groups = list(model.parameters()) + list(action_encoder.parameters())
    if proprio_encoder is not None:
        parameter_groups.extend(proprio_encoder.parameters())
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)

    if args.overfit_one_batch:
        cached_batch = first_batch
    else:
        cached_batch = None

    overfit_start_loss = None
    if args.overfit_one_batch:
        overfit_start_loss = _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            z_past=z_past,
            z_future=z_future,
            a_plan=a_plan,
            q_last=q_last,
            k=args.k,
            t_min=args.t_min,
            t_max=args.t_max,
            weight_mode=args.weight_mode,
        )
        print(f"Overfit baseline loss: {overfit_start_loss:.6f}")

    loop_iter = itertools.count(start=1)
    for step in loop_iter:
        if step > args.max_steps:
            break
        started = time.time()

        if cached_batch is None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
        else:
            batch = cached_batch

        z_past, z_future, a_plan, q_last, latent_shape = _prepare_packed_batch(
            batch=batch,
            vae=vae,
            device=device,
            video_key=args.video_key,
            context_len=args.context_len,
            horizon_len=args.horizon_len,
        )

        metrics = train_chunkwise_batch(
            model=model,
            action_encoder=action_encoder,
            optimizer=optimizer,
            z_past=z_past,
            z_future=z_future,
            a_plan=a_plan,
            q_last=q_last,
            proprio_encoder=proprio_encoder,
            k=args.k,
            t_min=args.t_min,
            t_max=args.t_max,
            weight_mode=args.weight_mode,
            grad_clip_norm=args.grad_clip_norm,
        )

        step_time_s = time.time() - started
        log_payload = metrics.to_log_dict(step=step)
        log_payload["lr"] = float(optimizer.param_groups[0]["lr"])
        log_payload["step_time_s"] = float(step_time_s)
        append_jsonl(metrics_path, log_payload)

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:06d} loss={metrics.loss:.6f} grad={metrics.grad_norm:.4f} "
                f"time={step_time_s:.3f}s chunks={metrics.per_chunk_losses}"
            )

        if step % args.checkpoint_every == 0:
            path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                action_encoder=action_encoder,
                optimizer=optimizer,
                proprio_encoder=proprio_encoder,
                extra_state={"args": vars(args)},
            )
            print(f"checkpoint={path}")

    final_ckpt = save_checkpoint(
        output_dir=output_dir,
        step=args.max_steps,
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        proprio_encoder=proprio_encoder,
        extra_state={"args": vars(args)},
    )
    print(f"final_checkpoint={final_ckpt}")

    if args.overfit_one_batch:
        overfit_end_loss = _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            z_past=z_past,
            z_future=z_future,
            a_plan=a_plan,
            q_last=q_last,
            k=args.k,
            t_min=args.t_min,
            t_max=args.t_max,
            weight_mode=args.weight_mode,
        )
        assert overfit_start_loss is not None
        print(
            f"overfit_loss_start={overfit_start_loss:.6f} "
            f"overfit_loss_end={overfit_end_loss:.6f}"
        )
        if args.save_overfit_artifacts:
            model.eval()
            action_encoder.eval()
            if proprio_encoder is not None:
                proprio_encoder.eval()
            action_conditioning = action_encoder(a_plan)
            proprio_conditioning = None if proprio_encoder is None else proprio_encoder(q_last)
            pred_clean_tokens = _predict_clean_future_tokens(
                model=model,
                z_past=z_past,
                z_future=z_future,
                action_conditioning=action_conditioning,
                proprio_conditioning=proprio_conditioning,
                k=args.k,
                eval_t=args.overfit_eval_t,
            )
            _save_overfit_artifacts(
                vae=vae,
                pred_clean_tokens=pred_clean_tokens,
                target_clean_tokens=z_future,
                latent_shape=latent_shape,
                output_dir=output_dir,
            )
            print(f"saved_overfit_artifacts={output_dir / 'overfit_preview'}")


if __name__ == "__main__":
    main()
