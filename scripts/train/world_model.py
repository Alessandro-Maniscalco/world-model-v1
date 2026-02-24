"""Train the world model with chunkwise teacher-forced flow matching.

This entrypoint uses typed YAML-backed config plus CLI overrides and the
canonical shared data-preparation pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torch import nn

# Ensure local `src/` package imports work when run as `python scripts/train/world_model.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    # Prevent this file (`world_model.py`) from shadowing the `world_model` package.
    sys.path.remove(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
loaded_world_model = sys.modules.get("world_model")
if loaded_world_model is not None and not hasattr(loaded_world_model, "__path__"):
    # Drop incorrectly loaded module objects so package import can succeed.
    sys.modules.pop("world_model", None)

from world_model.conditioning import ActionEncoder, ProprioEncoder
from world_model.config import TrainScriptConfig, apply_namespace_overrides, load_train_config
from world_model.data import build_lerobot_dataloader, prepare_packed_batch
from world_model.latents import WanVAE
from world_model.models import WanDiTWrapper
from world_model.training import (
    append_jsonl,
    chunkwise_teacher_forcing_loss,
    save_checkpoint,
    train_chunkwise_batch,
)


def _config_parser() -> argparse.ArgumentParser:
    """Create parser for config-file bootstrap args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    return parser


def _build_parser(defaults: TrainScriptConfig) -> argparse.ArgumentParser:
    """Create full CLI parser using dataclass defaults."""
    parser = argparse.ArgumentParser(description=__doc__, parents=[_config_parser()])
    parser.add_argument("--repo-id", default=defaults.repo_id)
    parser.add_argument("--video-key", default=defaults.video_key)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--context-len", type=int, default=defaults.context_len, help="frame-time context length (l)")
    parser.add_argument("--horizon-len", type=int, default=defaults.horizon_len, help="frame-time horizon length (H)")
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--k", type=int, default=defaults.k, help="K in K+1 chunk schedule")
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=defaults.grad_clip_norm)
    parser.add_argument("--weight-mode", choices=["uniform", "snr", "clipped_snr"], default=defaults.weight_mode)
    parser.add_argument("--t-min", type=float, default=defaults.t_min)
    parser.add_argument("--t-max", type=float, default=defaults.t_max)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--disable-proprio", action="store_true", default=defaults.disable_proprio)
    parser.add_argument("--enable-proprio", dest="disable_proprio", action="store_false")
    parser.add_argument("--disable-amp", action="store_true", default=defaults.disable_amp)
    parser.add_argument("--enable-amp", dest="disable_amp", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=defaults.gradient_checkpointing)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--subset-size", type=int, default=defaults.subset_size, help="0 uses full dataset")
    parser.add_argument("--overfit-one-batch", action="store_true", default=defaults.overfit_one_batch)
    parser.add_argument("--no-overfit-one-batch", dest="overfit_one_batch", action="store_false")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def _load_args() -> TrainScriptConfig:
    """Load YAML config and apply CLI overrides into a final train config."""
    config_args, _ = _config_parser().parse_known_args()
    defaults = load_train_config(config_args.config)
    parser = _build_parser(defaults)
    args = parser.parse_args()
    return apply_namespace_overrides(defaults, args)


def _set_seed(seed: int) -> None:
    """Set Python and torch RNG seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Compute one eval-mode chunkwise loss for overfit diagnostics."""
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
    """Run chunkwise world-model training."""
    cfg = _load_args()
    _set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Training config: steps={cfg.max_steps} batch={cfg.batch_size} "
        f"k={cfg.k} l={cfg.context_len} H={cfg.horizon_len}"
    )

    loader = build_lerobot_dataloader(
        repo_id=cfg.repo_id,
        video_key=cfg.video_key,
        context_len=cfg.context_len,
        horizon_len=cfg.horizon_len,
        dt=cfg.dt,
        batch_size=cfg.batch_size,
        subset_size=cfg.subset_size,
        shuffle=not cfg.overfit_one_batch,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    vae = WanVAE.from_pretrained(device=device, deterministic=True)
    data_iter = iter(loader)
    first_batch = next(data_iter)
    prepared = prepare_packed_batch(
        batch=first_batch,
        encoder=vae,
        device=device,
        video_key=cfg.video_key,
        context_len=cfg.context_len,
        horizon_len=cfg.horizon_len,
        proprio_mode="last",
    )

    latent_dim = prepared.z_future.shape[-1]
    action_dim = prepared.a_plan.shape[-1]
    model = WanDiTWrapper(
        hidden_dim=cfg.hidden_dim,
        latent_dim=latent_dim,
        cond_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        mixed_precision=not cfg.disable_amp,
        gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    action_encoder = ActionEncoder(
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        pool="mean",
    ).to(device)

    proprio_encoder = None
    if not cfg.disable_proprio and prepared.q_last is not None:
        proprio_encoder = ProprioEncoder(
            proprio_dim=prepared.q_last.shape[-1],
            hidden_dim=cfg.hidden_dim,
            enabled=True,
        ).to(device)

    parameter_groups = list(model.parameters()) + list(action_encoder.parameters())
    if proprio_encoder is not None:
        parameter_groups.extend(proprio_encoder.parameters())
    optimizer = torch.optim.AdamW(parameter_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

    cached_batch = first_batch if cfg.overfit_one_batch else None

    overfit_start_loss = None
    if cfg.overfit_one_batch:
        overfit_start_loss = _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            z_past=prepared.z_past,
            z_future=prepared.z_future,
            a_plan=prepared.a_plan,
            q_last=prepared.q_last,
            k=cfg.k,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
        )
        print(f"Overfit baseline loss: {overfit_start_loss:.6f}")

    for step in itertools.count(start=1):
        if step > cfg.max_steps:
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

        prepared = prepare_packed_batch(
            batch=batch,
            encoder=vae,
            device=device,
            video_key=cfg.video_key,
            context_len=cfg.context_len,
            horizon_len=cfg.horizon_len,
            proprio_mode="last",
        )

        metrics = train_chunkwise_batch(
            model=model,
            action_encoder=action_encoder,
            optimizer=optimizer,
            z_past=prepared.z_past,
            z_future=prepared.z_future,
            a_plan=prepared.a_plan,
            q_last=prepared.q_last,
            proprio_encoder=proprio_encoder,
            k=cfg.k,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
            grad_clip_norm=cfg.grad_clip_norm,
        )

        step_time_s = time.time() - started
        log_payload = metrics.to_log_dict(step=step)
        log_payload["lr"] = float(optimizer.param_groups[0]["lr"])
        log_payload["step_time_s"] = float(step_time_s)
        append_jsonl(metrics_path, log_payload)

        if step % cfg.log_every == 0 or step == 1:
            print(
                f"step={step:06d} loss={metrics.loss:.6f} grad={metrics.grad_norm:.4f} "
                f"time={step_time_s:.3f}s chunks={metrics.per_chunk_losses}"
            )

        if step % cfg.checkpoint_every == 0:
            path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                model=model,
                action_encoder=action_encoder,
                optimizer=optimizer,
                proprio_encoder=proprio_encoder,
                extra_state={"config": asdict(cfg)},
            )
            print(f"checkpoint={path}")

    final_ckpt = save_checkpoint(
        output_dir=output_dir,
        step=cfg.max_steps,
        model=model,
        action_encoder=action_encoder,
        optimizer=optimizer,
        proprio_encoder=proprio_encoder,
        extra_state={"config": asdict(cfg)},
    )
    print(f"final_checkpoint={final_ckpt}")

    if cfg.overfit_one_batch:
        overfit_end_loss = _evaluate_loss(
            model=model,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            z_past=prepared.z_past,
            z_future=prepared.z_future,
            a_plan=prepared.a_plan,
            q_last=prepared.q_last,
            k=cfg.k,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            weight_mode=cfg.weight_mode,
        )
        assert overfit_start_loss is not None
        print(
            f"overfit_loss_start={overfit_start_loss:.6f} "
            f"overfit_loss_end={overfit_end_loss:.6f}"
        )


if __name__ == "__main__":
    main()
