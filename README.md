# world-model-v1

Latent-space world model for LIBERO data using a frozen Wan VAE, chunkwise
teacher-forced flow matching, and AdaLN-conditioned DiT blocks.

## Architecture Snapshot

- Visual backbone: Wan VAE encoder/decoder (frozen).
- Prediction backbone: Wan DiT-style wrapper in latent token space.
- Conditioning: null conditioning for the current future-observation stage, with action conditioning kept available for a later stage.
- Temporal logic: all masking/splitting/chunking is defined in latent time.

## Canonical Layout

- `src/world_model/training/`: flow matching + train-step/checkpoint utilities.
- `src/world_model/data/`: temporal helpers, packing, batch preparation, loaders.
- `src/world_model/models/`: model wrappers.
- `src/world_model/latents/`: VAE interface.
- `scripts/train/`: canonical train/infer entrypoints.
- `scripts/check/`: canonical diagnostics and smoke checks.
- `configs/train/`, `configs/eval/`: YAML config defaults.


## Canonical Commands

Activate environment:

```bash
source .venv/bin/activate
```

Run tests:

```bash
pytest -q
```

Train:

```bash
python scripts/train/world_model.py --config configs/train/world_model.yaml
```

Training optimization controller:

```bash
python scripts/train/training_optimizer.py --train-config configs/train/aloha_fork_pick_up.yaml --memory-path docs/training_optimizer.md
```

Local overfit smoke run:

```bash
python scripts/train/world_model.py \
  --config configs/train/droid_local_smoke.yaml \
  --video-path runs/check_droid_preview_start25/preview.mp4
```

Infer:

```bash
python scripts/train/infer_world_model.py --config configs/eval/infer_world_model.yaml --checkpoint <path>
```

Forward smoke check:

```bash
python scripts/check/forward_real_batch.py --help
```

## Mixed Precision

`scripts/train/world_model.py` now chooses the runtime dtype automatically:

- CUDA + bf16 support: `torch.bfloat16`
- CUDA without bf16 support: `torch.float16`
- CPU or `--disable-amp`: `torch.float32`

On the current development machine, PyTorch reports `bf16_supported=True` for the `NVIDIA GeForce RTX 3080 Laptop GPU`, so training logs should show:

```text
Training dtype: torch.bfloat16
```

This reduces activation memory, but it does not make full 1.3B fine-tuning cheap. On a 16 GB GPU, the canonical `480x832` DROID config can still OOM in backward even with bf16 and gradient checkpointing. Use the local smoke config first, then scale up.

The local smoke config also switches `trainable_backbone: head`, which trains only the VACE control patch embedder plus the output head instead of the full 1.3B backbone. That is a plumbing/overfit mode for a 16 GB GPU. The broader `trainable_backbone: vace` option is still available when you want to train the full VACE-side control stack on a larger GPU budget.

## Latent-Time Schedule

Training chunking is defined in latent time, not raw frame time. After the Wan VAE temporally compresses the clip, the future window must still have at least `k + 1` latent steps for K+1 chunking.

Example:

- `k=1` means 2 future chunks are required.
- If `horizon_len=4` raw frames compresses to only 1 latent future step, training is invalid and the script now fails fast with a clear error.

The training script now prints the prepared latent window:

```text
Latent window: context=<...> future=<...> total=<...>
```
