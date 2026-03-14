# Training Notes

## Status

This document records the current practical training path for the Wan VACE
world-model repo as of March 6, 2026.

## Current Scope

- Predict future observations in Wan VAE latent space.
- Use null conditioning for the current stage.
- Keep action conditioning code available for a later stage.
- Treat raw-frame horizon as an input convenience, not a strict training-time
  contract.

## Main Findings

### 1. Train in latent space, not RGB frame space

The Wan VAE temporally compresses clips before the world model sees them. That
means the real training contract is latent time:

- `context_len` and `horizon_len` are specified in raw frames
- the model actually predicts over `context_latent_steps` and
  `horizon_latent_steps`
- valid Wan windows must satisfy:
  - `context_len = 4m + 1`
  - `horizon_len = 4h`
  - `total_frames = context_len + horizon_len = 4n + 1`
- for valid windows:
  - `context_latent_steps = 1 + (context_len - 1) / 4`
  - `horizon_latent_steps = horizon_len / 4`

For that reason, all chunking, masking, and teacher forcing should be reasoned
about in latent time.

### 2. Latent-time chunking can invalidate short horizons

K+1 chunking requires at least `k + 1` latent future steps. A short raw-frame
horizon can compress below that threshold.

Example:

- `k=1` requires 2 latent future chunks
- `horizon_len=4` raw frames can compress to only 1 latent future step
- that configuration is invalid for training

`scripts/train/world_model.py` now fails fast with an explicit message and
prints the prepared latent window:

```text
Latent window: context=<...> future=<...> total=<...>
```

### 3. bf16 is supported on the current GPU

PyTorch reports bf16 support on the current development machine:

- GPU: `NVIDIA GeForce RTX 3080 Laptop GPU`
- capability: `(8, 6)`
- `torch.cuda.is_bf16_supported() == True`

The training script now auto-selects:

- `torch.bfloat16` on CUDA with bf16 support
- `torch.float16` on CUDA without bf16 support
- `torch.float32` on CPU or when `--disable-amp` is set

This is logged as:

```text
Training dtype: torch.bfloat16
```

### 4. Full 1.3B fine-tuning still OOMs on 16 GB

Even with bf16, gradient checkpointing, cached latents, and smaller spatial
resolution, full fine-tuning of the pretrained Wan VACE 1.3B backbone still
OOMs on a 16 GB laptop GPU.

This was true for:

- `trainable_backbone=full`
- `trainable_backbone=vace`

The `vace` mode still exposed too many trainable parameters for Adam optimizer
state on this hardware.

### 5. A workable local overfit path now exists

The practical local-overfit mode on this machine is:

- `conditioning_mode=none`
- local clip training via `--video-path`
- `trainable_backbone=head`
- bf16 enabled
- small spatial resolution via `configs/train/droid_local_smoke.yaml`

`trainable_backbone=head` trains only:

- `vace_patch_embedding`
- `norm_out`
- `proj_out`
- `scale_shift_table`

This keeps the trainable parameter count low enough for a short overfit run on
the laptop GPU.

## Recommended Overfit Command

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train/world_model.py \
  --config configs/train/droid_local_smoke.yaml \
  --video-path runs/check_droid_preview_start25/preview.mp4
```

## ALOHA Fork-Pick-Up Dataset Path

The same 16 GB constraint still applies when moving from DROID to cached ALOHA
episodes on the `NVIDIA GeForce RTX 3080 Laptop GPU`. The practical ALOHA path
keeps the same conservative spatial settings but now uses a stronger training
structure for the full fork-pick-up dataset:

- `trainable_backbone=lora`
- `conditioning_mode=action`
- `gradient_checkpointing=true`
- small spatial resolution (`128x224`)
- batch size `1`
- all episodes once the full fork-pick-up dataset is cached locally

Design choice: keep `dt=0.1` for ALOHA even though the source data is 50 Hz.
That intentionally samples a 10 Hz-equivalent training window so the model sees
larger visible motion per step instead of nearly duplicate adjacent frames.

Design choice: the overnight ALOHA preset now targets the full dataset,
restores action conditioning, and uses LoRA adapters on the Wan backbone
instead of head-only tuning. This keeps the trainable footprint much lower than
full fine-tuning while letting the model adjust more than just the final output
layers.

Design choice: the overnight ALOHA preset still checks for continuation every
5000 steps. It compares the mean loss of the latest 5000-step block against the
prior 5000-step block and stops when the relative improvement falls below `5%`.
This makes the run long when the loss is still improving, but prevents wasting
the rest of the night on a flat curve.

Smoke-verify the dataset-backed path with:

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train/world_model.py \
  --config configs/train/aloha_fork_pick_up_smoke.yaml
```

For a one-command episode-0 fetch and training run, use:

```bash
source .venv/bin/activate
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train/run_aloha_episode0.py
```

Design choice: keep the episode-0 workflow separate from the full-dataset
overnight config. The helper above only fetches the backing files needed for
episode `0`, then launches the existing smoke recipe unchanged:

- `repo_id=lerobot/aloha_static_fork_pick_up`
- `episodes=[0]`
- `video_key=observation.images.cam_high`
- `128x224`
- `dt=0.1`
- `trainable_backbone=head`
- `conditioning_mode=none`

The first fetch must run without `HF_HUB_OFFLINE` and without
`TRANSFORMERS_OFFLINE`. In this environment, `TRANSFORMERS_OFFLINE=1` also puts
`huggingface_hub` into offline mode. After episode `0` is cached locally, rerun
the helper with offline mode if you want a deterministic cached launch:

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train/run_aloha_episode0.py
```

For an overnight run on the 3080, start with:

```bash
source .venv/bin/activate
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train/world_model.py \
  --config configs/train/aloha_fork_pick_up.yaml
```

This full-dataset preset requires the complete `lerobot/aloha_static_fork_pick_up`
cache and therefore cannot be bootstrapped with offline env vars enabled. It
now uses all episodes, `conditioning_mode=action`, and `trainable_backbone=lora`.
The earlier dataset-backed smoke verification still uses the episode-0-only
smoke config.

The saved ALOHA smoke config completed successfully with:

- `Training dtype: torch.bfloat16`
- `Latent window: context=2 future=2 total=4`
- `Trainable backbone mode: head (692800 params)`
- `overfit_loss_start=6.126202`
- `overfit_loss_end=4.798404`

This confirms that the cached ALOHA fork-pick-up episode-0 path is viable on
the local 16 GB 3080 in the same head-only recipe described above.

## Most Recent Verified Result

The saved local smoke config completed successfully with:

- `Training dtype: torch.bfloat16`
- `Latent window: context=2 future=2 total=4`
- `Trainable backbone mode: head (692800 params)`
- `overfit_loss_start=6.161936`
- `overfit_loss_end=4.004341`

This confirms:

- bf16 is active
- latent-time schedule validation passes
- the local 16 GB overfit path is viable

## Recommended Next Step

Use the `head` mode only for plumbing and overfit validation. After that:

1. decide whether the next target is `trainable_backbone=vace` or a PEFT/LoRA
   path
2. keep null conditioning until future-observation prediction is stable
3. add actions only after the latent-only/null-conditioned path is behaving
   predictably
