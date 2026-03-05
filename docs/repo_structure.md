# Repo Structure Ownership

**Documentation Rule**: Every time a code change is made, `docs/repo_structure.md` and `docs/architecture.md` must be checked and updated if necessary to reflect the changes.

## Configuration

- `configs/train/world_model.yaml`
Canonical training preset. This file is the default runtime source for train
settings when `scripts/train/world_model.py` is invoked without `--config`.

- `configs/eval/infer_world_model.yaml`
Canonical inference preset. This file is the default runtime source for eval
settings when `scripts/train/infer_world_model.py` is invoked without
`--config`.

## Source Packages

- `src/world_model/config.py`
Defines typed `dataclass` schemas for training (`TrainScriptConfig`) and
inference (`InferScriptConfig`), along with YAML loading, key validation, and
CLI override helpers. Runtime defaults now come from the canonical YAML presets
under `configs/`, while this module owns the Python-side config contract,
including inference-only prompt-conditioning and single-chunk rollout toggles.

- `src/world_model/data/`
  - `dataset.py`: Gets raw frames from disk.
  - `temporal.py`: Expands frame-rate signals into latent-time sequences and
    computes latent-time splits.
  - `schema.py`: Defines the canonical prepared batch for the Wan VACE path,
    including structured latent videos, aligned actions, and latent metadata.
  - `prepare.py`: Runs the VAE, computes latent-time splits, aligns actions,
    and prepares `z_past_video`/`z_future_video` for the runtime path.

- `src/world_model/latents/vae.py`
Owns Wan VAE encode/decode interfaces and range/layout normalization.

- `src/world_model/chunking/schedule.py`
Owns K+1 latent-time chunk schedules and chunk-id generation.

- `src/world_model/masking/block_causal.py`
Owns block-causal attention mask construction.

- `src/world_model/models/`
  - `wan_vace_world_model.py`: Adapts repo chunkwise tensors to the vendored
    Wan VACE backbone, including control-tensor assembly and latent-frame to
    patch-token mask expansion.
  - `wan_vace_conditioning.py`: Builds Wan cross-attention action tokens and
    VACE control tensors.
  - `wan_vace_factory.py`: Builds runtime Wan VACE modules from config and
    optionally overlays local fine-tune checkpoints.

- `src/world_model/vendor/wan/`
  - `transformer_wan.py`: Vendors the base Wan transformer components shared by
    both the plain Wan and VACE paths, including attention processors,
    attention layers, rotary position embeddings, timestep/text embedding
    modules, transformer blocks, and the base 3D video transformer.
  - `transformer_wan_vace.py`: Vendors the Wan VACE-specific transformer stack,
    including VACE control blocks and the `WanVACETransformer3DModel` that
    combines patch embedding, time/text conditioning, control-hint injection,
    and output projection for video-latent denoising.

- `src/world_model/training/`
  - `flow_matching.py`: Owns straight-line flow matching and the chunkwise
    teacher-forced loss for the structured latent-video Wan VACE path.
  - `chunkwise_training.py`: Orchestrates a single optimizer step, including
    gradient clipping across both the Wan VACE backbone and the action-token
    encoder, and handles metrics/checkpoints.

- `src/world_model/eval/`
Owns inference-time chunkwise sampling for structured latent videos via
`infer_future_videos_chunkwise`, including flow-matching scheduler stepping,
optional classifier-free guidance, and a single-chunk rollout mode for
upstream-style smoke tests.

## Scripts

- `scripts/train/world_model.py`
Canonical Wan VACE training entrypoint. Builds the pretrained or config-shaped
VACE backbone, action-token encoder, and chunkwise flow-matching loop.

- `scripts/train/infer_world_model.py`
Canonical Wan VACE inference and visualization entrypoint. Loads the pretrained
VACE backbone, optionally overlays a local fine-tune checkpoint, supports
either action-token conditioning or upstream-style prompt conditioning for
smoke tests, runs chunkwise latent-video rollout with optional single-chunk
mode, keeps prompt encoding on CPU to reduce GPU memory, respects Hugging Face
offline-cache env settings for local-only loading, uses reduced-precision CUDA
inference when AMP is enabled, and saves a comparison grid.

- `scripts/check/`
Canonical diagnostics (`dataset`, `forward_real_batch`, `latent_cache`,
`masking_leakage`, `vae_roundtrip`, `latents_summary`).
