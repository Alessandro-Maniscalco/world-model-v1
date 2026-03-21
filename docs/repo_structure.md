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

- `configs/train/aloha_fork_pick_up_smoke.yaml`
Small dataset-backed ALOHA smoke preset for the Wan VACE path. This keeps the
low-memory head-only recipe for quick cache-backed sanity checks on a 16 GB
workstation GPU.

## Documentation

- `docs/training_optimizer.md`
Persistent experiment memory for the current training-optimization loop. The
controller appends `[controller ...]` findings to `Current Signal`, rewrites
the current controller recommendation under `Next Work`, and logs completed
stages under `Training runs`, including comparison-video paths plus manual
visual-review commands for the saved sweep artifacts. In Codex mode it also
stores the latest model-side planning summary under `Codex Analysis` and keeps
the audit trail for bounded controller or validated repo edits under
`Controller Edits`.

## Source Packages

- `src/world_model/config.py`
Defines typed `dataclass` schemas for training (`TrainScriptConfig`) and
inference (`InferScriptConfig`), along with YAML loading, key validation, and
CLI override helpers. Runtime defaults now come from the canonical YAML presets
under `configs/`, while this module owns the Python-side config contract,
including trainable-backbone policy, inference-only prompt-conditioning,
single-chunk rollout toggles, and ordered full-plan action-conditioning flags.

- `src/world_model/data/`
  - `dataset.py`: Gets raw frames from disk.
  - `temporal.py`: Validates Wan's exact `4n+1` raw-frame packing, expands
    frame-rate signals into latent-time sequences, and computes latent-time
    splits.
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
    Wan VACE backbone, including control-tensor assembly, action-derived future
    control bias injection, and latent-frame to patch-token mask expansion.
  - `wan_vace_conditioning.py`: Builds Wan cross-attention action tokens,
    learned action-order features, latent control priors, and VACE control
    tensors.
  - `wan_vace_factory.py`: Builds runtime Wan VACE modules from config,
    including the action encoder plus action-control projector, and optionally
    overlays local fine-tune checkpoints.

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
    teacher-forced loss for the structured latent-video Wan VACE path,
    including chunk-vs-full action token selection and aligned action-control
    priors.
  - `chunkwise_training.py`: Orchestrates a single optimizer step, including
    gradient clipping across the Wan VACE backbone, action-token encoder, and
    action-control projector, and handles metrics/checkpoints.

- `src/world_model/optimization/`
Owns the staged training-optimization controller helpers. This layer reads
`docs/training_optimizer.md`, chooses the next conservative experiment,
launches the canonical train/eval/check scripts, stores machine-readable state
under `runs/training_optimizer/`, and writes concise findings back into the
markdown memory file. The current implementation uses one persistent
ChatGPT-authenticated Codex session for short inspection/edit turns while
keeping long-running training and sweep commands outside the chat session.
  - `controller.py`: Owns the shared-session controller loop, including state
    persistence, snapshot-based rollback protection for in-session edits,
    bounded external command execution, and stop-summary reporting.
  - `codex_runner.py`: Wraps the local `codex` CLI for fail-closed
    ChatGPT-login checks plus structured `codex exec` calls with JSON schema
    enforcement and optional image inputs for artifact inspection.

- `src/world_model/eval/`
Owns inference-time chunkwise sampling for structured latent videos via
`infer_future_videos_chunkwise`, including flow-matching scheduler stepping,
optional classifier-free guidance, action-control prior slicing, and a
single-chunk rollout mode for upstream-style smoke tests.

## Scripts

- `scripts/train/world_model.py`
Canonical Wan VACE training entrypoint. Builds the pretrained or config-shaped
VACE backbone, configured conditioning encoder, action-control projector, and
chunkwise flow-matching loop. Also owns local-video overfit mode, latent-time
schedule validation, automatic bf16/fp16 selection, ordered full-plan action
conditioning toggles, and the `full`/`vace`/`head` trainable-backbone policies.

- `scripts/train/training_optimizer.py`
Canonical optimization-loop controller. Reuses the existing training,
validation, checkpoint sweep, and plausibility scripts; plans one staged
experiment at a time from `docs/training_optimizer.md`; persists structured
history to `runs/training_optimizer/controller_state.json`; and updates the
markdown memory with the latest finding plus next recommendation. The CLI
manages shared-session Codex options such as resume/fresh-session control,
turn timeout, dry-run mode, and external-command iteration limits.

- `scripts/train/infer_world_model.py`
Canonical Wan VACE inference and visualization entrypoint. Loads the pretrained
VACE backbone, optionally overlays a local fine-tune checkpoint, supports
either action-token conditioning or upstream-style prompt conditioning for
smoke tests, runs chunkwise latent-video rollout with optional single-chunk
mode, can switch action conditioning between chunk-local and ordered full-plan
reuse, keeps prompt encoding on CPU to reduce GPU memory, respects Hugging Face
offline-cache env settings for local-only loading, uses reduced-precision CUDA
inference when AMP is enabled, and saves a comparison grid.

- `scripts/check/`
Canonical diagnostics (`dataset`, `forward_real_batch`, `latent_cache`,
`masking_leakage`, `vae_roundtrip`, `latents_summary`) plus manual sweep tools
such as `sweep_infer_resolutions.py` and
`sweep_vae_roundtrip_resolutions.py`. The controller-driven evaluation path
centers on `sweep_local_repo_resolutions.py`,
`check_generated_video_plausibility.py`, and
`check_arm_motion_alignment.py`.
