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

- `configs/train/droid_local_smoke.yaml`
Small local-overfit training preset for the exported DROID clip path. This
config exists specifically to validate the bf16/null-conditioning/latent-time
training path on a 16 GB workstation GPU.

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
including trainable-backbone policy, inference-only prompt-conditioning, and
single-chunk rollout toggles.

- `src/world_model/data/`
  - `dataset.py`: Gets raw frames from disk.
  - `droid_video.py`: Shared helpers for exporting reusable local preview clips
    from DROID/LeRobot episodes, including PNG frame dumps and MP4 preview
    creation.
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

- `src/world_model/optimization/`
Owns the staged training-optimization controller helpers. This layer reads
`docs/training_optimizer.md`, chooses the next conservative experiment,
launches the canonical train/eval/check scripts, stores machine-readable state
under `runs/training_optimizer/`, and writes concise findings back into the
markdown memory file. It now supports either the deterministic rule-based
planner or a local ChatGPT-authenticated Codex planner with budgets, lockfiles,
artifact inspection, and validated repo edits.
  - `controller.py`: Parses `Next Work` hints from the optimizer markdown into
    staged experiment plans, resumes or starts the selected branch, runs the
    train/sweep/plausibility checks for one stage, summarizes the result,
    records comparison-video review guidance, can either follow the
    deterministic recommendation path or run a Codex-authenticated
    decide/inspect/edit/run/stop loop, can rewrite its bounded controller-policy
    block when repeated outcomes suggest a better process, and writes the next
    conservative recommendation into both JSON state and markdown memory.
  - `codex_runner.py`: Wraps the local `codex` CLI for fail-closed
    ChatGPT-login checks plus structured `codex exec` calls with JSON schema
    enforcement and optional image inputs for artifact inspection.

- `src/world_model/eval/`
Owns inference-time chunkwise sampling for structured latent videos via
`infer_future_videos_chunkwise`, including flow-matching scheduler stepping,
optional classifier-free guidance, and a single-chunk rollout mode for
upstream-style smoke tests.

## Scripts

- `scripts/train/world_model.py`
Canonical Wan VACE training entrypoint. Builds the pretrained or config-shaped
VACE backbone, configured conditioning encoder, and chunkwise flow-matching
loop. Also owns local-video overfit mode, latent-time schedule validation,
automatic bf16/fp16 selection, and the `full`/`vace`/`head` trainable-backbone
policies.

- `scripts/train/training_optimizer.py`
Canonical optimization-loop controller. Reuses the existing training,
validation, checkpoint sweep, and plausibility scripts; plans one staged
experiment at a time from `docs/training_optimizer.md`; persists structured
history to `runs/training_optimizer/controller_state.json`; and updates the
markdown memory with the latest finding plus next recommendation. The CLI now
exposes `--planner codex|deterministic` and Codex-loop budget flags for real
runs, Codex calls, failures, edit cycles, and wall-clock limits.

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
`masking_leakage`, `vae_roundtrip`, `latents_summary`) plus manual sweep tools
such as `sweep_infer_resolutions.py`, `sweep_vae_roundtrip_resolutions.py`,
and `diagnose_dit_conditioning.py` for visual comparison across resize settings
and null-vs-prompt conditioning diagnostics. The diagnosis script can now
auto-export a first-episode DROID preview clip for local-video checks and can
fall back to the pretrained backbone when no fine-tuned checkpoint exists.
