# Repo Structure Ownership

**Documentation Rule**: Every time a code change is made, `docs/repo_structure.md` and `docs/architecture.md` must be checked and updated if necessary to reflect the changes.

## Source Packages

- `src/world_model/config.py`
Defines typed `dataclass` configurations for training (`TrainScriptConfig`) and 
inference (`InferScriptConfig`). It contains default parameters, provides helpers to load settings from YAML 
files, and merges command-line overrides from `argparse`.

- `src/world_model/data/`
  - `dataset.py`: Gets raw frames from disk.
  - `temporal.py`: Makes sure the timestamps are correct.
  - `pack.py`: Splits and reshapes the tensors.
  - `schema.py`: Defines what the output "looks like."
  - `prepare.py`: Runs the VAE and executes the full pipeline: VAE ecnoncoding, temporal splitting, signal aligment, batch construction.

- `src/world_model/latents/vae.py`
Owns Wan VAE encode/decode interfaces and range/layout normalization.

- `src/world_model/chunking/schedule.py`
Owns K+1 latent-time chunk schedules and chunk-id generation.

- `src/world_model/masking/block_causal.py`
Owns block-causal attention mask construction.

- `src/world_model/conditioning/`
Owns action/proprio encoders and AdaLN-Zero modulation primitives.

- `src/world_model/models/wan_dit_wrapper.py`
Owns model wrappers used for flow-matching velocity prediction.

- `src/world_model/training/`
  - `flow_matching.py`: Owns the core math for straight-line flow matching (timestep sampling, noisy state generation, weighting) and computing the chunkwise teacher-forced MSE loss.
  - `chunkwise_training.py`: Orchestrates a single optimizer step (forward, backward passes, grad clipping) and handles metrics logging and checkpoint saving.

- `src/world_model/eval/`
Owns inference-time token sampling and token-to-latent conversion.

## Scripts

- `scripts/train/world_model.py`
Canonical training entrypoint.

- `scripts/train/infer_world_model.py`
Canonical inference/visualization entrypoint.

- `scripts/check/`
Canonical diagnostics (`dataset`, `forward_real_batch`, `latent_cache`,
`masking_leakage`, `vae_roundtrip`, `latents_summary`).
