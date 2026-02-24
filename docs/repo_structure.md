# Repo Structure Ownership

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

- `src/world_model/masking/`
Owns no-leak/block-causal attention mask construction.

- `src/world_model/conditioning/`
Owns action/proprio encoders and AdaLN-Zero modulation primitives.

- `src/world_model/models/`
Owns model wrappers used for flow-matching velocity prediction.

- `src/world_model/training/`
Owns flow-matching objective and batch training/checkpoint orchestration.

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
