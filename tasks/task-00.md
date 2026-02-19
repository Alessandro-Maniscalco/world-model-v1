# Task 00 - Minimal OpenPI-style Structure

## Goal
Add maintainable module boundaries without adopting heavy packaging/distribution complexity.

## Scope
- Central config in `src/world_model/config.py`
- Separate dataset reading from model-ready packing in `src/world_model/data/`
- First-class masking module in `src/world_model/masking/`
- Architecture-aligned package layout under `src/world_model/`

## Done when
- New subpackages exist: `data`, `latents`, `chunking`, `masking`, `conditioning`, `models`, `training`
- Masking logic is imported from shared module in checks/scripts
- Tests cover no-future mask semantics and packed-batch shape contracts
