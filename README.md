# world-model-v1

Latent-space world model for LIBERO data using a frozen Wan VAE, chunkwise
teacher-forced flow matching, and AdaLN-conditioned DiT blocks.

## Architecture Snapshot

- Visual backbone: Wan VAE encoder/decoder (frozen).
- Prediction backbone: Wan DiT-style wrapper in latent token space.
- Conditioning: action and optional proprio embeddings injected via AdaLN only.
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

Infer:

```bash
python scripts/train/infer_world_model.py --config configs/eval/infer_world_model.yaml --checkpoint <path>
```

Forward smoke check:

```bash
python scripts/check/forward_real_batch.py --help
```
