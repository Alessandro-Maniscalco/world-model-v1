# Dreamdojo World Model Run Guide

This folder provides a local helper around the `world-model-eval` checkpoint flow from:
https://github.com/world-model-eval/world-model-eval/blob/master/README.md#world-model-checkpoint

## 1) Activate this repo virtualenv

```bash
cd /home/amaniscalco/world-model-v1
source .venv/bin/activate
```

## 2) One-time setup (clone + editable install + optional checkpoint download)

```bash
python -m scripts.Dreamdojo.world_model_eval_runner setup --download-checkpoint
```

This uses the checkpoint file id `1uiRP2BuavapMsyP9Cbr25mi_ymk9SEJb` and writes to:
`runs/dreamdojo/checkpoints/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt`
On Python 3.13, the helper automatically falls back to `pip install -e ... --no-deps` because one pinned upstream dependency is unavailable on 3.13.

## 3) Run a smoke rollout directly on the world model

Use any starting RGB image:

```bash
hf auth login
python -m scripts.Dreamdojo.world_model_eval_runner smoke \
  --start-image /absolute/path/to/start_frame.png \
  --steps 8 \
  --output-path runs/dreamdojo/smoke_rollout.mp4
```

This smoke run uses random actions (no OpenVLA/Octo dependency) so you can quickly verify the checkpoint and video generation path.
If you do not want to log in interactively, pass `--hf-token <your_token>` to the `smoke` command.
