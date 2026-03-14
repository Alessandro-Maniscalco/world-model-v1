# Blur Investigation: Base Wan VACE on ALOHA

## Goal

Determine where visible blur is introduced when running the public Wan VACE base pipeline on ALOHA fork-pick-up frames.

## Scope

- Dataset: `lerobot/aloha_static_fork_pick_up`
- Episode: `0`
- Start frame: `30`
- Camera: `observation.images.cam_high`
- Primary resolution under investigation: `832x480`

## What We Checked

### 1. Raw dataset preview

Command:

```bash
python scripts/check/preview_aloha_sequence.py \
  --repo-id lerobot/aloha_static_fork_pick_up \
  --episode-index 0 \
  --frame-offset 30 \
  --video-key observation.images.cam_high \
  --num-frames 17 \
  --output-dir runs/check_aloha_fork_preview_start30
```

Result:

- The exported preview frames are visually sharp.
- Conclusion: the dataset decode path is not the blur source.

### 2. VAE-only roundtrip

Command:

```bash
python scripts/check/sweep_vae_roundtrip_resolutions.py \
  --video-path runs/check_aloha_fork_preview_start30/preview.mp4 \
  --output-dir runs/check_aloha_vae_roundtrip_start30 \
  --context-len 9 \
  --horizon-len 8 \
  --resolutions 832x480
```

Artifacts:

- `runs/check_aloha_vae_roundtrip_start30/832x480/vae_roundtrip_vs_raw_grid.png`
- `runs/check_aloha_vae_roundtrip_start30/832x480/sharpness_report.json`

Result:

- The VAE roundtrip remains visually sharp.
- `sharpness_report.json` shows:
  - `raw_future_aligned`: `0.0009567738452460617`
  - `vae_roundtrip`: `0.001431095355655998`
  - `relative_to_vae_roundtrip.raw_future_aligned`: `0.668560513081595`

Interpretation:

- The VAE is not the dominant blur source in this ALOHA setup.
- Blur must be introduced after the VAE-only path, likely in the base VACE denoising / conditioning path.

### 3. Existing base VACE output at 832x480

Artifact:

- `runs/check_wan_vace_base_resolution_sweep/832x480.mp4`

Measured comparison between:

- source condition frame resized to `832x480`
- frame 0 extracted from `runs/check_wan_vace_base_resolution_sweep/832x480.mp4`

Measured values:

- source condition gradient energy: `0.0011410649167373776`
- base output frame 0 gradient energy: `0.0009021928999572992`
- base/source sharpness ratio: `0.7906586967347309`
- pixel MSE: `31.83503532409668`

Interpretation:

- The blur is already present in the generated frame, not just in user perception of the MP4.
- The base output frame is materially softer than the input-conditioned frame.

## Important Finding: Prompt Mismatch

The current base resolution sweep script imports its prompt from:

- `scripts/check/wan_vace_diffuser_generate_video.py`

That helper currently defines:

```python
PROMPT = "Guitarist playing guitar"
```

The resolution sweep script uses that inherited prompt when generating ALOHA fork-pick-up outputs.

Implication:

- The current ALOHA blur tests are confounded by an obviously unrelated text prompt.
- Even if the model preserves layout, the denoiser is being asked to move the output toward a semantically different scene.
- This can plausibly cause both blur and unwanted rewriting of conditioned content.

## Current Conclusions

1. Dataset decode is sharp.
2. The Wan VAE roundtrip is sharp enough and does not explain the observed blur.
3. The public base VACE pipeline output is softer than the conditioned source frame.
4. The current base sweep is using an unrelated inherited prompt, so the blur diagnosis is not yet text-conditioning clean.

## Next Checks

1. Patch the base resolution sweep to accept explicit `--prompt` and `--negative-prompt` values.
2. Re-run a cheap comparison with identical seed/resolution/steps:
   - inherited `guitarist` prompt
   - empty prompt
3. If empty prompt remains blurry, investigate other VACE-side causes:
   - conditioning scale behavior
   - whether VACE preserves conditioned frames softly rather than pixel-exactly
   - effect of inference step count on conditioned-frame fidelity
