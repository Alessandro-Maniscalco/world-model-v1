## Goal
The process is to start from the simplest meaningful setup, validate visible task-relevant motion, then increase complexity one axis at a time.

## Proven Complexity ladder
List only proven rungs. For each rung, include a short description of the
complexity and the best video link.

## Next complexity to test
Only one, including the rung name and why it is next. It is flexible.
- Rung: observation-only short-window residual scout.
- `conditioning_mode=none`, `context_len=17`, `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`, `single_chunk_rollout=true`, `future_latent_residual_mode=last_context_frame`, `gradient_checkpointing=true`
- Why next: the harder `ctx21/h8` residual run stayed frozen until frame `21` and then collapsed into a late color-shifted smear on the main clip plus held-out episodes `1` and `2`, so the best remaining residual-target test is the cheaper scout rung before rejecting the reformulation globally.

## Best rung for current complexity
Only one for the current complexity being researched, including the mp4 link
and a short description of the run.
- None yet for the short-window residual scout rung.

## Rung Findings for current complexity
Clear when complexity increases. Use one point per rung.
- Hard benchmark residual probe rejected: `conditioning_mode=none`, `ctx21/h8`, `future_latent_residual_mode=last_context_frame`, `gradient_checkpointing=true`, `fresh400` stayed nearly static through frames `14-21` and then broke into a late brown/green smear from frame `21` onward on the main clip and both held-outs; all three windows failed plausibility on frames `21-25`, and the arm-motion reports stayed `misaligned` (`late_motion_ratio≈3.62/2.55/5.94`).

## Stable Findings
- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation,
  treat plausibility as a safety gate, and rank runs visual first
- In this repo, the closest local Wan/VACE-style inference contract uses
  `single_chunk_rollout=true` with at least `50` integration steps.
- The late-motion failure is not obviously action-specific on the harder
  benchmark geometry: observation-only `conditioning_mode=none` also stayed
  late-heavy on `ctx21/h8`, so the backbone/objective must be treated as a
  first-class suspect, not only the action path.
- Longer context helped stability on the harder benchmark geometry, so wins on
  short-window scout rungs should still be rechecked before promoting them to
  the main benchmark.

## Kept Code Changes
Still-relevant code-changing commits that remain available as structural levers.
- Commit `0f50064` (`Add residual future latent training mode`): adds
  checkpoint-compatible `future_latent_residual_mode=last_context_frame` to
  train/infer config, flow-matching training, and rollout sampling so the model
  can denoise future latents relative to the last observed latent frame instead
  of absolute latents.
