## Goal
The process is to start from the simplest meaningful setup, validate visible task-relevant motion, then increase complexity one axis at a time.

## Proven Complexity ladder
List only proven rungs. For each rung, include a short description of the
complexity and the best video link.

## Next complexity to test
Only one, including the rung name and why it is next. It is flexible.
- Rung: observation-only short-window absolute-target scout.
- `conditioning_mode=none`, `context_len=17`, `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `max_steps=200`
- Why next: the short-window residual scout also stayed static through frames `14-16` and then exploded into a pale/brown-green blob on frame `17` for the main clip plus held-out episodes `1` and `2`, so residual targets are exhausted. The simplest remaining discriminator before another code-side redesign is the same cheap short-window scout without residualization, to test whether the easier geometry itself can produce earlier fork motion or whether the whole observation-only family is structurally late. Per operator guidance, simple low-latent scout runs stop at `200` steps.

## Best rung for current complexity
Only one for the current complexity being researched, including the mp4 link
and a short description of the run.
- None yet for the short-window absolute-target scout rung.

## Rung Findings for current complexity
Clear when complexity increases. Use one point per rung.
- Short-window residual scout rejected: `conditioning_mode=none`, `ctx17/h4`, `single_chunk_rollout=true`, `future_latent_residual_mode=last_context_frame`, `gradient_checkpointing=true`, `step_0000200` stayed essentially static through frames `14-16` and then collapsed into a pale/brown-green blob on frame `17` in the main clip plus held-out episodes `1` and `2`; the arm crops never formed a readable fork shape or contact trajectory, all three windows failed plausibility on frame `17`, and the arm-motion reports stayed `misaligned` (`late_motion_ratio≈2.57/3.55/5.89`).

## Stable Findings
- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation,
  treat plausibility as a safety gate, and rank runs visual first
- Stay on the complexity ladder: find the simplest rung that works, then
  promote upward one axis at a time.
- In this repo, the closest local Wan/VACE-style inference contract uses
  `single_chunk_rollout=true` with at least `50` integration steps.
- For simple runs with few latent steps, cap the scout at `200` training steps
  before deciding whether that rung earns promotion.
- The late-motion failure is not obviously action-specific on the harder
  benchmark geometry: observation-only `conditioning_mode=none` also stayed
  late-heavy on `ctx21/h8`, so the backbone/objective must be treated as a
  first-class suspect, not only the action path.
- Residual-target reformulation is exhausted: the hard `ctx21/h8` residual run
  stayed frozen through frames `14-21` before a late smear, and the cheaper
  `ctx17/h4` scout also stayed static through frames `14-16` before exploding
  into an implausible blob on frame `17` for the main clip and both held-outs.
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
