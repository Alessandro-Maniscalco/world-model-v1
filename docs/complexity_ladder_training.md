## Goal
The process is to start from the simplest meaningful setup, validate visible task-relevant motion, then increase complexity one axis at a time.

## Proven Complexity ladder
List only proven rungs. For each rung, include a short description of the
complexity and the best video link.

## Next complexity to test
Only one, including the rung name and why it is next. It is flexible.
- Rung: action-conditioned two-latent scout.
- `conditioning_mode=action`, `context_len=17`, `horizon_len=8`, `k=1`, `chunk_schedule_mode=k_chunks`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `max_steps=200`, `no_action_input_layernorm=true`, `action_mlp_dim=128`, `action_mlp_residual=true`
- Why next: the observation-only `ctx17/h4` scout stayed parked through frames `14-16` on the main clip plus held-out episodes `1` and `2`, then made only a single late jump on frame `17`. Because `horizon_len=4` compresses to just one future latent step under Wan packing, another `h4` follow-up is a weak test for multi-step timing. The next meaningful ladder rung is therefore the smallest action-conditioned setup with more than one future latent step: keep the cheap `ctx17` geometry, promote only the future horizon to `h8`, and add the established action path to test whether action cues can pull motion earlier across two latent steps without giving up plausibility.

## Best rung for current complexity
Only one for the current complexity being researched, including the mp4 link
and a short description of the run.
- None yet for the action-conditioned two-latent scout rung.

## Rung Findings for current complexity
Clear when complexity increases. Use one point per rung.
- Observation-only short-window absolute-target scout rejected as a working rung: `conditioning_mode=none`, `ctx17/h4`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `step_0000200` stayed essentially static through frames `14-16` on the main clip and held-out episodes `1` and `2`, then jumped only on frame `17`; the generated fork shape stayed plausible, but it still missed the reference timing and contact path, especially on held-outs where the final-frame move is abrupt and slightly blurry (`late_motion_ratio≈1.72/2.20/2.53`, all `misaligned`).

## Stable Findings
- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation,
  treat plausibility as a safety gate, and rank runs visual first
- Stay on the complexity ladder: find the simplest rung that works, then
  promote upward one axis at a time.
- In this repo, the closest local Wan/VACE-style inference contract uses
  `single_chunk_rollout=true` with at least `50` integration steps.
- For simple runs with few latent steps, cap the scout at `200` training steps
  before deciding whether that rung earns promotion.
- Under Wan temporal packing, `horizon_len=4` gives only one future latent
  step, so `h4` is useful as a cheapest final-block scout but not as a strong
  test of multi-step temporal commitment.
- Observation-only is now exhausted as the simplest ladder family: the hard
  `ctx21/h8` control stayed late-heavy, the residual `ctx17/h4` scout imploded
  on frame `17`, and the absolute-target `ctx17/h4` scout stayed plausible but
  still waited until frame `17` to move on the main clip and both held-outs.
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
