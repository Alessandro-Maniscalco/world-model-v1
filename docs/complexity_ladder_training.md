## Goal
The process is to find the easiest setup that can produce a good-looking plausible video with visible task-relevant motion, then increase difficulty one axis at a time from that anchor.

## Proven Complexity ladder
- None yet.

## Next complexity to test
- Rung: max-context single-generated-frame action scout.
- `conditioning_mode=action`, `context_len=21`, `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `max_steps=200`, `no_action_input_layernorm=true`, `action_mlp_dim=128`, `action_mlp_residual=true`
- Why next: the `ctx17/h8` action scout finally generated more than one future frame, but all five generated frames `17-21` on the main clip and held-out episodes `1` and `2` were visibly blurred and ghosted, with episode `2` failing plausibility on every generated frame. Since the operator wants the easiest path to a good video first, the next rung should favor visual stability over longer rollout: keep the action path, collapse back to the cheapest one-latent future block, and increase context only so the model has the strongest visual anchor while synthesizing just one final frame.

## Best rung for current complexity
- None yet for the max-context single-generated-frame action scout rung.

## Rung Findings for current complexity
- Action-conditioned short-window two-latent scout rejected as an easy-video rung: `conditioning_mode=action`, `ctx17/h8`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `max_steps=200` generated frames `17-21` instead of only the last frame, but all five generated frames on the main clip were brown/green ghosted smears around the fork and plate; held-out episode `1` showed the same persistent blur across frames `17-21`, and held-out episode `2` was worst, with every generated frame failing plausibility and the plate/fork region washing out into a bright blob (`late_motion_ratio≈2.17/1.38/5.04`, all `misaligned`).

## Stable Findings
- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation,
  treat plausibility as a safety gate, and rank runs visual first
- Stay on the complexity ladder: find the easiest rung that produces a good
  video, then promote upward one axis at a time.
- In this repo, the closest local Wan/VACE-style inference contract uses
  `single_chunk_rollout=true` with at least `50` integration steps.
- For simple runs with few latent steps, cap the scout at `200` training steps
  before deciding whether that rung earns promotion.
- Lower context is simpler, but not easier for image quality; more context has
  helped stability, so the easy-video ladder should prefer short horizons
  before aggressively cutting context.
- Under Wan temporal packing, `horizon_len=4` gives one future latent step and
  `horizon_len=8` gives two; on the validated `ctx17/h4` scouts only frame
  `17` changed, while on the validated `ctx17/h8` action scout frames
  `17-21` were all generated.
- Observation-only is exhausted for easy-video scouting on this task: the hard
  `ctx21/h8` control stayed late-heavy, the residual `ctx17/h4` scout imploded
  on frame `17`, and the absolute-target `ctx17/h4` scout was readable but
  still only synthesized the final frame and missed contact timing.
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
