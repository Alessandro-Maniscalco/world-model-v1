## Goal
The process is to find the easiest setup that can produce a good-looking plausible video with visible task-relevant motion, then increase difficulty one axis at a time from that anchor.

## Proven Complexity ladder
- None yet.

## Next complexity to test
- Rung: max-context single-generated-frame action residual scout.
- `conditioning_mode=action`, `context_len=21`, `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `future_latent_residual_mode=last_context_frame`, `max_steps=200`, `no_action_input_layernorm=true`, `action_mlp_dim=128`, `action_mlp_residual=true`
- Why next: the `ctx21/h4` action checkpoint is still the best easy-video base, but both its standard `50`-step and rescue `100`-step comparisons copy frames `14-20` on the main clip and held-out episodes `1` and `2` and only generate frame `21`; that single generated frame remains a smear on the main clip and episode `1` and still fails plausibility on episode `2`. Because the sampler rescue did not change the visible failure, the highest-value next action is now the strongest remaining structural lever inside the same easy-video base: keep the short one-latent horizon and action path, but train the future as a residual from the last observed latent frame to test whether frame `21` can become a clean fork-contact frame instead of a blob.

## Best rung for current complexity
- Best current run: [`step_0000200` comparison](../runs/training_optimizer/eval/optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h4_lora32_action_noinputln_mlp128resid_gradckpt_singlechunk_fresh200_final_for_eval/optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h4_lora32_action_noinputln_mlp128resid_gradckpt_singlechunk_fresh200_final_for_eval_comparison.mp4). It holds the scene stable through frames `14-20` on the main clip and both held-outs, but only frame `21` is generated and that final frame still smears instead of producing a clean fork contact.

## Rung Findings for current complexity
- Action-conditioned short-window two-latent scout rejected as an easy-video rung: `conditioning_mode=action`, `ctx17/h8`, `single_chunk_rollout=true`, `gradient_checkpointing=true`, `max_steps=200` generated frames `17-21` instead of only the last frame, but all five generated frames on the main clip were brown/green ghosted smears around the fork and plate; held-out episode `1` showed the same persistent blur across frames `17-21`, and held-out episode `2` was worst, with every generated frame failing plausibility and the plate/fork region washing out into a bright blob (`late_motion_ratio≈2.17/1.38/5.04`, all `misaligned`).
- Max-context one-latent action scout is the right easy-video base but not yet a keep: `conditioning_mode=action`, `ctx21/h4`, `single_chunk_rollout=true`, `step_0000200` copies frames `14-20` exactly on the main clip and held-out episodes `1` and `2`, then generates only frame `21`; that frame turns into a green/brown smear on the main clip, the same tool-tip blur on episode `1`, and a brighter washed-out blob on episode `2`, which fails plausibility on frame `21` (`late_motion_ratio≈1.70/1.40/3.35`, motion verdict `misaligned`/`overactive`/`misaligned`).
- Plain sampler-step rescue is rejected for this rung: rerunning the same `ctx21/h4` checkpoint at `100` inference steps leaves frames `14-20` copied and frame `21` still collapsed on all three windows, with slightly worse scalar checks than the `50`-step baseline (`max_frame_mae≈20.95/19.11/27.22` vs. `≈20.51/18.74/26.92` and the same episode-`2` plausibility failure on frame `21`).

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
- On the validated `ctx21/h4` action scout, frames `14-20` are still copied
  and only frame `21` is synthesized.
- Raising the sampler from `50` to `100` inference steps does not materially
  change that `ctx21/h4` failure, so more plain inference-step sweeps are not
  justified in this rung.
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
