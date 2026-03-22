## Goal
The process is to hold one short-window anchor fixed, trace the first stage
that visibly breaks the arm/fork motion, and only validate changes that
improve that same anchor on the same canonical evaluation window.

## Fixed Anchor
- Hard constraints for every run in this loop: `context_len=17`,
  `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`,
  `single_chunk_rollout=true`, `frame_width=320`, `frame_height=240`.
- Historical stage-diagnostic window: `repo_id=lerobot/aloha_static_fork_pick_up`,
  `episode_index=0`, `start_frame=60`,
  `video_key=observation.images.cam_high`.
- Operator-directed evaluation window from this point onward:
  `repo_id=lerobot/aloha_static_fork_pick_up`, `episode_index=1`,
  `start_frame=60`, `video_key=observation.images.cam_high`.

## Proven Stage Checks
- Raw-window selection, preprocessing, VAE encode/decode, and export counts are
  not the first visible failure on the canonical anchor. Best artifact:
  `runs/training_optimizer/fixed_anchor_stage_probe/ctx17_h4_step400_ep0_start60/infer/comparison_grid.png`
- Control/residual construction is not the first visible failure on the
  canonical anchor: the saved stage-state report shows the single future latent
  step uses the expected `[0,1]` chunk boundary and the last-context
  future-control path cancels to exact zeros after residual subtraction. Best
  artifact:
  `runs/training_optimizer/fixed_anchor_stage_probe/ctx17_h4_step400_ep0_start60/stage_state_report.json`
- Repeated scheduler integration is not the first visible failure on the
  canonical anchor. At `integration_steps=1`, the generated future already
  shows a thickened, softened fork that misses a clean plate-edge contact, and
  `10/25/50` mainly increase brightness and double-edge ghosting rather than
  changing the motion timing. Best artifact:
  `runs/training_optimizer/fixed_anchor_denoising_step_sweep/ctx17_h4_step400_ep0_start60/steps_1/comparison_grid.png`
- Sampling drift is not required to create the canonical failure. The
  teacher-forced `t=1000` clean estimate on the true noisy target path already
  reproduces the same widened, softened fork and missed plate-edge contact
  before any scheduler rollout. Best artifact:
  `runs/training_optimizer/fixed_anchor_teacher_forced_clean_estimates/ctx17_h4_step400_ep0_start60/steps1_idx0_t1000/comparison_grid.png`

## Next Diagnostic Step
- Rung: checkpoint selection inside the first action-conditioned fixed-anchor
  branch, under the operator's `ep1`-only evaluation rule.
- Evaluate `step_0000200.pt` on episode `1` only, then compare it against the
  validated `step_0000300.pt` current best and the older `step_0000400.pt`
  baseline from the same `ctx17/h4`, `k=1`, single-chunk, residual+filllastctx
  action-conditioned run.
- Why next: the operator explicitly asked whether `step_0000200.pt` is
  materially worse than `step_0000400.pt`; if not, the branch should use the
  earlier checkpoint because it halves training time. The newly validated
  `step_0000300.pt` clip is now slightly better than `step_0000400.pt` on the
  operator's `ep1` decision window, so the next comparison should keep the
  operator's `200 vs 400` question while also checking against the strongest
  current checkpoint.

## Best current result for fixed anchor
- `runs/training_optimizer/inspection/optimizer_aloha_static_fork_pick_up_full_320x240_ctx17_h4_lora32_action_gradckpt_residual_lastctx_filllastctx_singlechunk_fresh400_step300_ep1_start60/optimizer_aloha_static_fork_pick_up_full_320x240_ctx17_h4_lora32_action_gradckpt_residual_lastctx_filllastctx_singlechunk_fresh400_step_0000300_comparison.mp4`

## Stage Findings for current anchor
- The current checkpoint-mode sweep compares the full `context + future`
  rollout. For `ctx17/h4`, that means many scalar diagnostics are diluted by
  `17` copied context frames even when the visible failure is concentrated in
  the final `4` future frames.
- On the canonical stage probe, both raw and generated windows stay static
  through the last `4` context frames and motion starts only in the final `4`
  future frames. The raw future and VAE roundtrip stay visually crisp enough,
  but the generated future blooms into a bright fork/contact blur and never
  reaches a clean contact pose.
- `frame_report.json` shows `21` preprocessed raw frames, `6` total latent
  steps, a `5`-step context / `1`-step future split, and `4` decoded future
  frames. `sharpness_report.json` stays close to the VAE baseline
  (`generated/roundtrip≈0.974`, `raw/roundtrip≈1.014`), so the VAE is not the
  first visible failure.
- The saved stage-state report shows the chunk schedule is the single boundary
  `[0,1]` and the future control video becomes exact zero after last-context
  fill plus residual subtraction. That keeps the earliest remaining suspicious
  stage on denoising of the single future latent residual step.
- The denoising-step sweep keeps the same frame accounting at every
  `integration_steps` setting, so the visible change is in the denoised future
  content rather than in frame packing or decode/export length.
- At `integration_steps=1`, the generated future still stays static through the
  copied context and only moves in the last `4` future frames, but the fork is
  already thicker and softer than the VAE roundtrip and never lands in a clean
  plate-edge contact pose.
- At `integration_steps=10`, `25`, and `50`, the motion onset still does not
  move earlier; instead the future fork brightens and develops stronger doubled
  edges across frames `2-4` of the horizon. `25` and `50` are the most ghosted
  variants, so extra steps amplify a bad future instead of rescuing it.
- The sharpness scalar rises from `generated/roundtrip≈0.873` at
  `integration_steps=1` to `≈0.963/0.977/0.974` at `10/25/50`, but the visible
  fork/contact behavior still worsens. On this anchor, sharper does not mean
  more correct.
- The teacher-forced probe on the actual dataset window produced the first
  `t=1000` comparison before running out of GPU memory on later timesteps. That
  first on-path clean estimate already matches the sampled `integration_steps=1`
  failure: the fork is visibly widened and softened in all `4` future frames,
  stays static until the horizon begins, and still misses a crisp plate-edge
  contact pose.
- Because the teacher-forced `t=1000` future is already wrong before any
  scheduler updates, the first failing stage is the checkpoint's direct
  velocity prediction for the single future latent residual step rather than
  rollout drift, export, or control assembly.
- The first action-conditioned `ctx17/h4` scout changes the family-level
  behavior. Motion now starts with the first future frame instead of staying
  nearly frozen until the horizon ends, and all three reviewed clips move
  through all `4` future frames while staying plausible.
- The branch is not clean yet. On ep0, the fork reaches toward the plate
  across the whole horizon but carries a blue/purple ghost trail and ends as a
  widened, brightened contact blur. On ep1, frames `17-20` also move through
  the whole horizon, but the gripper/fork region is the most overactive and
  washed out of the reviewed set, with blue/white/purple doubling already
  obvious in frame `17` and a missed clean contact by frame `20`
  (`profile_correlation≈0.285`, `late_motion_ratio≈1.962`).
- Held-out ep2 is still the cleanest clip in this branch: the fork advances
  through the full horizon with much less ghosting and a near-correct contact
  approach, though it remains somewhat too energetic late in the horizon
  (`motion_verdict=good`).
- On the operator's `ep1` decision clip, `step_0000300.pt` is only a modest
  visible improvement over `step_0000400.pt`, not a clean fix. Motion still
  starts in frame `17` and runs through frames `18-20`, but the step-300 fork
  smear is slightly less saturated and less widened in the arm crop, with lower
  frame-20 error (`max_frame_mae≈12.76` vs `≈15.31`) and slightly better full
  clip plausibility scalars. It still misses a crisp contact and remains
  `motion_verdict=misaligned` (`profile_correlation≈0.243`).
- `metrics.jsonl` shows the action-conditioned branch improves sharply between
  `step 100` and `step 300`, then regresses by `step 400`. The operator-directed
  checkpoint question is now whether `step_0000200.pt` is already close enough
  to this small step-300 improvement to justify the shorter run time.

## Stable Findings
- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation,
  treat plausibility as a safety gate, and rank runs visual first.
- Stay on the fixed anchor until the first failing stage is identified or the
  operator says to pivot.
- Fresh training runs are allowed, but the hard anchor stays fixed and every
  new run must test one concrete hypothesis.
- Current operator-directed testing is single-video only: use
  `episode_index=1`, `start_frame=60` as the decision clip and do not spend
  time on episodes `0` or `2` unless explicitly requested.
- When using checkpoint-mode sweep artifacts for this anchor, visually inspect
  the final `horizon_len` frames first and do not rely on full-window scalar
  summaries alone.
- For temporal bugs, always check raw window length, latent-time shapes,
  decoded future frame counts, and exported video frame counts together.
- Under Wan temporal packing, `horizon_len=4` gives one future latent step, so
  the current anchor isolates the easiest single-step future denoising case.
- Earlier held-out reviews remain decision-relevant: episode `1` has the worst
  white/black ghosting, episode `2` is cleaner but still diffuse and
  undercommitted, and neither held-out clip rescues this checkpoint family.
- Human operator override: future checkpoint comparisons in this loop should
  use episode `1` only unless a later operator message changes that rule.

## Kept Code Changes
Still-relevant code-changing commits that remain available as structural
levers.
- Commit `0f50064` (`Add residual future latent training mode`): adds
  checkpoint-compatible `future_latent_residual_mode=last_context_frame` to
  train/infer config, flow-matching training, and rollout sampling so the model
  can denoise future latents relative to the last observed latent frame instead
  of absolute latents.
- Commit `6323a3c` (`Add last-context future control fill mode`): adds
  `future_control_fill_mode=last_context_frame` through train/infer configs,
  checkpoint restore, and Wan/VACE control assembly so masked future control
  slots can copy the last observed latent frame instead of gray filler latents,
  giving residual-mode runs a true zero-change future control prior.
