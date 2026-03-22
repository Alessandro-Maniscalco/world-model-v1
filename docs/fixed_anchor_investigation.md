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
- Rung: curriculum-transfer test from the best ep1 overfit checkpoint into the
  full dataset.
- Resume the validated ep1-only `ctx17/h4`, `k=1`, single-chunk,
  residual+filllastctx, action-conditioned LoRA checkpoint
  `step_0000200.pt` on the full dataset for `200` more steps, then evaluate
  the same episode-1 window at `start_frame=60` at `step 300` and `step 400`.
- Why next: the one-trajectory overfit branch is now answered and saturated.
  On ep1, `step 200` is the clean peak: the copied context still stays static
  through frame `16`, motion starts at frame `17`, and the frame-17-to-20 fork
  approach nearly matches the reference. Continuing that same branch to
  `300/400` brings back blue/white ghosting, thicker arm-crop blur, and worse
  contact. The new stage question is whether that good ep1 solution can
  survive limited full-dataset mixing at all, or whether broader training
  immediately washes it out.

## Best current result for fixed anchor
- `runs/training_optimizer/inspection/optimizer_aloha_static_fork_pick_up_ep1only_ctx17_h4_lora32_action_tokenscale075_gradckpt_residual_lastctx_filllastctx_singlechunk_overfit200_step200_ep1_start60/optimizer_aloha_static_fork_pick_up_ep1only_ctx17_h4_lora32_action_tokenscale075_gradckpt_residual_lastctx_filllastctx_singlechunk_overfit200_step_0000200_comparison.mp4`

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
- `step_0000200.pt` answers the operator's checkpoint question in the useful
  direction. On ep1, motion still starts at frame `17` and continues through
  frames `18-20`, but the future fork/gripper region is visibly cleaner than
  both step `300` and step `400`: much less blue/purple ghosting, a narrower
  arm-crop silhouette, and a less blown-out final contact frame. The contact is
  still soft rather than crisp, but the clip looks less distorted overall and
  improves motion alignment scalars (`profile_correlation≈0.371`,
  `late_motion_ratio≈1.659`) relative to both later checkpoints.
- `metrics.jsonl` shows the action-conditioned branch improves sharply between
  `step 100` and `step 300`, then regresses by `step 400`. The operator-directed
  checkpoint question is now resolved: future runs in this family should treat
  `200` steps as the default cap unless a later result clearly beats it on the
  same ep1 clip.
- Lowering inference-time action gain below `1.0` changes the ep1 clip without
  changing its timing. At both scales, the copied context stays static through
  frame `16`, motion still starts at frame `17`, and the fork/gripper keeps
  moving through frames `18-20`.
- `action_token_scale=0.75` is the current best ep1 artifact. It keeps the same
  frame-17 motion onset and still misses a clean plate-edge contact by frame
  `20`, but the future fork halo is slightly thinner, the arm-crop silhouette
  is a bit less widened, and the final contact frame is less blown out than the
  scale-`1.0` baseline. The reports move in the same direction
  (`profile_correlation≈0.413`, `late_motion_ratio≈1.637`,
  future-frame MAE `17:8.84`, `18:13.94`, `19:13.27`, `20:12.89`).
- `action_token_scale=0.50` is a regression on the same ep1 clip. Motion still
  starts at frame `17` and runs through frame `20`, but the future fork becomes
  softer and less well aligned, with worse motion scalars than both scale
  `1.0` and `0.75` (`profile_correlation≈0.350`,
  `late_motion_ratio≈1.707`, `max_frame_mae≈15.60`). Lowering gain helps only
  up to a point; halving it does not rescue contact.
- The token-scale sweep does not fix the visible failure in inference alone.
  At both scales the copied context stays static through frame `16`, motion
  starts at frame `17`, and the model still misses clean plate-edge contact by
  frame `20`; `0.75` only thins the same fork/gripper blur slightly, while
  `0.50` softens and misaligns it again.
- The fresh full-dataset train-time `action_token_scale=0.75` scout changes
  the same ep1 clip only modestly. `step 100` keeps the copied context static
  through frame `16`, starts moving at frame `17`, and reaches through frames
  `18-20` with a slimmer, less blown-out fork than the earlier inference-only
  best, but it still misses a crisp plate-edge landing by frame `20`.
- That improvement does not survive continuation inside the same branch.
  By `step 200`, the ep1 clip still starts moving at frame `17`, but the fork
  brightens and thickens again across frames `18-20`, the arm crop looks more
  widened, and the reports reintroduce `overactive_motion`
  (`profile_correlation≈0.338`, `late_motion_ratio≈1.677`,
  `max_frame_mae≈15.16`) compared with the cleaner `step 100`
  (`≈0.237`, `≈1.414`, `≈11.52`).
- The operator-directed ep1-only overfit run is the first branch that nearly
  fits the decision clip. `step 100` still keeps frame `16` static and starts
  moving at frame `17`, but it remains softer and more washed out than the
  reference through frames `18-20`. By `step 200`, the same ep1-only branch
  still stays static through frame `16`, starts moving at frame `17`, and then
  follows the fork approach with much weaker blur and a narrower arm-crop
  silhouette than any earlier checkpoint, leaving only a slight frame-20
  softness instead of the old blown-out missed-contact smear.
- The supporting reports move in the same direction on the ep1-only overfit
  run. Relative to the previous best full-dataset `step 100` ep1 artifact,
  overfit `step 200` lowers full-window `mean_frame_mae` from `≈1.936` to
  `≈1.353`, lowers `max_frame_mae` from `≈11.52` to `≈8.58`, and keeps the
  clip plausible while improving arm-motion alignment
  (`profile_correlation≈0.345`, `late_motion_ratio≈1.254`). The scalar labels
  still say `misaligned`, but the visible future-horizon error is much smaller.
- The same ep1-only overfit branch regresses when continued past `step 200`.
  `step 300` and `step 400` still keep the copied context static through frame
  `16` and start moving at frame `17`, but both bring back immediate
  blue/white haloing on the fork by frame `17`, thicken the arm crop through
  frames `18-20`, and end with a more overactive, smeared contact than
  `step 200`. `step 400` is the worst of the three, with the strongest doubled
  fork silhouette and the widest final-frame blur.
- The reports confirm the visual regression inside the resumed overfit branch.
  Relative to overfit `step 200`, `step 300` worsens to
  `mean_frame_mae≈2.741`, `max_frame_mae≈17.29`,
  `late_motion_ratio≈1.681`, and `overactive_motion`; `step 400` remains
  worse than `step 200` at `mean_frame_mae≈1.774`,
  `max_frame_mae≈11.89`, `late_motion_ratio≈1.825`, and even lower
  arm-motion `spatial_motion_iou≈0.660`. For this neighborhood, `step 200` is
  the best stop point.

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
- On the operator clip, the best current inference-time action gain is `0.75`:
  it modestly reduces frame-17-to-20 fork/gripper ghosting without pushing the
  motion back toward the frozen no-action failure.
- The action-conditioned residual+filllastctx family can fit the operator clip
  substantially better when every training window comes from episode `1`. The
  remaining fixed-anchor gap now looks like optimization or data-mixing, not a
  hard inability to represent the frame-17-to-20 fork approach at all.
- Inside the ep1-only overfit neighborhood, continuing past `step 200` is not
  beneficial. `step 300` and `step 400` regress back toward overactive blur,
  so `step 200` should be treated as the best checkpoint in that branch.
- When a fresh action-conditioned branch regresses between checkpoints on ep1,
  do not continue it plainly; change the training neighborhood instead. The
  new full-dataset train-time `0.75` scout follows that rule: `step 100` is
  the only keepable point and `step 200` is already worse.

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
