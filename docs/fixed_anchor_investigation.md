# Fixed-Anchor Root-Cause Investigation

Decision memory for one fixed-anchor debugging loop. Keep the short-window
contract fixed until the first failing stage is clear enough to justify a fix
or a pivot.

## Goal
Find the first pipeline stage that causes the arm/fork morphing failure under
the fixed short-window contract, then validate only changes that improve that
same anchor on the same eval window.

## Fixed Anchor
- Hard constraints for every run in this loop: `context_len=17`,
  `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`,
  `single_chunk_rollout=true`, `frame_width=320`, `frame_height=240`.
- Canonical eval window: `repo_id=lerobot/aloha_static_fork_pick_up`,
  `episode_index=0`, `start_frame=60`,
  `video_key=observation.images.cam_high`.

## Current First-Failing-Stage Hypothesis
- The first visible failure is in denoising the single future latent residual
  step. Raw/preprocess/VAE/export counts are consistent, and the last-context
  future control path cancels to the intended zero-change prior.

## Stage Findings
- The current checkpoint-mode sweep compares the full `context + future`
  rollout. For `ctx17/h4`, that means many scalar diagnostics are diluted by
  `17` copied context frames even when the visible failure is concentrated in
  the final `4` future frames.
- The starting artifact is still visibly bad in the arm/fork region even though
  plausibility passes, so future-horizon visual failure overrides aggregate
  full-window metrics.
- The canonical stage probe at
  `runs/training_optimizer/fixed_anchor_stage_probe/ctx17_h4_step400_ep0_start60`
  reproduced the same ep0 failure: both raw and generated windows stay static
  through the last `4` context frames, then motion starts only in the final
  `4` future frames; raw future and VAE roundtrip stay crisp enough, but the
  generated future blooms into a bright fork/contact blur and misses clean
  contact.
- Frame accounting on the canonical window is internally consistent:
  `21` preprocessed raw frames -> `6` latent steps -> split `5` context +
  `1` future latent step -> `4` decoded future frames.
- The VAE roundtrip is not the first visible failure on this anchor:
  raw-vs-roundtrip future grids stay visually aligned, and sharpness stays
  close (`raw/roundtrip≈1.014`, `generated/roundtrip≈0.974`).
- The stage-state dump confirms the structural control path is behaving as
  intended for this anchor: `future_control_fill_mode=last_context_frame` and
  `future_latent_residual_mode=last_context_frame` cancel the future control
  video to exact zeros after residual subtraction, while the true future
  residual target stays comparatively small (`abs_mean≈0.122`) and the chunk
  schedule is the single latent boundary `[0,1]`.

## Open Hypotheses
- The model's first clean estimate for that one future latent step may already
  point toward the blurred fork/contact state.
- The scheduler trajectory may drift from a reasonable early estimate as the
  integration steps accumulate.
- Packing the whole `4`-frame horizon into one future latent step may be too
  coarse for clean last-horizon contact geometry even when control alignment is
  correct.

## Next Diagnostic Step
- Sweep `scripts/train/infer_world_model.py` on the same canonical raw clip at
  `integration_steps={1,10,25,50}` and compare the future-only grids to decide
  whether blur/ghosting is present in the earliest denoising estimate or only
  appears after repeated scheduler updates.

## Stable Findings
- Stay on this fixed anchor until the first failing stage is identified or the
  operator says to pivot.
- Prefer short diagnostic probes, instrumentation, and future-only inspection
  over another training continuation when the current failure is not yet
  localized.
- Keep the short-window contract fixed and let other settings move only when
  they test a concrete hypothesis about the visible failure.
- A turn is not complete unless it accounts for the whole pipeline on the
  current anchor.
- When using checkpoint-mode sweep artifacts for this anchor, visually inspect
  the final `horizon_len` frames first and do not rely on full-window scalar
  summaries alone.
- For temporal bugs, always check raw window length, latent-time shapes,
  decoded future frame counts, and exported video frame counts together.
- `ctx17/h4` packs the entire future horizon into exactly one latent step, so
  denoising or residual-target errors there affect all `4` future frames at
  once.
- Earlier held-out reviews still matter when judging motion-first behavior:
  episode `1` has the worst ghosting, episode `2` is the cleanest but still
  undercommitted, and neither held-out clip rescues this checkpoint family.

## Kept Code Changes
- Commit `0f50064` (`Add residual future latent training mode`): keeps
  `future_latent_residual_mode=last_context_frame` available for residual-target
  debugging.
- Commit `6323a3c` (`Add last-context future control fill mode`): keeps
  `future_control_fill_mode=last_context_frame` available for true zero-change
  future-control debugging.
- Commit `dd8dccd` (`Add world-model stage-state dump script`): adds
  `scripts/check/dump_world_model_stage_state.py` so fixed-anchor probes can
  dump latent split, chunk schedule, residual-target size, and VACE control
  assembly before denoising.
