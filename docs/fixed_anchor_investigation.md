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
- Starting artifact:
  `runs/training_optimizer/eval/optimizer_aloha_static_fork_pick_up_full_320x240_ctx17_h4_lora32_noaction_gradckpt_residual_lastctx_filllastctx_singlechunk_fresh400_step400_ep0_start60/optimizer_aloha_static_fork_pick_up_full_320x240_ctx17_h4_lora32_noaction_gradckpt_residual_lastctx_filllastctx_singlechunk_fresh400_step_0000400_comparison.mp4`
- Starting checkpoint:
  `runs/optimizer_aloha_static_fork_pick_up_full_320x240_ctx17_h4_lora32_noaction_gradckpt_residual_lastctx_filllastctx_singlechunk_fresh400/checkpoints/step_0000400.pt`

## Current First-Failing-Stage Hypothesis
- Not yet proven.

## Stage Findings
- The current checkpoint-mode sweep compares the full `context + future`
  rollout. For `ctx17/h4`, that means many scalar diagnostics are diluted by
  `17` copied context frames even when the visible failure is concentrated in
  the final `4` future frames.
- The starting artifact is still visibly bad in the arm/fork region even though
  plausibility passes, so future-horizon visual failure overrides aggregate
  full-window metrics.
- Existing checkpoints are starting evidence, not a ceiling. Fresh training
  runs are allowed when they test one concrete hypothesis under the same fixed
  contract.

## Open Hypotheses
- The failure may start in preprocessing, VAE encode/decode, latent packing,
  control or residual construction, masking, denoising, or export.
- The failure may only become obvious in future-only views or stage-boundary
  reports.
- The last-context residual target and future-control fill may still be
  mismatched in scale, masking, or temporal alignment.

## Next Diagnostic Step
- Reproduce the fixed anchor with direct checkpoint inference on the canonical
  window and collect stage-boundary artifacts first: raw frame window, latent
  split, decoded frame counts, exported frame counts, VAE roundtrip, generated
  future, and any saved frame/sharpness reports.
- Then keep iterating in order: raw input, preprocessing, latent packing,
  control or residual construction, denoising, decode, export.

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

## Kept Code Changes
- Commit `0f50064` (`Add residual future latent training mode`): keeps
  `future_latent_residual_mode=last_context_frame` available for residual-target
  debugging.
- Commit `6323a3c` (`Add last-context future control fill mode`): keeps
  `future_control_fill_mode=last_context_frame` available for true zero-change
  future-control debugging.
