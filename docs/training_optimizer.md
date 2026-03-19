## Stable Findings

- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation and `scripts/check/check_generated_video_plausibility.py` only as a safety gate or tie-breaker.
- Rank runs motion-first: arm and tool movement outrank whole-frame sharpness and aggregate MAE when the clips remain plausible.
- The strongest motion-first branch found so far is `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` at step `800`.
- The plain `horizon_len=16` family remains relevant as a reduced-distortion reference, but plain h16 continuation and checkpoint selection stayed undercommitted and never produced a motion-first win.
- The best h16 motion-assisted reference so far is `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p5_resume1000to1200` at step `1200`: cleaner than the stronger h16 motion branches, still undercommitted on the main clip and episode `1`.
- The local h16 `motion_loss_alpha=0.75` cap and excess-only neighborhood is exhausted. It produced useful diagnostics, but no clean winner over h16 `motion_loss_alpha=0.5`.
- The temporal-difference action residual was only a partial improvement over plain h16, and its direct interaction with h16 `motion_loss_alpha=0.5` regressed.
- The temporal action-token mixer produced one plausible structural baseline at step `1200`, but plain continuation to step `1400` regressed, so the remaining issue is still action-magnitude and temporal conditioning rather than color fidelity.

## Best Run

- Motion-first best run: `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` at step `800`.
- Image-quality reference: `optimizer_aloha_static_fork_pick_up_full_320x240_lora8_action` at step `1400`.
- Clean h16 resume point: `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1` at step `1000`.
- Best h16 motion-assisted reference: `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p5_resume1000to1200` at step `1200`.
- Ranking takeaway: use residual step `800` as the motion reference, use normalized step `1400` as the image-quality reference, and treat h16 as a cleaner-but-still-undercommitted family that now needs a different temporal-control lever rather than more plain continuation.

## Active Decision

- `Status`: paused by operator.
- `Question`: can stronger temporal action conditioning plus a mild motion assist improve commitment on the clean h16 branch without reintroducing the distortion and late-heavy failures from the local h16 scalar sweeps?
- `Next step`: if optimization resumes, run the clean h16 step-`1000` branch with `action_temporal_mixer_kernel_size=3`, `action_temporal_mixer_scale=0.5`, and `motion_loss_alpha=0.5`, then re-evaluate the standard main clip plus held-out episodes `1` and `2`.
- `Success signal`: main clip and episode `1` no longer carry `stops_early`, episode `2` remains plausible, and the result beats h16 `motion_loss_alpha=0.5` step `1200` on motion-first ranking.
- `Exit condition`: if that mixer-plus-motion-loss run still undercommits on the main clip and episode `1` or makes episode `2` implausible, stop spending budget inside this local h16 neighborhood and escalate to a larger action-conditioning change or a distinct longer-context experiment.

## Exhausted Families

- Residual-family continuations and nearby width, LR, and action-input-LayerNorm tweaks around residual step `800`.
- Plain `horizon_len=16` continuation and checkpoint selection inside the clean h16 rerun.
- Local h16 `motion_loss_alpha=0.75` cap and excess-only sweep.
- `action_temporal_difference_scale=0.5` as a standalone h16 follow-up.
- `action_temporal_difference_scale=0.5` plus h16 `motion_loss_alpha=0.5` interaction test.
- Plain continuation of the temporal action-token mixer from step `1200` to step `1400`.
- Subset restriction and related dataset-filtering side branches.

## Kept Code Changes

- Commit `17ba95f` (`Cap motion-aware loss weights`): added `motion_loss_max_weight` so motion-aware loss can stay active without letting a few high-motion regions dominate training.
- Commit `7fe8994` (`Add excess-only motion loss weighting`): added an excess-only weighting mode so motion emphasis can target above-average motion regions instead of boosting all regions equally.
- Commit `7832e4a` (`Add temporal-difference action residual`): added an optional temporal-difference residual over action tokens without breaking checkpoint compatibility.
- Commit `5537878` (`Add temporal action-token mixer`): added an optional zero-init temporal mixer over projected action tokens plus backward-compatible checkpoint loading for structural action-conditioning tests.

## Resume From

- Clean h16 resume checkpoint: `runs/optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1/checkpoints/step_0001000.pt`
- Motion-first reference checkpoint: `runs/optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid/checkpoints/step_0000800.pt`
- Best h16 motion-assisted reference checkpoint: `runs/optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p5_resume1000to1200/checkpoints/step_0001200.pt`
- Latest temporal-mixer checkpoint family: `runs/optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_actiontempmixk3s0p5_resume1000to1200/checkpoints/step_0001400.pt`
- Validation ledger: `runs/training_optimizer/experiment_ledger.md`
