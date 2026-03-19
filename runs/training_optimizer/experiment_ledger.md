# Experiment Ledger

Detailed validation chronology lives here. Use `docs/training_optimizer.md` for the
current decision state and use this ledger only when a turn needs older context.

## Current Reference Set

- Motion-first reference: `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` step `800`
- Image-quality reference: `optimizer_aloha_static_fork_pick_up_full_320x240_lora8_action` step `1400`
- Clean h16 resume point: `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1` step `1000`
- Best h16 motion-assisted reference: `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p5_resume1000to1200` step `1200`

## Family Summaries

- Residual motion-first family:
  - `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` step `800` remains the strongest motion-first result.
  - Nearby continuations (`825`, `850`, `900`, `1000`, `1200`) and nearby width or LR variants did not beat step `800`.

- Plain h16 family:
  - `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid` step `800` and rerun step `1000` stayed cleaner-looking than the stronger-motion branches but remained undercommitted on the main clip and episode `1`.
  - Checkpoint selection inside the rerun was exhausted when step `900` clearly regressed.

- h16 motion-loss family:
  - `motion_loss_alpha=1.0` improved late motion relative to plain h16 but overshot, blurred more, and generalized worse.
  - `motion_loss_alpha=0.5` from the clean h16 step-`1000` checkpoint to step `1200` is the best h16 motion-assisted reference so far: cleaner and more stable than the stronger h16 motion-loss branches, but still undercommitted on the main clip and episode `1`.
  - The local `motion_loss_alpha=0.75` family, including capped and excess-only variants, is exhausted. It provided diagnostics about spike control, but no variant beat the h16 `motion_loss_alpha=0.5` reference cleanly.

- h16 temporal-difference action family:
  - `action_temporal_difference_scale=0.5` alone slightly improved main-clip motion over plain h16 step `1000`, but episode `2` remained implausible.
  - Combining that temporal-difference residual with h16 `motion_loss_alpha=0.5` overswung the main clip and episode `2` and is exhausted as a local interaction test.

- h16 temporal-mixer family:
  - `action_temporal_mixer_kernel_size=3`, `action_temporal_mixer_scale=0.5` to step `1200` produced the first plausible structural-action baseline on all three windows.
  - Direct continuation to step `1400` regressed: the main clip and episode `1` remained undercommitted and episode `2` became implausible again, so plain continuation of the temporal-mixer branch is exhausted.

## Recent Validation Notes

- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p5_resume1000to1200` step `1200`
  - `arm_motion_verdict`: cleaner h16 motion-assisted trade, still undercommitted on the main clip and episode `1`
  - `image_quality_verdict`: best h16 motion-assisted image and stability trade so far
  - `continue_training`: no direct continuation; keep as reference
  - Why: improved main MAE and overlap over plain h16 step `1000` and reduced episode-`2` overactivity, but never broke out of the early-stop family on the main clip and episode `1`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_actiontempmixk3s0p5_resume1000to1200` step `1200`
  - `arm_motion_verdict`: plausible structural-action baseline, still undercommitted on the main clip and episode `1`
  - `image_quality_verdict`: clean and plausible on all three windows
  - `continue_training`: yes at the time; one direct continuation probe was justified
  - Why: this was the first structural-action branch that stayed plausible on all three windows and slightly improved main-clip MAE over the h16 `motion_loss_alpha=0.5` reference.

- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_actiontempmixk3s0p5_resume1000to1200` step `1400`
  - `arm_motion_verdict`: regressed continuation, still undercommitted on the main clip and episode `1`
  - `image_quality_verdict`: mostly clean on the main clip and episode `1`, but episode `2` fails plausibility again
  - `continue_training`: no; plain continuation is exhausted
  - Why: main `late_motion_ratio` rose to `0.540` while overlap and MAE worsened, episode `1` regressed on MAE, and episode `2` became implausible again.

## Code-Change Ledger

- `17ba95f` `Cap motion-aware loss weights`
- `7fe8994` `Add excess-only motion loss weighting`
- `7832e4a` `Add temporal-difference action residual`
- `5537878` `Add temporal action-token mixer`
