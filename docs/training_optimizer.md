## Goal

Keep only the information that should change the next optimization decision.

## Stable Findings

- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation and `scripts/check/check_generated_video_plausibility.py` only as a safety gate or tie-breaker.
- Rank runs motion-first: arm and tool movement outrank whole-frame sharpness and aggregate MAE when both clips remain plausible.
- The current bottleneck is motion and control fidelity, not color. Earlier whitening concerns were mostly evaluation-path issues, and the FP32 VAE roundtrip is close to neutral.
- The normalized action branch, `optimizer_aloha_static_fork_pick_up_full_320x240_lora8_action`, remains the cleanest image branch at step `1400`, but it is undercommitted and stops early enough to lose under the motion-first ranking. Extending it to step `1500` did not produce a meaningful improvement.
- The normalized step-`1400` action-conditioning probes narrow the failure source:
  - `auto` and `sequence` action-source modes are bit-identical, so action-source selection is not the issue.
  - Zeroing actions worsens the rollout, so the model does use action tokens.
  - Nominal and `1.5x` actions are bit-identical on that branch, which points to poor action-magnitude sensitivity rather than total action ignorance.
- Repo inspection explains that action-scale invariance: the current `ActionTokenEncoder` applies per-token `LayerNorm` before projection, so positive action rescaling is normalized away at input.
- Removing action-input LayerNorm is directionally useful. Full-dataset `no_action_input_layernorm` branches consistently beat the older normalized step-`800` baseline early, but plain continuation on those branches does not keep improving and often regresses after the early peak.
- The strongest motion-first branch found so far is `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` at step `800`.
- That residual step-`800` branch is not just a main-clip fluke:
  - It remains competitive on held-out episodes and alternate windows.
  - It usually improves task-motion commitment and max-error behavior over normalized step `1400`.
  - It still has a mixed trade against normalized step `1400` because temporal stability can be worse, especially on some windows.
- Increasing the future horizon alone did not improve motion-first ranking at step `800`. A fresh rerun of the best residual recipe at `horizon_len=16` stayed plausible, but it regressed into an undercommitted or misaligned family: the main clip and episode `1` stop early, while episode `2` becomes more overactive late.
- Human review adds an important secondary signal: the plain `horizon_len=16` branch appears to reduce image distortion relative to the stronger-motion families. That means the branch should get direct follow-up tests before it is dropped.
- A fresh plain-`horizon_len=16` rerun to step `1000` preserved that cleaner-looking family, but it still did not produce a clear motion breakthrough. The main clip and episode `1` remained undercommitted, episode `2` stayed late-overactive, and the branch only gained a small amount of motion over plain h16 step `800` while worsening MAE.
- Checkpoint selection inside that plain-h16 rerun is now exhausted too. Evaluating rerun step `900` was clearly worse than both plain h16 step `800` and rerun step `1000`: on the main clip it stopped even earlier (`late_motion_ratio` `0.287`, `profile_correlation` `0.364`, MAE `13.328`), episode `1` also regressed sharply (`late_motion_ratio` `0.272`, MAE `14.258`), and episode `2` became the noisiest late-sweep variant (`mean_frame_mae_rgb_0_255` `15.860`, `max_frame_mae_rgb_0_255` `30.422`).
- Adding `motion_loss_alpha=1.0` on top of the h16 branch does change the failure mode, but not cleanly enough to replace the plain-branch follow-up. On the main clip it removes the worst early-stop behavior and raises profile alignment (`0.636` vs `0.437`) and late motion (`1.597` vs `0.487`), but it also overshoots harder (`total_motion_ratio` `1.712`, `peak_motion_ratio` `2.835`), degrades spatial overlap (`0.386`), and raises error on the main and held-out windows.
- A mild h16 motion-weighted continuation from the clean rerun checkpoint is directionally better than both plain checkpoint probing and the older h16 `motion_loss_alpha=1.0` branch, but it still does not solve commitment. Resuming clean h16 step `1000` with `motion_loss_alpha=0.5` to step `1200` improved main-clip MAE (`7.491` vs `8.426`) and spatial overlap (`0.697` vs `0.661`) and made episode `2` much less overactive (`late_motion_ratio` `1.935` vs `3.311`), while keeping the cleaner h16 look. But the main clip and episode `1` still stayed `undercommitted` with `stops_early`.
- The h16 scalar `motion_loss_alpha` sweep now looks exhausted. Raising the clean h16 continuation from `0.5` to `0.75` did not land at a better trade point; it snapped back into the unstable overactive family, with visible distortion on the main clip and episode `2`, main-clip MAE `15.171`, and an explicit plausibility fail on episode `2`.
- That `0.75` failure points to a narrower next lever than more raw alpha tuning: the current motion-aware loss can let a few high-motion latent regions dominate. The smallest grounded code change is to keep motion weighting but cap the maximum per-region multiplier.
- The first capped h16 retry confirms that direction is real but not solved yet. Resuming clean h16 step `1000` with `motion_loss_alpha=0.75` and `motion_loss_max_weight=2.0` removed the uncapped `0.75` blow-up and restored plausibility on all three windows, but it still stayed too motion-heavy on the main clip and episode `2` relative to the cleaner uncapped `0.5` reference (`main late_motion_ratio` `1.074` vs `0.517`, `ep2 late_motion_ratio` `2.514` vs `1.935`) while improving episode `1` fidelity (`mean_frame_mae_rgb_0_255` `6.008` vs `6.506`).
- The residual branch peaks early. Nearby checkpoint probes did not reveal a clear takeover beyond step `800`:
  - Step `900` was mixed rather than better overall.
  - Steps `825`, `850`, `1000`, and `1200` did not beat step `800`.
- The remaining nearby recipe tweaks around that best residual branch are now exhausted:
  - Lower-LR continuation and a fresh lower-LR rerun did not beat the original branch.
  - Higher LoRA capacity at rank `64` did not help.
  - Residual MLP bottlenecks at `64` and `96` did not beat `mlp128resid`; `64` regressed toward undercommitted and stops-early behavior.
  - Re-enabling action-input LayerNorm on the residual branch made motion alignment worse again.
- Subset restriction has not earned its complexity so far. The hand-picked subset run and the top-motion subset run both failed to beat the full-dataset branch.
- `conditioning_mode=none` is a useful image-quality control baseline, but it is not the preferred path for solving the motion-control problem.
- The most likely remaining issue is still weak action-magnitude preservation and/or an architecture-level limitation in how action tokens steer manipulator motion.

## Current Exploration

- The operator-directed reduced-distortion signal on the clean h16 branch still matters, and the new cap is helping: `motion_loss_max_weight=2.0` eliminates the catastrophic uncapped `0.75` instability, but the result is still too overactive on the main clip and episode `2`.
- Best next test: keep the stronger `motion_loss_alpha=0.75` continuation from clean h16 step `1000`, but tighten the cap from `2.0` to `1.5` so the branch can be compared directly against the too-loose capped `2.0` run and the too-weak uncapped `0.5` reference.

## Future Explorations

- If tighter capped motion weighting still cannot improve commitment without reintroducing distortion, switch to a code-level action-conditioning change such as a temporal action encoder per Wan block instead of reopening older width, rank, or continuation sweeps.
- If a future operator review says the plain h16 image-quality gain matters more than motion on a specific downstream use case, compare the clean h16 checkpoints against the normalized image-first branch directly instead of against the motion-first residual branch.

## What Changed

- Added `motion_loss_alpha` to the train config, train CLI, and flow-matching loss so training can upweight moving latent regions directly instead of only evaluating motion after the fact; `source .venv/bin/activate && pytest tests/test_flow_matching.py tests/test_train_world_model_wan_vace.py` passed (`34 passed`).
- Commit `17ba95f` (`Cap motion-aware loss weights`): added an optional `motion_loss_max_weight` cap to the train config, train CLI, and flow-matching loss so motion weighting can stay active without letting a few high-motion latent regions dominate the loss; `source .venv/bin/activate && pytest tests/test_flow_matching.py tests/test_train_world_model_wan_vace.py` passed (`36 passed`).
- Kept the earlier motion-first evaluation stack and prompt updates, then validated that the nearby residual-family sweep remained exhausted:
  - step-`850` continuation from the best residual branch,
  - step-`825` continuation from the best residual branch,
  - fresh rank-`64` residual run,
  - fresh `action_mlp_dim=64` residual run,
  - fresh `action_mlp_dim=96` residual run,
  - fresh residual run with action-input LayerNorm re-enabled.

## What Was Run / Video Verification

- `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid`:
  - step `800`: `arm_motion_verdict: best arm movement so far`; `image_quality_verdict: blurrier than normalized step 1400 but plausible`; `continue_training: no clear continuation win yet`. Why: this is still the strongest motion-first branch even though temporal stability remains mixed.
  - step `825`: `arm_motion_verdict: misaligned and not better than step 800`; `image_quality_verdict: acceptable`; `continue_training: no`. Why: worse than steps `800` and `850` on the explicit gate without a cleaner task-motion win.
  - step `850`: `arm_motion_verdict: same stronger-motion family, slightly more overactive`; `image_quality_verdict: acceptable`; `continue_training: no`. Why: plausible, but no cleaner or more committed task-motion win over step `800`.
  - step `900`: `arm_motion_verdict: mixed`; `image_quality_verdict: mixed`; `continue_training: no`. Why: some temporal trade improvement, but not enough to displace step `800` overall.
  - step `1000`: `arm_motion_verdict: partial recovery only`; `image_quality_verdict: acceptable`; `continue_training: no`. Why: better than step `1200`, still clearly worse than step `800`.
  - step `1200`: `arm_motion_verdict: degraded late`; `image_quality_verdict: regressed`; `continue_training: no`. Why: plain continuation does not preserve the early peak.
- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid`:
  - step `800`: `arm_motion_verdict: undercommitted on the main clip and worse on held-out windows`; `image_quality_verdict: plausible and visibly less distorted`; `continue_training: yes, one direct continuation probe`. Why: later comparison frames show the generated arm pausing near the bowl edge instead of matching the reference's longer inward path, but operator review says this branch keeps a cleaner image regime that still merits direct follow-up.
  - rerun1 step `900`: `arm_motion_verdict: worse undercommitted checkpoint`; `image_quality_verdict: plausible early, but late frames degrade more than step 800 or step 1000`; `continue_training: no`. Why: the arm stops even earlier on the main and episode-1 windows, and episode `2` turns into the noisiest late-sweep variant, with MAE spikes far above the neighboring checkpoints.
  - rerun1 step `1000`: `arm_motion_verdict: still undercommitted on the main clip and episode 1, still overactive on episode 2`; `image_quality_verdict: plausible and still cleaner-looking than the more overactive h16 motion-loss branch`; `continue_training: yes, as the clean resume point for mild motion-weighted follow-ups`. Why: visible main-clip frames still show the arm pausing near the bowl edge instead of taking a clearly longer inward path, and the report changes are only incremental versus step `800` (`late_motion_ratio` `0.525` vs `0.487`, `profile_correlation` `0.422` vs `0.437`, MAE `8.426` vs `8.003`).
- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_motionloss1p0`:
  - step `800`: `arm_motion_verdict: more committed than plain h16, but too overactive and still misaligned`; `image_quality_verdict: plausible, but less clean than plain h16`; `continue_training: no, not before the plain h16 continuation probe`. Why: on the main clip the branch no longer stops early and does move farther along the reference timing, but the arm-crop comparison shows a larger late overshoot and blurrier tool path, while both held-out windows also get higher error and stay in the overactive family.
- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p5_resume1000to1200`:
  - step `1200`: `arm_motion_verdict: cleaner h16 motion-weighted trade, but still undercommitted on the main clip and episode 1`; `image_quality_verdict: best h16 motion-weighted image/stability trade so far`; `continue_training: no, keep as the best uncapped h16 motion-weighted reference`. Why: this continuation improves main-clip MAE and spatial overlap over plain h16 step `1000` and cuts episode-`2` overactivity substantially, but it still stops early on the main and episode-`1` windows; after the failed uncapped `0.75` interpolation it is now the comparison point for capped-motion follow-ups rather than a branch to keep extending directly.
- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p75_resume1000to1200`:
  - step `1200`: `arm_motion_verdict: overactive and artifact-heavy, not a better interpolation point`; `image_quality_verdict: regressed badly with visible distortion and an explicit plausibility fail on episode 2`; `continue_training: no, not without bounding the motion-loss spikes first`. Why: main-clip and episode-`2` comparison frames show late-frame warping and stronger ghosting than the clean h16 family, while the reports jump to main-clip MAE `15.171`, episode-`2` late-motion ratio `3.043`, and a failed plausibility frame with `extreme_color_shift`.
- `optimizer_aloha_static_fork_pick_up_full_320x240_h16_lora32_action_noinputln_mlp128resid_rerun1_motionloss0p75_cap2p0_resume1000to1200`:
  - step `1200`: `arm_motion_verdict: stabilized but still too overactive on the main clip and episode 2`; `image_quality_verdict: much cleaner than uncapped 0.75 and plausible on all three windows, but still not as clean or spatially aligned as the uncapped 0.5 reference`; `continue_training: yes, one tighter-cap follow-up`. Why: the capped run removes the uncapped `0.75` blow-up and restores plausibility on every window, but the main comparison still shows a longer late swing than the reference and episode `1` still stops short; quantitatively it lands between the two uncapped references, with much better plausibility than uncapped `0.75` but worse main and episode-`2` MAE and motion alignment than uncapped `0.5`.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora64_action_noinputln_mlp128resid`:
  - step `800`: `arm_motion_verdict: same family, slightly worse than rank-32 residual`; `image_quality_verdict: acceptable`; `continue_training: no`. Why: extra adapter capacity did not improve the best residual branch.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp64resid`:
  - step `800`: `arm_motion_verdict: undercommitted and stops early`; `image_quality_verdict: cleaner but less useful`; `continue_training: no`. Why: numerically cleaner, but it regresses toward the sharper-but-static family.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp96resid`:
  - step `800`: `arm_motion_verdict: between mlp64 and mlp128, still not better than mlp128`; `image_quality_verdict: acceptable`; `continue_training: no`. Why: stays in the stronger-motion family, but without a better motion-first trade than `mlp128resid`.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_mlp128resid`:
  - step `800`: `arm_motion_verdict: more overactive and less aligned than noinputln residual`; `image_quality_verdict: acceptable`; `continue_training: no`. Why: re-enabling action-input LayerNorm did not produce a better motion-first regime.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid_motionloss1p0`:
  - launch attempt from `configs/train/world_model.yaml`: `arm_motion_verdict: not available`; `image_quality_verdict: not available`; `continue_training: rerun with corrected memory settings`. Why: the first motion-weighted run never reached a checkpoint because it hit CUDA OOM during training. This was a command-shape problem, not a model verdict: the generic base config defaults left gradient checkpointing disabled and did not carry over the ALOHA-safe memory settings used by the stronger earlier runs.
  - corrected rerun at step `800`: `arm_motion_verdict: stronger motion than normalized step 1400 and close to residual step 800, but still slightly overactive`; `image_quality_verdict: plausible, a bit less clean than residual step 800`; `continue_training: no clear continuation win yet`. Why: the motion-weighted branch preserved the desired arm-movement family and slightly increased total and late motion over residual step `800`, but it stayed `misaligned` with `overactive_motion` and did not produce a clean visual takeover.
  - held-out episodes `1` and `2`, start `60`: `arm_motion_verdict: more overactive and less aligned than residual step 800`; `image_quality_verdict: episode 1 acceptable, episode 2 unstable`; `continue_training: no`. Why: on episode `1` the branch moved more but deviated farther from the reference path, and on episode `2` the explicit plausibility backfill failed with `temporal_instability`, so the motion-loss branch does not generalize better than the residual step-`800` anchor.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid_motionloss0p5`:
  - step `800`: `arm_motion_verdict: even more motion, but more misaligned than alpha=1.0`; `image_quality_verdict: plausible, slightly dirtier than alpha=1.0`; `continue_training: no`. Why: lowering the motion-loss strength did not moderate the branch; it increased total and late motion further while reducing profile alignment and spatial overlap, so it is not the clean next continuation target.
- `optimizer_aloha_static_fork_pick_up_full_320x240_lora8_action`:
  - step `1400`: `arm_motion_verdict: undercommitted and stops early`; `image_quality_verdict: best image quality branch`; `continue_training: no`. Why: still the image-first reference, but no longer the preferred training target.
  - step `1500`: `arm_motion_verdict: same undercommitted family`; `image_quality_verdict: no meaningful improvement`; `continue_training: no`. Why: plain continuation on the normalized branch looks exhausted.

## Best Run

- Best current branch under the motion-first ranking: `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` at step `800`.
- Best current branch under the older image-first ranking: `optimizer_aloha_static_fork_pick_up_full_320x240_lora8_action` at step `1400`.
- Practical ranking takeaway:
  - use residual step `800` as the motion reference,
  - use normalized step `1400` as the image-quality reference,
  - use clean h16 rerun step `1000` as the reduced-distortion resume point and h16 `motion_loss_alpha=0.5` step `1200` as the best h16 motion-weighted reference so far,
  - treat capped h16 `motion_loss_alpha=0.75`, `motion_loss_max_weight=2.0` as proof that the cap helps, but not yet as the best h16 trade,
  - do not spend another loop on nearby continuation, LR tuning, width sweeps, subset restriction, or action-input-LayerNorm restoration around the old `horizon_len=8` residual family.
- If optimization resumes later, the next grounded move is the same capped h16 continuation but with `motion_loss_max_weight=1.5`; if that still fails, switch to a stronger action-conditioning path.
