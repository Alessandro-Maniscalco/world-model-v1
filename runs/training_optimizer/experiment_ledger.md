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

- Longer-context short-horizon family:
  - `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800` is the first longer-context probe worth continuing.
  - It kept all three windows plausible, improved held-out stability over both the residual and h16 references, and produced a `good` arm-motion verdict on episode `2`.
  - It still lagged early and then caught up late on the main clip and episode `1`, so it did not beat the residual step-`800` motion-first anchor.
  - The follow-up context-only continuation to `context_len=29` is exhausted after the step-`400` comparison: it stayed even more static through the rollout than `ctx21/h8` step `400`, so its lower error came from default-pose persistence rather than earlier task motion.
  - The follow-up short-horizon `action_temporal_difference_scale=0.5` resume from step `800` to step `1000` is also exhausted. It reduced main-clip MAE and pulled some motion off the very end, but it stayed misaligned on the main clip, failed plausibility on episode `1`, and downgraded episode `2` from `good` to `misaligned`.
  - The follow-up short-horizon temporal action-token mixer resume from step `800` to step `1000` is also exhausted. It restored plausibility on all three windows and looked cleaner than the action-delta branch, but the rollout still held too static for too long and then finished with the same late-heavy motion pattern on the main clip, episode `1`, and episode `2`.
  - The plain short-horizon `motion_loss_alpha=0.5` resume from step `800` to step `1000` did not break that pattern either. It stayed plausible on all three windows and improved image metrics slightly, but the generated fork still waited too long to commit on the main clip and episode `1`, while episode `2` remained visibly late and still graded `misaligned`.
  - The excess-only short-horizon `motion_loss_alpha=0.5` resume from step `800` to step `1000` is also exhausted. It did not pull motion earlier than the plain motion-loss run, worsened main and episode-`1` image error, and still finished with the same late-heavy misalignment, so the next distinct move needs a direct code-level timing bias instead of another motion-loss-shaping retry.

## Recent Validation Notes

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_linearinit_resume800to1000` step `1000`
  - `arm_motion_verdict`: nonzero-init projector rerun still late-heavy and `misaligned` on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: all three windows remain plausible and relatively clean, but there is no visible earlier commitment win
  - `continue_training`: no more projector-init-only retries; move to a train-side projector-supervision change
  - Why: the last-horizon comparison sheets and arm crops still show the same long static hold followed by late fork motion. The reports confirm that the branch stayed `misaligned` across all three windows (`late_motion_ratio≈2.00/1.43/2.56`, `mean_frame_mae≈2.86/2.66/2.39`), and validation was much weaker than the zero-init rerun (`best_val_loss≈0.1186` at step `950`, `val_loss≈0.3950` at step `1000` versus `best_val_loss≈0.0283` on the zero-init branch). That closes projector-init alone as a rescue and justifies the new direct projector-supervision lever in commit `81fd6b1`.

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

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800`
  - `arm_motion_verdict`: stronger and cleaner short-horizon memory probe, but still late-heavy on the main clip and episode `1`
  - `image_quality_verdict`: clean and plausible on all three windows, with clearly better held-out stability than the old residual reference
  - `continue_training`: yes, but only as one more context-only follow-up at `context_len=29`
  - Why: the main clip stayed visually coherent and plausible while the generated fork still lagged through the middle of the rollout and then surged late near the plate, episode `1` showed the same late catch-up pattern, and episode `2` was the best held-out result in this neighborhood so far with a `good` arm-motion verdict and no plausibility issues.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `400` versus `optimizer_aloha_static_fork_pick_up_full_320x240_ctx29_h8_lora32_action_noinputln_mlp128resid` step `400`
  - `arm_motion_verdict`: both are still misaligned, but `ctx29/h8` is not the wanted kind of improvement
  - `image_quality_verdict`: `ctx29/h8` is cleaner and lower-MAE than `ctx21/h8` at step `400`
  - `continue_training`: no direct resume of the paused `ctx29/h8` step-`600` branch
  - Why: in the comparison video, `ctx29/h8` holds the fork near the default pose for essentially the whole window, while `ctx21/h8` at least shows some late fork drift near the end. The metrics agree on the trade: `ctx29/h8` reduced main MAE from about `4.04` to `3.01` and improved profile correlation from about `0.17` to `0.47`, but it still failed motion-first ranking because the visible task motion did not start earlier and the generated rollout remained undercommitted by mid-training.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_actiondelta0p5_resume800to1000` step `1000`
  - `arm_motion_verdict`: slight main-clip cleanup but still misaligned, with worse held-out behavior
  - `image_quality_verdict`: cleaner than plain `ctx21/h8` on the main clip, but episode `1` now hits a plausibility failure and late-rollout artifact
  - `continue_training`: no; close the short-horizon action-delta interaction neighborhood
  - Why: the main comparison video still holds the fork mostly static until late in the rollout, only then drifting toward the plate, while episode `1` develops a visible end-of-rollout burst and frame-21 plausibility failure. Episode `2` remains passable but no longer carries the earlier branch's `good` motion verdict, so the overall motion-first trade regressed relative to the plain `ctx21/h8` anchor.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_actiontempmixk3s0p5_resume800to1000` step `1000`
  - `arm_motion_verdict`: plausible and cleaner than the action-delta follow-up, but still misaligned and late-heavy on all three windows
  - `image_quality_verdict`: clean and plausible on the main clip plus held-out episodes `1` and `2`
  - `continue_training`: no; close the short-horizon action-conditioning neighborhood and move to a loss-side lever
  - Why: the comparison sheets show the same basic failure mode as the plain `ctx21/h8` anchor: the generated fork stays near the default pose for most of the rollout, then adds a late burst near the plate instead of committing earlier. The metrics support that read rather than overturning it: all three windows stayed plausible, but the main clip still had `late_motion_ratio≈2.06` and `profile_correlation≈0.30`, episode `1` still had `late_motion_ratio≈1.48`, and episode `2` remained `misaligned` with `late_motion_ratio≈2.39`, so there is no motion-first win that justifies another action-token follow-up.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_motionloss0p5_resume800to1000` step `1000`
  - `arm_motion_verdict`: plausible loss-side follow-up, but still misaligned and late-heavy on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: clean and plausible on all three windows, with slightly better main and episode-`2` MAE than the short-horizon temporal-mixer follow-up
  - `continue_training`: one more narrow loss-shaping follow-up is justified, but only as `motion_loss_excess_only=true`
  - Why: the comparison sheets still show the generated fork holding near the default pose through most of the rollout, then only moving late near the end. The arm-crop view does not reveal an earlier commitment win hidden by the full frame. The metrics agree with that motion-first read: the main clip stayed `misaligned` with `late_motion_ratio≈2.16`, episode `1` stayed `misaligned` with `late_motion_ratio≈1.53`, and episode `2` stayed `misaligned` with `late_motion_ratio≈2.35`, even though all three windows remained plausible and episode `2` MAE improved to about `2.10`. That keeps the loss-side branch alive for one final excess-only weighting probe, but not for another plain scalar continuation.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_motionloss0p5_excessonly_resume800to1000` step `1000`
  - `arm_motion_verdict`: excess-only loss shaping still misaligned and late-heavy on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: plausible on all three windows, but worse than the plain `motion_loss_alpha=0.5` resume on the main clip and clearly worse on episode `1`
  - `continue_training`: no; close the short-horizon motion-loss-shaping neighborhood and move to a direct timing-bias code lever
  - Why: the comparison sheets and arm crops still show the fork holding near the default pose for most of the rollout, with no earlier commitment win over the plain motion-loss run. The summaries confirm that regression: the main clip stayed `misaligned` with `late_motion_ratio≈2.19` and MAE rising from about `2.87` to `3.00`, episode `1` stayed `misaligned` with MAE jumping from about `2.71` to `3.17`, and episode `2` stayed `misaligned` with worse late-motion ratio and MAE than the plain motion-loss run. That closes the short-horizon loss-shaping branch.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_futurelossbias0p5_resume800to1000` step `1000`
  - `arm_motion_verdict`: plausible timing-bias probe, but still misaligned and late-heavy on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: plausible on all three windows, but not a motion-first improvement over the plain `ctx21/h8` anchor or the plain `motion_loss_alpha=0.5` follow-up
  - `continue_training`: no direct continuation from step `1000`; first evaluate step `900` inside the same validated branch
  - Why: the full comparison sheets and arm crops still show a long static hold near the default pose followed by late motion only near the end of the rollout, so the new timing bias did not visibly pull commitment earlier. The summaries support that read: all three windows stayed plausible, but the main clip remained `misaligned` with `late_motion_ratio≈2.14` and `profile_correlation≈0.26`, episode `1` remained `misaligned` with `late_motion_ratio≈1.56`, and episode `2` remained `misaligned` with `late_motion_ratio≈2.48`. The validation history changes the next decision, though: `best_val_loss` improved from about `0.199` at step `850` to `0.0586` at step `900` to `0.0332` at step `950`, then regressed sharply to `0.2686` at step `1000`, so the next bounded action is checkpoint selection at step `900` rather than another new scalar.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_futurelossbias0p5_resume800to1000` step `900`
  - `arm_motion_verdict`: failed checkpoint-selection rescue; still misaligned on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: worse than the same branch at step `1000` on the main clip and episode `1`, and episode `2` now fails plausibility
  - `continue_training`: no; close the short-horizon timing-bias neighborhood and move to a broader temporal-window change
  - Why: the comparison sheets and arm crops still show long static persistence followed by late motion rather than earlier commitment, so the checkpoint chosen before the step-`1000` validation collapse did not recover a motion-first win. The summaries confirm that regression: the main clip stayed `misaligned` with `late_motion_ratio≈1.79` but MAE worsened to about `4.03`, episode `1` stayed `misaligned` with MAE worsening to about `4.50`, and episode `2` became implausible with `mean_frame_mae≈4.13` and `temporal_delta_ratio≈2.31`. That exhausts the `future_loss_early_bias=0.5` timing-bias branch, including checkpoint selection, and makes a broader horizon change the next distinct test.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_lora32_action_noinputln_mlp128resid` step `800`
  - `arm_motion_verdict`: broader temporal-window probe is still misaligned and more late-heavy than the `ctx21/h8` anchor on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: plausible on all three windows, but clearly worse than `ctx21/h8` step `800` on MAE and especially worse on episode `2`
  - `continue_training`: no direct continuation from step `800`; first evaluate the validation-best checkpoint at step `700`
  - Why: the comparison sheets and arm crops still show the fork holding too long and then moving late, without any visible early-commitment gain from the longer horizon. The summaries align with that read: the main clip stayed `misaligned` with `late_motion_ratio≈2.83` and MAE rising to about `3.80` versus about `2.17` on `ctx21/h8`, episode `1` stayed `misaligned` with MAE worsening to about `4.00`, and episode `2` lost the earlier branch's `good` verdict and regressed to `misaligned` with MAE around `2.64`. The validation trace explains the next action: `best_val_loss` reached about `0.0303` at step `700`, then worsened to about `0.1001` at step `750` and `0.2271` at step `800`, so checkpoint selection at step `700` is the only justified follow-up before closing the `ctx21/h12` neighborhood.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_lora32_action_noinputln_mlp128resid` step `700`
  - `arm_motion_verdict`: failed checkpoint-selection rescue; still misaligned and late-heavy on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: plausible on all three windows, but not a recovery over step `800` and still clearly worse than `ctx21/h8` step `800`
  - `continue_training`: no; close the broader `ctx21/h12` temporal-window neighborhood and move to a different temporal-coverage change
  - Why: the comparison sheets and arm crops still show the same basic failure mode as step `800`: long static persistence and late task motion, with no earlier commitment win over the `ctx21/h8` anchor. The summaries confirm that read: the main clip stayed `misaligned` with MAE worsening further to about `4.26`, episode `1` improved only marginally on MAE but remained visibly late-heavy, and episode `2` regressed further with `late_motion_ratio≈3.20` and `mean_frame_mae≈3.30`. Because both step `800` and the validation-best step `700` failed to improve the family anchor, the `ctx21/h12` temporal-window sweep is exhausted. The next distinct test should keep the stable `ctx21/h8` geometry but widen temporal coverage by increasing `k` from `1` to `2`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_k2_lora32_action_noinputln_mlp128resid` launch attempt
  - `arm_motion_verdict`: none; training never started far enough to produce a checkpoint or validation video
  - `image_quality_verdict`: none; no generated artifacts were created
  - `continue_training`: blocked in this exact form; resume later with a valid schedule such as `ctx21/h12/k2` or after a code-level schedule change
  - Why: the latest long command exited immediately during schedule validation, before any checkpoint or evaluation artifact existed. The stderr log shows the concrete blocker: `horizon_len=8` compresses to `horizon_latent_steps=2`, but `k=2` requires at least `3` latent future steps, so the run raises `ValueError: Invalid latent-time schedule ... Increase horizon_len or reduce k.` This means there was no new model result to rank, only a configuration blocker to record before stopping.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_k2_lora32_action_noinputln_mlp128resid` step `400`
  - `arm_motion_verdict`: plausible multi-chunk probe, but still late-heavy and `misaligned` on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: clean and plausible on all three windows, but not a motion-first win over the `ctx21/h8` step-`800` anchor
  - `continue_training`: no plain continuation; move to one bounded structural follow-up in the same neighborhood
  - Why: the comparison sheets and arm crops still show a long static hold followed by late fork motion, even though the multi-chunk rollout remains plausible. The reports support that read: main `late_motion_ratio≈2.85`, episode `1≈1.54`, episode `2≈2.57`, while validation peaked around `0.0704` at step `300` before regressing by step `400`. That keeps the neighborhood alive only for one direct structural follow-up.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_k2_lora32_action_noinputln_mlp128resid_chunkbias1p0` step `400`
  - `arm_motion_verdict`: chunk-position early weighting failed to pull motion earlier; the main clip plus held-out episodes `1` and `2` all stayed `misaligned`
  - `image_quality_verdict`: plausible on all three windows, but still visibly late-heavy and not better than the plain `ctx21/h12/k2` probe
  - `continue_training`: no more weighting tweaks in this neighborhood; pivot to reducing future-prefix teacher forcing
  - Why: the comparison sheets and arm crops still show the same long static hold and late catch-up pattern. The reports confirm there is no motion-first win: main `late_motion_ratio≈2.85`, episode `1≈1.48`, episode `2≈2.68`, and validation worsened versus the unmodified `ctx21/h12/k2` run, bottoming near `0.0926` at step `300` before regressing to about `0.1960` at step `400`. That exhausts chunk-position weighting and makes future-prefix leakage the next structural target.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_k2_lora32_action_noinputln_mlp128resid_pastonly` step `400`
  - `arm_motion_verdict`: removing the ground-truth future prefix without rollout feedback made timing worse; the main clip plus held-out episodes `1` and `2` all stayed `misaligned`
  - `image_quality_verdict`: plausible on all three windows, but still visibly late-heavy and not a motion-first win over the plain or chunk-biased `ctx21/h12/k2` runs
  - `continue_training`: no more `past_only` follow-ups; the only justified next move in this family is a stronger rollout-prefix structural change
  - Why: the comparison sheets still show a long static hold and only late fork motion, with no visible early-commitment gain. The reports make that regression explicit: main `late_motion_ratio≈3.01`, episode `1≈1.84`, and episode `2≈3.93`, all worse than the plain multi-chunk probe, even though all three windows remained plausible. Validation recovered somewhat versus `chunkbias1p0` (`best_val_loss≈0.0788` at step `300`) but still regressed to about `0.1293` at step `400`, so plain future-prefix removal is exhausted and rollout-style prefix feedback becomes the next structural target.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_k2_lora32_action_noinputln_mlp128resid_predprefix` step `400`
  - `arm_motion_verdict`: detached predicted-prefix feedback still failed to pull motion earlier; the main clip plus held-out episodes `1` and `2` all stayed `misaligned`
  - `image_quality_verdict`: plausible on all three windows, with slightly cleaner MAE than earlier `ctx21/h12/k2` follow-ups, but still not a motion-first win
  - `continue_training`: no more `ctx21/h12/k2` rollout-structure follow-ups; move the same structural lever to a different neighborhood or pivot away from this family
  - Why: the comparison sheets and arm crops still show a long static hold followed by late fork motion, so detached rollout-prefix feedback did not recover visible early commitment. The reports confirm that read: main `late_motion_ratio≈3.36`, episode `1≈2.05`, and episode `2≈3.55`, all still clearly `misaligned`, while validation only matched the `past_only` branch near step `300` (`best_val_loss≈0.0781`) before regressing to about `0.1225` at step `400`. That exhausts the `ctx21/h12/k2` rollout-structure family and justifies moving the lever to the stronger `ctx21/h8` anchor instead of spending another follow-up here.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_predprefix_resume800to1000` step `1000`
  - `arm_motion_verdict`: rollout-prefix teacher forcing still failed to pull motion earlier; the main clip plus held-out episodes `1` and `2` all stayed `misaligned`
  - `image_quality_verdict`: plausible on the main clip and episode `2`, but episode `1` now fails plausibility on frame `21`
  - `continue_training`: no; treat rollout-prefix teacher forcing as exhausted on the best short-horizon anchor and pivot to a different lever
  - Why: the comparison sheets and arm crops still show the same long static hold followed by late fork motion that blocked the earlier branches, with no motion-first gain over the plain `ctx21/h8` step-`800` anchor. The reports confirm that read: main `late_motion_ratio≈2.46`, episode `1≈1.55`, and episode `2≈3.54`, all `misaligned`, while episode `1` fails plausibility and validation only improves to about `0.0332` at step `950` before regressing sharply to about `0.2456` at step `1000`. That closes rollout-prefix teacher forcing on both the `ctx21/h12/k2` and `ctx21/h8` neighborhoods. The next bounded action should keep the still-plausible `ctx21/h12/k2` coverage branch but switch to a different family, starting with `motion_loss_alpha=0.5`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h12_k2_lora32_action_noinputln_mlp128resid_motionloss0p5` step `400`
  - `arm_motion_verdict`: motion-loss weighting keeps the multi-chunk run plausible, but it still leaves the main clip plus held-out episodes `1` and `2` visibly late-heavy and `misaligned`
  - `image_quality_verdict`: plausible on all three windows; main-clip MAE improves versus plain `ctx21/h12/k2`, but episode `1` image error worsens and episode `2` does not recover a cleaner held-out timing trade
  - `continue_training`: no checkpoint-selection or continuation follow-up; close the low-risk `ctx21/h12/k2` neighborhood and move to a longer valid `k=2` schedule
  - Why: the comparison sheets and arm crops still show a long static hold followed by late fork motion, with no clear visual commitment win over the plain `ctx21/h12/k2` probe. The reports support that read: main `late_motion_ratio≈2.25` improves over the plain branch's `≈2.85`, but episode `1` only nudges to `≈1.45` while MAE worsens from about `4.87` to `5.22`, and episode `2` regresses to `late_motion_ratio≈2.94` from `≈2.57`. The validation curve is also weaker than the plain branch at every checkpoint, bottoming near `0.1063` at step `300` versus about `0.0704`. That makes this a safety-preserving but non-decisive trade, not a motion-first winner. The next distinct bounded action is a longer valid two-chunk schedule, `ctx21/h16/k2`, to test whether `h12` simply compressed away too much future coverage.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid` step `400`
  - `arm_motion_verdict`: best motion timing so far inside the valid `k=2` family, but still `misaligned` on the main clip plus held-out episodes `1` and `2`
  - `image_quality_verdict`: main clip and episode `1` stay plausible, but the run is much blurrier than earlier `k=2` probes and episode `2` now fails plausibility on frame `21`
  - `continue_training`: no plain continuation to step `800`; first evaluate the validation-best checkpoint at step `300`
  - Why: the comparison sheets and arm crops show somewhat earlier fork drift on the main clip and episode `1` than the `ctx21/h12/k2` family, and the reports support that timing gain (`late_motion_ratio≈1.90` main and `≈1.08` episode `1`). But that motion win comes with a major image-quality penalty: main `mean_frame_mae≈7.33`, episode `1≈6.09`, and episode `2` fails plausibility with `mean_frame_mae≈7.68`. The validation curve also explains the next action: `best_val_loss` improved to about `0.0577` at step `300` before regressing to about `0.1052` at step `400`, so checkpoint selection at step `300` is the smallest decisive follow-up before either preserving or closing this remaining `k=2` branch.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid` step `300`
  - `arm_motion_verdict`: checkpoint selection preserves the branch's earlier timing on the main clip and episode `1`, but all three windows still stay `misaligned`
  - `image_quality_verdict`: main clip and episode `1` remain plausible but still noticeably blurry, and episode `2` still fails plausibility on frame `21`
  - `continue_training`: no more plain chunk-local `k=2` follow-ups; pivot to a different conditioning family while keeping the earlier-moving `ctx21/h16/k2` geometry
  - Why: the step-`300` comparison sheets still show a long static hold followed by late fork motion, even though timing improves slightly versus step `400`. The reports make the trade concrete: main `late_motion_ratio≈1.62` and episode `1≈0.95` are the best timing values seen inside the valid `k=2` family, but main `mean_frame_mae≈6.88` and episode `1≈9.07` stay blurry, and episode `2` still fails plausibility with `failing_frame_indices=[21]` despite a lower `mean_frame_mae≈5.93`. Because the validation-best checkpoint still misses the held-out safety gate, the plain chunk-local `k=2` branch is exhausted. The next bounded action should keep the promising `ctx21/h16/k2` geometry but switch to full-plan ordered action conditioning without the broadcast prior.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid_fullplan_ordered_noprior` step `350`
  - `arm_motion_verdict`: ordered full-plan action conditioning did not preserve the earlier timing from plain `ctx21/h16/k2`; the main clip plus held-out episodes `1` and `2` all stayed `misaligned`
  - `image_quality_verdict`: main clip and episode `1` remain plausible, but both still look blurry and episode `2` still fails plausibility on frame `21`
  - `continue_training`: no more local `k=2` full-plan action-conditioning follow-ups; pivot to the train/infer mismatch in teacher-forced future inputs
  - Why: the comparison sheets still show the same long static hold followed by late fork motion, and this branch is visibly later than plain `ctx21/h16/k2` step `300` on the main clip and episode `1`. The reports confirm that regression: main `late_motion_ratio≈2.39` versus `≈1.62` on plain step `300`, episode `1≈1.41` versus `≈0.95`, and episode `2≈3.79` still fails plausibility with `failing_frame_indices=[21]`. Validation also did not justify another continuation: it bottomed at about `0.0311` on step `200`, then regressed to about `0.1753` and triggered early stop at step `350`. Because full-plan visibility did not preserve the only earlier-moving valid `k=2` geometry, the next distinct bounded action is a code-level rollout-matching change rather than another action-plan scalar.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid_activechunktf` step `400`
  - `arm_motion_verdict`: rollout-matched future inputs recover some timing versus the ordered full-plan branch, but the main clip plus held-out episodes `1` and `2` all still stay `misaligned`
  - `image_quality_verdict`: episode `1` is materially cleaner than plain `ctx21/h16/k2` step `300`, but the main clip is still blurry and episode `2` still fails plausibility on frame `21`
  - `continue_training`: no blind continuation beyond step `400`; first evaluate the validation-best checkpoint at step `300`
  - Why: the comparison sheets still show a long static hold followed by late fork motion, but the active-chunk run partially reverses the timing regression from ordered full-plan conditioning. The reports make the trade concrete: main `late_motion_ratio≈1.85` improves over ordered full-plan `≈2.39` but still trails plain `ctx21/h16/k2` step `300` at `≈1.62`; episode `1≈1.12` improves over ordered full-plan `≈1.41` but still trails plain step `300` at `≈0.95`; and episode `2≈2.80` still fails plausibility with `failing_frame_indices=[21]`. Validation again peaks earlier than the final checkpoint, bottoming near `0.0576` at step `300` before regressing to about `0.1050` at step `400`. That makes step-`300` checkpoint selection the smallest decisive follow-up before closing or keeping the new lever.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid_activechunktf` step `300`
  - `arm_motion_verdict`: validation-best checkpoint modestly improves main and episode-`1` timing over plain `ctx21/h16/k2` step `300`, but all three windows still stay `misaligned`
  - `image_quality_verdict`: main and episode `1` are slightly cleaner than plain `ctx21/h16/k2` step `300`, but episode `2` still fails plausibility on frame `21`
  - `continue_training`: no more teacher-forcing-only follow-ups in this `k=2` geometry; pivot to a non-teacher-forcing follow-up
  - Why: the step-`300` reports show the best trade inside the `active_chunk` branch: main `late_motion_ratio≈1.31` improves over plain `ctx21/h16/k2` step `300` at `≈1.62`, episode `1≈0.91` improves slightly over plain `≈0.95`, and main / episode-`1` MAE both edge down. But the held-out blocker does not move: episode `2` still fails plausibility with `failing_frame_indices=[21]`, `late_motion_ratio≈2.08`, and `mean_frame_mae≈5.93`, which trips the active decision's exit condition. That closes rollout-matched future inputs as a teacher-forcing-only fix and makes `motion_loss_alpha=0.5` on plain `ctx21/h16/k2` the smallest next bounded test.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid_motionloss0p5` step `400`
  - `arm_motion_verdict`: the motion-loss follow-up regresses sharply on the main clip and still leaves all three windows `misaligned`
  - `image_quality_verdict`: main and episode `1` stay plausible, but the main clip gets blurrier and later while episode `2` still fails plausibility on frame `21`
  - `continue_training`: no more training continuation from step `400`; first evaluate the validation-best checkpoint at step `300`
  - Why: the reports show no final-checkpoint rescue. Main timing swings far later than either plain or `active_chunk` step `300` (`late_motion_ratio≈3.00` versus `≈1.62` and `≈1.31`), main MAE rises to about `7.36`, episode `1` regresses to `late_motion_ratio≈1.15`, and episode `2` still fails plausibility with `failing_frame_indices=[21]`, `late_motion_ratio≈2.71`, and `mean_frame_mae≈7.01`. The validation curve still peaks earlier at step `300` (`best_val_loss≈0.0858`) before regressing to about `0.1527` at step `400`, so checkpoint selection is the only remaining bounded rescue before closing the local scalar neighborhood.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k2_lora32_action_noinputln_mlp128resid_motionloss0p5` step `300`
  - `arm_motion_verdict`: checkpoint-selection failed to rescue the branch; the main clip plus episode `1` stayed `misaligned` and later than both plain and `active_chunk` step `300`, while episode `2` remained `misaligned`
  - `image_quality_verdict`: main and episode `1` stayed plausible but blurrier and higher-MAE than the plain and `active_chunk` step-`300` baselines, and episode `2` still failed plausibility on frame `21`
  - `continue_training`: no; close the low-risk `ctx21/h16/k2` scalar neighborhood and move to a new chunk-local temporal-coverage geometry
  - Why: the validation-best checkpoint still misses the branch exit gate. Main `late_motion_ratio≈1.78`, `profile_correlation≈0.07`, and `mean_frame_mae≈8.19` trail plain `ctx21/h16/k2` step `300` (`≈1.62`, `≈0.31`, `≈6.88`) and `activechunktf` step `300` (`≈1.31`, `≈0.39`, `≈6.62`). Episode `1` likewise regresses to `late_motion_ratio≈1.14` and `mean_frame_mae≈10.73` versus plain `≈0.95` / `≈9.07` and `active_chunk≈0.91` / `≈8.96`. Episode `2` still fails plausibility on frame `21`, with `late_motion_ratio≈2.46` and `mean_frame_mae≈6.97`, so the local scalar follow-up is exhausted. The smallest remaining plain chunk-local test is `ctx21/h16/k3`, which changes chunk granularity on the same valid four-step latent horizon without another scalar or teacher-forcing tweak.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k3_lora32_action_noinputln_mlp128resid` step `300`
  - `arm_motion_verdict`: the plain `k=3` geometry restores held-out plausibility, but all three windows stay `misaligned` and the main clip plus episode `1` move later than the best `active_chunk` `k=2` checkpoint
  - `image_quality_verdict`: all three windows are plausible and cleaner than the best `k=2` checkpoints, especially on episode `2`, but the motion-first ranking still loses because task-relevant movement starts too late
  - `continue_training`: no more plain chunk-local coverage follow-ups; keep the safer `k=3` geometry and change the future-input lever instead
  - Why: the run early-stopped at step `300` with `best_val_loss≈0.1143`, then regressed to `≈0.1401`, so the final checkpoint is already the validation-best checkpoint. That checkpoint does improve held-out safety versus the `k=2` family: main, episode `1`, and episode `2` all pass plausibility, with MAE around `5.12`, `7.87`, and `5.66`, whereas both plain and `active_chunk` `k=2` step `300` still failed episode `2` on frame `21`. But the motion trade is not good enough: main `late_motion_ratio≈1.67` regresses from plain `k=2≈1.62` and `active_chunk k=2≈1.31`, episode `1≈1.91` regresses badly from plain `≈0.95` and `active_chunk≈0.91`, and episode `2≈3.61` is still visibly late despite the plausibility pass. That closes plain chunk-local temporal coverage as an architecture family and makes `teacher_forcing_future_input_mode=active_chunk` on the safer `ctx21/h16/k3` geometry the smallest next bounded test.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h16_k3_lora32_action_noinputln_mlp128resid_activechunktf` step `300`
  - `arm_motion_verdict`: rollout-matched future inputs do not rescue the safer `k=3` geometry; all three windows stay `misaligned`, and timing is slightly worse than plain `k=3`
  - `image_quality_verdict`: all three windows remain plausible, but MAE also ticks up slightly versus plain `k=3`, so the run does not win on either motion-first ranking or image quality
  - `continue_training`: no; h16 rollout-matching on the current chunk schedule is exhausted, so the next resumed move should be a code-level schedule or architecture change rather than another long run in this family
  - Why: the run early-stopped at step `300` with the same validation pattern as plain `k=3`: `best_val_loss≈0.1142` at step `150`, then drift to `≈0.1401` by step `300`. The held-out safety gain is preserved, because main, episode `1`, and episode `2` all still pass plausibility, but the timing trade worsens instead of improving: main `late_motion_ratio≈1.69` versus plain `k=3≈1.67`, episode `1≈2.11` versus `≈1.91`, and episode `2≈3.76` versus `≈3.61`. MAE also rises slightly on all three windows, from about `5.12/7.87/5.66` to `5.25/8.09/5.76`. That closes rollout-matched future inputs on the safer `ctx21/h16/k3` geometry and, together with the exhausted `ctx21/h16/k2` `active_chunk` branch, leaves code-level schedule or architecture changes as the next rational move when this loop resumes.

- `e271c40` `Add exact-k chunk schedule mode`
- `703a306` `Add action-token output scale`
- `16c8f47` `Plumb action token scale through local sweeps`

- `703a306` `Add action-token output scale`
  - `hypothesis`: raw action scaling may be getting attenuated before Wan cross-attention, so a direct post-projection action-token gain is a smaller and more relevant structural lever than another raw action sweep.
  - `implementation`: added `action_token_scale` to train/infer configs, CLI plumbing, runtime checkpoint restoration, and `ActionTokenEncoder`, where it scales projected action tokens after optional order-conditioning, temporal-difference, and temporal-mixer logic but before Wan cross-attention.
  - `validation`: `source .venv/bin/activate && pytest tests/test_config_defaults.py tests/test_wan_vace_conditioning.py tests/test_wan_vace_factory.py tests/test_train_world_model_wan_vace.py tests/test_infer_world_model_wan_vace.py` passed (`108 passed`), and both `python scripts/train/world_model.py --help` and `python scripts/train/infer_world_model.py --help` expose `--action-token-scale`.
  - `next action`: run the canonical single-chunk `ctx21/h8` step-`800` checkpoint with `--action-token-scale 2.0` on episode `0` / start `60` to test whether stronger post-projection action tokens can finally change the late-heavy rollout.
  - `hypothesis`: the strongest remaining chunk-count test is still the blocked short-horizon `ctx21/h8/k2` probe, and that block came from the hardcoded `k+1` latent schedule rather than model quality. Allowing exact-`k` chunk schedules should unlock that test on the best short-horizon anchor without disturbing the default training or inference path.
  - `implementation`: added `chunk_schedule_mode` to chunk scheduling, train/infer configs, CLI plumbing, chunkwise training, inference rollout, and checkpoint sweeps; preserved `k_plus_one` as the default; and added focused pytest coverage for schedule construction, train-side validation, flow-matching chunk windows, config defaults, and infer parser wiring.
  - `validation`: `source .venv/bin/activate && pytest tests/test_chunking_schedule.py tests/test_config_defaults.py tests/test_flow_matching.py tests/test_train_world_model_wan_vace.py tests/test_infer_world_model_wan_vace.py` passed (`103 passed`), `python scripts/train/world_model.py --help | rg "chunk-schedule-mode|teacher-forcing-future-input-mode"` showed the new CLI, and `PYTHONPATH=src python - <<'PY' ... build_chunk_schedule(future_steps=2, k=2, chunk_schedule_mode=\"k_chunks\") ...` returned `((0, 1), (1, 2))`.
  - `next action`: run the first exact-`k` short-horizon probe by resuming the validated `ctx21/h8` step-`800` anchor to step `1000` with `--k 2 --chunk-schedule-mode k_chunks`, then evaluate the main clip plus held-out episodes `1` and `2`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_k2_exactchunks_lora32_action_noinputln_mlp128resid_resume800to1000` step `1000`
  - `arm_motion_verdict`: exact-`k` chunking keeps the short-horizon branch plausible on the main clip plus held-out episodes `1` and `2`, but all three windows still stay `misaligned` with the same long static hold followed by late fork motion
  - `image_quality_verdict`: cleaner than most earlier short-horizon follow-ups and plausible on all three windows, but not yet a motion-first win over the plain `ctx21/h8` anchor
  - `continue_training`: no blind continuation beyond step `1000`; first evaluate the saved step-`900` checkpoint before deciding whether the exact-`k` rescue stays alive
  - Why: the reports show a safety-preserving but still late-heavy trade. Main `late_motion_ratio≈2.04`, episode `1≈1.47`, and episode `2≈2.38` all remain `misaligned`, even though plausibility stays `PASS` on all three windows and MAE remains relatively low at about `2.91/2.66/2.28`. The training trace explains the next action: validation improves from about `0.1421` at step `850` to `0.0541` at step `900` to `0.0282` at step `950`, then collapses to `0.2363` at step `1000`, but only checkpoints `900` and `1000` were saved. That makes step-`900` checkpoint selection the smallest decisive follow-up before either keeping or closing the exact-`k` short-horizon branch.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_k2_exactchunks_lora32_action_noinputln_mlp128resid_resume800to1000` step `900`
  - `arm_motion_verdict`: checkpoint-selection fails the plain exact-`k` rescue; main and episode `1` move slightly earlier than step `1000`, but all three windows remain `misaligned` and episode `2` fails plausibility
  - `image_quality_verdict`: main and episode `1` remain cleaner than many older short-horizon branches, but all three windows regress on MAE versus step `1000`, and episode `2` now fails on frames `21` and `22`
  - `continue_training`: no more plain exact-`k` checkpoint selection or continuation; keep the short-horizon exact-`k` neighborhood alive only for one distinct structural follow-up
  - Why: step `900` improves main `late_motion_ratio` from `≈2.04` to `≈1.83` and episode `1` from `≈1.47` to `≈1.35`, but main / episode `1` / episode `2` MAE all worsen from about `2.91/2.66/2.28` to `3.94/4.37/4.10`, and episode `2` becomes implausible with `failing_frame_indices=[21, 22]`. That misses the checkpoint-selection success signal, so the plain exact-`k` rescue is exhausted. The only remaining bounded hypothesis in this neighborhood is rollout-matched future inputs (`teacher_forcing_future_input_mode=active_chunk`) on the same short-horizon exact-`k` anchor.

## Code-Change Ledger

- `17ba95f` `Cap motion-aware loss weights`
- `7fe8994` `Add excess-only motion loss weighting`
- `7832e4a` `Add temporal-difference action residual`
- `5537878` `Add temporal action-token mixer`
- `7cba14d` `Add early-horizon loss bias`
- `614c605` `Add early chunk loss bias`
- `07281db` `Add past-only teacher forcing mode`
- `48f2883` `Add predicted-prefix teacher forcing`
- `cc64dce` `Match teacher forcing future inputs to rollout`
- `e271c40` `Add exact-k chunk schedule mode`

## Latest Controls

- `base_prompt_singlechunk_ep0_start60`
  - `arm_motion_verdict`: plausible upstream-style smoke path, but still `undercommitted`
  - `image_quality_verdict`: coherent and plausible, but much blurrier and higher-MAE than the trained action checkpoint
  - `continue_training`: no direct continuation; keep only as the no-checkpoint single-chunk control
  - Why: the prompt-conditioned base path with `single_chunk_rollout=true` passed plausibility on the canonical episode-`0` / start-`60` window, but it stopped early rather than solving the task motion (`late_motion_ratio≈0.43`, `profile_correlation≈0.72`, `mean_frame_mae≈7.49`). That is enough to say the upstream-style local Wan/VACE path basically works, but not enough to justify spending more turns on prompt-only smoke variants.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800` with `single_chunk_rollout=true`
  - `arm_motion_verdict`: still `misaligned` and late-heavy even without chunked rollout
  - `image_quality_verdict`: plausible and relatively clean on the main window (`mean_frame_mae≈2.41`), but not a motion-first win
  - `continue_training`: no more chunk-count or teacher-forcing retries until action-path causality is tested
  - Why: removing chunked rollout did not rescue timing. The control checkpoint stayed plausible, but its arm-motion summary remained late-heavy and `misaligned` (`late_motion_ratio≈2.11`, `profile_correlation≈0.31`, `peak_motion_ratio≈2.55`). That is enough to reject chunking as the main blocker. The smallest next control is the same single-chunk run with `action_scale=0.0` to test whether the learned action path is materially steering the rollout at all.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800` with `single_chunk_rollout=true`, `action_scale=0.0`
  - `arm_motion_verdict`: still `misaligned` and almost unchanged from the action-on single-chunk control
  - `image_quality_verdict`: plausible, but blurrier and higher-MAE than the action-on version
  - `continue_training`: no more plain zero-action or chunking controls; one overscaled-action control is the smallest remaining causality check in this neighborhood
  - Why: zeroing actions barely changed the motion profile on the canonical window: `late_motion_ratio≈2.08` versus `≈2.11`, `profile_correlation≈0.31` versus `≈0.31`, and `total_motion_ratio≈1.31` versus `≈1.32`, while `mean_frame_mae` worsened from about `2.41` to `3.95`. The last-horizon comparison frames keep the same late-motion pattern with slightly worse image quality, which argues that chunking is not the blocker and the default action path is weakly coupled at best. The smallest next control is therefore the same single-chunk rollout with `action_scale=2.0` to test whether the action signal is merely underweighted before spending budget on a code-level redesign.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800` with `single_chunk_rollout=true`, `action_scale=2.0`
  - `arm_motion_verdict`: still `misaligned`, with timing slightly later than the default action-on single-chunk control
  - `image_quality_verdict`: plausible and close to the default action-on image quality, much cleaner than the zero-action control
  - `continue_training`: no more inference-side raw action-scale retries in this single-chunk neighborhood; return to structural action-path experiments
  - Why: doubling raw action scale still did not change the canonical late-motion failure. The arm-motion summary stayed almost the same shape as the `action_scale=1.0` control, only slightly worse on timing (`late_motion_ratio≈2.21` versus `≈2.11`, `profile_correlation≈0.30` versus `≈0.31`), while plausibility remained `PASS` and `mean_frame_mae≈2.53` stayed close to the default action-on value (`≈2.41`). The last-horizon comparison frames remain visually the same late-heavy rollout. That closes the borrowed simplicity checks cleanly: prompt-only smoke works, chunking is not the blocker, and raw action scaling is not the missing lever. The next justified step is a code-level action-path change, starting with a post-projection action-token gain rather than another inference-side raw-scale probe.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800` with `single_chunk_rollout=true`, `action_token_scale=2.0`
  - `arm_motion_verdict`: still `misaligned`, and visually almost unchanged from the default action-on single-chunk control
  - `image_quality_verdict`: plausible and close to the default action-on case, but not a motion-first win
  - `continue_training`: no more positive token-gain retries in this neighborhood; the only remaining cheap causality check is a post-projection token-scale ablation
  - Why: after the wrapper fix in `16c8f47`, the rerun completed successfully and still kept the same late-heavy motion pattern as the default action-on control. The arm-motion report stayed `misaligned` with `late_motion_ratio≈2.17`, `profile_correlation≈0.31`, and `peak_motion_ratio≈2.52`, versus the default single-chunk control at roughly `2.11`, `0.31`, and `2.55`. The last-horizon comparison frames and arm crop are visually near-identical, and plausibility remained `PASS` with `mean_frame_mae≈2.51`, only slightly blurrier than the default `≈2.41`. That means stronger post-projection token gain does not wake up the current action path on the canonical window. The smallest remaining causality check is the matching `action_token_scale=0.0` ablation to see whether projected action tokens matter at all.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` step `800` with `single_chunk_rollout=true`, `action_token_scale=0.0`
  - `arm_motion_verdict`: still `misaligned`, with the same visible late-motion pattern as the default and `action_token_scale=2.0` controls
  - `image_quality_verdict`: plausible and slightly cleaner than the `action_token_scale=2.0` run, but still not a motion-first win
  - `continue_training`: no more single-window token-scale sweeps; the next bounded move should switch to the latent action-control prior path
  - Why: zeroing projected action tokens still did not change the canonical rollout in any meaningful visual way. The arm-motion report remained `misaligned` with `late_motion_ratio≈1.92`, `profile_correlation≈0.34`, and `peak_motion_ratio≈2.48`, while plausibility stayed `PASS` and `mean_frame_mae≈2.48` stayed close to the default `≈2.41`. The last-horizon comparison frames and arm-crop strip remain visually near-identical across `action_token_scale=0.0`, `1.0`, and `2.0`. That closes the local token-path causality question: the checkpoint is effectively ignoring projected action tokens on this window. The next distinct bounded experiment is to resume the same `ctx21/h8` anchor with a nonzero `action_control_prior_scale`, which tests a different existing conditioning path rather than another token sweep.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_actionctrlprior0p5_resume800to1000` step `1000`
  - `arm_motion_verdict`: the default one-sided latent prior stays `misaligned` on the main clip plus held-out episodes `1` and `2`, with the same long static hold followed by late fork motion
  - `image_quality_verdict`: all three windows stay plausible and relatively clean, but there is no motion-first win and the main clip / episode `1` smear slightly more than the plain step-`800` control
  - `continue_training`: no more one-sided latent-prior retries; keep the `ctx21/h8` anchor but change the routing, not the scalar
  - Why: the main comparison and arm-crop sheets remain visually close to the default single-chunk control, and the reports confirm that the motion pattern barely moved even after train-side latent-prior exposure. Main `late_motion_ratio≈1.98`, episode `1≈1.47`, and episode `2≈2.39` all remain `misaligned`, while plausibility stays `PASS` on all three windows and `mean_frame_mae≈2.79/2.57/2.26` stays in the same range as the safe short-horizon anchor. Training also collapses late again (`best_val_loss≈0.0281`, `val_loss≈0.2371` at step `1000`), so another scalar continuation in this routing is not justified. The smallest new hypothesis is to keep the same projector and anchor but route the latent prior through both future VACE control branches instead of only the reactive branch.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_actionctrlprior0p5_dualfill_resume800to1000` final checkpoint
  - `arm_motion_verdict`: routing the latent prior through both future VACE control branches still leaves the main clip plus held-out episodes `1` and `2` `misaligned`, with the same late fork commitment pattern
  - `image_quality_verdict`: all three windows stay plausible and close in quality to the one-sided latent prior, but there is no visible motion-first gain
  - `continue_training`: no more latent-prior routing retries on this anchor; pivot to a different conditioning path instead of another routing or scalar tweak
  - Why: the last-horizon comparison sheets and arm crop are visually almost unchanged from the one-sided latent prior: the fork stays static for too long and only moves late. The reports confirm that there is no useful shift in behavior: main `late_motion_ratio≈2.06`, episode `1≈1.43`, and episode `2≈2.53` all remain `misaligned`, while plausibility stays `PASS` on all three windows and MAE stays in the same band at about `2.91/2.72/2.34`. Training also follows the same late-collapse trace (`best_val_loss≈0.0281`, `val_loss≈0.2359` at step `1000`). Because the stronger routing still looks inert, the latent-prior routing family is exhausted. The smallest remaining structural probe is to reuse the same action-derived latent signal but add it directly to future latent hidden states before the Wan backbone.

## Latest Structural Edits

- `d73b3e9` `Route latent action priors through both VACE branches`
  - Added `action_control_prior_mode` with checkpoint-compatible defaults in train/infer configs and CLI parsing.
  - Added `dual_fill` routing in `WanVACEWorldModel` so the existing action-derived latent prior can modulate both future VACE control branches instead of only the reactive future stream.
  - Validated with `103` focused pytest passes plus train/infer CLI smoke checks for `--action-control-prior-mode`.

- `6006617` `Add direct action bias to future latents`
  - Added `action_hidden_state_bias_scale` to train/infer configs, checkpoint restore, and CLI parsing.
  - Reused the existing action-derived latent projector so its output can bias future latent hidden states directly before the Wan backbone, while leaving the control-stream prior path optional and default-off.
  - Validated with `106` focused pytest passes plus train/infer CLI smoke checks for `--action-hidden-state-bias-scale`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_resume800to1000`
  - `arm_motion_verdict`: no model verdict yet; the run was blocked by validation plumbing before any sweep artifacts were produced
  - `image_quality_verdict`: not available because the run never reached checkpoint evaluation
  - `continue_training`: rerun the exact same hidden-state-bias command after the plumbing fix; do not change the hypothesis yet
  - Why: training resumed successfully from step `800` and advanced through step `850`, but the first validation pass crashed with `TypeError: _evaluate_loss() got an unexpected keyword argument 'action_hidden_state_bias_scale'`. That is a tooling-only blocker, not a model result. The correct response is to fix validation plumbing and rerun the same bounded experiment.

- `6ccb840` `Fix validation plumbing for action hidden-state bias`
  - Forwarded `action_hidden_state_bias_scale` through `_evaluate_loss` and aligned the validation-side projector gating with the train-step path.
  - Added a regression assertion in the train-entrypoint tests so validation-loss evaluation keeps receiving the new bias scale.
  - Validated with the same focused `106`-test bundle plus a train CLI smoke check.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_rerun_resume800to1000` final checkpoint
  - `arm_motion_verdict`: still `misaligned` on the main clip plus held-out episodes `1` and `2`, with the same long static hold followed by late fork motion
  - `image_quality_verdict`: all three windows stay plausible and fairly clean, but there is no visible motion-first gain over the latent-prior baselines
  - `continue_training`: no plain continuation; the only remaining cheap rescue in this neighborhood is checkpoint selection at step `900`
  - Why: the last-horizon comparison sheets and arm-crop strips remain visibly late-heavy in all three windows. The reports confirm the same pattern: main `late_motion_ratio≈1.99`, episode `1≈1.41`, and episode `2≈2.47`, all still `misaligned`, while plausibility stays `PASS` with `mean_frame_mae≈2.85/2.59/2.33`. Training also collapsed late again, with `best_val_loss≈0.0283` but `val_loss≈0.2363` at step `1000`. Since only steps `900` and `1000` were saved, step `900` is the smallest remaining rescue before the branch should pivot to a stronger train-side action redesign.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_rerun_resume800to1000` step `900`
  - `arm_motion_verdict`: still `misaligned` on the main clip plus held-out episodes `1` and `2`, with the same late fork commitment pattern and no motion-first win
  - `image_quality_verdict`: step `900` looks a bit earlier than step `1000` on the main clip and episode `1`, but it is blurrier there and it reintroduces an episode-`2` plausibility failure on frame `21`
  - `continue_training`: no more zero-init hidden-state-bias retries in this neighborhood; pivot to a projector redesign instead of another scalar or checkpoint follow-up
  - Why: the last-horizon comparison sheets still hold too static for too long before late motion. Main `late_motion_ratio≈1.83` and episode `1≈1.37` improve only slightly relative to step `1000`, but both windows get blurrier (`mean_frame_mae≈3.92/4.38` versus `≈2.85/2.59`), and episode `2` regresses from plausible to implausible with `failing_frame_indices=[21]`, `late_motion_ratio≈2.54`, and `mean_frame_mae≈4.06`. That closes the checkpoint-selection rescue. The most plausible explanation is structural: the fresh `ActionControlProjector` on these resumed branches still starts from an exact zero mapping, so it has almost no chance to become useful in only `200` extra steps.

- `cf7ebc9` `Add configurable action-control projector init`
  - Added `action_control_projector_init_mode` to train/infer configs, CLI parsing, checkpoint-config restore, and runtime projector construction.
  - Kept the old exact-zero projector start as the default for reproducibility, but added `linear_default` so resumed latent-prior and hidden-state-bias branches can start from a real linear mapping when old checkpoints have no projector weights.
  - Validated with `116` focused pytest passes plus train/infer CLI smoke checks for `--action-control-projector-init-mode`.

- `81fd6b1` `Add action-control aux loss`
  - Added `action_control_aux_loss_scale` to train config, CLI parsing, validation-loss evaluation, and chunkwise training so the fresh `ActionControlProjector` can receive direct train-only supervision against the clean future latent summary.
  - Kept inference behavior unchanged and preserved checkpoint compatibility by storing the aux-loss scale only in config metadata, not in runtime sampling state.
  - Validated with `127` focused pytest passes plus a train CLI smoke check for `--action-control-aux-loss-scale`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_linearinit_auxloss1p0_resume800to1000` final checkpoint
  - `arm_motion_verdict`: still loses motion-first on the main clip plus held-out episode `1`; main and episode `1` remain visibly late-heavy and `misaligned`, while only episode `2` improves to `motion_verdict=good`
  - `image_quality_verdict`: all three windows remain plausible and fairly clean, but there is no visible earlier commitment win where it matters most
  - `continue_training`: no more action-only projector-local retries; pivot to a broader projector redesign that adds observed-state context
  - Why: the last-horizon comparison sheets and arm-crop strips still show the same long static hold followed by late fork motion on the main clip and episode `1`. The reports make that concrete: main `late_motion_ratio≈2.147`, `profile_correlation≈0.283`, and `mean_frame_mae≈2.44`; episode `1≈1.470`, `≈0.565`, and `≈2.88`; episode `2≈2.129`, `≈0.739`, and `≈2.08`, with plausibility `PASS` on all three windows. Validation is also much weaker than the earlier short-horizon anchor (`best_val_loss≈0.2918` at step `950`, `final≈0.4600` at step `1000`), so step-`900` checkpoint selection is not a compelling rescue. Aux loss improved held-out safety on episode `2`, but it did not wake up motion-first behavior on the main clip or episode `1`.

- `0d714da` `Add observed-context action control projector`
  - `hypothesis`: the fresh latent projector bolted onto old checkpoints is not just undertrained, it is missing state context. If it only sees action plus progress features, short resumed runs cannot learn how the current scene state should modulate future latent biasing.
  - `implementation`: added `action_control_projector_observed_context_mode` to train/infer configs, CLI parsing, checkpoint-config restore, runtime factory construction, training, inference, and checkpoint sweeps. In the new `last_frame` mode, `ActionControlProjector` pools the last observed latent frame across space, projects it through a new `context_projection`, and adds that state-conditioned bias to the broadcast future latent prior. Old behavior remains the default `none` path, and older checkpoints are still allowed to omit the new projector weights.
  - `validation`: `source .venv/bin/activate && pytest tests/test_config_defaults.py tests/test_wan_vace_conditioning.py tests/test_wan_vace_factory.py tests/test_chunkwise_training_wan_vace.py tests/test_train_world_model_wan_vace.py tests/test_infer_world_model_wan_vace.py` passed (`127 passed in 3.50s`); `python scripts/train/world_model.py --help` and `python scripts/train/infer_world_model.py --help` both expose `--action-control-projector-observed-context-mode`; and a direct `.venv` smoke check with `PYTHONPATH=src` exercised `train_chunkwise_batch(..., action_control_projector_observed_context_mode='last_frame')` without interface errors.
  - `next action`: rerun the `ctx21/h8` step-`800` to `1000` hidden-state-bias branch with `action_hidden_state_bias_scale=0.5`, `action_control_projector_init_mode=linear_default`, `action_control_aux_loss_scale=1.0`, and `action_control_projector_observed_context_mode=last_frame`, then evaluate the main clip plus held-out episodes `1` and `2`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_linearinit_auxloss1p0_obslastframe_resume800to1000` final checkpoint
  - `arm_motion_verdict`: observed-state context does not rescue the resumed branch; main, episode `1`, and episode `2` all stay visibly late-heavy and `misaligned`
  - `image_quality_verdict`: all three windows remain plausible and reasonably clean, but there is no visible motion-first win and timing regresses sharply on the main clip plus episode `2`
  - `continue_training`: no more resumed projector-local follow-ups; the only remaining bounded check is whether the same architecture helps when co-trained from step `0`
  - Why: the last-horizon comparison sheets and arm-crop strips still show a long static hold followed by late fork motion on all three windows. The reports are worse than the already-bad aux-loss resume on the main clip and episode `2`: main `late_motion_ratio≈2.949`, `profile_correlation≈0.175`, and `mean_frame_mae≈2.57`; episode `1≈1.789`, `≈0.580`, and `≈2.69`; episode `2≈3.104`, `≈0.582`, and `≈2.18`, all with plausibility `PASS`. Training is also weaker (`best_val_loss≈0.3806` at step `950`, `final≈0.5594` at step `1000`). That closes observed-context resume as a rescue and points to the remaining training-regime hypothesis: the projector may need to co-train from step `0` instead of being attached to a mature checkpoint for only `200` extra steps.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_hiddenstatebias0p5_linearinit_auxloss1p0_obslastframe_fresh400` final checkpoint
  - `arm_motion_verdict`: fresh observed-context projector training still fails decisively; main, episode `1`, and episode `2` all stay visibly late-heavy and `misaligned`
  - `image_quality_verdict`: the rollout is visibly bad enough that metrics no longer rescue it, and all three windows fail plausibility
  - `continue_training`: no more projector-path experiments in this family; pivot to a distinct backbone-conditioning route
  - Why: reviewing the last `8` frames first, then the arm crops, then plausibility, showed the same long static hold followed by late fork motion plus visible artifacts/tearing. The canonical reports confirm the rejection: main `late_motion_ratio≈2.979`, `profile_correlation≈0.154`, `failing_frame_indices=[21,22]`, and `mean_frame_mae≈4.24`; episode `1≈1.648`, `≈0.476`, `failing_frame_indices=[21]`, and `mean_frame_mae≈4.80`; episode `2≈2.576`, `≈0.446`, `failing_frame_indices=[21]`, and `mean_frame_mae≈3.17`. Validation improved steadily through step `400` (`best_val_loss≈0.2501` at the final checkpoint), so checkpoint selection is not a rescue. This exhausts the whole latent-projector family, including fresh training from step `0`.

- `action added-K/V backbone path` code pivot
  - `hypothesis`: the repo has exhausted latent-projector and token-gain tweaks because action is still entering Wan through weakly coupled side paths. Mirroring the existing action tokens into Wan's native added-K/V image-conditioning slot is the smallest distinct backbone-conditioning route.
  - `implementation`: added `action_backbone_added_kv_mode` with `reuse_action_tokens` in train/infer configs and CLI parsing, rebuilt pretrained Wan VACE backbones with `image_dim=text_dim` and `added_kv_proj_dim=inner_dim` when this mode is active, kept the new image-path weights trainable under LoRA, threaded optional `action_image_tokens` through training/inference/model protocols, and reused the existing action tokens for that path.
  - `validation`: `source .venv/bin/activate && pytest tests/test_config_defaults.py tests/test_wan_vace_factory.py tests/test_wan_vace_world_model.py tests/test_infer_world_model_wan_vace.py tests/test_train_world_model_wan_vace.py` passed (`112 passed`); both train and infer CLIs expose `--action-backbone-added-kv-mode`; and a direct pretrained-backbone smoke build under `.venv` with `PYTHONPATH=src` produced `image_dim=4096` and `added_kv_proj_dim=1536` on the real Wan VACE checkpoint.
  - `next action`: run a fresh `ctx21/h8` `400`-step training job with `action_backbone_added_kv_mode=reuse_action_tokens`, then evaluate the main clip plus held-out episodes `1` and `2`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid_addedkv_fresh400`
  - `arm_motion_verdict`: no model result; training failed before any checkpoint or sweep artifacts were produced
  - `image_quality_verdict`: not applicable because there are no videos to review
  - `continue_training`: retry the same added-K/V backbone hypothesis with a smaller LoRA budget, not a new architecture branch
  - Why: the run failed during training with `torch.OutOfMemoryError` inside the Wan/VACE LoRA path before the first checkpoint, while trying to allocate about `20 MiB`. The controller result confirms that `metrics.jsonl`, `final_for_eval.pt`, and all eval/inspection artifacts are absent, so there is nothing to rank visually and no model conclusion to draw. Because the failure happened immediately and batch size, resolution, context, and horizon are already fixed at the canonical `ctx21/h8` neighborhood, the smallest same-hypothesis recovery is to rerun the added-K/V backbone branch with `lora_rank=16` instead of `32`.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora16_action_noinputln_mlp128resid_addedkv_fresh400`
  - `arm_motion_verdict`: no model result; training failed before any checkpoint or sweep artifacts were produced
  - `image_quality_verdict`: not applicable because there are no videos to review
  - `continue_training`: one final low-risk fit retry in the same neighborhood is justified with the base-config LoRA budget; do not change architecture yet
  - Why: reducing `lora_rank` from `32` to `16` did not clear the blocker. The run still failed inside the Wan/VACE added-K/V path with `torch.OutOfMemoryError` before the first checkpoint, now while trying to allocate about `16 MiB`, and the controller result again confirms that `metrics.jsonl`, `final_for_eval.pt`, and all eval/inspection artifacts are absent. Since the geometry and hypothesis are still untested, the smallest remaining same-hypothesis recovery is to rerun the branch at `lora_rank=8`, which matches the base config's LoRA budget and is the last cheap memory cut before a more structural memory or architecture pivot.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora8_action_noinputln_mlp128resid_addedkv_fresh400`
  - `arm_motion_verdict`: no model result; training failed before any checkpoint or sweep artifacts were produced
  - `image_quality_verdict`: not applicable because there are no videos to review
  - `continue_training`: no more plain LoRA-rank cuts in this neighborhood; switch to activation-memory reduction while keeping the same added-K/V hypothesis
  - Why: the `lora_rank=8` retry still failed before the first checkpoint or any evaluation artifacts, this time with `torch.OutOfMemoryError` while trying to allocate about `42 MiB`. Since `32`, `16`, and `8` all failed inside the same Wan/VACE added-K/V path, plain rank reduction is exhausted as a fit strategy. The smallest distinct recovery that still tests the same architecture is to keep `lora_rank=8` and enable `gradient_checkpointing`, which the repo already supports for the Wan backbone.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora8_action_noinputln_mlp128resid_addedkv_gradckpt_fresh400` final checkpoint
  - `arm_motion_verdict`: plausible but still `misaligned` on the main clip plus held-out episodes `1` and `2`, with the same late-heavy commitment pattern
  - `image_quality_verdict`: all three windows stay plausible and fairly coherent, but the last-horizon comparison sheets and arm-crop strip still show the fork moving too late and too aggressively near the end
  - `continue_training`: no new train-side follow-up yet; take the cheap checkpoint-selection rescue first at step `300`
  - Why: gradient checkpointing finally made the added-K/V backbone branch runnable, but the visual result is not a win. The main sheet still shows a long static hold followed by late, overactive fork motion; episode `1` does the same, and episode `2` stays the latest of the three. The arm-motion reports confirm that all three windows remain `misaligned`: main `late_motion_ratio≈2.275`, `profile_correlation≈0.277`, `mean_frame_mae≈3.03`; episode `1≈2.031`, `≈0.522`, `≈4.49`; episode `2≈3.173`, `≈0.467`, `≈3.15`, all with plausibility `PASS`. Training, however, improved steadily through step `350` (`best_val_loss≈0.0659`) before regressing at the final step (`val_loss≈0.0981` at step `400`), and the best saved checkpoint is step `300` (`val_loss≈0.0846`). That makes step-`300` checkpoint selection the only cheap remaining rescue before this whole added-K/V family should pivot out.

- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora8_action_noinputln_mlp128resid_addedkv_gradckpt_fresh400` step `300`
  - `arm_motion_verdict`: all three windows remain visibly late-heavy and `misaligned`; the held-out windows are not safe enough to keep this family alive
  - `image_quality_verdict`: the main clip is only marginally acceptable, but held-out episodes `1` and `2` both fail plausibility and still show unstable late fork motion
  - `continue_training`: no more added-K/V follow-ups; close Wan-side routing and pivot to a train-side action representation intervention
  - Why: the step-`300` rescue does not recover the branch. Reviewing the last `8` frames first, then the arm-crop strip and reports, showed the same long static hold followed by late motion, and the controller result confirms the safety failure: main plausibility `PASS`, episode `1` plausibility `FAIL`, episode `2` plausibility `FAIL`, with all three arm-motion reports still `misaligned`. That means the only runnable backbone-route test inside Wan still fails the held-out safety gate even before its late regression at step `400`, so the whole added-K/V family is now exhausted.

- `560fa96` `Add action-token latent aux loss`
  - `hypothesis`: Wan-side routing changes have been exhausted because the learned action tokens themselves are weakly informative. The smallest intervention outside routing is to supervise those tokens directly against the clean future latent summaries so the existing cross-attention path has a more causal representation to work with.
  - `implementation`: added `action_token_latent_aux_loss_scale` to train/infer config restore and the train CLI; extended `ActionTokenEncoder` with an optional checkpoint-compatible latent-summary head; threaded the new train-only auxiliary loss through chunkwise training, validation-loss evaluation, and metrics logging; and kept inference behavior unchanged because the new head is only used for training supervision.
  - `validation`: `source .venv/bin/activate && pytest tests/test_config_defaults.py tests/test_wan_vace_conditioning.py tests/test_wan_vace_factory.py tests/test_chunkwise_training_wan_vace.py tests/test_train_world_model_wan_vace.py` passed (`106 passed in 3.49s`); `source .venv/bin/activate && python -m py_compile scripts/train/world_model.py scripts/train/infer_world_model.py src/world_model/config.py src/world_model/models/wan_vace_conditioning.py src/world_model/models/wan_vace_factory.py src/world_model/training/chunkwise_training.py` succeeded; `python scripts/train/world_model.py --help` exposes `--action-token-latent-aux-loss-scale`; and `python scripts/train/infer_world_model.py --help` still succeeds with checkpoint-restore support intact.
  - `next action`: resume the best held-out-safe `ctx21/h8` step-`800` action checkpoint for `200` more steps with `action_token_latent_aux_loss_scale=1.0`, then evaluate the main clip plus held-out episodes `1` and `2`.
