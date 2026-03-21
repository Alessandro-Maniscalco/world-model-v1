## Stable Findings
Durable facts that should survive multiple controller turns.

- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation, treat plausibility as a safety gate, and rank runs motion-first over sharpness or aggregate MAE when the clips stay plausible.
- The upstream Wan/VACE contract is whole-window inference. In this repo, the closest smoke path is `conditioning_mode=prompt` with `single_chunk_rollout=true` and at least `50` integration steps.
- The single-chunk simplicity controls are now complete on the canonical episode-`0` / start-`60` window: prompt-only smoke works as a plausible upstream-style path, but action-conditioned runs with `action_scale=0.0`, `1.0`, and `2.0` all keep the same late-heavy motion pattern, so chunking and raw action amplitude are not the main blockers.
- On the best held-out-safe `ctx21/h8` step-`800` anchor, projected action-token scaling and both tested latent-prior routes leave the same late-heavy single-chunk rollout, so the current action-conditioning routes into Wan/VACE look weakly coupled.
- The late-motion failure has survived scalar `ctx21/h8` tweaks, ordered full-plan conditioning, h12/h16 multi-chunk coverage, rollout-prefix and past-only teacher forcing, rollout-matched future inputs, and short-horizon exact-`k` chunk scheduling.

## Best Run
Current winners and the ranking takeaway to anchor comparisons.

- Motion-first best overall: `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` at step `800`.
- Best held-out-safe action anchor: `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` at step `800`.
- Ranking takeaway: use `ctx21/h8` step `800` as the control checkpoint for rollout-structure tests, because it is the strongest action-conditioned anchor that still kept episode-`2` behavior relatively safe.

## Findings
Important but less-stable takeaways that may change as new experiments land.

- Longer context helped stability. The `context_len=21`, `horizon_len=8` branch was a meaningful improvement over shorter-memory baselines.
- The strongest action-conditioned checkpoints remain `optimizer_aloha_static_fork_pick_up_full_320x240_lora32_action_noinputln_mlp128resid` at step `800` for raw motion and `optimizer_aloha_static_fork_pick_up_full_320x240_ctx21_h8_lora32_action_noinputln_mlp128resid` at step `800` for held-out safety.
- The prompt-conditioned single-chunk smoke on episode `0` / start `60` stayed plausible but undercommitted (`motion_verdict=undercommitted`, `late_motion_ratio≈0.43`, `profile_correlation≈0.72`), so the upstream-style Wan/VACE path basically works but does not solve task motion by itself.
- The action-conditioned `ctx21/h8` step-`800` single-chunk control on the same window still stayed late-heavy and `misaligned` (`late_motion_ratio≈2.11`, `profile_correlation≈0.31`) while remaining plausible, so chunked rollout is not the main cause of the late-motion failure.
- The zero-action version of that same single-chunk control stayed almost unchanged on motion (`late_motion_ratio≈2.08`, `profile_correlation≈0.31`) while getting blurrier (`mean_frame_mae≈3.95` versus `≈2.41`), so the default action path affects image quality more than it affects the late-motion pattern on the canonical window.
- Doubling raw action scale on that same single-chunk control also kept the same late-heavy `misaligned` motion (`late_motion_ratio≈2.21`, `profile_correlation≈0.30`) with image quality close to the default action-on case (`mean_frame_mae≈2.53`), so inference-side raw action scaling is exhausted as a causality test.
- The first `action_token_scale=2.0` probe was initially blocked by sweep-wrapper plumbing, but the rerun completed and stayed effectively unchanged from the default single-chunk control (`late_motion_ratio≈2.17`, `profile_correlation≈0.31`, `mean_frame_mae≈2.51`), so post-projection token gain does not rescue the late-motion failure on the canonical window.
- The matching `action_token_scale=0.0` ablation also stayed visually near-identical to the default single-chunk control while remaining plausible (`late_motion_ratio≈1.92`, `profile_correlation≈0.34`, `mean_frame_mae≈2.48`), so projected action tokens are effectively inert on the canonical `ctx21/h8` step-`800` checkpoint.
- Resuming the same `ctx21/h8` step-`800` anchor with `action_control_prior_scale=0.5` on the default `reactive_only` latent-prior path also stayed plausible but still late-heavy and `misaligned` across the main clip plus held-out episodes `1` and `2` (`late_motion_ratio≈1.98/1.47/2.39`, `mean_frame_mae≈2.79/2.57/2.26`), so the one-sided latent prior does not rescue the branch either.
- The stronger `dual_fill` latent-prior routing also stayed plausible but effectively unchanged from the one-sided prior (`late_motion_ratio≈2.06/1.43/2.53`, `mean_frame_mae≈2.91/2.72/2.34`), so the whole latent-prior routing family is now exhausted on the `ctx21/h8` anchor.
- The first `action_hidden_state_bias_scale=0.5` resume from the `ctx21/h8` step-`800` anchor was blocked by validation-loss plumbing after training resumed to step `850`, not by model behavior, so the correct next action remains a rerun of that exact hidden-state-bias probe.

## Active Questions
The one question to answer next, broken down into the minimum parts.

- Can the existing action-derived latent signal steer the model once it is added directly to the future latent hidden states before the Wan backbone?
- Smallest next structural move: resume `ctx21/h8` step `800` to step `1000` with `action_hidden_state_bias_scale=0.5`, keeping `action_control_prior_scale=0.0`, then evaluate the main clip plus held-out episodes `1` and `2`.
- If the direct future-latent bias still leaves the same late-heavy pattern, the next iteration should pivot from conditioning-route tweaks to a stronger train-side action objective or projection redesign.

## Future Questions
Questions to revisit only after the simplicity check is answered.

- If the direct future-latent bias still barely changes motion timing, what is the smallest justified next redesign: simpler action projection, a stronger train-side action objective, or a broader backbone-conditioning change?
- If the direct future-latent bias changes behavior sharply, should the next step be a calibrated bias-scale sweep or a cleaner train-side ablation of latent-bias vs. token conditioning before revisiting multi-chunk rollout?
- Should held-out single-chunk checks on episodes `1` and `2` wait until the action-path causality question is answered on the canonical main window?

## Exhausted Families
Branches that should not get another near-duplicate retry.

- Residual-family scalar continuations around residual step `800`.
- Short-horizon `ctx21/h8` scalar shaping: temporal difference, temporal mixer, motion-loss weighting, and early-horizon loss bias.
- Ordered full-plan conditioning on the `ctx21/h8` anchor.
- h12/h16 multi-chunk expansion families, including plain `k=2` and `k=3`, chunk-position weighting, motion-loss follow-ups, and checkpoint-selection rescues.
- Teacher-forcing rollout variants: `past_only`, `predicted_prefix`, and `teacher_forcing_future_input_mode=active_chunk`.
- Short-horizon exact-`k` rescue on `ctx21/h8/k2`, including checkpoint selection.
- Single-window inference-side raw action-scale controls on `ctx21/h8` step `800`.
- Single-window projected-token scale sweeps on `ctx21/h8` step `800`.
- Latent control-prior routing on `ctx21/h8` step `800`, including `reactive_only` and `dual_fill`.
- Dataset-subset restriction side branches.

## Kept Code Changes
Still-relevant code-changing commits that remain available as structural levers.

- Commit `17ba95f` (`Cap motion-aware loss weights`): added `motion_loss_max_weight` so motion-aware loss can stay active without letting a few high-motion regions dominate training.
- Commit `7fe8994` (`Add excess-only motion loss weighting`): added an excess-only weighting mode so motion emphasis can target above-average motion regions instead of boosting all regions equally.
- Commit `7832e4a` (`Add temporal-difference action residual`): added an optional temporal-difference residual over action tokens without breaking checkpoint compatibility.
- Commit `5537878` (`Add temporal action-token mixer`): added an optional zero-init temporal mixer over projected action tokens plus backward-compatible checkpoint loading for structural action-conditioning tests.
- Commit `7cba14d` (`Add early-horizon loss bias`): added an optional linear temporal loss bias so earlier future frames can be upweighted directly when timing, not plausibility, is the blocking failure.
- Commit `614c605` (`Add early chunk loss bias`): added a training-only `future_chunk_early_bias` so earlier autoregressive chunks can receive more loss mass without changing inference behavior.
- Commit `07281db` (`Add past-only teacher forcing mode`): added a training-only `teacher_forcing_observation_mode` / `--teacher-forcing-observation-mode` so later teacher-forced chunks can observe only the true past, directly testing whether future-prefix leakage is driving the late-motion failure.
- Commit `48f2883` (`Add predicted-prefix teacher forcing`): added a `predicted_prefix` teacher-forcing mode that feeds detached model-predicted clean chunks back as the observed prefix for later chunks, directly testing whether rollout-style prefix feedback is required to fix the late-motion failure.
- Commit `cc64dce` (`Match teacher forcing future inputs to rollout`): keeps the `teacher_forcing_future_input_mode=active_chunk` lever available if the single-chunk control shows chunking is the real problem and multi-chunk train/infer mismatch is still worth revisiting.
- Commit `e271c40` (`Add exact-k chunk schedule mode`): keeps exact-`k` rollout available if the single-chunk control wins and chunk-count becomes worth revisiting from a cleaner baseline.
- Commit `703a306` (`Add action-token output scale`): adds a checkpoint-compatible `action_token_scale` lever so post-projection action-token gain can be tested directly at train or infer time without changing old checkpoints.
- Commit `16c8f47` (`Plumb action token scale through local sweeps`): adds `--action-token-scale` to `scripts/check/sweep_local_repo_resolutions.py` so the canonical checkpoint-evaluation path can actually run the new token-gain control.
- Commit `d73b3e9` (`Route latent action priors through both VACE branches`): adds `action_control_prior_mode=dual_fill` so the existing latent prior can modulate both future VACE control branches instead of only the reactive branch.
- Commit `6006617` (`Add direct action bias to future latents`): adds `action_hidden_state_bias_scale` so the existing action-derived latent signal can bias future latent hidden states directly before the Wan backbone.
- Commit `6ccb840` (`Fix validation plumbing for action hidden-state bias`): forwards `action_hidden_state_bias_scale` through validation-loss evaluation so hidden-state-bias training runs can complete instead of failing at the first validation step.
