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
- The hidden-state-bias checkpoint rescue also failed: step `900` stayed visibly late-heavy, worsened main and episode-`1` MAE versus step `1000`, and reintroduced an episode-`2` plausibility failure on frame `21`, so the zero-init hidden-state-bias branch is exhausted.
- The nonzero-init hidden-state-bias rerun also failed: `action_control_projector_init_mode=linear_default` kept the main clip plus held-out episodes `1` and `2` plausible, but all three windows stayed visibly late-heavy and `misaligned` (`late_motion_ratio≈2.00/1.43/2.56`) and validation was much weaker than the zero-init rerun (`best_val_loss≈0.1186` vs. `≈0.0283`), so projector init alone does not wake up the latent route.
- The linear-init hidden-state-bias rerun with direct projector supervision also failed to rescue the motion-first ranking: main and episode `1` still stayed visibly late-heavy and `misaligned` (`late_motion_ratio≈2.15/1.47`), only episode `2` improved to `motion_verdict=good`, and validation stayed very weak (`best_val_loss≈0.2918`, `final≈0.4600`). Aux loss alone is therefore exhausted as an action-only projector fix.
- The observed-context rerun also failed as a resumed-branch rescue: main, episode `1`, and episode `2` all stayed `misaligned` (`late_motion_ratio≈2.95/1.79/3.10`) despite plausibility passing on all three windows, and validation degraded further (`best_val_loss≈0.3806`, `final≈0.5594`).
- The fresh `ctx21/h8` observed-context projector run from step `0` also failed decisively: the main clip plus held-out episodes `1` and `2` all stayed visibly late-heavy and `misaligned` (`late_motion_ratio≈2.98/1.65/2.58`) and all three windows failed plausibility (`failing_frame_indices main=[21,22], ep1=[21], ep2=[21]`) even though validation improved steadily to `best_val_loss≈0.2501` at step `400`. That exhausts the whole latent-projector family, not just the resumed variants.
- The next unused action path is inside the Wan backbone itself, not another latent projector: mirroring the existing action tokens into Wan's added-K/V image-conditioning slot is a distinct stronger conditioning route that does not depend on a fresh latent projector learning from scratch.
- The first fresh `ctx21/h8` added-K/V backbone run with `lora_rank=32` did not produce a model result at all: training OOMed inside the Wan backbone LoRA path before the first checkpoint or evaluation artifacts were written (`torch.OutOfMemoryError`, missing about `20 MiB`). That means the architecture question is still open; only the initial memory budget was too large.
- The follow-up fresh added-K/V retry with `lora_rank=16` also failed before checkpointing or evaluation, this time missing about `16 MiB`. So the added-K/V hypothesis is still untested on model quality; the only information gained is that this branch still needs a smaller memory footprint on the 16 GB RTX 3080.

## Active Questions
The one question to answer next, broken down into the minimum parts.

- Can a stronger backbone-conditioning route help where all latent-projector variants failed?
- Smallest next move: rerun the same fresh `ctx21/h8` `400`-step added-K/V backbone job with `action_backbone_added_kv_mode=reuse_action_tokens`, but reduce `lora_rank` from `16` to `8` so the run can finally fit and produce evaluation artifacts.
- If the added-K/V backbone route is still late-heavy or implausible, stop local action-conditioning tweaks in this architecture family and pivot out of Wan-side action routing entirely.

## Future Questions
Questions to revisit only after the simplicity check is answered.

- If the added-K/V backbone route still barely changes motion timing, what is the smallest justified next redesign outside Wan-side action routing entirely?
- If the added-K/V backbone route changes behavior sharply, should the next step be a calibrated fresh-training sweep or a cleaner ablation of cross-attention text tokens vs. added-K/V image tokens?
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
- Zero-init hidden-state-bias rescue on `ctx21/h8` step `800`, including checkpoint selection.
- Linear-init hidden-state-bias rescue without direct projector supervision on `ctx21/h8` step `800`.
- Linear-init hidden-state-bias rescue with direct projector supervision but no observed latent context on `ctx21/h8` step `800`.
- Linear-init hidden-state-bias rescue with direct projector supervision and observed latent context on `ctx21/h8` step `800`.
- Fresh latent-projector training on `ctx21/h8`, including the observed-context `fresh400` branch.
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
- Commit `cf7ebc9` (`Add configurable action-control projector init`): adds `action_control_projector_init_mode` so resumed latent-prior and hidden-state-bias branches can opt out of the exact-zero projector start that likely left the fresh latent route inert.
- Commit `81fd6b1` (`Add action-control aux loss`): adds a train-only `action_control_aux_loss_scale` that directly supervises the fresh action-control projector against the clean future latent summary, so short resumed latent branches are no longer learning only through indirect denoising gradients.
- Commit `0d714da` (`Add observed-context action control projector`): adds `action_control_projector_observed_context_mode` so the fresh latent projector can condition on the pooled last observed latent frame during train, infer, and checkpoint sweeps while keeping the old action-only path unchanged by default.
- Commit `1717b5f` (`Add action added-K/V backbone path`): mirrors action tokens into Wan's added-K/V image-conditioning slot and keeps those newly introduced image-path weights trainable under LoRA, creating the first non-projector backbone-conditioning route for action.
