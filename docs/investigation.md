## Goal
First, enforce a zero-step safety invariant on the fixed operator slice:
with `conditioning_mode=action` and zero training steps, the model should
behave as close as possible to `conditioning_mode=none`.

Second, only after that invariant is validated, continue the low-resolution
architecture search with `context_len=9`, `horizon_len=8`, using a fixed
untouched base DiT parent for every new architecture branch.

## Fixed Policy
- Run Python and pytest inside `.venv`.
- Review only `episode_index=1`, `start_frame=60` unless the operator changes
  it.
- Start every new architecture branch from the same untouched base DiT
  checkpoint.
- Evaluate every checkpoint against the untouched base-DiT baseline on the
  fixed slice and against best-so-far motion.
- If a branch is worse than the base baseline after `step_0000100`, stop it.
- If a branch is worse than base on both motion and visible quality at
  multiple spaced checkpoints, stop it.

## Base Geometry
- Use the bounded smoke geometry unless there is a strong reason not to:
  `episode=0`, `224x128`, `context_len=9`, `horizon_len=8`, `k=1`,
  `gradient_checkpointing=true`
- Startup auto batch probing has repeatedly found a real fitting batch size of
  `8` on this machine for checkpointed runs.
- Training time is now the bottleneck, so new training probes should cap at
  `100` steps first and only continue past that if the short run is clearly
  promising.

## Canonical Baselines
- Untouched base-DiT parent:
  keep this as the required parent for every new architecture branch.
- Literal zero-step comparator artifact:
  `runs/training_optimizer/eval/untouched_base_none_subset8_step0_baseline_224x128_ep1_start60_step0000_operator/untouched_base_none_subset8_step0_baseline_224x128_step_0000000_comparison.mp4`
- Best decision-driving artifact so far:
  `runs/training_optimizer/eval/fullft_subset8_spread_resume200_lr5e5_step400_ep1_start60_step0350_operator/fullft_subset8_spread_resume200_lr5e5_step400_step_0000350_comparison.mp4`

## Baseline Video Read
- In the zero-step untouched-base clip, `f0-f8` are copied context, then the
  future fails immediately at `f9`:
  `f9-f16` turn into a blue OOD wash with a blown-out plate blob and no
  coherent arm/tool trajectory. Plausibility fails there.
- In the best baseline clip, `f0-f8` are copied context.
- Visible motion starts at `f9`.
- The rollout stays coherent through `f16`.
- It is still a late no-contact near-miss, but it has the strongest clean
  last-horizon commitment seen so far.
- On the harder held-out none-conditioned slice at `start_frame=226`, the same
  family appears again: static through `f8`, motion from `f9`, coherent but
  undercommitted through `f16`.

## Exhausted Family
- The bounded low-scale action-conditioned LoRA family is exhausted for now.
- Tried and rejected in this neighborhood:
  frozen action-only, resumed LoRA, higher aux loss, zero-init off, added-K/V,
  base-start, temporal mixer, temporal difference, no-input-LN, residual MLP.
- Common pattern across those branches:
  `f0-f8` stay as context, motion starts at `f9`, clips stay coherent enough to
  avoid full confetti, but the arm still fails to commit into clean contact by
  `f16`.
- Strongest alternative branch:
  temporal-difference `step_0000300` added more late motion, but it was still
  blurrier than baseline and still missed contact.
- Cleanest post-trained base-start branch:
  `lora_action_addedkv_frombase_subset8_tok005_aux1_nozeroinit_step400`
  `step_0000200`; cleaner than later variants, but still flatter than the
  baseline and still undercommitted.
- Cheap eval-only gain probes did not rescue this family:
  higher gain either reopened blur/ghosting or stayed in the same
  undercommitted family.

## Architecture Outcomes
- `trainable_backbone=vace` from the untouched base parent is non-improving.
  On `step_0000100`, `0000200`, `0000300`, and `0000400`, `f0-f8` stay copied
  context, visible motion starts at `f9`, and the rollout stays coherent
  through `f16` with only mild local haze, but every checkpoint still misses
  clean plate contact.
- `step_0000300` is the strongest VACE checkpoint, but it is still flatter and
  less committed than the best none-conditioned baseline `step_0000350`.
  `step_0000400` regresses again, so do not spend another plain continuation
  here.
- The tail-focused late-heavy VACE pivot is also non-improving and worse than
  the earlier evenly spaced VACE branch.
- On salvaged late-heavy checkpoints `step_0000100`, `0000200`, and
  `0000300`, `f0-f8` stay copied context, visible motion starts at `f9`, and
  the rollout stays coherent through `f16`, but every checkpoint still stops
  short of contact.
- `step_0000200` is the best late-heavy VACE checkpoint:
  it adds slightly more late-horizon sweep than `step_0000100`, but the fork
  still undershoots the plate by `f16` and remains weaker than both the
  earlier evenly spaced VACE `step_0000300` and the best none-conditioned
  baseline `step_0000350`.
- `step_0000300` regresses hard:
  motion fades after roughly `f11` and the last horizon becomes visibly
  flatter than either `step_0000100` or `step_0000200`.
- Treat the VACE family as exhausted for this investigation:
  shifting the control sites later preserved fidelity but reduced
  last-horizon commitment instead of improving it, so do not spend another
  VACE-layer schedule or resume this branch to `step_0000400`.
- The first fair full-backbone action-conditioned branch from the untouched
  base parent is materially better than the LoRA and VACE action branches, but
  it still does not beat the baseline.
- On `step_0000100`, `0000200`, `0000300`, and `0000400`, `f0-f8` stay copied
  context, visible motion starts at `f9`, and all four checkpoints remain
  globally coherent through `f16` without the old `f9` blue OOD collapse.
- `step_0000100` is too weak:
  motion starts at `f9` but the arm stops early and still misses contact.
- `step_0000200` is the strongest full-action checkpoint:
  it adds the most last-horizon arm motion in this branch and gets closer to
  the plate by `f16`, but from roughly `f11-f16` the arm/tool region broadens
  into a duplicated blur plume and still stops short of clean contact.
- `step_0000300` backs off again:
  it stays coherent and a bit cleaner than `step_0000200`, but late motion
  drops and the fork undershoots sooner.
- `step_0000400` is the cleanest later checkpoint, but it is flatter than
  `step_0000200` and still less committed than the best none-conditioned
  baseline `step_0000350`.
- Treat the direct full-action continuation as answered:
  this family can stay coherent, but the in-run peak arrives at
  `step_0000200` and later checkpoints regress, so do not spend another plain
  continuation here.
- The first `trainable_backbone=head` action-conditioned branch with
  `--no-gradient-checkpointing` is non-improving and visibly worse than the
  full-action branch.
- On `step_0000100`, `0000200`, `0000300`, and `0000400`, `f0-f8` stay copied
  context and motion still begins at `f9`, but the predictions immediately
  drift into a blocky blue-tinted low-detail smear across the arm/tool region
  and never recover clean late-horizon commitment.
- `step_0000300` is the least bad head checkpoint:
  it keeps slightly more structure than `step_0000100`, `0000200`, or
  `0000400`, but it is still visibly undercommitted and far below both the
  full-action `step_0000200` branch peak and the none-conditioned baseline
  `step_0000350`.
- Treat the head/no-checkpointing family as structurally wrong for this task:
  do not spend another run on `trainable_backbone=head`, and do not treat
  `gradient_checkpointing=false` as a standalone goal.
- Disabling action input LayerNorm in the strongest full-action family is also
  non-improving.
- On `step_0000100`, `0000200`, `0000300`, and `0000400`, `f0-f8` stay copied
  context, visible motion starts at `f9`, and the rollout stays globally
  coherent through `f16` without the zero-step blue collapse, but every
  checkpoint still stops short of clean contact.
- `step_0000100` is the most animated no-input-LN checkpoint, but it fattens
  the lower arm/tool region into a blurrier smear by the last horizon.
- `step_0000200` and `step_0000400` are cleaner than `step_0000100`, but both
  flatten sooner and leave even more distance before contact.
- `step_0000300` is the best no-input-LN checkpoint because it keeps slightly
  cleaner arm/tool structure than `step_0000100` while preserving more late
  motion than `step_0000200` or `step_0000400`, but it is still flatter and
  less committed than the earlier full-action `step_0000200` checkpoint and
  the none-conditioned baseline `step_0000350`.
- Treat the no-input-LN branch as answered:
  it does not beat the earlier full-action peak, so do not spend another plain
  training continuation there.

## Current Decision
- The zero-step untouched-base `conditioning_mode=none` comparator is the
  current safety reference:
  it stays copied through `f8` and then collapses into a blue OOD wash from
  `f9-f16`.
- The latest validated zero-step `conditioning_mode=action` probe using
  added-K/V mirroring is visually almost identical to that none-conditioned
  reference:
  `f0-f8` stay copied, `f9-f16` collapse into the same blue OOD wash, and the
  full-frame plus arm-crop clips differ only slightly.
- Treat that mirrored action result as off-policy, not as the invariant pass:
  the operator explicitly disallowed
  `action_backbone_added_kv_mode=reuse_action_tokens` for this safety check.
- The compliant zero-step no-added-K/V `conditioning_mode=action` probe now
  validates that safer invariant:
  `f0-f8` stay copied, `f9-f16` collapse into the same blue OOD wash as the
  none-conditioned reference, and direct generated-video comparison stays very
  close with overall MAE about `0.73` and late-frame MAE about `1.55` on the
  `0-255` RGB scale.
- Treat the zero-step safety invariant as passed for the standard action path:
  untouched base parent, `trainable_backbone=full`,
  `action_backbone_added_kv_mode=none`,
  `action_token_scale=0.05`, and `action_output_zero_init=true`.
- Operator override:
  do not use `action_backbone_added_kv_mode=reuse_action_tokens` for this
  invariant test because it does not follow the official VACE Wan usage path.
- To make that zero-step test meaningful, the added-K/V upgrade path now
  zero-initializes the newly introduced image-conditioning and added-K/V
  projection parameters; targeted `pytest tests/test_wan_vace_factory.py -q`
  passed.
- Next action:
  run the first short action-conditioned training continuation from that exact
  safe checkpoint, capped at `100` steps, with `conditioning_mode=action`,
  `action_backbone_added_kv_mode=none`, `action_output_zero_init=true`, and
  checkpoint evaluation at `step_0000050` and `step_0000100`.
- Keep the `100`-step cap for any fresh training probe after this because
  training time is the bottleneck.
- Continue training from the validated zero-step action checkpoint rather than
  rebuilding a different action architecture first.

## Storage
- Storage is within comfortable headroom.
- Delete current run roots if they are dominated and no
  longer needed for baseline comparison

## Kept Code Changes
- `099a2e7`: shuffled `subset_size` training now spreads windows across the
  dataset instead of truncating to the first contiguous windows.
- `9cef3a8`: resumed runs reapply configured `lr` and `weight_decay` after
  loading optimizer state.
- `8d23255`: explicit `trainable_backbone=none` mode for true frozen-backbone
  action-only runs.
- `a41401c`: resume-path coverage for frozen action-only tuning from a
  none-conditioned checkpoint.
- `f921cdd`: resume logic now remaps PEFT `base_layer` keys so a
  none-conditioned full checkpoint can seed action-conditioned LoRA tuning.
