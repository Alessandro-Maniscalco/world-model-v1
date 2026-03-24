## Goal
Find the safest prompt-free starting point for action training on the fixed
slice. The action path should begin as a no-op, preserve the base video
continuation behavior, and then be used for bounded action-conditioned
training.

## Fixed Policy
- Run Python and pytest inside `.venv`.
- Review only `episode_index=1`, `start_frame=60` unless the operator changes
  it.
- Do not use a text prompt.
- Keep the same geometry unless there is a strong reason to change it:
  `224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8`.
- Start every new architecture branch from the same untouched pretrained
  backbone parent.
- Before any training continuation, validate the zero-step base safety
  invariant on the operator slice.
- Prefer architecture choices that preserve the pretrained VACE path as
  closely as possible before introducing extra action-specific backbone
  routing.

## Best Run
- Best usable prompt-free parent for action training is the existing
  none-conditioned checkpoint
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`.
- Its zero-step `conditioning_mode=action` override with a fresh zero-init
  action encoder stays plausible and close to the matching none-conditioned
  reference (`overall MAE≈2.54`, `late-frame MAE≈4.06`) with no new artifact
  family.
- The latest bounded training branch
  `fullft_subset8_spread_resume200_lr5e5_step400_actionfreeze_zeroinit_subset8_step450`
  is promising but not fully ranked yet because only the `step_0000400`
  operator eval completed before the shell failed.

## Canonical Baselines
- Untouched pretrained zero-step `conditioning_mode=none` on the operator slice
  is the floor:
  `f0-f8` copy context and `f9-f16` collapse into a blue OOD wash.
- The best prompt-free trained behavior reference is
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`:
  it stays plausible through `f16` but remains undercommitted and misses
  contact.
- The latest untouched-parent zero-step `conditioning_mode=action` probe is a
  clean no-op relative to that floor, not an improvement:
  it outputs only the 8 future frames, and all `f0-f7` stay in the same blue
  failure family as baseline `f9-f16` with future-only MAE≈`1.88`.
- Treat the untouched-parent action no-op as a safety pass only. It does not
  earn another architecture branch because it preserves a bad base model.

## Active Question
- Which first bounded prompt-free action-conditioned training continuation best
  preserves the `step_0000350` base while starting to learn from actions on
  `episode_index=1`, `start_frame=60`?
- Keep the action path minimal:
  fresh zero-init output projection, no prompt, no added-K/V routing, and no
  extra action-temporal modules unless a bounded run proves they are needed.

## Proven Outcomes
- Prompt-conditioned and inference-anchor explorations answered earlier
  diagnosis questions, but they are not the target branch for this controller.
- `action_backbone_added_kv_mode=reuse_action_tokens` is off-policy for the
  standard zero-step safety check.
- The repo can now probe `conditioning_mode=action` from a none-trained
  checkpoint with a fresh zero-init action encoder, so zero-step action
  comparisons are meaningful.
- The latest untouched-parent zero-step action no-op probe shows the fresh
  action encoder does not introduce a new failure family on its own; it stays
  nearly identical to the matching untouched none future horizon.
- The `step_0000350` zero-step action override proves the same minimal action
  path can stay plausible on a good prompt-free base, so the base checkpoint
  rather than the action architecture is now the dominant lever.
- The first frozen-backbone action-training continuation does not reopen the
  blue failure family at `step_0000400`. The completed operator eval is
  future-only (`8` frames), but all `f0-f7` stay coherent and non-blue with
  only mild arm blur/ghosting and a missed-contact near-miss. Motion remains
  undercommitted (`late_motion_ratio≈0.64`), and future-only comparison
  against the `step_0000350` none baseline is still close enough to keep
  (`overall MAE≈4.15`).
- The latest shell failure was in the ad-hoc equivalence step, not training or
  checkpoint evaluation. The completed `step_0000400` eval artifacts decode
  cleanly, and the failing comparison script only assumed a `17`-frame output
  when the current checkpoint-mode action eval artifact is future-only.

## Current Decision
- Retire further zero-step architecture sweeps from the untouched pretrained
  parent.
- Use the minimal fresh zero-init action path already validated on
  `step_0000350` as the training architecture.
- Next run:
  finish the interrupted frozen-backbone branch by evaluating
  `step_0000450` on the fixed operator slice and comparing both completed eval
  clips as future-only horizons against the `step_0000350` none baseline.
- Do not spend another training run until `step_0000450` is reviewed video-
  first. The current branch may already be good enough to continue or may have
  regressed after `step_0000400`; the missing evidence is only the final eval.
- Rank new training runs by:
  visual inspection first,
  whether `f0-f8` remain copied and `f9-f16` stay coherent without a new
  artifact family,
  whether the arm shows more committed late motion without blur/ghosting,
  and only then scalar closeness to the matching none-conditioned reference.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata, which keeps zero-step comparison probes honest.
- Validated worktree change in `src/world_model/models/wan_vace_factory.py`:
  when probing `conditioning_mode=action` from a none-trained checkpoint, keep
  a fresh zero-initialized action encoder instead of loading null-conditioning
  weights. This change was validated before the latest `step_0000350` action
  probe and is required for meaningful zero-step action comparisons.
