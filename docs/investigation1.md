## Goal
Make the prompt-free base model continue video cleanly on the fixed slice
before any action conditioning matters. Actions should be absent or exact
no-op until the base continuation path is visually sane.


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
- Best prompt-free behavior reference remains the trained none-conditioned
  checkpoint
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`:
  it stays coherent through `f16`, shows only mild late blur/ghosting, and
  still just misses contact.
- No untouched-parent zero-step base fix is selected yet.

## Canonical Baselines
- Untouched pretrained zero-step `conditioning_mode=none` on the operator slice
  is the floor:
  `f0-f8` copy context and `f9-f16` collapse into a blue OOD wash.
- The best trained prompt-free behavior reference is
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`:
  it stays plausible through `f16` but remains undercommitted and misses
  contact.
- The latest untouched-parent zero-step `conditioning_mode=action` probe stays
  nearly identical to that bad none-conditioned floor, so actions are not the
  reason the untouched base goes blue.

## Active Question
- Which prompt-free base architecture change makes the untouched parent stop
  collapsing into the blue OOD wash on `episode_index=1`, `start_frame=60`?
- Keep actions out of the diagnosis path. If actions are enabled at all, they
  must remain exact no-op and must not be part of the explanation for success.

## Proven Outcomes
- Prompt-conditioned and inference-anchor explorations answered earlier
  diagnosis questions, but they are not the target branch for this controller.
- `action_backbone_added_kv_mode=reuse_action_tokens` is off-policy for the
  standard zero-step safety check.
- The untouched-parent action no-op probe proved the blue failure is already in
  the base continuation path, not in the action path.
- The strongest evidence from earlier control probes is that the pretrained Wan
  backbone can stay non-blue when its conditioning contract is closer to native
  VACE behavior; the current prompt-free repo `none` path is the mismatched
  part.
- A kept repo change now replaces the literal all-zero null-conditioning
  tokens with a pretrained-compatible empty-prompt summary token in
  `conditioning_mode=none`, while preserving the old zero-token fallback for
  explicit tests and ablations.

## Current Decision
- Pause action-branch work until the untouched prompt-free base is fixed.
- Validate the new null-conditioning contract on the untouched zero-step
  `conditioning_mode=none` operator slice first.
- If the new base path stays non-blue and coherent through `f9-f16`, keep it
  as the new Stage-0 architecture; if it is still visibly blue, the next lever
  should be the future-control / residual contract, not action training.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata, which keeps zero-step comparison probes honest.
- Validated worktree change in `src/world_model/models/wan_vace_factory.py`:
  when probing `conditioning_mode=action` from a none-trained checkpoint, keep
  a fresh zero-initialized action encoder instead of loading null-conditioning
  weights. This change was validated before the latest `step_0000350` action
  probe and is required for meaningful zero-step action comparisons.
- Validated worktree changes in
  `src/world_model/models/wan_vace_conditioning.py` and
  `src/world_model/models/wan_vace_factory.py`:
  `conditioning_mode=none` now uses a reusable pretrained-compatible null
  token instead of literal zero cross-attention by default. Targeted pytest
  coverage passed before operator-slice revalidation.
