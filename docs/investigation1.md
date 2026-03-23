## Goal
Research a safer architecture for the fixed operator slice.

Stage 0 comes first:
find a `conditioning_mode=none` zero-step path that does not collapse into the
blue OOD failure on the operator slice.

Only after that base is acceptable should Stage 1 be enforced:
with `conditioning_mode=action` and zero training steps, the model should
behave as close as possible to that safer `conditioning_mode=none` path on the
same slice. Only then should action-conditioned training continue.

## Fixed Policy
- Run Python and pytest inside `.venv`.
- Review only `episode_index=1`, `start_frame=60` unless the operator changes
  it.
- Keep the same geometry unless there is a strong reason to change it:
  `224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8`.
- Start every new architecture branch from the same untouched pretrained
  backbone parent.
- Before any training continuation, validate the zero-step base safety
  invariant on the operator slice.
- Do not treat the current blue zero-step `conditioning_mode=none` artifact as
  a sufficient target. It is evidence of the current failure, not the desired
  base behavior.
- Prefer architecture choices that preserve the pretrained VACE path as
  closely as possible before introducing extra action-specific backbone
  routing.
- If a candidate zero-step `conditioning_mode=none` path is still blue or
  visibly implausible, do not move on to action training yet.
- If a candidate action-conditioned zero-step path is visibly worse than the
  best available zero-step `conditioning_mode=none` path, do not spend a long
  training run on it yet. First identify which architectural change caused the
  drift.

## Known Bad Baseline
- Current literal zero-step `conditioning_mode=none` comparator artifact:
  `runs/training_optimizer/eval/untouched_base_none_subset8_step0_baseline_224x128_ep1_start60_step0000_operator/untouched_base_none_subset8_step0_baseline_224x128_step_0000000_comparison.mp4` This artifact currently goes blue at the future boundary and is not good
  enough to serve as the final desired base behavior.

## Desired Zero-Step Properties
- First build a zero-step `conditioning_mode=none` path on the same slice and
  checkpoint-mode path that stays non-blue, plausible, and as visually stable
  as possible through the future horizon.
- Then build a zero-step `conditioning_mode=action` artifact on the same slice
  and same checkpoint-mode path.
- The action-conditioned zero-step artifact should stay as close as possible to
  the best available zero-step `conditioning_mode=none` path in visible
  quality, plausibility, and motion profile.
- If either zero-step property fails, treat that as an architecture problem,
  not a training problem.

## First Required Work
- Determine why the current zero-step `conditioning_mode=none` path turns blue
  on the operator slice.
- Find the minimal none-conditioned architecture or inference-path change that
  preserves a non-blue zero-step future rollout.
- Determine which action-path settings preserve the zero-step invariant best.
- Explicitly test whether the invariant holds when:
  `conditioning_mode=action` is enabled but action outputs are initialized to
  behave like a near-no-op.
- Identify which local additions break the invariant:
  examples include action-token projection choices, added-K/V reuse, or other
  action-specific backbone modifications.
- Only after the zero-step action path is acceptably close to a non-blue
  `conditioning_mode=none` base should longer action-training branches be
  compared.

## Open Questions
- Does the zero-step failure come from the base world-model path itself,
  from the current checkpoint-mode inference path, or from using the
  pretrained VACE path without the semantic prompt guidance it expects?
- Which minimal none-conditioned path avoids the blue collapse while staying as
  close as possible to the pretrained behavior?
- Which minimal action-conditioned configuration stays closest to the
  best non-blue none-conditioned zero-step path?
- Once the zero-step invariant is satisfied, does action training improve
  late-horizon commitment without reopening blur, ghosting, or OOD collapse?

## Proven Outcomes
- The literal zero-step `conditioning_mode=none` checkpoint baseline on the
  operator slice is still the floor, not the target:
  `f0-f8` copy context and `f9-f16` collapse into a blue OOD wash with a
  blown-out plate blob and no coherent fork trajectory.
- The canonical pretrained-base dense-prefix control on the same `17`-frame
  window does not rescue that failure. `f0-f8` stay roughly copied, but the
  first generated frame `f9` explodes into saturated rainbow block confetti
  and `f10-f16` remain unstable and trajectory-free.
- The native pretrained-base `9`-frame / `5`-condition control is less
  catastrophic but still not a safe zero-step base on this slice. `f0-f4`
  stay near the reference, but `f5-f8` turn into a bright white orb with
  green/purple halos around the plate/fork region and no clean contact path.
- These two base-path controls reject the simple wrapper-only explanation. The
  pretrained no-prompt base family is now non-improving on this slice even
  when the frame contract is moved closer to native VACE usage.

## Current Decision
- Do not spend another long run on plain no-prompt pretrained-base sweeps.
  That local neighborhood is exhausted after the `17/9` confetti failure and
  the softer but still implausible native `9/5` orb/halo failure.
- The next bounded major lever is prompt-aware base-mode validation on the same
  operator slice. The local sweep tool was dropping the caller prompt and
  always using `prompt=""`; commit `aa230a1` now forwards the configured base
  prompt into the actual base-mode inference path and is pytest-validated.
- Use that fixed prompt-aware path for one operator-slice control before any
  deeper redesign. If a task-relevant prompt still fails visibly, treat the
  pure pretrained-base family as structurally wrong for Stage 0 and pivot to a
  code-level none-conditioned redesign rather than more base-path sweeps.

## Kept Code Changes
- `aa230a1`: base-mode local sweeps now forward the configured prompt and
  guidance settings into `_run_local_pipeline`, with focused pytest coverage in
  `tests/test_sweep_local_repo_resolutions.py`.
