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
- The first prompt-aware native `9/5` control does not visibly rescue that
  failure. `f0-f4` still stay near-copied, and `f5-f8` still smear into the
  same bright orb/halo failure with no recognizable fork approach or contact.
  The prompted output is only slightly different from the no-prompt native
  output, so prompt text alone did not earn a keep decision on this slice.
- The true-CFG prompt-aware native `9/5` control is the best base-path result
  so far. `f0-f5` stay coherent and near-copied; `f6-f8` keep the plate and
  fork geometry recognizable with only a warm plate bloom and a thin
  turquoise/rainbow smear near the right edge. The arm crop stays structurally
  coherent through `f8`, but the rollout is still undercommitted and never
  makes contact.
- These two base-path controls reject the simple wrapper-only explanation. The
  pretrained no-prompt base family is now non-improving on this slice even
  when the frame contract is moved closer to native VACE usage.
- That prompted result was only a partial guidance test: the local base sweep
  was forwarding `guidance_scale`, but `_run_local_pipeline` still hardcoded
  `do_classifier_free_guidance=False`, so CFG was never actually enabled.

## Current Decision
- Keep the no-prompt pretrained-base family marked exhausted. The `17/9`
  confetti path and native `9/5` orb/halo path are still bad local neighbors.
- Keep the true-CFG prompted native `9/5` run as the current safety reference
  for pretrained behavior on this slice. Real CFG meaningfully improves the
  base path, but the result is still mostly static and misses contact.
- The next bounded major lever is to move from the canonical diffusers base
  path to the repo world-model inference path without a checkpoint, using
  prompt conditioning on the same operator slice. If prompt-conditioned repo
  inference also stays coherent and non-blue, then the null-token
  `conditioning_mode=none` path is the specific failure family. If repo prompt
  inference regresses into blur, ghosting, or collapse, focus the redesign on
  the repo inference/chunking path itself before more training.
- Commit `3e80f6c` now adds a checkpoint-free `repo_prompt` mode to
  `scripts/check/sweep_local_repo_resolutions.py`, so the next run can stay on
  the fixed operator slice and produce the same MP4 comparison artifacts while
  using the repo world-model inference path with prompt CFG.

## Kept Code Changes
- `aa230a1`: base-mode local sweeps now forward the configured prompt and
  guidance settings into `_run_local_pipeline`, with focused pytest coverage in
  `tests/test_sweep_local_repo_resolutions.py`.
- `3497d07`: base-mode local sweeps now honor classifier-free guidance when
  `guidance_scale > 1.0`, including a direct `_run_local_pipeline` regression
  test.
- `3e80f6c`: local sweeps now support checkpoint-free `repo_prompt` inference,
  including prompt CFG on the repo world-model path and pytest coverage for
  the new mode.
