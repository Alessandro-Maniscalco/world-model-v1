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
- Do not treat semantic prompt conditioning as a target solution. The operator
  wants a prompt-free base DiT that already continues the video before
  action-conditioned training begins.
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
- The checkpoint-free `repo_prompt` bridge fails when the repo future-sampling
  path takes over. `f0-f8` copy the context exactly, but the first generated
  frame `f9` already shifts into a washed blue/purple desk view with the
  arm/fork fading out; `f10-f16` keep that purple wash with the plate still
  centered but no coherent tool trajectory or contact. The arm crop shows the
  same `f9` boundary failure and then loses the fork into a purple haze.
- Residualizing repo future latents around the last context frame does not fix
  the repo-path boundary failure. The prompt-conditioned residualized rerun
  still copies `f0-f8`, then visibly breaks at `f9`: the arm crop turns into a
  ghosted blue-white blob, and `f10-f16` remain blurred blue/purple with the
  fork effectively gone and no contact. It is slightly less washed out than
  the plain `repo_prompt` run, but the first bad frame and the missed-contact
  outcome do not improve.
- Replacing the gray future control template with the last context frame is the
  best repo-prompt result so far, but it is still not safe. The run stays
  coherent through `f10`: `f9-f10` keep the plate and arm structure
  recognizable, and the arm crop still shows the gripper/fork region instead of
  immediate purple wash. But `f11-f16` turn into blue-white ghosting and bloom,
  the fork never reaches contact, and the late horizon is still implausible.
- Combining both last-context anchors is the first repo-path prompt result that
  stays plausible through the full `f9-f16` horizon. The clip remains static
  through `f8`, then `f9-f16` stay non-blue and structurally coherent: the
  plate, arm, and fork remain recognizable with only mild dark-blue tinting and
  soft blur, and the arm crop keeps the tool geometry visible through `f16`.
  But the rollout is still undercommitted: the fork barely advances, never
  reaches contact, and never picks up.
- The same dual-anchor contract does not transfer to the literal zero-step
  `conditioning_mode=none` checkpoint path. The dual-anchor rerun on
  `untouched_base_none_subset8_step0_baseline_224x128/checkpoints/step_0000000.pt`
  is visually identical to the original blue floor: `f0-f8` copy context,
  `f9-f16` collapse into the same blue OOD wash, and the arm crop loses the
  fork at the same boundary. The generated MP4 is bitwise identical to the
  original baseline export (`overall MAE=0.0`, `late-frame MAE=0.0`).
- After `0456e04` fixed checkpoint-mode runtime overrides, the corrected
  checkpoint-path prompt+dual-anchor rerun still changed nothing. It again
  copies `f0-f8`, then collapses into the same blue future wash at `f9-f16`;
  the arm crop loses the fork at the same boundary, and the generated MP4 is
  again bitwise identical to the original zero-step none baseline
  (`overall MAE=0.0`, `late-frame MAE=0.0`).
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
- Keep the new `repo_prompt` result as evidence that prompt CFG alone is not
  enough once the repo future-sampling path is used. The first bad frame is
  `f9`, exactly where generated future latents begin.
- Keep the residualized `repo_prompt` rerun as a non-improving local neighbor.
  It changes the late-frame texture but does not move the first bad frame off
  `f9`, so absolute future-latent residualization alone is not enough.
- Keep the `future_control_fill_mode=last_context_frame` repo-prompt rerun as
  the best repo-path local neighbor so far. It delays the visible collapse from
  `f9` to about `f11`, but the late horizon still ghosts blue/white and misses
  contact.
- Keep the combined-anchor prompt repo run as the best repo future-sampling
  result so far. It is plausible through `f16` and proves the repo path can be
  stabilized, but it is still prompt-conditioned and misses contact, so it does
  not satisfy Stage 0 yet.
- Keep the dual-anchor none-checkpoint transfer as a failed local neighbor. It
  changed nothing at all, so inference-only control/residual anchoring is not
  enough to rescue the literal zero-token none path.
- Retire checkpoint-path prompt reruns as a target branch. Even the corrected
  prompt+dual-anchor checkpoint probe stayed bitwise identical to the blue
  none-conditioned floor, and the operator does not want prompt conditioning
  in the intended training path.
- The next bounded major lever is prompt-free and checkpoint-local: rerun the
  best trained none-conditioned checkpoint
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`
  on the fixed operator slice with the same dual-anchor inference contract.
  Distinct hypothesis: if the already-plausible non-prompt checkpoint improves
  further under dual anchors, then the right base DiT path is to keep training
  this prompt-free branch and only add action conditioning later as an exact
  or near-no-op. If `step_0000350` is unchanged or worse under dual anchors,
  then inference-contract tweaks are exhausted and the next move should shift
  to prompt-free training/design changes on the none-conditioned branch.
- Commit `3e80f6c` remains the bridge that lets these repo-path prompt probes
  stay on the fixed operator slice with the same MP4 comparison artifacts.

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
- `0456e04`: local sweeps now apply prompt/guidance and explicit runtime
  overrides after loading checkpoint metadata, including checkpoint-mode
  overrides for conditioning mode and both future-anchor settings plus focused
  pytest coverage.
