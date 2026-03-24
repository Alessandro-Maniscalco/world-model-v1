## Goal
Make the prompt-free base model produce a decent future video on the fixed slice
before any action-conditioned branch resumes. Actions should initially have no
effect, so they are off-policy until the base continuation path is visually
acceptable.

## Fixed Policy
- Run Python and pytest inside `.venv`.
- Review only `episode_index=1`, `start_frame=60`.
- Do not use a text prompt.
- Do not run new action-conditioned experiments until the no-action base path is
  visibly good.
- Keep the same geometry unless a result proves it is the blocker:
  `224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8`.
- Keep new base-path architecture branches anchored to the same untouched
  pretrained Wan/VACE parent.

## Canonical Baselines
- Untouched pretrained zero-step `conditioning_mode=none` is the floor. In the
  full-window export, `f0-f8` copy context and the first future frame (`f9`)
  collapses into the blue/purple failure family.
- The best prompt-free trained reference remains
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`.
  It stays coherent through `f16` with mild late soft blur/ghosting and still
  misses contact. Use it only as a reference for what a decent prompt-free base
  looks like, not as the active parent for base-architecture diagnosis.
- The old untouched-parent no-action probes are only diagnosis evidence now.
  They proved the blue collapse is already in the base continuation path; they
  are not current candidate solutions.

## Proven Outcomes
- The current prompt-free `conditioning_mode=none` contract is not neutral:
  it still routes the untouched backbone through null conditioning plus future
  control defaults that are off-distribution for pretrained Wan/VACE.
- Correct checkpoint runtime overrides now apply after checkpoint metadata load
  (`0456e04`), so zero-step checkpoint probes are trustworthy.
- Replacing literal zero tokens with a pretrained empty-prompt summary token
  helped but did not fix the base path. In the validated
  `untouched_base_none_pretrainednull_dualanchor_...` run, prediction starts
  immediately at generated `f0` because the artifact is future-only; `f0-f2`
  keep the plate and fork recognizable with less catastrophic wash than the old
  zero-token dual-anchor probe, but `f3-f7` still pick up strong blue/yellow
  color shift, arm-crop ghosting, and missed contact. There were no held-out
  clips. Plausibility fails on `f3-f7`, and the generated video still sits far
  from the original none floor (`overall MAE≈41.1`) even though it is closer to
  the earlier zero-token dual-anchor probe (`overall MAE≈13.45`).
- The remaining architectural mismatch after that run was concrete: the repo
  was still treating `conditioning_mode=none` as chunk-conditioned
  per-future-step tokens, so the empty prompt was collapsed into one repeated
  token instead of a global text-token sequence like the pretrained Wan prompt
  path.
- The repo now uses Wan's full empty-prompt token sequence for
  `conditioning_mode=none` in the local checkpoint-eval path instead of a
  repeated summary token. A real-model smoke check confirms the
  null-conditioning tensor shape is `512x4096`, and the runtime encoder emits
  `1x512x4096` tokens for an 8-step future horizon.
- That full empty-prompt token sequence materially improves the untouched-parent
  base path on the dual-anchor contract. In the validated
  `untouched_base_none_pretrainedseq_dualanchor_...` run, prediction still
  starts immediately at generated `f0` because the artifact is future-only, but
  `f0-f7` keep the plate, fork, and gripper recognizable and stay out of the
  catastrophic blue/purple collapse family. The remaining defect is milder:
  the prediction is darker and cooler than the reference from `f0`, soft late
  ghosting appears by the end of the horizon, and the fork still never reaches
  contact. There were no held-out clips. Plausibility passes on all 8 frames,
  the generated video differs from the old original none floor by
  `overall MAE≈39.64`, differs from the older zero-token dual-anchor probe by
  `≈8.31`, and differs from the summary-token dual-anchor probe by `≈17.55`.
- Removing the future latent residual anchor while keeping only
  `future_control_fill_mode=last_context_frame` regresses the same
  untouched-parent branch. In the validated
  `untouched_base_none_pretrainedseq_controlfilllastctx_...` run, prediction is
  future-only and starts immediately at generated `f0`; `f0` is still
  recognizable, but by `f1-f7` the arm/fork region turns into a bright blue
  overexposed blob with a blown-out plate edge and obvious arm-crop haloing.
  The fork remains barely recognizable, motion stops early, and contact is
  still missed. There were no held-out clips. Plausibility fails on `f1-f7`,
  `overall MAE≈18.18` versus the validated dual-anchor run, and the branch is
  visibly worse despite staying on the same token sequence.
- The older eval root
  `runs/training_optimizer/eval/untouched_base_none_nulltokenfix_224x128_ep1_start60_step0000_operator`
  still contains only `eval_stdout.log`. Treat it as unfinished and
  unvalidated.

## Active Question
- Can the prompt-free no-action base path keep the recovered quality with
  `future_latent_residual_mode=last_context_frame` alone, or does the
  untouched parent really need both anchors together?

## Current Decision
- Keep the validated code change that makes prompt-free none conditioning use
  Wan's full empty-prompt token sequence as global conditioning instead of
  chunk-conditioned repeated tokens.
- Control-fill-only is no longer an open candidate: it visibly regressed.
- The next controller pass should do one final bounded simplification test on
  the same untouched parent and fixed operator slice: rerun the zero-step none
  eval with `future_latent_residual_mode=last_context_frame` but without
  `future_control_fill_mode=last_context_frame`, then compare that result
  directly against the validated
  `untouched_base_none_pretrainedseq_dualanchor_...` clip. The goal is to find
  out whether the residual anchor alone is enough or whether the dual-anchor
  contract is the minimal stable prompt-free base path.
- Rank the result by:
  visual inspection first,
  whether the fork/gripper region stays recognizable through `f7`,
  whether the blue overexposed arm/fork blob returns,
  whether late ghosting and the cool tint get better or worse,
  and only then scalar MAE.

## Not Needed Now
- Do not resume action-conditioned branches until the prompt-free base path is
  visually acceptable.
- Do not spend time on prompt-conditioned branches; they answered earlier
  diagnosis questions but are off-policy for the target model.
- Do not expand beyond `episode_index=1`, `start_frame=60`.
- Do not treat the trained `step_0000350` checkpoint as the active parent for
  this stage; it is only the quality reference.
- Do not rerun the older zero-token or summary-token none branches unless a new
  regression makes the comparison necessary again.
- Do not rerun the control-fill-only branch unless the residual-only result
  creates a contradiction that needs a direct three-way visual comparison.
- Do not do broader repository exploration before the single-anchor none
  ablation is answered; the unresolved question is already concrete.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata, which keeps base-path comparison probes honest.
- `0348c65`: prompt-free none conditioning stopped using literal zero tokens
  and began reusing Wan's empty-prompt text embedding instead.
- `872de58`: prompt-free none conditioning now uses Wan's full empty-prompt
  token sequence in the local checkpoint-eval path, and none-mode rollout
  treats that sequence as global conditioning instead of chunk slicing.
