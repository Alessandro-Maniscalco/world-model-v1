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

## Proven Outcomes
- The legacy prompt-free `conditioning_mode=none` contract is not neutral:
  literal/summary null tokens plus gray future control and no latent residual
  are off-distribution for pretrained Wan/VACE, which is why the untouched
  parent collapses into the blue/purple failure family.
- Correct checkpoint runtime overrides now apply after checkpoint metadata load
  (`0456e04`), so zero-step checkpoint probes are trustworthy.
- `872de58` fixed the none-token path so checkpoint evals use Wan's full
  empty-prompt token sequence as global conditioning instead of chunked repeated
  summary tokens.
- With that token fix plus both anchors
  (`future_control_fill_mode=last_context_frame` and
  `future_latent_residual_mode=last_context_frame`), the untouched-parent base
  path becomes the first none-conditioned branch that stays out of the
  catastrophic blue/purple collapse family. In the validated
  `untouched_base_none_pretrainedseq_dualanchor_...` run, prediction is
  future-only and starts immediately at generated `f0`, but `f0-f7` keep the
  plate, fork, and gripper recognizable. The remaining defect is milder: the
  prediction is darker and cooler than the reference from `f0`, soft late
  ghosting appears by the end of the horizon, and the fork still never reaches
  contact. There were no held-out clips. Plausibility passes on all 8 frames.
- Both single-anchor simplifications regress visibly on the same untouched
  parent and fixed slice:
  `untouched_base_none_pretrainedseq_controlfilllastctx_...` keeps only
  last-context control fill and by `f1-f7` turns the arm/fork region into a
  bright blue overexposed blob with a blown-out plate edge and arm-crop haloing;
  `untouched_base_none_pretrainedseq_residuallastctx_...` keeps only the latent
  residual anchor and is even worse, failing from `f0` with full-frame
  blue/purple smear, extreme color shift, and unreadable arm-crop blobs. There
  were no held-out clips in either run. This exhausts the single-anchor
  neighborhood.
- The plain no-override untouched-parent eval,
  `untouched_base_none_pretrainedseq_defaultdualanchor_...`, is bitwise
  identical to the validated manual dual-anchor clip
  `untouched_base_none_pretrainedseq_dualanchor_...` (`overall MAE=0.0`,
  `per-frame MAE=0.0`). Prediction is still future-only and starts immediately
  at generated `f0`, but `f0-f7` keep the plate, fork, and gripper
  recognizable, stay out of the catastrophic blue/purple collapse family, and
  only retain the milder darker cool tint plus soft late ghosting from the
  validated dual-anchor branch. There were no held-out clips. The saved summary
  metadata still reports the legacy `gray` / `none` values, but that is now
  only a reporting mismatch, not a model-path mismatch.

## Active Question
- With the prompt-free none contract now repaired and reaching the real
  untouched-parent eval path, does a short no-action training continuation from
  the untouched pretrained parent improve motion and scene fidelity without
  reopening the blue-collapse family?

## Current Decision
- Keep the validated code change that upgrades only untouched pretrained none
  checkpoints (`max_steps == 0` in checkpoint metadata) from the legacy
  `gray`/`none` contract to the validated dual-anchor contract while leaving
  trained none checkpoints alone. The implementation lives in
  `src/world_model/models/wan_vace_factory.py`, targeted validation passed with
  `76` tests across `tests/test_wan_vace_factory.py`,
  `tests/test_sweep_local_repo_resolutions.py`, and
  `tests/test_infer_world_model_wan_vace.py`, and the plain no-override
  operator-slice eval now confirms that the default reaches the real
  checkpoint-eval path.
- The next controller pass should spend the long-run budget on one bounded
  no-action base-training continuation from the untouched pretrained none
  parent, not on more zero-step architecture ablations. Keep the fixed geometry
  (`224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8`), keep
  `conditioning_mode=none`, keep actions out of the run, cap the continuation
  at `400` steps, and then evaluate the resulting checkpoints on the fixed
  operator slice. The question is whether the repaired prompt-free base
  contract can now learn useful future motion without reopening the blue/purple
  failure family.
- Rank the result by:
  visual inspection first,
  whether the future horizon stays out of the catastrophic blue/purple collapse
  family,
  whether the fork/gripper region stays recognizable and motion becomes more
  committed,
  whether the darker cool tint and late ghosting shrink relative to the current
  zero-step dual-anchor/default-dual-anchor baseline,
  and only then scalar MAE.

## Not Needed Now
- Do not resume action-conditioned branches until the prompt-free base path is
  visually acceptable.
- Do not spend time on prompt-conditioned branches; they answered earlier
  diagnosis questions but are off-policy for the target model.
- Do not expand beyond `episode_index=1`, `start_frame=60`.
- Do not treat the trained `step_0000350` checkpoint as the active parent for
  this stage; it is only the quality reference.
- Do not rerun the older zero-token, summary-token, control-fill-only,
  residual-only, manual-dual-anchor, or plain-default-dual-anchor zero-step
  branches unless the first bounded training continuation creates a
  contradiction that needs a direct comparison.
- Do not spend time fixing the saved summary metadata mismatch before the first
  bounded no-action training continuation is answered; the model-path question
  is already resolved.
- Do not do broader repository exploration before the first bounded untouched-
  parent training continuation is answered; the unresolved question is already
  concrete.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata, which keeps base-path comparison probes honest.
- `0348c65`: prompt-free none conditioning stopped using literal zero tokens
  and began reusing Wan's empty-prompt text embedding instead.
- `872de58`: prompt-free none conditioning now uses Wan's full empty-prompt
  token sequence in the local checkpoint-eval path, and none-mode rollout
  treats that sequence as global conditioning instead of chunk slicing.
- `7896373`: untouched pretrained none checkpoints (`max_steps == 0`) now
  default to the validated dual-anchor contract in the runtime factory, while
  trained none checkpoints keep their saved contract.
