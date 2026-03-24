## Goal
Make the prompt-free base model produce a decent future video on the fixed
operator slice before any action-conditioned branch resumes. Actions should
initially have no effect, so the base no-action continuation path must be
visually acceptable first.

## Fixed Policy
- Run Python and pytest inside `.venv`.
- Review only `episode_index=1`, `start_frame=60`.
- Do not use a text prompt.
- Do not spend long-run budget on new action-conditioned experiments until the
  no-action base path is visibly good.
- Keep the same geometry unless a result proves it is the blocker:
  `224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8`.
- Keep new base-path branches anchored to the same untouched pretrained
  Wan/VACE parent.

## Best Run
- Current best base-only checkpoint: `step_0000300` from
  `runs/untouched_base_none_pretrainedseq_dualanchor_fullft_subset8_adafactor_lr5e5_bs1gc_resume0200_step300_ckpt50/checkpoints/step_0000300.pt`.
- On the fixed operator slice it exports a full-window `17`-frame clip:
  `f0-f8` copy context, visible future motion begins at `f9`, and `f9-f16`
  keep the plate, fork, and gripper recognizable without reopening the
  catastrophic blue/purple collapse family.
- The late horizon is still not solved: mild bright bloom/ghosting appears
  around the gripper by `f14-f16`, the fork never reaches contact, and there
  were no held-out clips in this operator-only pass.

## Findings
- The legacy prompt-free `conditioning_mode=none` contract was off-distribution
  for pretrained Wan/VACE. Literal or summary null tokens plus gray future
  control caused the untouched parent to collapse into the blue/purple failure
  family at the first future frame.
- `0456e04` made checkpoint-path local sweeps apply runtime overrides after
  checkpoint metadata load, keeping zero-step comparison probes honest.
- `0348c65` stopped using literal zero tokens for none conditioning and instead
  reused Wan's empty-prompt embedding.
- `872de58` switched the none path to Wan's full empty-prompt token sequence
  and treated it as global conditioning instead of chunk-sliced repeated
  summary tokens.
- `7896373` upgraded only untouched pretrained none checkpoints (`max_steps ==
  0`) to the validated dual-anchor contract by default, while trained none
  checkpoints keep their saved contract.
- With the full empty-prompt sequence plus both anchors
  (`future_control_fill_mode=last_context_frame` and
  `future_latent_residual_mode=last_context_frame`), the untouched parent was
  the first prompt-free base branch to stay out of the catastrophic blue/purple
  collapse family on the fixed operator slice.
- Both single-anchor simplifications regressed visibly on the untouched parent.
  Control-fill-only turned the arm/fork region into a bright blue overexposed
  blob by `f1-f7`; residual-only was even worse and failed from `f0` with
  blue/purple smear and unreadable arm-crop blobs. That neighborhood is
  exhausted.
- The first startup-safe no-action continuation attempts exposed training-path
  blockers, not model-quality blockers: none tokens were still being treated as
  horizon-aligned action tokens, resume optimizer state could mismatch the
  current module layout, and `AdamW` moment buffers OOMed at `batch_size=1`.
  The accepted fixes are now in the shared worktree and the fit-proven training
  profile is `Adafactor`, `batch_size=1`, `--no-auto-batch-size`, and
  `--gradient-checkpointing`.
- The first successful bounded no-action continuation under the repaired none
  contract reached `step_0000200` and `step_0000400`. Both stayed in the safe
  visual family. `step_0000200` had better motion-first behavior than the saved
  `step_0000400`, which looked cleaner but more frozen.
- The narrow resume-from-`step_0000200` recovery run to `max_steps=300`
  succeeded through training and saved `step_0000250` and `step_0000300`. The
  long-command `returncode=1` came only from a post-run comparison helper that
  incorrectly assumed future-only `8`-frame exports.
- `step_0000250` is plausible on all `17` frames and keeps the scene coherent:
  `f0-f8` are near-exact copies, visible motion starts at `f9`, and `f9-f16`
  keep the plate, fork, and gripper recognizable with mild blur/ghosting and a
  small inward fork drift, but the late horizon is still only `misaligned`
  (`late_motion_ratio≈0.741`, `profile_correlation≈0.542`,
  `mean MAE≈6.62`, `generated_temporal_mae≈2.33`) and still misses contact.
- `step_0000300` is the best checkpoint in this branch so far by motion-first
  ranking. It is plausible on all `17` frames, still copies `f0-f8`, and
  visible future motion begins at `f9`. From `f9-f16`, the fork/gripper region
  stays recognizable and more committed than `step_0000250`, with no return to
  the blue/purple collapse family; the tradeoff is mild late bright
  bloom/ghosting around the gripper in the arm crop. The motion report finally
  turns `good` (`late_motion_ratio≈0.894`, `profile_correlation≈0.708`,
  `mean MAE≈5.06`, `generated_temporal_mae≈2.14`). There were no held-out
  clips in either resume-eval pass.
- Shape-aware diffs back the visual ranking: `step_0000300` is much closer to
  the earlier good-motion `step_0000200` future tail than `step_0000250` is
  (`overall MAE≈4.96` vs `≈10.49`). The older saved `step_0000400` remains the
  more frozen endpoint of this same training family.
- The first `step_0000300 -> step_0000350` continuation attempt did not reach a
  checkpoint or any eval artifacts, so there were no new videos or held-out
  clips to review from that long command. The recorded `train_stdout.log`
  stopped after the initial device/dtype/resume lines, but an exact local
  reproduction from the same `step_0000300` checkpoint with the same training
  flags and `max_steps=301` completed successfully through
  `final_checkpoint=...step_0000301.pt`. That means the failed `step_0000350`
  command does not currently look like a deterministic model-path or
  configuration bug; the highest-probability explanation is a transient
  runner-side interruption during startup or early model load.
- The clean rerun of the same continuation did answer the checkpoint question.
  `step_0000350` is still plausible on all `17` frames and stays out of the
  blue/purple collapse family, but it regresses on motion-first ranking
  relative to `step_0000300`. It still copies `f0-f8` and visible future motion
  begins at `f9`, yet `f9-f16` stall earlier: the fork/gripper remains
  recognizable and slightly cleaner than `step_0000300`, but late motion drops
  enough that the arm report falls back to `undercommitted`
  (`late_motion_ratio≈0.539`, `profile_correlation≈0.452`). There were no
  held-out clips. `step_0000350` is closer to `step_0000300` than to the older
  `step_0000400` freeze (`overall MAE≈3.45` vs `step_0000300`, `≈8.16` vs
  `step_0000400` future tail), but the plain continuation family has now
  visibly peaked at `step_0000300`.
- The first zero-step `conditioning_mode=action` probe on top of
  `step_0000300` did not reach video export, so there were no new videos or
  held-out clips to review from that long command. The first failing stage was
  checkpoint overlay: the none-conditioned `step_0000300` checkpoint saves
  `action_encoder_state_dict = {'base_token': ...}` from
  `NullConditioningEncoder`, and the action probe incorrectly tried to load
  that null-conditioning token into `ActionTokenEncoder`, aborting with
  `RuntimeError: Unexpected action-encoder checkpoint keys: ['base_token']`.
  Commit `7296a3a` now treats a base-token-only none checkpoint the same way as
  the earlier empty-state none checkpoints and preserves the fresh zero-init
  action encoder. That fix is validated by `47` passing tests across
  `tests/test_wan_vace_factory.py` and
  `tests/test_sweep_local_repo_resolutions.py`, plus a real `step_0000300`
  checkpoint smoke that now builds an `ActionTokenEncoder` with zero nonzero
  entries in `output_proj.weight` and `output_proj.bias`.
- The clean rerun of the same zero-step action probe did answer the invariant
  question. It exports a full-window `17`-frame clip, so `f0-f8` are copied
  context and visible future motion starts at `f9`. From `f9-f16`, the
  action-conditioned rollout stays in the same safe visual family as the
  `step_0000300` none baseline: the plate, fork, and gripper remain
  recognizable, there is no return to the blue/purple collapse family, and
  there were no held-out clips. The perturbation is small but not literally
  zero. Visually, the late horizon looks slightly flatter and slightly less
  bloomy than the none baseline, while the arm report falls from `good` to
  `misaligned` on a narrow margin (`late_motion_ratio≈0.819`,
  `profile_correlation≈0.690`, `total_motion_ratio≈0.976`). The direct
  full-window diff against the none baseline is still small
  (`overall MAE≈1.20` on the `0-255` RGB scale), so the current action path is
  close enough to neutral to justify the first bounded action-training run.

## Active Question
- Starting from the repaired `step_0000300` base checkpoint and a nearly
  neutral zero-init action path, can a first bounded action-conditioned
  continuation improve task-relevant fork/gripper motion without reopening blue
  wash, heavy ghosting, or collapse?

## Current Decision
- The checkpoint-selection question is answered: `step_0000350` is cleaner but
  more frozen, so `step_0000300` remains the best base-only checkpoint and this
  plain continuation neighborhood is exhausted.
- The action no-op question is answered well enough to move on. The next
  controller pass should spend the long-run budget on the first bounded
  action-conditioned continuation from `step_0000300` under the same repaired
  dual-anchor contract: keep the fresh zero-init action path, use
  `trainable_backbone=none` so the backbone stays frozen, and evaluate the
  branch at small checkpoints before `400` total steps.
- Rank that result by:
  visual inspection first,
  whether task-relevant fork/gripper motion becomes more committed than the
  zero-step action probe and the none-conditioned `step_0000300` baseline,
  whether the rollout stays in the same safe visual family without reopening
  blue wash, heavy ghosting, or early stop,
  and only then scalar MAE.

## Not Needed Now
- Do not resume action-conditioned branches until the prompt-free base path is
  visually acceptable.
- Do not spend long-run budget on prompt-conditioned branches.
- Do not spend more budget reranking `step_0000250` versus `step_0000300`; that
  comparison is already resolved in favor of `step_0000300`.
- Do not spend more long-run budget on plain `step_0000300 -> step_0000x` full
  backbone continuations without changing a major lever; `step_0000350`
  already showed the branch freezing again.
- Do not spend more long-run budget on additional zero-step action no-op probes
  unless a later contradiction appears; the current rerun is already close
  enough to the none baseline to justify the first action-training branch.
- Do not revisit the exhausted zero-token, summary-token, control-fill-only, or
  residual-only zero-step neighborhoods unless a later contradiction appears.
- Do not treat the trained `fullft_subset8_spread_resume200_lr5e5_step400`
  `step_0000350` checkpoint as the active parent for this stage; it remains
  only the quality reference.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata.
- `0348c65`: prompt-free none conditioning stopped using literal zero tokens
  and now reuses Wan's empty-prompt embedding.
- `872de58`: prompt-free none conditioning now uses Wan's full empty-prompt
  token sequence and treats it as global conditioning instead of chunk slicing.
- `7896373`: untouched pretrained none checkpoints (`max_steps == 0`) now
  default to the validated dual-anchor contract, while trained none checkpoints
  keep their saved contract.
- `7296a3a`: action probes from none-conditioned checkpoints now treat a
  base-token-only `action_encoder_state_dict` as null-conditioning state and
  keep the fresh zero-init `ActionTokenEncoder`.
- `validated in worktree`: training-side chunkwise flow matching now accepts
  global prompt-free none-conditioning token sequences, and
  `_resume_training_state` can fall back to a fresh optimizer when saved
  optimizer parameter groups do not match the current module layout.
