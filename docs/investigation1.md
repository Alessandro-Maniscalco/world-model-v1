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
  full-window export, `f0-f8` copy context and the first future frame
  (`f9`) collapses into the blue/purple failure family.
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
- The strongest code-level lever found so far is to stop feeding literal zero
  text tokens to the prompt-free none path. A real-model smoke check confirmed
  Wan's empty prompt produces a dense nonzero `4096`-d null token, and the repo
  now uses that pretrained-compatible null token by default in
  `conditioning_mode=none`.
- The interrupted eval root
  `runs/training_optimizer/eval/untouched_base_none_nulltokenfix_224x128_ep1_start60_step0000_operator`
  currently contains only `eval_stdout.log`. Treat it as unfinished and
  unvalidated.

## Active Question
- Can the prompt-free no-action base path be made visually acceptable by
  replacing literal zero null-conditioning tokens with a pretrained-compatible
  empty-prompt token on the untouched parent checkpoint?

## Current Decision
- Keep the validated code change that initializes prompt-free none conditioning
  from Wan's empty-prompt text embedding instead of literal zero tokens.
- The next controller pass should do only one research loop:
  inspect the interrupted
  `runs/training_optimizer/eval/untouched_base_none_nulltokenfix_224x128_ep1_start60_step0000_operator`
  directory, validate it if complete, and otherwise rerun that exact operator-
  only checkpoint eval from scratch. Do not branch away before answering this
  question.
- Rank the result by:
  visual inspection first,
  whether the first future frame loses the blue/cyan tint,
  whether the fork/gripper region stays sharp and recognizable without ghosting,
  and only then scalar MAE.

## Not Needed Now
- Do not resume action-conditioned branches until the prompt-free base path is
  visually acceptable.
- Do not spend time on prompt-conditioned branches; they answered earlier
  diagnosis questions but are off-policy for the target model.
- Do not expand beyond `episode_index=1`, `start_frame=60`.
- Do not treat the trained `step_0000350` checkpoint as the active parent for
  this stage; it is only the quality reference.
- Do not do broader repository exploration before the null-token base eval is
  validated; the unresolved question is already concrete.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata, which keeps base-path comparison probes honest.
- `0348c65`: prompt-free none conditioning now reuses Wan's empty-prompt text
  embedding instead of literal zero tokens, while the action path keeps its
  validated zero-init no-op behavior and legacy checkpoint compatibility.
