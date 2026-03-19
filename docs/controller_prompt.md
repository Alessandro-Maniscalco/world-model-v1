# Shared-Session Controller Prompt

Use this file as the persistent operating contract for the shared Codex session.
The first session-start prompt inlines this file. Resumed turns refer back to the
section headings below instead of repeating the whole file.

- When deciding what to do next, read `docs/training_optimizer.md` first.
- Treat `docs/training_optimizer.md` as decision memory only.
- Use `runs/training_optimizer/experiment_ledger.md` only when you need detailed
  chronology, older validation notes, or a tie-break against older branches.
- Default policy: keep making progress until the controller disallows more long
  commands or the operator asks to stop.
- Prefer one concrete next action over extended exploration. If progress is still
  possible, do not stop just to summarize.

## Repo Edits And Rollback

- You may edit tracked repo files directly when a repo edit is the strongest next step.
- At the start of each fresh shared session, create a session-start git checkpoint
  commit before making repo edits. Use an empty commit if needed so you do not
  accidentally include unrelated worktree changes.
- After every validated repo code change you keep, create a git commit that
  includes only the files you changed for that step. Do not bundle unrelated user
  changes into these commits.
- Validate any repo edits you want to keep before returning your final JSON.
- Return `repo_edit_status=validated` only when kept repo edits were validated
  in-session.
- Return `repo_edit_status=rollback_requested` only when you want the controller
  to undo every repo edit made during the current turn.
- Otherwise return `repo_edit_status=none`.
- Rollback only restores editable repo files. It does not delete or restore
  artifacts under `runs/`.
- Record kept code-changing commits in `docs/training_optimizer.md` under
  `## Kept Code Changes`.
- Record detailed validation chronology in `runs/training_optimizer/experiment_ledger.md`.
- Do not record memory-only or ledger-only doc commits under `## Kept Code Changes`.
- You may delete artifacts under `runs/` only when they are clearly dominated and
  no longer needed for resume, validation, the latest run, the best run in any
  branch, or referenced summaries. When unsure, keep them.

## Long Experiment Commands

- Do not start long-running work inside the shared Codex session.
- If training or another long experiment is needed, return exactly one shell
  command string via `run_long_command`.
- Prefer one bounded shell chain that activates `.venv`, runs
  `scripts/train/world_model.py` when needed, then runs
  `scripts/check/sweep_local_repo_resolutions.py` and
  `scripts/check/check_generated_video_plausibility.py`.
- Run Python and pytest commands inside `.venv`.
- The command must produce all artifacts needed for the next validation step: the
  concrete checkpoint path, the generated MP4, the side-by-side comparison MP4,
  `plausibility_report.json`, and any arm-motion artifacts emitted by the sweep
  such as `*_arm_motion_report.json` and `*_arm_crop_comparison.mp4`.
- Include the concrete checkpoint path, output directory, repo id, episode index,
  start frame, video key, context length, horizon length, `k`, and resolution in
  the sweep command.
- `long_command.reason` must explain why this is the highest-value next bounded action.
- `long_command.expected_artifacts` must list the concrete artifact paths needed
  for validation.
- If a latest run result still needs validation, validate it before launching
  another long command unless there is a true blocker.

## Validation

- Validation stays inside the shared Codex session and should use short commands only.
- Validate the latest completed result first when one is available.
- Use `docs/training_optimizer.md` to decide. Use
  `runs/training_optimizer/experiment_ledger.md` only to recover detailed history.
- Review artifacts in this order:
  - full side-by-side comparison video first (`left=reference`, `right=generated`),
  - then `*_arm_crop_comparison.mp4` and `*_arm_motion_report.json` if present,
  - then metrics, logs, extracted frames, or other supporting artifacts as needed.
- Default video review command shape: `ffplay -loop 0 <comparison_video>`. If the
  clip is hard to judge in real time, extract frames with
  `ffmpeg -i <comparison_video> /tmp/<experiment_name>_%03d.png`.
- Start from concrete visible observations only. Describe what is happening in the
  generated rollout relative to the reference before proposing causes.
- Separate observations from hypotheses. Use only a small number of plausible
  causes and say what remains uncertain.
- Use these failure classes only as anchors, not as a rigid checklist: motion path
  mismatch, collapse to a common or default pose, temporal drift, late-rollout
  degradation, ghosting or instability on moving parts, contact or kinematic
  inconsistency, and brightness or color artifacts only when actually relevant.
- Even if a clip looks acceptable, explain the visible evidence that supports that conclusion.

### Motion-First Ranking

- Rank runs by visible task-relevant motion first, not by image sharpness first.
- For this repo, use this ranking order:
  - `1. arm/tool movement and commitment`
  - `2. contact and trajectory correctness`
  - `3. temporal stability on moving parts`
  - `4. overall scene fidelity and sharpness`
  - `5. aggregate metrics such as MAE`
- Treat `check_generated_video_plausibility.py` as a safety gate and tie-break input.
- If `*_arm_motion_report.json` is present, use its verdict and flags as
  supporting evidence, but do not let the JSON override an obvious human-visible
  motion win or motion failure.
- Use motion language that is decision-relevant, for example:
  `best arm movement so far`, `undercommitted`, `stops early`, `distorted late`,
  `good motion but blurry`, or `sharp but mostly static`.

### Memory Maintenance

- Keep `docs/training_optimizer.md` short and decision-oriented. It should usually
  fit on one screen without scrolling much.
- `docs/training_optimizer.md` must contain only these sections:
  - `## Stable Findings`
  - `## Best Run`
  - `## Active Decision`
  - `## Exhausted Families`
  - `## Kept Code Changes`
  - `## Resume From`
- Section contracts for `docs/training_optimizer.md`:
  - `## Stable Findings`: only durable facts that still change the next decision;
    target at most `8` bullets.
  - `## Best Run`: current winner, required comparison references, and the ranking
    takeaway that should guide the next move.
  - `## Active Decision`: the single active question. Use these bullets:
    `Question`, `Next step`, `Success signal`, and `Exit condition`. If the loop
    is paused, include `Status` first and keep the paused `Next step` as the
    concrete resume action.
  - `## Exhausted Families`: one bullet per branch family or local neighborhood
    that should not receive another near-duplicate follow-up.
  - `## Kept Code Changes`: only code-changing commits that still matter to future
    interpretation. Do not include memory-only doc commits.
  - `## Resume From`: only the concrete checkpoints, references, and artifact
    paths needed to restart quickly.
- `runs/training_optimizer/experiment_ledger.md` is the chronology file. Put
  detailed validation summaries, archived family notes, and per-run observations there.
- Delete repeated conclusions instead of restating them in multiple sections.
- If the latest result is the new best run, update `## Best Run` and remove stale
  references rather than preserving a full chronology in memory.

### Local Neighborhood Control

- A local neighborhood is a branch family where only one narrow lever is moving,
  such as checkpoint selection, one scalar sweep, one cap sweep, one context-only
  sweep, or plain continuation of the same checkpoint family.
- Every proposed long command must name the local neighborhood it belongs to and
  the distinct hypothesis it tests beyond the previous run.
- Do not spend more than `2` non-improving follow-ups inside the same local
  neighborhood after the current anchor or baseline.
- If a structural baseline is plausible and its direct continuation regresses, do
  not run another plain continuation in that same branch. Change one major lever
  or mark the family exhausted.
- Do not stack more than one new lever in a run unless the memory explicitly says
  the single-lever space is exhausted and the new run is an intentional interaction test.
- If the last `2` runs in a neighborhood both fail to improve the family anchor on
  the motion-first ranking, add that family to `## Exhausted Families` and move on.
- Prefer changing neighborhoods over making a third near-duplicate attempt.

### Decision Rule

- After validation, choose among only these outcomes:
  - make a short validated repo edit if a code change is the strongest next step,
  - return one bounded `run_long_command` if a sensible next experiment still exists,
  - return `stop` only if the operator asked to stop, the controller disallows
    more long commands, or there is a truly unfixable blocker that cannot be
    resolved in the current turn by short inspection, repo edits, or tests.
- Treat `stop` as exceptional.
- Do not use `stop` just because the current branch family, scalar sweep, or
  local hyperparameter neighborhood looks exhausted.
- If the latest validation says the next improvement should come from a code-level
  change, make that repo edit and validate it instead of stopping.
- Prefer converting "no sensible next experiment" into one small repo edit, test,
  or validation improvement when there is a plausible code-level lever to try.

## Operator Control

- Human messages added directly to the same Codex session are authoritative.
- If the operator asks to stop after the full loop, finish post-run validation and
  then return `stop`.

## Ending

- Return one raw JSON object only as your final answer.
- Always include these top-level keys:
  - `action_type`
  - `summary`
  - `session_work_summary`
  - `repo_edit_status`
  - `long_command`
  - `stop`
- `action_type` must be either `run_long_command` or `stop`.
- `summary` must be non-empty and should capture the overall outcome of the latest
  completed run or latest validated result in one paragraph.
- `session_work_summary` must be an ordered list of short paragraphs, not fragments.
- `repo_edit_status` must be one of `none`, `validated`, or `rollback_requested`.
- Always include both `long_command` and `stop` objects:
  - When `action_type=run_long_command`, fill `long_command.command`,
    `long_command.reason`, and `long_command.expected_artifacts`, and leave
    `stop.reason` empty unless needed.
  - When `action_type=stop`, set `stop.reason` to the concrete stop reason, and
    leave `long_command.command` and `long_command.reason` empty with
    `long_command.expected_artifacts=[]`.
- Use `action_type=stop` only for these cases:
  - the operator explicitly asked to stop,
  - the controller has no remaining long-command budget for this invocation,
  - a truly unfixable blocker prevents further progress in the current turn even
    after reasonable short inspection, repo edits, and tests.
- If you return `stop`, make it a full final summary that can be written directly
  into a markdown report file.
- If you return `stop`, make the final item in `session_work_summary` the overall
  takeaway and the next thing to do if optimization resumes later.
