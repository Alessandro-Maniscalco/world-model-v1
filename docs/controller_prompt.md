# Shared-Session Controller Prompt

Use this file as the persistent operating contract for the shared Codex session.
The first session-start prompt inlines it. Later turns should refer back to the
section headings instead of repeating the whole file.

- Read `docs/complexity_ladder_training.md` first.
- Treat `docs/complexity_ladder_training.md` as the decision memory for the
  active ladder process.
- Treat `docs/training_optimizer.md` as legacy background only when older
  findings are still useful for context, not as the primary next-step driver.
- Treat `runs/training_optimizer/experiment_ledger.md` as chronology only.

Everything is run on a RTX 3080 with 16GB of VRAM.

## Repo Edits And Rollback

- Repo edits are allowed when they are the strongest next step, including larger
  structural edits that test one coherent hypothesis better than another narrow
  sweep.
- Before editing, create a session-start git checkpoint commit.
- Keep only validated edits. After any kept validated code change, create a
  commit containing only your files.
- Use `repo_edit_status=validated` only for kept validated edits,
  `rollback_requested` when the controller should undo all edits from this turn,
  otherwise `none`.
- Use rollback freely for speculative edits that fail validation or do not earn
  a better next decision.
- Rollback affects repo files only, not `runs/`.
- Record kept code-changing commits in `docs/complexity_ladder_training.md`
  under `## Kept Code Changes`. Put detailed chronology in
  `runs/training_optimizer/experiment_ledger.md`.
- Delete `runs/` artifacts only when they are clearly dominated and no longer
  needed. When unsure, keep them.

## Long Experiment Commands

- Never start long-running work inside the shared session. If needed, return
  exactly one shell command via `run_long_command`.
- Run Python and pytest inside `.venv`.
- Prefer one bounded shell chain: train if needed, then
  `scripts/check/sweep_local_repo_resolutions.py`.
- Treat the sweep's `plausibility_report.json` as the canonical plausibility
  output unless you need separate ad hoc validation.
- The command must produce the checkpoint path and the artifacts needed for
  review: generated video, comparison video, plausibility report, and any
  arm-motion artifacts.
- Include the concrete checkpoint path, output directory, repo id, episode
  index, start frame, video key, context length, horizon length, `k`, and
  resolution in sweep commands.
- `long_command.reason` should say why this is the highest-value bounded action,
  justified with concrete video visual description.
- In-session code edits are effectively free relative to another training or
  evaluation run.
- `long_command.expected_artifacts` should list the concrete artifact paths to
  review.
- Validate the latest result before launching another long command unless there
  is a true blocker.
- Cap new exploratory runs at `400` steps first. Continue to `800` only if the
  run is still plausible and clearly improving.
- Prefer the highest expected-value next action for the overall runtime budget.
  If one structural edit plus one run is more likely to change the outcome than
  a cheap rescue check, prefer the structural edit.

## Validation

- Use short commands only.
- Review artifacts in this order:
  - Visual inspection of all videos. Never skip this. `left=target/reference` and `right=prediction`. Begin by reviewing the last `horizon_len` frames.
  - then `*_arm_crop_comparison.mp4` and `*_arm_motion_report.json` if present,
  - then `plausibility_report.json`,
  - then `metrics.jsonl` and logs
  - inspect what you think is useful
- Watch enough of each reviewed clip to describe the
  visible motion pattern in sentences, not just labels or metrics.
- If the video visibly goes bad, `*_arm_motion_report.json` is not needed. Visible collapse, incoherent motion, or late-horizon failure is enough to reject it even when scalar metrics look acceptable.
- Use `docs/complexity_ladder_training.md` to choose the next step and the
  ledger only for detail.

### Motion-First Ranking

- Rank runs by visible task-relevant motion first.
- Use this order:
  - `1. visual inspection`
  - `2. arm/tool movement and commitment`
  - `3. contact and trajectory correctness`
  - `4. temporal stability on moving parts`
  - `5. overall scene fidelity and sharpness`
  - `6. aggregate metrics such as MAE`
- Treat plausibility as a safety gate and tie-break input.
- When visual quality and metrics disagree, trust the video for keep/drop
  decisions.
- Use `*_arm_motion_report.json` as supporting evidence, not as a replacement
  for obvious human-visible judgment.
- Reasons and findings should mention the main clip and each held-out clip that
  was reviewed when their visible behavior differs in a decision-relevant way.

## Memory Maintenance

- Keep `docs/complexity_ladder_training.md` short and decision-oriented.
- It must contain only:
  - `## Goal`: the ladder objective in one short paragraph.
  - `## Proven Complexity ladder`: only proven rungs, each with a short
    description and the best video link.
  - `## Next complexity to test`: exactly one active rung, with why it is next.
  - `## Best rung for current complexity`: the current rung's best run only.
  - `## Rung Findings for current complexity`: compact findings for the active
    rung, one point per rung transition.
  - `## Stable Findings`: durable facts that apply across multiple rungs.
  - `## Kept Code Changes`: still-relevant code-changing commits only.
- Put detailed validation summaries and chronology in
  `runs/training_optimizer/experiment_ledger.md`.
- Re-review `## Next complexity to test` after every validated run and before
  choosing the next action. Rewrite it whenever the latest evidence changes
  which rung should be active next.
- In ladder findings, record the specific visible behavior that changed the
  decision, especially when motion starts, how long it stays static, and any
  blur, ghosting, or missed contact in held-out clips.
- Delete repeated conclusions instead of restating them.

## Ladder Promotion

- Work from the easiest rung likely to generate a good video upward, not the
  absolute minimum-complexity rung.
- Prefer the shortest future horizon, enough observed context to stabilize the
  scene, and the simplest conditioning path that visibly helps motion when
  choosing the base rung.
- Change only one major complexity axis at a time when promoting to the next
  rung.
- Do not add a rung to `## Proven Complexity ladder` until it has a validated
  best run with visible task-relevant motion and acceptable plausibility.
- If a short-window scout rung wins, keep it as a proven rung but do not treat
  it as a project win until a later harder rung shows the gain transfers
  upward.

## Local Neighborhood Control

- Every proposed run must name its local neighborhood and the distinct
  hypothesis it tests, and identify which ladder rung it belongs to.
- Do not spend more than `2` non-improving follow-ups inside the same
  neighborhood after the current anchor.
- If a direct continuation regresses, do not run another plain continuation in
  that branch. Change one major lever or mark the family exhausted.
- If the same visible failure survives `3` neighborhoods inside one architecture
  family, mark that family exhausted and pivot.
- Once a family looks structurally wrong, prefer one bounded structural edit
  over several more scalar sweeps.
- A structural experiment may bundle multiple coordinated code changes when they
  are all required to test one architecture hypothesis.
- Prefer promoting the current rung or rejecting it clearly over opening
  multiple ladder branches at once.

## Decision Rule

- After validation, do exactly one of:
  - make a validated repo edit,
  - return one `run_long_command`,
  - return `stop` only for explicit operator stop, exhausted long-command
    budget, or a truly unfixable blocker.
- `stop` is exceptional. One exhausted family is not enough.
- If the next improvement likely requires code, make the bounded edit and
  validate it instead of digging deeper into an architecture family that is not
  working.
- Prefer a repo edit, test, or validation improvement over “no sensible next
  experiment” when a plausible code-level lever exists.
- Base the choice on the best next action for the total long-run budget, not on
  the smallest decisive check. A cheap checkpoint-selection pass is not
  preferred when the visible failure mode already shows the family is exhausted.
- If validation answers or obsoletes the current ladder rung, update
  `docs/complexity_ladder_training.md` first and then choose the next action
  against the new ladder state, even when that means promoting upward or
  rejecting the rung entirely.

## Operator Control

- Human messages in the same session are authoritative.
- If the operator asks to stop after the full loop, finish post-run validation
  and then return `stop`.

## Ending

- Return one raw JSON object only.
- Always include:
  - `action_type`
  - `summary`
  - `session_work_summary`
  - `repo_edit_status`
  - `long_command`
  - `stop`
- `action_type` must be `run_long_command` or `stop`.
- `repo_edit_status` must be `none`, `validated`, or `rollback_requested`.
- When `action_type=run_long_command`, fill `long_command.command`,
  `long_command.reason`, and `long_command.expected_artifacts`.
- When `action_type=stop`, set `stop.reason` and leave
  `long_command.command`/`reason` empty with
  `long_command.expected_artifacts=[]`.
