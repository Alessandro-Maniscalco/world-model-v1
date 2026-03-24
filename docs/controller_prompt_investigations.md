## Repo Edits And Rollback

- Broad architecture changes are encouraged.
- Before editing, create a session-start git checkpoint commit.
- Keep only validated edits. After any kept validated code change, create a
  commit containing only your files.
- Use `repo_edit_status=validated` only for kept validated edits,
  `rollback_requested` when the controller should undo all edits from this turn,
  otherwise `none`.
- Use rollback freely for speculative edits that fail validation or do not earn
  a better next decision.
- Rollback affects repo files only, not `runs/`.
- Record kept code-changing commits in `docs/investigation.md` under
  `## Kept Code Changes`. Put detailed chronology in
  `runs/training_optimizer/investigation_ledger.md`.
- Delete `runs/` artifacts only when they are clearly dominated and no longer
  needed. When unsure, keep them.

## Long Experiment Commands

- Never start long-running work inside the shared session. If needed, return
  exactly one shell command via `run_long_command`.
- Run Python and pytest inside `.venv`.
- Prefer one bounded shell chain that directly answers the current question in
  `docs/investigation.md`. 
- If `scripts/check/sweep_local_repo_resolutions.py` only see `episode_index=1`, `start_frame=60`.
- The command must produce the concrete artifacts needed for review.
- Include the concrete checkpoint path, output directory, dataset slice, and
  any key config overrides in experiment commands.
- `long_command.reason` should say why this is the highest-value bounded action
  for the current investigation.
- In-session code edits are effectively free relative to another training or
  evaluation run.
- `long_command.expected_artifacts` should list the concrete artifact paths to
  review.
- Cap new exploratory runs at `400` steps first. Continue only if the result is
  clearly answering the current question.

## Validation

- Use short commands only.
- After every run, validate the newest result before choosing the next action.
- Trace the run end to end to find the first failing stage, not just the final artifact.
- Review artifacts in this order:
  - Visual inspection of all videos. Never skip this. `left=target/reference` and `right=prediction`. Begin with the last `horizon_len` frames.
  - then raw-window, frame-count, or sharpness reports if present,
  - then `*_arm_crop_comparison.mp4` and `*_arm_motion_report.json` if present,
  - then `plausibility_report.json`,
  - then metrics/logs and any tensor-shape dumps you created.
- Watch enough of each reviewed clip to describe the visible motion pattern in sentences, not just labels or metrics.
- If the video visibly goes bad, that is enough to reject the hypothesis even when scalar metrics look acceptable.
- At the end of the run, make a hypothesis why it acted that way. Look deep in the code to find the root cause. 

### Motion-First Ranking

- Rank runs by visible task-relevant motion first.
- Use this order:
  - `1. visual inspection`
  - `2. arm/tool movement and commitment`
  - `3. overall scene fidelity and sharpness`
  - `4. aggregate metrics such as MAE`
- Treat plausibility as a safety gate and tie-break input.
- When visual quality and metrics disagree, trust the video for keep/drop decisions.
- Analyze only `episode_index=1`, `start_frame=60`.


## Local Neighborhood Control

- Every proposed run must name its local neighborhood and the distinct
  hypothesis it tests.
- Do not spend more than `2` non-improving follow-ups inside the same
  neighborhood.
- If a direct continuation regresses, do not run another plain continuation in
  that branch. Change one major lever or mark the family exhausted.
- Once a family looks structurally wrong, prefer one bounded structural edit
  over several more scalar sweeps.

## Decision Rule

- After validation, do exactly one of:
  - make a validated repo edit,
  - return one `run_long_command`,
  - return `stop` only for explicit operator stop, exhausted long-command
    budget, or a truly unfixable blocker.
- `stop` is exceptional. One exhausted family is not enough.

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
