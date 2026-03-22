# Shared-Session Controller Prompt For Fixed-Anchor Root-Cause Work

Persistent operating contract for a fixed-anchor Codex debugging session.

- Read `docs/fixed_anchor_investigation.md` first.
- Treat `docs/fixed_anchor_investigation.md` as the primary decision memory.
- Treat `docs/complexity_ladder_training.md` and
  `docs/training_optimizer.md` as background only.
- Treat `runs/training_optimizer/fixed_anchor_investigation_ledger.md` as
  chronology only.

Everything is run on a RTX 3080 with 16GB of VRAM.

## Fixed-Anchor Objective

- Stay locked on the anchor described in `docs/fixed_anchor_investigation.md`.
- Do not change context length, horizon length, chunk count, rollout mode,
  resolution, or dataset window unless the memory is updated with a clear
  reason or the operator explicitly asks for a pivot.
- Other settings may change when they test one concrete root-cause hypothesis.
- The goal is to find the first failing stage that produces the visible
  arm/fork morphing, not to promote a ladder rung.

## Repo Edits And Rollback

- Repo edits are allowed when they are the strongest next step, especially
  instrumentation that exposes hidden stage-boundary state.
- Before editing, create a session-start git checkpoint commit.
- Keep only validated edits. After any kept validated code change, create a
  commit containing only your files.
- Use `repo_edit_status=validated` only for kept validated edits,
  `rollback_requested` when the controller should undo all edits from this turn,
  otherwise `none`.
- Rollback affects repo files only, not `runs/`.
- Record kept code-changing commits in `docs/fixed_anchor_investigation.md`
  under `## Kept Code Changes`. Put detailed chronology in
  `runs/training_optimizer/fixed_anchor_investigation_ledger.md`.
- Delete `runs/` artifacts only when they are clearly dominated. When unsure,
  keep them.

## Long Experiment Commands

- Never start long-running work inside the shared session. If needed, return
  exactly one shell command via `run_long_command`.
- Run Python and pytest inside `.venv`.
- Long commands may be direct checkpoint inference, targeted evaluation, or one
  bounded fresh training/resume run.
- Prefer inference and diagnostic generation before training when the first
  failing stage is still unknown.
- Existing checkpoints are starting evidence, not a fixed ceiling. New
  checkpoints are allowed if they keep the fixed short-window contract and test
  one concrete hypothesis at a time.
- When the command generates videos, produce the checkpoint path and the
  artifacts needed for review: generated video, comparison video, future-only
  or horizon-focused views if available, plausibility report, arm-motion
  artifacts, and any frame/sharpness/boundary reports the command supports.
- Include the concrete checkpoint path, output directory, repo id, episode
  index, start frame, video key, context length, horizon length, `k`, and
  resolution in sweep or infer commands.
- `long_command.reason` should say which pipeline stage or hypothesis the
  command is testing, justified with concrete visible behavior from the anchor.
- In-session code edits are effectively free relative to another training or
  evaluation run.
- `long_command.expected_artifacts` should list the concrete artifact paths to
  review.
- Cap new exploratory training runs at `400` steps first unless the operator
  asks otherwise.

## Validation

- Use short commands only.
- After every run, validate the newest result before choosing the next action.
- Trace the run end to end to find the first failing stage, not just the final
  artifact.
- Review artifacts in this order:
  - Visual inspection of all videos. Never skip this. `left=target/reference`
    and `right=prediction`. Begin with the last `horizon_len` frames.
  - then raw-window, frame-count, or sharpness reports if present,
  - then `*_arm_crop_comparison.mp4` and `*_arm_motion_report.json` if present,
  - then `plausibility_report.json`,
  - then metrics/logs and any tensor-shape dumps you created.
- For video or temporal failures, check all relevant boundaries: raw frame
  window, context/horizon packing, latent-time shapes, chunk schedule, decoded
  frame counts, and exported/comparison video frame counts.
- Record where the failure first appears and what looks wrong there in plain
  language.
- Watch enough of each reviewed clip to describe the visible motion pattern in
  sentences, not just labels or metrics.
- If the video visibly goes bad, that is enough to reject the hypothesis even
  when scalar metrics look acceptable.
- For this fixed anchor, remember that checkpoint-mode sweep metrics may include
  the copied context prefix. Do not let full-window scalar summaries outweigh
  obvious future-horizon failure.

## End-To-End Iteration Loop

- Treat the investigation as a repeated fault-localization loop.
- On each cycle: reproduce or reuse the current failing artifact, inspect
  stages from earliest to latest, pick the earliest unverified or suspicious
  stage, take the smallest action that exposes or tests it, rerun the same
  anchor, and update the memory.
- Do not skip ahead to later stages when an earlier stage is still unverified.
- Do not declare success because the final video changed; identify which stage
  changed first.

## Required Stage Coverage

- Account for all stages in order: raw source-frame window and count,
  post-preprocess frames and sizing, VAE encode shapes and latent split, chunk
  schedule and masks, action/control plan when applicable, residual targets
  when applicable, denoising inputs and outputs, decoded future frames, and
  exported videos and counts.
- A stage counts as checked only when there is direct evidence: an artifact, a
  saved report, a tensor dump, or a code-path verification tied to the current
  anchor.
- If a stage cannot be checked with existing artifacts, prefer a small
  instrumentation edit or targeted command that exposes it over another broad
  training sweep.

## Memory Maintenance

- Keep `docs/fixed_anchor_investigation.md` short and decision-oriented.
- It should contain only:
  - `## Goal`
  - `## Fixed Anchor`
  - `## Current First-Failing-Stage Hypothesis`
  - `## Stage Findings`
  - `## Open Hypotheses`
  - `## Next Diagnostic Step`
  - `## Stable Findings`
  - `## Kept Code Changes`
- Put detailed validation summaries and chronology in
  `runs/training_optimizer/fixed_anchor_investigation_ledger.md`.
- After every run, record the trace result in the ledger, including the first
  failing stage and why.
- Delete repeated conclusions instead of restating them.

## Decision Rule

- After validation, do exactly one of:
  - make a validated repo edit,
  - return one `run_long_command`,
  - return `stop` only for explicit operator stop, exhausted long-command
    budget, or a truly unfixable blocker.
- `stop` is exceptional. One failed hypothesis is not enough.
- If the next improvement likely requires code, make the bounded edit and
  validate it instead of repeating a near-duplicate run.

## Operator Control

- Human messages in the same session are authoritative.
- If the operator asks to stop after the full loop, finish post-run validation
  and then return `stop`.

## Ending

- Return one raw JSON object only.
- Always include `action_type`, `summary`, `session_work_summary`,
  `repo_edit_status`, `long_command`, and `stop`.
- `action_type` must be `run_long_command` or `stop`.
- `repo_edit_status` must be `none`, `validated`, or `rollback_requested`.
- When `action_type=run_long_command`, fill `long_command.command`,
  `long_command.reason`, and `long_command.expected_artifacts`.
- When `action_type=stop`, set `stop.reason` and leave
  `long_command.command`/`reason` empty with
  `long_command.expected_artifacts=[]`.
