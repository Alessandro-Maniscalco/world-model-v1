## Goal

This file records only the context that still changes the next training decision.

## Training Goal

The goal is to find the correct training and check architecture is correct so that the model predict the correct latent-space future denoising velocity under chunkwise teacher forcing.

## What can be changed

- Architecture
- Learning rate
- Number of parameters trained
- Number of epochs
- Number of episodes
- Batch size
- Number of frames

Would the best be to have as many parameters trained and highest learning rate as possible?

## Stable Findings

- `scripts/check/sweep_local_repo_resolutions.py` is still the main local smoke-check entrypoint, but checkpoint mode must use the repo's direct chunkwise world-model inference path, not a plain Wan VACE pipeline overlay.
- [`scripts/check/check_generated_video_plausibility.py`](/home/amaniscalco/world-model-v1/scripts/check/check_generated_video_plausibility.py) remains the acceptance gate for collapse, color drift, posterization, flat frames, and temporal instability.
- VACE control semantics are fixed: future placeholders should use gray control latents, and masked fill regions should use black control latents.

## Current Signal

- Pretrained base inference at `320x240` is still the reference-good path.
- After the flow-matching fix, `episode 0 + 320x240 + conditioning_mode=none` trains without immediate collapse.
- `trainable_backbone=head` learns recognizable images, but a whitening / bright-side artifact remains. The artifact is weaker around `100` steps and still present by `400-800` steps. In `runs/test_100to200_320x240_head_none/metrics.jsonl`, the 100-step loss windows move from `0.112` at steps `101-200` to `0.077` at steps `701-800`, so the model is still optimizing even while colors drift.
- Changing only the learning rate did not change the qualitative failure mode enough to justify further LR-focused tuning first.
- LoRA is the first change that improved the artifact directionally: at `100` steps the whitening is still visible but clearly reduced relative to the head-only run. The single-episode LoRA smoke run reaches a much lower short-horizon loss (`0.050` average over steps `101-111`) than the full multi-episode LoRA rank-8 run.
- The full multi-episode `LoRA rank 8 + conditioning_mode=none` run is not the right next baseline yet. By step `400` and `800` the images are slightly less clear in color, and the loss is flatter and much higher than the single-episode path: `0.218` over steps `301-400` and `0.215` over steps `701-800` in `runs/test_full_multi_320x240_lora8_none/metrics.jsonl`.
- The FP32 Wan VAE roundtrip works well enough and is not the main source of whitening. On the same `episode 0` clip, raw mean luma is `0.375` and the FP32 VAE roundtrip is `0.381`, so the decoder introduces only a small brightness shift.
- The apparent whitening in the earlier checkpoint sweeps was mostly an evaluation bug, not a confirmed training failure. The old checkpoint path in `scripts/check/sweep_local_repo_resolutions.py` overlaid checkpoint weights into the generic Wan VACE pipeline, changed the total frame contract, and used prompt-style conditioning instead of the repo's null/action token path.
- Under the correct direct world-model inference path, the same `episode 0 + LoRA rank 8 + step 400` checkpoint is much closer to raw: direct inference mean luma is `0.396` with `edge_minus_center=-0.057`, compared with `0.497` and `+0.033` under the mismatched pipeline wrapper. The step-400 corrected output looks visually plausible and no longer shows the previous side whitening failure mode.
- The first action-conditioned LoRA run moved the bottleneck away from color. In `runs/sweep_local/test_400steps_320x240_episode0_lora8_action_step_0000300_comparison.mp4`, colors look fine, but the robot arm falls down instead of tracking the target motion. The next problem is motion/control fidelity, not whitening.

## Next Work

- Keep normal mixed precision for model inference/training. The VAE is not the main problem; only the decode-side diagnostic path needs FP32 when isolating artifacts.
- Use the corrected checkpoint sweep or `scripts/train/infer_world_model.py` for checkpoint evaluation. Do not use the old checkpoint-overlay pipeline path as evidence for color drift.
- The no-conditioning LoRA path is no longer the blocker. The next work should focus on why action-conditioned rollouts lose the arm trajectory even when colors stay plausible.
- Keep the stable recipe fixed for now: `episode 0 + 320x240 + trainable_backbone=lora + lora_rank=8 + conditioning_mode=action`. Do not change LR, frame count, or dataset scope yet.
- Compare `100/200/300/400` checkpoints from the action-conditioned run using the corrected evaluator and the side-by-side comparison MP4. Prioritize arm pose, tool path, and contact dynamics over color.
- If the arm still falls under the stable action recipe, inspect the action-conditioning path next: action tensor layout (`sample` vs `sequence`), temporal alignment into latent steps, and whether the null/action token swap changed behavior more than the backbone update.

## Training runs

Recent note: `test_400steps_320x240_episode0_lora8_action_step_0000300_comparison.mp4` has acceptable colors but bad arm motion; use it as the reference failure case for action-conditioning debugging.

## Codex Analysis

- [controller 2026-03-15T21:47:36+00:00] Codex chose `inspect_artifact`: Inspect the corrected action-conditioned checkpoint evidence and the action-conditioning train/eval path before spending the remaining run or edit budget. Current evidence says color is acceptable under corrected evaluation, but the corrected 100/200/400 comparison artifacts are missing and the remaining blocker is arm/control fidelity under the fixed `episode 0 + 320x240 + trainable_backbone=lora + lora_rank=8 + conditioning_mode=action` recipe. Next note: Resolve the missing corrected checkpoint comparisons and trace action-token layout/timestep use before any new run or repo edit. Only spend the final run or edit budget once the onset of the arm-drop failure and the train/infer conditioning contract are clear.
Use this section for the latest Codex planning summary when the optimizer runs
in `--planner codex` mode. The controller rewrites the newest `[controller ...]`
bullet here after each Codex decision so the markdown keeps the current model
analysis without replacing the human-written `Next Work` instructions.

## Controller Edits

Use this section as the audit log for controller-process changes. When the
controller applies a bounded self-edit to `src/world_model/optimization/controller.py`,
or Codex applies a validated repo edit before the next real run, it should
append a timestamped entry here describing what changed, why it changed, and
whether the edit applied cleanly.
