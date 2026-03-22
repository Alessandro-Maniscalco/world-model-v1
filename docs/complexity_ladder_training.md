## Goal
The process is to find the easiest setup that can produce a good-looking plausible video with visible task-relevant motion, then increase difficulty one axis at a time from that anchor. This ladder is reset after the recent code fixes, so earlier branch conclusions are intentionally cleared until they are rerun.

## Proven Complexity ladder
- None yet.

## Next complexity to test
- Rung: validation-best checkpoint rescue on the short-window residual scout with last-context future-control fill.
- `conditioning_mode=none`, `context_len=17`, `horizon_len=4`, `k=1`, `chunk_schedule_mode=k_chunks`, `single_chunk_rollout=true`, `future_latent_residual_mode=last_context_frame`, `future_control_fill_mode=last_context_frame`, `gradient_checkpointing=true`, `lora_rank=32`, checkpoint `step_0000300`
- Why next: the structural fill-mode rerun is the first easy rung after the reset that no longer collapses into one bad frame. On the main clip plus held-out episodes `1` and `2`, motion starts at frame `17` and continues through frames `18-20`, all three windows stay plausible, and episode `2` upgrades to `motion_verdict=good`. But the final step-`400` checkpoint still undercommits: on the main clip and held-out episode `1`, the fork stays too far right through the last horizon and retains visible tool/contact blur or ghosting by frame `20`, while the training trace peaks much earlier (`best_val_loss≈0.0544` at step `300`, `≈0.2366` at step `400`). Step-`300` checkpoint selection is therefore the highest-value next action before any continuation or promotion.

## Best rung for current complexity
- `optimizer_aloha_static_fork_pick_up_full_320x240_ctx17_h4_lora32_noaction_gradckpt_residual_lastctx_filllastctx_singlechunk_fresh400` step `400`

## Rung Findings for current complexity
- Plain short-window residual scouting with gray future control is rejected after checkpoint selection: on the main clip plus held-out episodes `1` and `2`, frames `14-16` stay copied/static and only frame `17` changes. That last frame still becomes a beige/green fork-contact smear on the main clip, a washed-out ghosted blur on episode `1`, and the brightest low-detail arm/contact blob on episode `2`, with held-outs `1` and `2` failing plausibility on frame `17`.
- Last-context future-control fill changes the visible behavior enough to keep the branch alive: on the main clip plus held-out episodes `1` and `2`, motion now starts at frame `17` instead of staying static until a single collapsing frame, and all three windows remain plausible through frame `20`. The main clip and held-out episode `1` still undercommit and keep the fork too far right with residual blur/ghosting by the final frame, while held-out episode `2` is the cleanest and closest to a usable trajectory. Because the run overtrains past step `300`, checkpoint selection is the right next move instead of a blind continuation.

## Stable Findings
- Use `scripts/check/sweep_local_repo_resolutions.py` for checkpoint evaluation, treat plausibility as a safety gate, and rank runs visual first.
- Stay on the complexity ladder: validate the easiest rung first, then promote upward one major axis at a time.
- In this repo, the closest local Wan/VACE-style inference contract uses `single_chunk_rollout=true` with at least `50` integration steps.
- For simple runs with few latent steps, cap the scout at `200` training steps before deciding whether that rung earns promotion.
- Under Wan temporal packing, `horizon_len=4` gives one future latent step and `horizon_len=8` gives two.
- `k` now means exactly `k` future chunks, so `chunk_schedule_mode` should stay `k_chunks`.
- The post-fix easy `ctx21/h4` action scout still copies through the last-horizon setup frames and only changes frame `21`, where the main clip and held-out episode `1` smear and held-out episode `2` collapses brighter and misses contact.
- The post-fix `ctx17/h4` residual scout with gray future control also only changes one final frame, and neither step `400` nor the validation-best step `300` produces a clean fork/contact pose.
- Aligning the future VACE control stream with the residual target is the first easy-rung change that materially improves behavior: with `future_control_fill_mode=last_context_frame`, motion begins at frame `17` and all three reviewed windows stay plausible through frame `20`, though the main clip and held-out episode `1` still undercommit.
- The fill-mode branch has a strong checkpoint-selection signal: `best_val_loss≈0.0544` at step `300` regresses to `≈0.2366` by step `400`, so step `300` is decision-relevant before any continuation or family kill.

## Kept Code Changes
Still-relevant code-changing commits that remain available as structural levers.
- Commit `0f50064` (`Add residual future latent training mode`): adds checkpoint-compatible `future_latent_residual_mode=last_context_frame` to train/infer config, flow-matching training, and rollout sampling so the model can denoise future latents relative to the last observed latent frame instead of absolute latents.
- Commit `6323a3c` (`Add last-context future control fill mode`): adds `future_control_fill_mode=last_context_frame` through train/infer configs, checkpoint restore, and Wan/VACE control assembly so masked future control slots can copy the last observed latent frame instead of gray filler latents, giving residual-mode runs a true zero-change future control prior.
