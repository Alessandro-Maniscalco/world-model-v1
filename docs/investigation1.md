## Goal
Find the best video given the constraints `224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8` and `conditioning_mode=action`.

Look deep in the architecture and find big repo edits to explore are encouraged. Search online what is the best process to find the best training. Always reflect on possible alternatives and chose one. Make hypothesis of why something is not working and test it.

- Run Python and pytest inside `.venv`.
- Review only `episode_index=1`, `start_frame=60`.
- Do not use a text prompt.
- Do not run new action-conditioned experiments until the no-action base path is
  visibly good.
- Keep the same geometry unless a result proves it is the blocker:
  `224x128`, `context_len=9`, `horizon_len=8`, `k=1`, `subset_size=8`.
- Keep new base-path architecture branches anchored to the same untouched
  pretrained Wan/VACE parent.

## Best run
- Quality reference:
  `fullft_subset8_spread_resume200_lr5e5_step400/checkpoints/step_0000350.pt`.
  Best prompt-free reference so far; coherent through `f16` with mild late
  blur/ghosting, but still misses contact.
- Best repaired none-training branch:
  `untouched_base_none_pretrainedseq_dualanchor_fullft_subset8_adafactor_lr5e5_bs1gc_resume0200_step300_ckpt50/checkpoints/step_0000300.pt`.
  Best motion-first checkpoint in the repaired none branch so far.

## Findings
- Run:
  `untouched_base_none_pretrainedseq_defaultdualanchor_224x128_ep1_start60_step0000_operator`
  MP4:
  `runs/training_optimizer/eval/untouched_base_none_pretrainedseq_defaultdualanchor_224x128_ep1_start60_step0000_operator/untouched_base_none_subset8_step0_baseline_224x128_step_0000000_comparison.mp4`
  Description:
  First safe untouched-parent `conditioning_mode=none` zero-step baseline. It
  avoids the catastrophic blue/purple collapse family, but stays darker/cooler
  with soft late ghosting.
- Run:
  `untouched_base_none_pretrainedseq_dualanchor_fullft_subset8_adafactor_lr5e5_bs1gc_resume0200_step300_ckpt50_ep1_start60_step0300_operator`
  MP4:
  `runs/training_optimizer/eval/untouched_base_none_pretrainedseq_dualanchor_fullft_subset8_adafactor_lr5e5_bs1gc_resume0200_step300_ckpt50_ep1_start60_step0300_operator/untouched_base_none_pretrainedseq_dualanchor_fullft_subset8_adafactor_lr5e5_bs1gc_resume0200_step300_ckpt50_step_0000300_comparison.mp4`
  Description:
  Best repaired none-training checkpoint so far. Motion is finally good, the
  fork/gripper stays recognizable, and there is no return to the catastrophic
  collapse family. Remaining defect is mild late bloom/ghosting.
- Run:
  `fullft_subset8_spread_resume200_lr5e5_step400_ep1_start60_step0350_operator`
  MP4:
  `runs/training_optimizer/eval/fullft_subset8_spread_resume200_lr5e5_step400_ep1_start60_step0350_operator/fullft_subset8_spread_resume200_lr5e5_step400_step_0000350_comparison.mp4`
  Description:
  Best prompt-free quality reference. It stays coherent through `f16` with mild
  late soft blur/ghosting and still misses contact.
- Run:
  `fullft_subset8_spread_resume200_lr5e5_step400_actionzeroinit_224x128_ep1_start60_step0350_operator`
  MP4:
  `runs/training_optimizer/eval/fullft_subset8_spread_resume200_lr5e5_step400_actionzeroinit_224x128_ep1_start60_step0350_operator/fullft_subset8_spread_resume200_lr5e5_step400_step_0000350_comparison.mp4`
  Description:
  Zero-step `conditioning_mode=action` sanity check on the good `step_0000350`
  parent. It stays plausible and close to the none reference, but action-driven
  motion is still misaligned, so this is useful for safe architecture screening
  rather than action understanding.

## Kept Code Changes
- `0456e04`: checkpoint-path local sweeps apply runtime overrides after loading
  checkpoint metadata.
- `0348c65`: prompt-free none conditioning stopped using literal zero tokens
  and now reuses Wan's empty-prompt embedding.
- `872de58`: prompt-free none conditioning now uses Wan's full empty-prompt
  token sequence, and none-mode rollout treats that sequence as global
  conditioning instead of chunk slicing.
- `7896373`: untouched pretrained none checkpoints (`max_steps == 0`) now
  default to the validated dual-anchor contract, while trained none checkpoints
  keep their saved contract.
- `validated in worktree`: training-side chunkwise flow matching now accepts
  global prompt-free none-conditioning token sequences, and
  `_resume_training_state` can fall back to a fresh optimizer when saved
  optimizer parameter groups do not match the current module layout.
