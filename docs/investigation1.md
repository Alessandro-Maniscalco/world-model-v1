## Goal
Understand every single process in the following command, be as specific as possible. The numbered process of the long command, with each step being the function called has been laid out. Loop the controller to go through each step and verify it. 
For each step, under ## Verified Number Process write: number, green checkmark if the function was verified, function name, what was tested. If it fails make changes to either the test, function or process.

You are free to make any changes and cleanups where you feel it is better to do so.

If you find that the numbered process could be split more deeply, do so.

At the end I want you to understand the entire process, have edited the architecture so that it works, and be sure that after training the model you know the exact cuase of every failure and success.

## Command
cd /home/amaniscalco/world-model-v1 &&
source .venv/bin/activate &&
set -euo pipefail &&
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE &&
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True &&
RUN_ROOT='/home/amaniscalco/world-model-v1/runs/basepretrained_fullaction_tok005_zeroinit_noaddedkv_actiontokenaux1_adafactor5e5_bs1gc_step400_rerun1' &&
RUN_NAME=$(basename "$RUN_ROOT") &&
TRAIN_LOG="$RUN_ROOT/train_stdout.log" &&
test ! -e "$RUN_ROOT" &&
mkdir -p "$RUN_ROOT" &&
python -u scripts/train/world_model.py \
  --config configs/train/world_model.yaml \
  --output-dir "$RUN_ROOT" \
  --repo-id lerobot/aloha_static_fork_pick_up \
  --video-key observation.images.cam_high \
  --frame-height 128 \
  --frame-width 224 \
  --context-len 9 \
  --horizon-len 8 \
  --k 1 \
  --subset-size 8 \
  --conditioning-mode action \
  --future-control-fill-mode last_context_frame \
  --future-latent-residual-mode last_context_frame \
  --trainable-backbone full \
  --optimizer-name adafactor \
  --lr 5e-5 \
  --batch-size 1 \
  --no-auto-batch-size \
  --gradient-checkpointing \
  --action-conditioning-window chunk \
  --action-token-scale 0.05 \
  --action-output-zero-init \
  --action-backbone-added-kv-mode none \
  --action-token-latent-aux-loss-scale 1.0 \
  --max-steps 400 \
  --checkpoint-every 50 \
  --validation-every 50 \
  --validation-max-batches 8 \
  --seed 0 2>&1 | tee "$TRAIN_LOG" &&
test -f "$RUN_ROOT/metrics.jsonl" &&
test -f "$RUN_ROOT/checkpoints/step_0000350.pt" &&
test -f "$RUN_ROOT/checkpoints/step_0000400.pt" &&
for STEP in 350 400; do
  STEP_PAD=$(printf '%07d' "$STEP")
  STEP4=$(printf '%04d' "$STEP")
  CKPT="$RUN_ROOT/checkpoints/step_${STEP_PAD}.pt"
  EVAL_ROOT="/home/amaniscalco/world-model-v1/runs/training_optimizer/eval/${RUN_NAME}_ep1_start60_step${STEP4}_operator"
  LOG_PATH="$EVAL_ROOT/eval_stdout.log"
  STEM="${RUN_NAME}_step_${STEP_PAD}"
  test -f "$CKPT"
  test ! -e "$EVAL_ROOT"
  mkdir -p "$EVAL_ROOT"
  python -u scripts/check/sweep_local_repo_resolutions.py \
    --mode checkpoint \
    --checkpoint "$CKPT" \
    --config configs/train/world_model.yaml \
    --output-dir "$EVAL_ROOT" \
    --repo-id lerobot/aloha_static_fork_pick_up \
    --episode-index 1 \
    --start-frame 60 \
    --video-key observation.images.cam_high \
    --context-len 9 \
    --horizon-len 8 \
    --k 1 \
    --num-inference-steps 50 \
    --resolutions 224x128 \
    --single-chunk-rollout \
    --conditioning-mode-override action \
    --action-source sequence \
    --action-token-scale 0.05 \
    --future-control-fill-mode-override last_context_frame \
    --future-latent-residual-mode-override last_context_frame \
    2>&1 | tee "$LOG_PATH"
  test -f "$EVAL_ROOT/${STEM}.mp4"
  test -f "$EVAL_ROOT/${STEM}_comparison.mp4"
  test -f "$EVAL_ROOT/${STEM}_arm_crop_comparison.mp4"
  test -f "$EVAL_ROOT/${STEM}_summary.json"
  test -f "$EVAL_ROOT/arm_motion_report.json"
  test -f "$EVAL_ROOT/plausibility_report.json"
done


## Number process of the command

The list below expands the current base-pretrained long command into the shell
actions, repo entrypoints, repo helper calls, and the directly invoked
third-party loaders or schedulers that the repo explicitly calls. Steps
`99-163` repeat for each optimizer update until `max_steps=400`. Steps
`170-291` repeat once for `STEP=350` and once for `STEP=400`.

1. `cd /home/amaniscalco/world-model-v1` changes the shell working directory to the repo root.
2. `source .venv/bin/activate` loads the repo virtualenv activation script.
3. `set -euo pipefail` makes the shell fail on unset vars, non-zero exits, and broken pipelines.
4. `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE` allows pretrained assets to resolve normally instead of forcing offline mode.
5. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` enables the PyTorch CUDA allocator mode used by the long run.
6. `RUN_ROOT=...` defines the training output directory for this run family.
7. `RUN_NAME=$(basename "$RUN_ROOT")` derives the short run name used later in eval artifact paths.
8. `TRAIN_LOG="$RUN_ROOT/train_stdout.log"` defines the captured training stdout path.
9. `test ! -e "$RUN_ROOT"` refuses to overwrite an existing run directory.
10. `mkdir -p "$RUN_ROOT"` creates the training output directory tree.
11. `python -u scripts/train/world_model.py ...` launches the training entrypoint.
12. `main()` in `scripts/train/world_model.py` starts the train-time control flow.
13. `_load_args()` bootstraps config loading and CLI override resolution.
14. `_config_parser()` parses the bootstrap `--config` argument.
15. `load_train_config()` asks `src/world_model/config.py` for the typed train config.
16. `_load_config()` resolves the YAML path and dispatches the YAML load.
17. `_load_yaml()` reads `configs/train/world_model.yaml` into a mapping.
18. `_coerce_dataclass()` materializes the YAML mapping as a `TrainScriptConfig`.
19. `_coerce_field_value()` converts YAML list payloads into tuple-backed dataclass fields where needed.
20. `_build_parser()` builds the full training CLI parser from the resolved defaults.
21. `argparse.parse_args()` applies the command-line overrides on top of the YAML defaults.
22. `apply_namespace_overrides()` overlays the parsed CLI values onto the typed config.
23. `normalize_chunk_schedule_mode()` canonicalizes `k_chunks`.
24. `_set_seed()` seeds Python RNG and Torch RNG state.
25. `_validate_auto_stop_config()` validates the requested training configuration.
26. `output_dir.mkdir(parents=True, exist_ok=True)` ensures the run root exists from inside Python too.
27. `_select_runtime_dtype()` chooses the train-time runtime dtype.
28. `torch.amp.GradScaler(...)` constructs the mixed-precision grad scaler object.
29. `WanVAE.from_pretrained()` loads the pretrained Wan VAE wrapper.
30. `_offline_mode_enabled()` checks whether Hugging Face offline loading is enabled for the VAE load.
31. `AutoencoderKLWan.from_pretrained()` loads the diffusers Wan VAE weights.
32. `resolve_lerobot_episode_ids()` enumerates the dataset episode ids because validation is enabled.
33. `_load_lerobot_dataset_class()` imports `LeRobotDataset`.
34. `split_train_validation_episode_ids()` splits the episode pool into train and validation ids.
35. `_build_train_loader()` builds the probe dataloader used to inspect a first batch.
36. `build_lerobot_dataloader()` constructs the actual PyTorch `DataLoader` for the probe path.
37. `build_frame_deltas()` computes the exact frame-time offsets used in the LeRobot query window.
38. `validate_wan_temporal_window()` checks that `context_len=9` and `horizon_len=8` obey Wan temporal packing.
39. `_load_lerobot_dataset_class()` imports `LeRobotDataset` for the probe loader construction.
40. `_select_subset_indices()` chooses the deterministic `subset_size=8` sample indices.
41. `next(iter(probe_loader))` requests the first probe batch from the dataloader.
42. `collate_tensor_dict()` stacks the per-sample tensors into the probe batch dict.
43. `prepare_packed_batch()` converts the probe batch into model-ready latent tensors and action-plan tensors.
44. `validate_wan_temporal_window()` re-checks the raw-frame window inside batch preparation.
45. `preprocess_video_for_vae()` resizes the batch video to `128x224` and normalizes its spatial shape for Wan.
46. `_center_crop_video_to_multiple()` enforces divisibility by the Wan spatial multiple.
47. `_get_constant_control_latents()` builds or fetches the cached black control-latent template.
48. `_control_video_range_key()` classifies the numeric range for the black control cache key.
49. `_make_constant_video_like()` creates the constant black control video tensor.
50. `WanVAE.encode()` encodes the black control video.
51. `_to_bcthw()` converts `[B,T,C,H,W]` input into the VAE-native `[B,C,T,H,W]` layout.
52. `_normalize_video()` maps the input video into the VAE `[-1,1]` range.
53. `_cast_to_vae_runtime()` moves the video to the loaded VAE device and dtype.
54. `_vae_runtime_device_dtype()` inspects the active VAE parameter device and dtype.
55. `self.vae.encode()` runs the diffusers Wan VAE encoder.
56. `_latent_dist_mode()` selects the deterministic posterior mean.
57. `_normalize_latents()` converts raw VAE latents into the repo’s Wan-normalized latent format.
58. `_latent_stats()` reads `latents_mean` and `latents_std` from the VAE config.
59. `_get_constant_control_latents()` builds or fetches the cached gray control-latent template.
60. `_control_video_range_key()` classifies the numeric range for the gray control cache key.
61. `_make_constant_video_like()` creates the constant gray control video tensor.
62. `WanVAE.encode()` encodes the gray control video through the same VAE encode path.
63. `WanVAE.encode()` encodes the real resized video through the same VAE encode path.
64. `latent_split_for_wan_frames()` converts total latent time into exact context and future latent lengths.
65. `validate_wan_temporal_window()` re-validates the exact Wan frame packing inside the latent split helper.
66. `wan_latent_steps_from_frame_count()` computes the expected total latent steps from the raw frame count.
67. `wan_latent_steps_from_frame_count()` computes the context latent-step count from the raw context frame count.
68. `build_future_action_plan()` converts raw robot actions into future latent-step conditioning tokens.
69. `flatten_action_chunks()` flattens contiguous 4-frame raw action groups into one latent-step action vector.
70. `_validate_chunk_schedule()` confirms the latent horizon can support `k=1`.
71. `_autotune_batch_size()` is still called, then immediately returns the configured batch size because `--no-auto-batch-size` is set.
72. `_build_train_loader()` builds the real training dataloader.
73. `build_lerobot_dataloader()` constructs the real training `DataLoader`.
74. `build_frame_deltas()` recomputes the frame offsets for the real train loader.
75. `validate_wan_temporal_window()` re-validates the train loader temporal window.
76. `_load_lerobot_dataset_class()` imports `LeRobotDataset` for the train loader.
77. `_select_subset_indices()` reselects the deterministic `subset_size=8` train subset.
78. `_build_validation_loader()` builds the held-out validation dataloader.
79. `build_lerobot_dataloader()` constructs the validation `DataLoader`.
80. `build_frame_deltas()` computes the validation frame offsets.
81. `validate_wan_temporal_window()` validates the validation temporal window.
82. `_load_lerobot_dataset_class()` imports `LeRobotDataset` for the validation loader.
83. `next(data_iter)` requests the first real training batch.
84. `collate_tensor_dict()` stacks the first real training batch.
85. `prepare_packed_batch()` runs again on the first real training batch, repeating the latent and action-plan preparation path from steps `43-69`.
86. `_validate_chunk_schedule()` re-checks the latent future schedule on the first real training batch.
87. `build_model_from_config()` dispatches model creation.
88. `build_wan_vace_model_from_config()` builds the Wan VACE world-model wrapper from config and prepared batch shapes.
89. `_expected_control_channels()` computes the `[inactive; reactive; mask]` control channel count.
90. `_uses_action_added_kv()` decides whether action tokens should also populate Wan’s added-K/V image-conditioning path.
91. `WanVACETransformer3DModel.from_pretrained()` loads the pretrained Wan VACE backbone because the run starts from base pretrained weights.
92. `WanVACEWorldModel.__init__()` wraps the backbone with repo-specific control-stream defaults.
93. `build_action_encoder_from_config()` dispatches conditioning encoder creation.
94. `build_conditioning_encoder_for_model()` chooses the action-conditioning path because `conditioning_mode=action`.
95. `_resolve_action_mlp_dim()` resolves `action_mlp_dim=0` to the legacy linear projection path.
96. `ActionTokenEncoder.__init__()` constructs the action-token encoder module.
97. `_move_train_modules_to_runtime()` moves the model and action encoder onto the active runtime device and dtype.
98. `_configure_trainable_parameters()` marks the requested model and action-encoder parameters as trainable.
99. `_build_optimizer()` constructs the requested optimizer.
100. `transformers.Adafactor(...)` creates the actual optimizer instance.
101. The optimizer-update loop begins and repeats until `step=400` or an earlier stop condition fires.
102. `next(data_iter)` pulls the next training batch each time the cached-batch path is not active.
103. `collate_tensor_dict()` stacks that active training batch.
104. `prepare_packed_batch()` runs on the active batch for the current optimizer step, repeating the latent preparation chain from steps `43-69`.
105. `train_chunkwise_batch()` runs one optimizer update with chunkwise teacher-forced flow matching.
106. `_build_training_autocast_context()` builds the train-time autocast context.
107. `ActionTokenEncoder.forward()` projects the packed action plan into Wan cross-attention tokens.
108. `ActionTokenEncoder._project_tokens()` applies the configured base projection path.
109. `ActionTokenEncoder._apply_temporal_mixer()` is called and returns the tokens unchanged because the temporal mixer is disabled here.
110. `ActionTokenEncoder._project_output()` maps the internal action features into Wan’s token space.
111. `ActionTokenEncoder._apply_output_scale()` scales the projected action tokens by `action_token_scale=0.05`.
112. `chunkwise_teacher_forcing_loss()` dispatches the core latent-space training loss.
113. `_chunkwise_teacher_forcing_video_loss()` executes the actual teacher-forced chunkwise loss computation.
114. `_validate_chunkwise_video_inputs()` validates the structured latent/video/action inputs.
115. `normalize_chunk_schedule_mode()` canonicalizes the chunk schedule mode inside the loss.
116. `_validate_motion_loss_max_weight()` validates the motion-loss cap.
117. `_validate_future_latent_residual_mode()` validates the residual-coordinate choice.
118. `_validate_future_loss_early_bias()` validates the per-frame early-future bias value.
119. `_validate_future_chunk_early_bias()` validates the per-chunk early bias value.
120. `_build_future_latent_residual_base()` builds the latent residual baseline that is subtracted before denoising.
121. `build_chunk_schedule()` builds the latent future chunk boundaries for the training loss.
122. `_validate_schedule_args()` checks that the chunk schedule is feasible.
123. `resolve_num_chunks()` resolves the exact number of future chunks from `k=1`.
124. `normalize_chunk_schedule_mode()` re-canonicalizes the schedule mode inside the chunk helper.
125. `sample_t()` samples one normalized flow-matching timestep per batch item.
126. `w()` computes the per-sample loss weighting for the sampled timesteps.
127. `make_noisy_and_target()` constructs the noisy latent chunk and its target velocity.
128. `_select_teacher_forcing_future_input()` builds the future window the model will denoise on this step.
129. `_select_observed_video()` builds the observed latent prefix for teacher forcing.
130. `_select_teacher_forcing_future_chunk_ids()` builds the latent chunk-id vector for the future window.
131. `build_block_causal_mask()` creates the additive block-causal attention mask.
132. `_bool_to_additive()` converts the boolean block mask into an additive `0/-inf` mask.
133. `_select_action_tokens()` chooses the active action-token slice for this chunk.
134. `normalized_t_to_scheduler_timestep()` maps normalized `t` onto Wan’s scheduler timestep scale.
135. `_select_teacher_forcing_future_residual_base()` aligns the residual baseline to the future window the model sees.
136. `WanVACEWorldModel.forward()` runs the repo wrapper around the Wan VACE backbone.
137. `_slice_control_latent_template()` slices the cached black control-latent template to the active rollout length.
138. `_slice_control_latent_template()` slices the cached gray control-latent template to the active rollout length.
139. `build_vace_control_tensor()` constructs the `[inactive; reactive; mask]` control tensor passed to the backbone.
140. `_resolve_control_fill_latents()` validates the inactive-fill control latents.
141. `_resolve_control_fill_latents()` validates the reactive-fill control latents.
142. `_patches_per_frame()` computes how many patch tokens the backbone emits per latent frame.
143. `expand_block_causal_mask_to_patch_tokens()` expands the latent-frame block mask into Wan patch-token space.
144. `_resolve_control_scale()` builds the per-layer control scale tensor.
145. `self.backbone(...)` runs the pretrained Wan VACE transformer itself.
146. `_update_predicted_future_prefix()` is still called even though `teacher_forcing_observation_mode=full_prefix`, and in this mode it leaves the predicted prefix unchanged.
147. `_compute_motion_loss_weight()` builds the per-region motion weighting multiplier.
148. `_compute_future_loss_early_weight()` builds the per-frame early-future weighting multiplier.
149. `_compute_future_chunk_early_weight()` builds the per-chunk early weighting multiplier.
150. `_compute_action_token_latent_aux_loss()` computes the direct auxiliary supervision on the action tokens.
151. `ActionTokenEncoder.predict_future_latent_summary()` predicts the latent summary from the projected action tokens.
152. `_build_future_latent_aux_target()` builds the target future latent summary in the same coordinate system.
153. `torch.nn.utils.clip_grad_norm_(...)` clips the combined model and action-encoder gradients.
154. `optimizer.step()` applies the Adafactor update.
155. `ChunkwiseStepMetrics.to_log_dict()` converts the step metrics into the JSONL log payload.
156. `_should_run_validation()` decides whether this step should also run validation.
157. `_evaluate_validation_loss()` runs on validation steps and averages a deterministic prefix of held-out batches.
158. `prepare_packed_batch()` runs inside validation-loss scoring, repeating the latent preparation chain from steps `43-69`.
159. `_evaluate_loss()` computes one eval-mode chunkwise loss on that prepared validation batch.
160. `ActionTokenEncoder.forward()` runs again inside validation scoring.
161. `chunkwise_teacher_forcing_loss()` runs again inside validation scoring.
162. `_compute_action_token_latent_aux_loss()` runs again inside validation scoring.
163. `_update_validation_best_only()` updates the best validation-loss tracker because validation patience is disabled here.
164. `_should_save_checkpoint()` decides whether the current train step should emit a checkpoint.
165. `_build_checkpoint_extra_state()` packs the resumable config and validation state for the checkpoint payload.
166. `save_checkpoint()` writes the step checkpoint when the save rule fires.
167. `append_jsonl()` appends the step payload to `metrics.jsonl`.
168. `save_checkpoint()` writes the final checkpoint after the training loop ends.
169. `_build_checkpoint_extra_state()` packs the final checkpoint metadata.
170. `test -f "$RUN_ROOT/metrics.jsonl"` verifies that the metrics log was produced.
171. `test -f "$RUN_ROOT/checkpoints/step_0000350.pt"` verifies that checkpoint `350` exists.
172. `test -f "$RUN_ROOT/checkpoints/step_0000400.pt"` verifies that checkpoint `400` exists.
173. `for STEP in 350 400; do ... done` begins the per-checkpoint evaluation loop. Steps `174-291` execute once for `STEP=350` and once again for `STEP=400`.
174. `printf '%07d' "$STEP"` derives the zero-padded seven-digit checkpoint step string.
175. `printf '%04d' "$STEP"` derives the zero-padded four-digit eval-label step string.
176. `CKPT=...` defines the specific checkpoint path being evaluated in this loop pass.
177. `EVAL_ROOT=...` defines the eval artifact directory for this loop pass.
178. `LOG_PATH=...` defines the eval stdout log path for this loop pass.
179. `STEM=...` defines the per-checkpoint artifact stem.
180. `test -f "$CKPT"` verifies that the requested checkpoint exists before launching eval.
181. `test ! -e "$EVAL_ROOT"` refuses to overwrite an existing eval directory.
182. `mkdir -p "$EVAL_ROOT"` creates the eval output directory.
183. `python -u scripts/check/sweep_local_repo_resolutions.py ...` launches the checkpoint-mode evaluation entrypoint.
184. `main()` in `scripts/check/sweep_local_repo_resolutions.py` starts the checkpoint-mode sweep.
185. `_parse_args()` parses the evaluation CLI.
186. `_parse_resolution()` converts `224x128` into integer width and height values.
187. `_resolve_output_artifacts()` derives the predicted-video path, comparison-video path, and summary path.
188. `_resolve_output_root()` resolves the active eval output root.
189. `_checkpoint_run_stem()` derives the stable artifact prefix from the checkpoint filename and its run directory.
190. `_resolve_plausibility_output_path()` resolves the plausibility JSON path.
191. `_resolve_motion_output_path()` resolves the arm-motion JSON path.
192. `_run_one_checkpoint_resolution()` executes the full checkpoint-mode eval at this one resolution.
193. `_load_checkpoint_runtime_config()` loads the checkpoint payload and reconstructs the saved runtime config namespace.
194. `torch.load()` loads the checkpoint from disk.
195. `normalize_chunk_schedule_mode()` canonicalizes the checkpoint’s saved chunk schedule mode.
196. `_apply_runtime_overrides()` applies the CLI-side runtime overrides on top of the saved checkpoint config.
197. `_resolve_device()` chooses the active eval device.
198. `_select_runtime_dtype()` chooses the eval runtime dtype.
199. `_load_checkpoint_clip()` loads the exact evaluation video clip and per-frame action sequence from LeRobot.
200. `LeRobotDataset(...)` opens the requested dataset episode for checkpoint-mode eval.
201. `_run_checkpoint_world_model()` executes the repo’s direct chunkwise world-model inference path.
202. `WanVAE.from_pretrained()` loads the Wan VAE wrapper used for checkpoint-mode inference.
203. `_infer_checkpoint_action_dim()` infers the expected action width from the saved action-encoder state.
204. `_select_action_tensor()` resolves whether eval should use the full raw action sequence or a broadcast single action source.
205. `prepare_packed_batch()` converts the eval clip into latents and action-plan tensors, repeating the latent preparation chain from steps `43-69`.
206. `build_wan_vace_runtime_modules()` builds the runtime model modules and overlays the checkpoint.
207. `_merge_runtime_backbone_config()` restores saved train-time backbone and conditioning settings from the checkpoint metadata.
208. `_make_default_config_like()` creates the default config object used to decide which saved keys should override runtime defaults.
209. `_apply_untouched_none_contract_defaults()` checks whether untouched none-conditioned checkpoints should be upgraded to the dual-anchor runtime contract.
210. `_should_upgrade_untouched_none_contract()` decides whether that untouched-none upgrade applies for this checkpoint.
211. `_copy_runtime_contract_defaults()` copies any effective runtime-contract defaults back to the caller-visible config object.
212. `build_wan_vace_model_from_config()` rebuilds the Wan VACE world-model wrapper for evaluation.
213. `_expected_control_channels()` recomputes the control tensor width for eval.
214. `_uses_action_added_kv()` checks whether action tokens should also enter the added-K/V image-conditioning path for eval.
215. `WanVACETransformer3DModel.from_pretrained()` loads the pretrained backbone weights for the eval runtime model.
216. `WanVACEWorldModel.__init__()` wraps the eval backbone in the repo runtime adapter.
217. `build_conditioning_encoder_for_model()` rebuilds the eval-time action encoder.
218. `_resolve_action_mlp_dim()` resolves the eval action-encoder projection width choice.
219. `ActionTokenEncoder.__init__()` constructs the eval action encoder.
220. `apply_wan_vace_checkpoint_overlay()` overlays the checkpoint weights onto the runtime model and action encoder.
221. `_checkpoint_uses_fresh_action_encoder()` decides whether the checkpoint should keep a fresh action encoder instead of loading saved action weights.
222. `_load_action_encoder_state_dict()` overlays the saved action-encoder weights when the fresh-encoder rule does not apply.
223. `ActionTokenEncoder.forward()` projects the eval action plan into cross-attention tokens.
224. `FlowMatchEulerDiscreteScheduler.from_pretrained()` loads the scheduler used for chunkwise latent rollout.
225. `infer_future_videos_chunkwise()` runs chunkwise latent-space autoregressive inference.
226. `_validate_video_infer_inputs()` validates the latent rollout inputs.
227. `normalize_chunk_schedule_mode()` canonicalizes the eval chunk schedule mode.
228. `_build_rollout_boundaries()` resolves the latent rollout chunk boundaries.
229. `resolve_num_chunks()` computes the number of future chunks implied by `k=1`.
230. `_build_future_latent_residual_base()` builds the latent baseline that gets added back after residual-space sampling.
231. `build_block_causal_mask()` builds the inference-time block-causal mask.
232. `_bool_to_additive()` converts that inference-time block mask to additive form.
233. `_select_chunk_conditioning_tokens()` selects the positive cross-attention tokens for the active rollout chunk.
234. `_select_chunk_conditioning_tokens()` selects the image-conditioning tokens for the active rollout chunk.
235. `_select_chunk_conditioning_tokens()` selects the negative-conditioning tokens for the active rollout chunk.
236. `scheduler.set_timesteps()` creates the explicit denoising schedule for this rollout.
237. `WanVACEWorldModel.forward()` predicts the latent velocity for the active inference chunk.
238. `_slice_control_latent_template()` slices the black control template for inference.
239. `_slice_control_latent_template()` slices the gray control template for inference.
240. `build_vace_control_tensor()` builds the inference control tensor.
241. `_resolve_control_fill_latents()` validates inactive-fill latents for inference.
242. `_resolve_control_fill_latents()` validates reactive-fill latents for inference.
243. `_patches_per_frame()` computes the patch-token expansion factor for inference masking.
244. `expand_block_causal_mask_to_patch_tokens()` expands the latent-frame mask into patch-token space for inference.
245. `_resolve_control_scale()` builds the eval control scale tensor.
246. `scheduler.step()` advances the latent chunk by one scheduler step.
247. `_decode_future_latents()` decodes the predicted future latents and the target future latents back to full-frame videos.
248. `WanVAE.decode()` decodes the predicted full latent window.
249. `_denormalize_latents()` maps the repo latent format back to raw VAE latents.
250. `_latent_stats()` reads the latent normalization stats again for decode.
251. `_cast_to_vae_runtime()` aligns the latent tensor with the VAE runtime device and dtype for decode.
252. `_vae_runtime_device_dtype()` reads the VAE runtime device and dtype for decode.
253. `self.vae.decode()` runs the diffusers VAE decoder.
254. `_extract_decoded_sample()` extracts the decoded tensor from the diffusers decode payload.
255. `_from_bcthw()` converts the decoded tensor back to `BTCHW`.
256. `_format_output_range()` maps the decoded tensor into the zero-to-one export range.
257. `WanVAE.decode()` decodes the target full latent window through the same decode path.
258. `preprocess_video_for_vae()` preprocesses the raw target video for export-side alignment.
259. `_center_crop_video_to_multiple()` re-applies the spatial multiple crop to the raw target video for export alignment.
260. `_build_rollout_video()` aligns the target and predicted videos to a common frame count for export.
261. `_normalize_video_for_export()` normalizes the target rollout into zero-to-one export range.
262. `_normalize_video_for_export()` normalizes the predicted rollout into zero-to-one export range.
263. `_tensor_video_to_frames()` converts the predicted rollout tensor into HWC uint8 frames.
264. `_export_video()` writes the predicted MP4.
265. `_build_side_by_side_video()` concatenates the target and prediction videos horizontally.
266. `_tensor_video_to_frames()` converts the side-by-side comparison tensor into HWC uint8 frames.
267. `_export_video()` writes the comparison MP4.
268. `_write_plausibility_report()` computes and saves the plausibility JSON.
269. `_load_plausibility_module()` loads `scripts/check/check_generated_video_plausibility.py` as a reusable module.
270. `_tensor_video_to_frames()` converts the reference rollout into numpy RGB frames for plausibility scoring.
271. `_tensor_video_to_frames()` converts the generated rollout into numpy RGB frames for plausibility scoring.
272. `checker.align_videos()` aligns the reference and generated frame sequences before plausibility scoring.
273. `checker.analyze_frame()` scores each aligned frame.
274. `checker.build_summary()` aggregates the frame-level plausibility metrics into one summary.
275. `checker.save_report()` writes `plausibility_report.json`.
276. `_write_arm_motion_report()` computes and saves the arm-motion JSON and crop artifacts.
277. `_load_arm_motion_module()` loads `scripts/check/check_arm_motion_alignment.py` as a reusable module.
278. `_tensor_video_to_frames()` converts the reference rollout into numpy RGB frames for arm-motion scoring.
279. `_tensor_video_to_frames()` converts the generated rollout into numpy RGB frames for arm-motion scoring.
280. `checker.align_videos()` aligns the videos for arm-motion scoring.
281. `checker.build_motion_summary()` computes the arm-motion summary and ROI.
282. `checker.motion_crop_video_path()` derives the arm-crop comparison video path.
283. `checker.roi_preview_path()` derives the ROI preview image path.
284. `checker.draw_roi_preview()` writes the ROI preview image.
285. `checker.save_motion_crop_comparison()` writes the arm-crop comparison MP4.
286. `checker.save_report()` writes `arm_motion_report.json`.
287. `_save_summary()` writes the resolution sweep summary JSON.
288. `test -f "$EVAL_ROOT/${STEM}.mp4"` verifies that the predicted MP4 exists.
289. `test -f "$EVAL_ROOT/${STEM}_comparison.mp4"` verifies that the comparison MP4 exists.
290. `test -f "$EVAL_ROOT/${STEM}_arm_crop_comparison.mp4"` verifies that the arm-crop comparison MP4 exists.
291. `test -f "$EVAL_ROOT/${STEM}_summary.json"` verifies that the per-checkpoint summary JSON exists.
292. `test -f "$EVAL_ROOT/arm_motion_report.json"` verifies that the arm-motion JSON exists.
293. `test -f "$EVAL_ROOT/plausibility_report.json"` verifies that the plausibility JSON exists.

## Verified numbered process

Verified entries below were exercised directly with short shell or Python probes in
this session. I am only marking steps that were actually checked.

- `1` ✅ `cd /home/amaniscalco/world-model-v1`: ran all investigation probes from the repo root and confirmed repo-relative paths resolved under `/home/amaniscalco/world-model-v1`.
- `2` ✅ `source .venv/bin/activate`: ran the shell, `pytest`, and Python verification probes inside `.venv`.
- `3` ✅ `set -euo pipefail`: enabled strict shell mode in the setup probe and confirmed the command chain completed without unset-variable or pipeline failures.
- `4` ✅ `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE`: verified both environment variables reported as `unset` after the shell preflight.
- `5` ✅ `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: verified the exported allocator setting printed back as `expandable_segments:True`.
- `6` ✅ `RUN_ROOT=...`: verified the shell preflight resolved `RUN_ROOT` to `/home/amaniscalco/world-model-v1/runs/investigation1_shell_setup_probe`.
- `7` ✅ `RUN_NAME=$(basename "$RUN_ROOT")`: verified `basename` derived `investigation1_shell_setup_probe` from the probe `RUN_ROOT`.
- `8` ✅ `TRAIN_LOG="$RUN_ROOT/train_stdout.log"`: verified the derived log path was `/home/amaniscalco/world-model-v1/runs/investigation1_shell_setup_probe/train_stdout.log`.
- `9` ✅ `test ! -e "$RUN_ROOT"`: verified the freshness guard succeeds on a brand-new probe directory before creation.
- `10` ✅ `mkdir -p "$RUN_ROOT"`: verified `mkdir -p` created the probe directory and that it could be removed cleanly after the check.
- `13` ✅ `_load_args()`: imported `scripts/train/world_model.py`, set `sys.argv` to investigation-style training overrides, and confirmed `_load_args()` returned a fully merged `TrainScriptConfig`.
- `14` ✅ `_config_parser()`: verified `_load_args()` successfully consumed `--config configs/train/world_model.yaml` from `sys.argv` before building the full parser.
- `15` ✅ `load_train_config()`: loaded `configs/train/world_model.yaml` directly and confirmed typed defaults including `context_len=9`, `horizon_len=8`, `validation_enabled=True`, `checkpoint_every=100`, and `conditioning_mode='none'`.
- `16` ✅ `_load_config()`: verified the train config loader resolved the canonical YAML path and returned a concrete `TrainScriptConfig` instance.
- `17` ✅ `_load_yaml()`: verified YAML loading succeeded from `configs/train/world_model.yaml` because `load_train_config()` returned the expected mapping-backed config without error.
- `18` ✅ `_coerce_dataclass()`: verified the loaded YAML materialized as `TrainScriptConfig`, not a raw dict.
- `19` ✅ `_coerce_field_value()`: verified YAML list-backed `vace_layers` was coerced into the tuple `(0, 5, 10, 15, 20, 25, 30, 35)`.
- `20` ✅ `_build_parser()`: built the full training CLI parser from `load_train_config('configs/train/world_model.yaml')` and confirmed it accepted the investigation command's override surface.
- `21` ✅ `argparse.parse_args()`: parsed the investigation-style training overrides and confirmed `optimizer_name='adafactor'`, `auto_batch_size=False`, `gradient_checkpointing=True`, `action_output_zero_init=True`, and `action_backbone_added_kv_mode='none'`.
- `22` ✅ `apply_namespace_overrides()`: verified `_load_args()` overlaid CLI values onto YAML defaults by returning `conditioning_mode='action'`, `future_control_fill_mode='last_context_frame'`, `future_latent_residual_mode='last_context_frame'`, `repo_id='lerobot/aloha_static_fork_pick_up'`, and `video_key='observation.images.cam_high'`.
- `23` ✅ `normalize_chunk_schedule_mode()`: verified `_load_args()` returned `chunk_schedule_mode='k_chunks'` after the normalization pass.
- `11` ✅ `python -u scripts/train/world_model.py ...`: ran the bounded 100-step reproduction successfully to completion and wrote the expected training artifacts under `runs/investigation1_basepretrained_action_tok005_zeroinit_noaddedkv_aux1_adafactor5e5_bs1gc_step100_p1`.
- `12` ✅ `main()` in `scripts/train/world_model.py`: verified the entrypoint completed the full bounded training loop because `train_stdout.log` reached `final_checkpoint=...step_0000100.pt`.
- `26` ✅ `output_dir.mkdir(parents=True, exist_ok=True)`: verified the run directory exists and contains `train_stdout.log`, `metrics.jsonl`, and `checkpoints/`.
- `27` ✅ `_select_runtime_dtype()`: verified the training runtime selected `torch.bfloat16` on `Device: cuda` from `train_stdout.log`.
- `32` ✅ `resolve_lerobot_episode_ids()`: verified episode enumeration happened because the bounded run printed a concrete validation split preview.
- `34` ✅ `split_train_validation_episode_ids()`: verified the split produced `train_episodes=90` and `val_episodes=10`.
- `38` ✅ `validate_wan_temporal_window()`: verified the successful batch-preparation path reached `Latent window: context=3 future=2 total=5`, which is the expected latent packing for `context_len=9` and `horizon_len=8`.
- `101` ✅ optimizer-update loop: verified the loop executed through `step=000100`, and `metrics.jsonl` contains exactly `100` rows.
- `156` ✅ `_should_run_validation()`: verified validation fired at `step=000050` and `step=000100`.
- `166` ✅ `save_checkpoint()`: verified checkpoints were written at `step_0000050.pt` and `step_0000100.pt`.
- `167` ✅ `append_jsonl()`: verified `metrics.jsonl` accumulated the per-step payloads through step `100`.
- `168` ✅ `save_checkpoint()` final: verified the bounded run wrote `final_checkpoint=/home/amaniscalco/world-model-v1/runs/investigation1_basepretrained_action_tok005_zeroinit_noaddedkv_aux1_adafactor5e5_bs1gc_step100_p1/checkpoints/step_0000100.pt`.
- `171` ✅ `test -f "$RUN_ROOT/checkpoints/step_0000350.pt"`: verified the exact original `step_0000350.pt` checkpoint exists after the resumed `100 -> 400` continuation.
- `172` ✅ `test -f "$RUN_ROOT/checkpoints/step_0000400.pt"`: verified the exact original `step_0000400.pt` checkpoint exists after the resumed `100 -> 400` continuation.
- `185` ✅ `_parse_args()`: imported `scripts/check/sweep_local_repo_resolutions.py`, set `sys.argv` to investigation-style checkpoint-eval arguments, and confirmed `mode='checkpoint'`, `episode_index=1`, `start_frame=60`, `action_source='sequence'`, and `single_chunk_rollout=True`.
- `186` ✅ `_parse_resolution()`: verified `224x128` parses to width `224` and height `128`.
- `187` ✅ `_resolve_output_artifacts()`: verified a single-resolution checkpoint probe resolves prediction, comparison, and summary outputs to `runs/tmp_eval_probe/tmp_probe_step_0000100.mp4`, `runs/tmp_eval_probe/tmp_probe_step_0000100_comparison.mp4`, and `runs/tmp_eval_probe/tmp_probe_step_0000100_summary.json`.
- `188` ✅ `_resolve_output_root()`: verified an explicit `--output-dir runs/tmp_eval_probe` is preserved as the active eval output root.
- `189` ✅ `_checkpoint_run_stem()`: verified the checkpoint stem contributed `tmp_probe_step_0000100` to the resolved eval artifact names.
- `190` ✅ `_resolve_plausibility_output_path()`: verified the plausibility report path resolves to `runs/tmp_eval_probe/plausibility_report.json`.
- `191` ✅ `_resolve_motion_output_path()`: verified the arm-motion report path resolves to `runs/tmp_eval_probe/arm_motion_report.json`.
- `173` ✅ `for STEP in 350 400; do ... done`: verified the exact original eval loop executed once for `STEP=350` and once for `STEP=400`.
- `174` ✅ `printf '%07d' "$STEP"`: verified the loop produced the expected padded checkpoint ids `0000350` and `0000400`.
- `175` ✅ `printf '%04d' "$STEP"`: verified the loop produced the expected eval labels `0350` and `0400`.
- `176` ✅ `CKPT=...`: verified the loop resolved concrete checkpoint paths for `step_0000350.pt` and `step_0000400.pt`.
- `177` ✅ `EVAL_ROOT=...`: verified the loop resolved distinct eval directories for `...step0350_operator` and `...step0400_operator`.
- `178` ✅ `LOG_PATH=...`: verified each eval pass wrote its own `eval_stdout.log`.
- `179` ✅ `STEM=...`: verified each eval pass resolved the expected artifact stem `investigation1_basepretrained_action_tok005_zeroinit_noaddedkv_aux1_adafactor5e5_bs1gc_step100_p1_step_0000350` or `...step_0000400`.
- `180` ✅ `test -f "$CKPT"`: verified the checkpoint existence guard passed for both exact-command eval passes.
- `181` ✅ `test ! -e "$EVAL_ROOT"`: verified both exact-command eval directories were absent before creation.
- `182` ✅ `mkdir -p "$EVAL_ROOT"`: verified the loop created both exact-command eval directories successfully.
- `183` ✅ `python -u scripts/check/sweep_local_repo_resolutions.py ...`: ran checkpoint-mode eval on `step_0000100.pt` successfully and produced the predicted MP4, comparison MP4, arm-crop comparison MP4, summary JSON, plausibility JSON, and arm-motion JSON.
- `268` ✅ `_write_plausibility_report()`: verified `plausibility_report.json` was written and passed all `17/17` compared frames with `video_flags=[]`.
- `276` ✅ `_write_arm_motion_report()`: verified `arm_motion_report.json` was written and flagged the rollout as `misaligned` due to `temporal_profile_mismatch`.
- `287` ✅ `_save_summary()`: verified the sweep summary JSON was written for the checkpoint eval run.
- `288` ✅ `test -f "$EVAL_ROOT/${STEM}.mp4"`: verified the predicted rollout MP4 exists after eval.
- `289` ✅ `test -f "$EVAL_ROOT/${STEM}_comparison.mp4"`: verified the side-by-side comparison MP4 exists after eval.
- `290` ✅ `test -f "$EVAL_ROOT/${STEM}_arm_crop_comparison.mp4"`: verified the arm-crop comparison MP4 exists after eval.
- `291` ✅ `test -f "$EVAL_ROOT/${STEM}_summary.json"`: verified the per-checkpoint summary JSON exists after eval.
- `292` ✅ `test -f "$EVAL_ROOT/arm_motion_report.json"`: verified the arm-motion JSON exists after eval.
- `293` ✅ `test -f "$EVAL_ROOT/plausibility_report.json"`: verified the plausibility JSON exists after eval.

Run validation note from the bounded `step_0000100` probe:

- Visual inspection of the comparison clip and arm-crop clip shows a coherent, non-collapsed rollout. The robot arm and tool stay recognizable and stay in roughly the correct workspace, but the active arm motion is slightly out of sync with the reference over the forecast window instead of committing at exactly the same time.
- The plausibility gate passed cleanly, but the arm-motion report marked the clip `misaligned` because the temporal motion profile correlation was only `0.682`, just under the configured `0.7` floor.
- Training itself looks healthy rather than broken: loss and validation both improved over the 100-step probe, ending at `val_loss=0.230510` with checkpoints at steps `50` and `100`.

Run validation note from the exact `step_0000350` and `step_0000400` endpoints:

- Both exact-command endpoint clips remain plausible and non-collapsed. The scene stays stable, the arm and tool remain recognizable, and neither checkpoint falls back into the catastrophic collapse family.
- `step_0000350` looks slightly cleaner in the late frames, while `step_0000400` shows slightly stronger late arm movement but also a brighter late gripper/arm flare. Both still fail the arm-motion alignment gate.
- The arm-motion metrics improved only marginally from `profile_correlation=0.6797` at `step_0000350` to `0.6993` at `step_0000400`, which is still just below the configured `0.7` pass floor.
- The resumed training trajectory was strongly non-monotonic: `val_loss` improved to `0.082952` at `step_0000250`, regressed hard to `0.569113` at `step_0000350`, then recovered to a new best `0.055560` at `step_0000400`. That pattern is consistent with a drift-and-recovery cycle inside the same neighborhood rather than a smooth quality climb.

Run validation note from the checkpoint-localization `step_0000250` and `step_0000300` sweep:

- `step_0000250` stays coherent and plausible, but the late forecast frames show more bright gripper/arm flare and softer motion timing than the reference. The arm-motion report still marks it `misaligned` with `profile_correlation=0.3991`, even though plausibility passed cleanly.
- `step_0000300` is the first checked checkpoint in this family that clears the arm-motion gate. Visual inspection of the full-frame and arm-crop contact sheets shows cleaner late-frame motion than `step_0000250`, with less flare and a closer match to the reference commitment timing.
- The scalar reports support the visual ranking rather than contradicting it: `step_0000300` improved `mean_frame_mae_rgb_0_255` from `5.8599` to `5.0353`, improved `temporal_delta_ratio` from `1.2712` to `1.1678`, and raised `profile_correlation` from `0.3991` to `0.7227`.
- The current best explanation for this run family is that the model is undertrained at `step_0000250`, reaches the best motion-timing regime around `step_0000300`, then drifts away from that visual optimum by `step_0000350` and `step_0000400` even though the later validation losses recover.

Run validation note from the earlier `step_0000150` and `step_0000200` sweep:

- `step_0000150` remains plausible and visually coherent, but it still misses the reference timing. The late frames show a softer, slightly smeared arm commitment, and the motion report still flags `temporal_profile_mismatch` with `profile_correlation=0.5673`.
- `step_0000200` is clearly worse than `step_0000150` by both video and motion metrics. The late forecast frames show a much brighter arm/gripper flare and more aggressive movement, and the motion report adds `overactive_motion` on top of `temporal_profile_mismatch`, with `profile_correlation=0.4122` and `total_motion_ratio=1.4657`.
- These earlier checkpoints show that the trajectory is not a simple monotonic climb toward `step_0000300`. `step_0000100` was already closer to the reference timing than `150`, `200`, or `250`, then `300` briefly becomes the best visible checkpoint before the run regresses again at `350` and `400`.
- The current causal picture is a narrow, unstable success window rather than steady improvement: the same full-backbone, subset-8, batch-size-1 training branch oscillates between plausible-but-misaligned motion, overactive motion, and one clearly good checkpoint at `step_0000300`.

Run validation note from the final saved checkpoint, `step_0000050`:

- `step_0000050` is plausible and stable, but it is clearly undercommitted. The late forecast frames keep the arm close to its starting posture instead of matching the reference pickup motion, and the arm-motion report marks it `undercommitted`.
- The scalar report matches the video: `profile_correlation=0.4163`, `late_motion_ratio=0.6957` just misses the configured `0.7` floor, and `total_motion_ratio=1.0306` shows that the issue is not runaway motion but insufficient late commitment.
- With `step_0000050` included, the full saved-checkpoint story for this exact command is now consistent: early undercommitment at `50`, a temporary timing improvement at `100`, a regression into mistimed or overactive motion at `150/200/250`, one clearly good checkpoint at `300`, and then a drift back to near-threshold but still failing motion alignment at `350/400`.
- The most defensible root-cause hypothesis for this command family is training instability inside an extremely small, high-variance optimization regime: full-backbone updates, `batch_size=1`, `subset_size=8`, and Adafactor `5e-5` produce plausible videos throughout, but only a narrow part of the trajectory lines up the arm-motion timing well enough to pass.


## Kept Code Changes
