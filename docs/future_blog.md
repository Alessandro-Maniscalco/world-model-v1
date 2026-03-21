What Changes It Is Testing

context_len and horizon_len: change how much past the model sees and how much future it predicts. This tests whether the model is failing because it lacks memory or because the prediction window is the wrong size.
k: changes how many future chunks training/inference roll through. This tests whether more autoregressive structure helps the model commit to motion earlier.
motion_loss_alpha: adds extra loss weight on moving regions. This asks: “is the model too happy to predict static frames?”
motion_loss_max_weight: caps that motion weighting so a few very active pixels do not dominate training.
motion_loss_excess_only: only boosts unusually high-motion regions, instead of boosting all motion.
future_loss_early_bias: gives earlier future timesteps more loss weight. This directly tests “can we force earlier movement by caring more about early mistakes?”
future_chunk_early_bias: same idea, but at the chunk level instead of per-frame.
action_temporal_difference_scale: feeds action changes over time, not just the raw action values. This tests whether the model needs to know “how the command is changing,” not just “what the command is.”
action_temporal_mixer_kernel_size and action_temporal_mixer_scale: adds a small temporal mixer over action tokens. This tests whether better local temporal processing of actions helps timing.
action_conditioning_window=full: gives the model the whole remaining action plan on every denoising step, instead of only the active chunk. This tests whether it is moving late because it cannot “see the whole plan.”
action_order_conditioning=true: adds positional/order information to action tokens so the model knows which action token is early vs late.
action_control_prior_scale: injects an action-derived latent prior into the future latent stream. This tests whether stronger action guidance inside the VACE path helps.
teacher_forcing_observation_mode=past_only: later chunks only see the real past, not the real future prefix. This tests whether teacher forcing is leaking too much help during training.
teacher_forcing_observation_mode=predicted_prefix: later chunks see the model’s own earlier predictions. This tests whether rollout-style feedback is needed during training.
teacher_forcing_future_input_mode=active_chunk: only the active chunk is denoised during teacher forcing, matching inference more closely. This tests train/infer mismatch.
