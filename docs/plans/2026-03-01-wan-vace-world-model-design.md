# Wan VACE World Model Design

## Goal

Replace the repo's current custom DiT wrapper with a Wan VACE-compatible inner
model while preserving the existing chunkwise teacher-forcing outer training
structure.

## Approved decisions

1. Keep the current K+1 chunkwise teacher-forcing loop as the outer trainer.
2. Vendor upstream Wan code locally instead of wrapping diffusers directly.
3. Follow the diffusers Wan VACE conditioning split exactly. Conceptually, the
   conditioning unit becomes `V_wm = [A; F; M]` instead of VACE's original
   `V = [T; F; M]`, but the implementation still preserves the diffusers split:
   - `encoder_hidden_states = action tokens`
   - `control_hidden_states = past latents + masks`
4. Keep latent videos structured as `[B, C, T, H, W]` through the model path.
5. Patch the vendored Wan/VACE attention stack to accept the repo's
   block-causal mask.
6. Defer proprio integration in the first migration so the Wan VACE path can be
   validated with fewer moving parts.

## Architecture summary

The outer trainer still owns latent-time scheduling, teacher forcing, timestep
sampling, noise injection, and chunk-local loss computation. The inner model
becomes a vendored Wan VACE backbone wrapped by a local adapter that maps
world-model inputs into Wan's expected conditioning split.

Action conditioning changes from the current pooled AdaLN vector to a token
sequence. Those action tokens replace text inside the conceptual VACE
conditioning unit, so the world-model conditioning becomes `V_wm = [A; F; M]`.
At execution time, action tokens become Wan `encoder_hidden_states`, while
teacher-forced clean latent observations plus masks remain in the VACE control
stream, matching diffusers `pipeline_wan_vace.py`.

The main local fork is mask threading. The repo's current chunkwise teacher
forcing depends on additive block-causal masks, but upstream Wan/VACE does not
expose that path through self-attention. The vendored copy must accept an
expanded patch-token mask so future chunks remain hidden during training.

## Module layout

### Vendored code

Copy upstream Wan code into:

1. `src/world_model/vendor/wan/`

The vendored tree should include the Wan VACE transformer and the files it
imports for attention, rotary embeddings, timestep conditioning, normalization,
and feed-forward layers.

### Local adapter code

Add repo-owned layers:

1. `src/world_model/models/wan_vace_world_model.py`
2. `src/world_model/models/wan_vace_conditioning.py`

The adapter owns:

1. action-token encoding and projection
2. VACE control-tensor assembly from latent videos and masks
3. patch-token mask expansion
4. Wan/VACE forward orchestration under the repo's chunkwise training scheme

## Data flow

### Training

1. Decode batch from dataset.
2. Encode full video window with frozen Wan VAE to `[B, C_lat, T_lat, H_lat, W_lat]`.
3. Split latent time into context and future windows.
4. For chunk step `i`:
   - noise only the active future chunk
   - keep earlier future chunks clean under teacher forcing
   - build action tokens aligned to the future horizon
   - build a VACE control tensor from the currently available clean latents
   - build masks that mark observed vs generated regions
   - expand the block-causal mask from latent frames to Wan patch tokens
   - run Wan VACE forward
   - compute flow-matching loss on the active chunk only

### Inference

1. Encode observed context frames.
2. Build action tokens for the rollout horizon.
3. Build VACE control tensors from observed context plus rollout masks.
4. Denoise chunk by chunk without teacher forcing.
5. Decode final future latents back to video.

## Scope decisions

### In scope

1. vendored Wan/VACE backbone
2. action-token cross-attention path
3. VACE control path from past latents + masks
4. chunkwise mask expansion into patch-token space
5. training config and trainer updates
6. test-first migration with mocked backbone coverage

### Out of scope for first pass

1. proprio as part of the Wan conditioning contract
2. non-chunkwise training alternatives
3. direct diffusers model ownership in the main training loop

## Risks

1. The vendored Wan/VACE stack is large, so the migration must be staged behind
   tests and narrow adapter seams.
2. Patch-token mask expansion is the main correctness risk because the repo's
   teacher-forcing guarantees depend on it.
3. Structured latent-video tensors will touch data prep, trainer code, and
   tests at the same time, so a compatibility layer may be needed during the
   migration.

## References

1. https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
2. https://github.com/Wan-Video/Wan2.1/tree/main/wan
3. https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video.py
4. https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py
5. https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan_vace.py
