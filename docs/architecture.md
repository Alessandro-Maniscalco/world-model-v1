# Architecture

## Status

This document describes the current default architecture for the Wan VACE
world-model path as of March 1, 2026. Training and inference now use the Wan
VACE-compatible path by default. The legacy custom `WanDiTWrapper` still exists
only as a quarantined comparison path and is no longer the default model
namespace export.

## Overview

The world model predicts future visual observations entirely in the latent space
of a frozen Wan video VAE. Training keeps the repo's existing chunkwise
teacher-forcing structure as the outer loop, but replaces the inner denoiser
with a Wan VACE-compatible backbone vendored from upstream Wan code.

Canonical checkpoint format:

1. base transformer weights come from Hugging Face Diffusers
   `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, subfolder `transformer`
2. repo `.pt` checkpoints are treated as optional local fine-tune overlays on
   top of that canonical pretrained backbone

Core objective:

pi_theta(o_{t:t+H} | o_{t-l:t}, a_{t:t+H-1})

Key decisions:

1. Keep chunkwise teacher forcing and K+1 scheduling as the outer training
   structure.
2. Replace the local DiT wrapper with a vendored Wan VACE backbone plus a local
   world-model adapter.
3. Follow the diffusers Wan VACE conditioning split exactly. Conceptually, the
   Video Condition Unit becomes `V_wm = [A; F; M]`, where actions replace the
   text slot used in VACE. Concretely, preserve the diffusers execution split:
   - `encoder_hidden_states = action tokens`
   - `control_hidden_states = past latents + masks`
4. Keep latent videos structured as `[B, C, T, H, W]` through the model path.
   Do not flatten latents to one token per timestep for the new backbone path.
5. Add local self-attention mask threading to the vendored Wan/VACE stack so
   chunkwise teacher forcing remains enforceable.

## High-level components

### Frozen latent video codec

The Wan VAE remains frozen for both training and inference:

1. `encode(video) -> latents [B, C_lat, T_lat, H_lat, W_lat]`
2. `decode(latents) -> video`

Latent time is authoritative. All chunk splits, teacher forcing, and masks are
defined after VAE encoding.

### Outer chunkwise trainer

The trainer still owns:

1. latent-time splitting into context and future windows
2. K+1 chunk schedule construction
3. teacher forcing of earlier future chunks
4. timestep sampling
5. noise injection for the active target chunk
6. loss computation on only the supervised chunk
7. optimizer step, logging, and checkpointing

This logic stays local to the repo rather than being pushed into the vendored
Wan backbone.

### Inner Wan VACE-compatible denoiser

The inner denoiser is a vendored Wan VACE backbone wrapped by a local adapter.
The vendored code should include the upstream model implementation and its
dependencies from the Wan repository so the local project stays structurally
aligned with Wan 2.1 VACE 1.3B.

The adapter is responsible for:

1. converting repo-level tensors into Wan/VACE input shapes
2. building action tokens for cross-attention
3. building VACE control tensors from teacher-forced latents and masks
4. expanding chunkwise block-causal masks from latent frames to Wan patch tokens
5. returning a velocity prediction in latent-video layout

## Conditioning split

### Cross-attention path

Action conditioning enters the backbone through Wan's main cross-attention path.
Conceptually, this fills the text slot inside VACE's Video Condition Unit. The
action encoder no longer returns a single pooled AdaLN vector. Instead, it
returns an action-token sequence that is projected to the embedding width
expected by Wan VACE `encoder_hidden_states`.

Target contract:

1. queries Q come from noisy video tokens inside the Wan backbone
2. keys and values K,V come from action tokens passed as
   `encoder_hidden_states`

This directly follows the diffusers Wan VACE interface, except that text tokens
are replaced by action tokens. In other words, the conditioning unit changes
from `V = [T; F; M]` to `V_wm = [A; F; M]`, but the implementation still uses
VACE's original split between cross-attention tokens and control hints.

### VACE control path

Past observations stay in the VACE hint stream rather than being moved into the
main cross-attention stream. Together with action tokens, they form the
world-model Video Condition Unit `V_wm = [A; F; M]`. The control stream itself
is built from:

1. teacher-forced clean latent observations available at the current chunk step
2. binary masks marking which latent regions are observed versus generated

At chunk step `i`, the observed set contains:

1. all latent context frames
2. all earlier future chunks that are still teacher-forced at step `i`

The unobserved set contains:

1. the active noisy chunk
2. later future chunks that should remain blocked

The local control builder must preserve the semantics of diffusers
`pipeline_wan_vace.py` and `transformer_wan_vace.py`: control tensors are
prepared as VACE inputs, then injected at configured `vace_layers`.

## Data flow

### Training

1. Load a decoded video window and aligned actions.
2. Encode the full window with the frozen Wan VAE to
   `[B, C_lat, T_lat, H_lat, W_lat]`.
3. Split latent time into:
   - `z_past_video`
   - `z_future_video`
4. Build the K+1 chunk schedule over future latent timesteps.
5. For each chunk step:
   - sample timestep `t`
   - noise only the active future chunk
   - keep earlier future chunks clean under teacher forcing
   - build action tokens for `encoder_hidden_states`
   - build VACE control tensors from available clean latents plus masks
   - run the Wan VACE adapter forward
   - compute flow-matching loss on only the active chunk

### Inference

Inference stays open-loop:

1. encode observed context frames
2. initialize the active future chunk from noise
3. build action tokens
4. build VACE control tensors from observed context plus rollout masks
5. iteratively denoise one chunk at a time
6. decode predicted future latents back to video

Unlike training, no future teacher forcing is available during inference. Only
real context and previously generated predictions can populate the control
stream.

## Masking

The repo's chunkwise block-causal logic remains defined in latent time, but the
Wan backbone attends over patch tokens after 3D patch embedding. Therefore:

1. latent-frame chunk ids must be expanded to patch-token chunk ids
2. additive block-causal masks must be threaded through Wan self-attention
3. masked self-attention must be implemented in the local vendored fork

This is the main intentional fork from upstream Wan/VACE behavior. Without it,
the repo cannot preserve its current chunkwise teacher-forcing guarantees.

## Model ownership boundaries

### Vendored upstream code

Vendor the Wan backbone implementation and the files it depends on, including:

1. Wan VACE transformer blocks
2. attention modules
3. rotary embeddings
4. timestep and conditioning embeddings
5. normalization and feed-forward utilities

The goal is to stay structurally close to upstream Wan rather than reimplement a
parallel local DiT.

### Local repo code

Local code should own:

1. action-token encoding
2. VACE control-tensor construction from latent videos and masks
3. chunkwise teacher-forcing schedule and loss
4. mask expansion from latent time to Wan patch tokens
5. training/inference entrypoints and configs

## Trainable vs frozen components

Frozen:

1. Wan VAE encoder
2. Wan VAE decoder

Trainable:

1. vendored Wan VACE backbone parameters
2. action-token encoder and projection layers
3. any local adapter layers required to match Wan input contracts

Deferred for the first migration:

1. proprio conditioning as part of the Wan-compatible path

The first migration should keep proprio disabled or isolated until the Wan VACE
training path is stable.

## Implementation map

Expected module additions and migrations:

1. vendor Wan code under `src/world_model/vendor/wan/`
2. add a local adapter such as `src/world_model/models/wan_vace_world_model.py`
3. add local conditioning helpers such as
   `src/world_model/models/wan_vace_conditioning.py`
4. update batch preparation to preserve latent-video structure
5. update training code to call the new Wan/VACE adapter
6. replace wrapper-centric tests with adapter and trainer integration tests

## References

Primary references for this architecture:

1. https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
2. https://github.com/Wan-Video/Wan2.1/tree/main/wan
3. https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video.py
4. https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan_vace.py
5. https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan_vace.py
