# Architecture

## Status

This document describes the current default architecture for the Wan VACE
world-model path as of March 3, 2026. Training and inference use the Wan
VACE-compatible path end to end. The legacy custom `WanDiTWrapper` has been
removed, along with its dedicated Wan T2V transformer download path.

## Overview

The world model predicts future visual observations entirely in the latent space
of a frozen Wan video VAE. Training keeps the repo's existing chunkwise
teacher-forcing structure as the outer loop, but replaces the inner denoiser
with a Wan VACE-compatible backbone vendored from upstream Wan code.

Canonical checkpoint format:

1. base transformer weights come from Hugging Face Diffusers
   `Wan-AI/Wan2.1-VACE-1.3B-diffusers`
2. repo `.pt` checkpoints are treated as optional local fine-tune overlays on
   top of that canonical pretrained backbone

### Parameter initialization and ownership

Current parameter sources in the default VACE path are:

1. Imported and frozen:
   - Wan VAE encoder weights loaded by `WanVAE.from_pretrained(...)`
   - Wan VAE decoder weights loaded by `WanVAE.from_pretrained(...)`
2. Imported and then fully fine-tuned:
   - all Wan VACE backbone parameters loaded by
     `WanVACETransformer3DModel.from_pretrained(...)` when
     `load_pretrained_backbone=true`
3. Locally initialized and then fully trained:
   - the action-token encoder in `wan_vace_conditioning.py`
4. Imported as local fine-tune overlays when a repo checkpoint is provided:
   - `model_state_dict` for the Wan VACE world-model module
   - `action_encoder_state_dict` for the action-token encoder

The current training code does not partially freeze the Wan VACE backbone, does
not use LoRA adapters, and does not isolate specific VACE layers. If the
pretrained backbone path is enabled, the imported VACE transformer weights are
optimized end to end. If `load_pretrained_backbone=false`, the backbone is
instantiated from config and still optimized end to end.

Core objective:

$$\pi_\theta(o_{t:t+H} \mid o_{t-l:t}, a_{t:t+H-1})$$

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
The vendored code includes the upstream model implementation and its direct
dependencies so the local project stays structurally aligned with Wan 2.1
VACE 1.3B while allowing local masking changes.

The adapter is responsible for:

1. converting repo-level tensors into Wan/VACE input shapes
2. building action tokens for cross-attention
3. building VACE control tensors from teacher-forced latents and masks
4. expanding chunkwise block-causal masks from latent frames to Wan patch tokens
5. returning a velocity prediction in latent-video layout

## Backbone mechanics from vendored Wan code

### Patchified video tokenization

The vendored Wan backbone is a true 3D video transformer rather than a
framewise transformer wrapped around latent timesteps:

1. inputs enter as latent videos with shape `[B, C, T, H, W]`
2. `patch_embedding` is a `Conv3d` with `kernel_size=stride=patch_size`
3. the patch grid is flattened to token shape `[B, N_patch, D]`
4. output tokens are projected back to `out_channels * prod(patch_size)` and
   unpatchified back to `[B, C, T, H, W]`

This matters for the local adapter because all masks, chunk ids, and control
features must align with Wan's post-patch token grid, not only with latent
frames.

### Positional and timestep modulation

Wan does not use a learned full 3D position table on the main video tokens.
Instead, the vendored implementation applies factorized rotary position
embeddings:

1. `WanRotaryPosEmbed` splits each attention-head dimension across time, height,
   and width axes
2. separate 1D RoPE frequencies are built for each axis and then concatenated
   per patch token
3. the resulting rotary embedding is applied inside self-attention to both Q and
   K

Time conditioning is also structurally important:

1. the scalar diffusion timestep is projected to `temb`
2. `temb` is expanded to a 6-way modulation vector per block
3. each transformer block uses that modulation for AdaLN-style
   shift/scale/gate control around self-attention and feed-forward sublayers
4. the model head uses a final 2-way shift/scale modulation before projection

### Transformer block structure

The main Wan backbone block is ordered as:

1. pre-norm self-attention with RoPE and residual gating
2. pre-norm cross-attention over `encoder_hidden_states`
3. pre-norm feed-forward with residual gating

Architecturally relevant details from the code:

1. self-attention is the only path that consumes the local `attention_mask`
2. cross-attention is unmasked in the vendored implementation
3. attention uses explicit Q/K RMS normalization before attention score
   computation
4. when image conditioning is enabled, image tokens are projected separately and
   prepended to the conditioning sequence, then consumed through an added K/V
   projection path

This is why the repo's chunkwise block-causal fork only needs to thread masks
through the self-attention path.

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

At the backbone level, the vendored conditioning embedder expects this sequence
to already be projected to Wan's hidden width. The embedder then:

1. applies the text-style token projection used by upstream Wan
2. optionally projects image tokens through a separate image embedder
3. concatenates projected image tokens ahead of the action/token sequence when
   image conditioning is present

Inference additionally supports an upstream-style verification path where the
cross-attention stream is populated by real Wan prompt embeddings from the
tokenizer and UMT5 text encoder. This path is inference-only and exists to
verify that the local Wan VACE integration behaves like upstream prompt-guided
sampling before the project fully commits to action-only conditioning.

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

The current local control builder follows the diffusers VACE split but uses the
repo's latent-space world-model semantics:

1. `control_hidden_states` are assembled from `[inactive; reactive; mask]`
   features, forming a 96-channel tensor (`16 + 16 + 64`).
2. future control latents are zero-filled.
3. future control masks are one-filled to mark them as unobserved.
4. these control tensors are processed by a parallel, specialized stack of
   `WanVACETransformerBlock` layers to generate "hints."
5. control injection happens at configured `vace_layers` (e.g., layers 0, 5, 10,
   etc.). At these specific depths in the main backbone, the corresponding hint
   is added directly (as a residual injection) to the main video representation.
   Past observations do not use cross-attention.

The vendored VACE implementation adds several concrete constraints:

1. `control_hidden_states` are patchified by a dedicated `vace_patch_embedding`
   conv, separate from the main video patch embedder
2. `vace_layers` must include layer `0` and cannot reference a layer index past
   `num_layers - 1`
3. there is one `WanVACETransformerBlock` per configured control injection
   point, not one per main backbone layer
4. the first VACE block applies an input projection before mixing control tokens
   with the main hidden states
5. every VACE block returns a projected control hint plus an updated internal
   control state
6. the model runs all VACE blocks first, stores their hints, then runs the main
   Wan blocks and injects the next hint whenever the block index matches
   `vace_layers`
7. each injected hint is scaled by a corresponding
   `control_hidden_states_scale` entry at runtime

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
2. initialize each active future chunk from Gaussian noise
3. build either action tokens or upstream-style prompt embeddings for the
   cross-attention stream
4. build VACE control tensors from observed context plus rollout masks
5. iteratively denoise one chunk at a time with the Wan flow-matching scheduler
6. optionally collapse the future window into one chunk for upstream-style
   smoke tests while preserving the same chunked code path
7. keep prompt tokenization and UMT5 prompt encoding on CPU, then move only the
   resulting embeddings to CUDA to keep the 1.3B VACE path within workstation
   memory limits
8. respect Hugging Face offline-cache env settings so cached Wan assets can be
   loaded without metadata network calls
9. run inference in reduced precision on CUDA when AMP is enabled
10. decode predicted future latents back to video

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

Vendor the Wan backbone implementation and its direct dependencies, including:

1. Wan VACE transformer blocks
2. attention modules
3. rotary embeddings
4. timestep and conditioning embeddings
5. normalization and feed-forward utilities

The goal is to stay structurally close to upstream Wan rather than reimplement a
parallel local DiT. The intentional local fork is the self-attention mask
threading needed for chunkwise teacher forcing.

### Local repo code

Local code should own:

1. action-token encoding
2. VACE control-tensor construction from latent videos and masks
3. chunkwise teacher-forcing schedule and loss
4. mask expansion from latent time to Wan patch tokens
5. training/inference entrypoints and config schema/validation

Configuration ownership:

1. canonical runtime presets live under `configs/`
2. the typed config contract defines the validated runtime schema
3. train and infer entrypoints load the canonical preset by default, then apply
   CLI overrides

## Trainable vs frozen components

Frozen:

1. Wan VAE encoder
2. Wan VAE decoder

Trainable:

1. all Wan VACE backbone parameters, whether initialized from pretrained
   diffusers weights or from local config
2. action-token encoder and its projection layers

Imported but not trainable:

1. pretrained Wan VAE parameters

Imported and then trainable:

1. pretrained Wan VACE backbone parameters from
   `Wan-AI/Wan2.1-VACE-1.3B-diffusers`
2. optional local repo checkpoint overlays for `model_state_dict` and
   `action_encoder_state_dict`

Excluded from the current runtime path:

1. proprio conditioning as part of the Wan-compatible path

## Runtime structure

Current runtime structure:

1. vendor the upstream Wan backbone and keep local changes minimal
2. keep a local world-model adapter layer between repo tensors and Wan/VACE
   inputs
3. keep local conditioning helpers responsible for action tokens and control
   tensors
4. preserve latent-video structure in batch preparation end to end
5. route training and inference through the Wan/VACE-compatible adapter

## References

Primary references for this architecture:

1. https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
2. https://github.com/Wan-Video/Wan2.1/tree/main/wan
3. https://github.com/modelscope/DiffSynth-Studio
4. https://github.com/huggingface/diffusers
