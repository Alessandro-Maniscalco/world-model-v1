# Architecture

## Status

This document describes the current default architecture for the Wan VACE
world-model path as of March 12, 2026. Training and inference use the Wan
VACE-compatible path end to end.

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
2. Imported and then trained according to the selected backbone policy:
   - all Wan VACE backbone parameters loaded by
     `WanVACETransformer3DModel.from_pretrained(...)` when
     `load_pretrained_backbone=true`
3. Locally initialized and then trained when applicable:
   - the action-token encoder in `wan_vace_conditioning.py`
   - or the null-conditioning encoder, which has no learned parameters
4. Imported as local fine-tune overlays when a repo checkpoint is provided:
   - `model_state_dict` for the Wan VACE world-model module
   - `action_encoder_state_dict` for the action-token encoder

The current training code supports four backbone policies:

1. `trainable_backbone=full`
   - optimize the full Wan VACE backbone
2. `trainable_backbone=vace`
   - optimize the VACE-side control path plus the output head
3. `trainable_backbone=head`
   - optimize only the VACE control patch embedder plus the output head
4. `trainable_backbone=lora`
   - attach PEFT LoRA adapters to the Wan backbone and keep the small VACE
     output-side modules trainable

The `head` mode is still the laptop-GPU smoke/overfit mode. The current
full-dataset ALOHA path uses `trainable_backbone=lora`.

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
defined after VAE encoding. Spatially, the frozen Wan VAE compresses by `8`.
Temporally, the rule is exact rather than a generic `T/4`:

1. the first raw frame stands alone
2. every later latent step consumes exactly `4` raw frames
3. valid full windows therefore require `total_frames = 4n + 1`
4. valid world-model splits require `context_len = 4m + 1` and
   `horizon_len = 4h`
5. for valid windows:
   - `total_latent_steps = 1 + (total_frames - 1) / 4`
   - `context_latent_steps = 1 + (context_len - 1) / 4`
   - `horizon_latent_steps = horizon_len / 4`


### Action window alignment

Action conditioning exactly matches the sampled video window. Because the Wan VAE compresses time by a factor of 4, a single latent step represents a 4-frame chunk. 

To prevent data loss (especially for high-frequency, continuous tasks like ALOHA), we do **not** use nearest-neighbor sampling. Instead, the action-plan builder flattens the 4 raw actions within each block into a single dense token:

1. `[B, 4*T_latent, A]` (frame-rate) -> flattened to $`a_{plan} \in \mathbb{R}^{B \times T_{hor}^{lat} \times (4A)}`$


Training inherits a spatial alignment constraint from this frozen codec and
the Wan backbone. The shared preprocessing path center-crops RGB frames so
pixel height and width are perfectly divisible by `8` before VAE encoding. The Wan
transformer then patchifies those latent videos with a spatial patch size of `(2, 2)`.
Combined, explicit training resolutions must use pixel dimensions divisible
by `16`. This keeps latent height and width even, which is an absolute requirement
for Wan's latent-space 3D patch grid.

Current practical resolution decision for ALOHA fork-pick-up:

1. the source camera stream is `640x480` (`4:3`)
2. training should preserve that `4:3` aspect ratio instead of stretching to
   `16:9`-like sizes such as `448x256`
3. the public Wan/VACE pipeline still works at several smaller `4:3` sizes
   (`512x384`, `384x288`, `320x240`, `256x192`, `192x144`). Check these sizes with sweep_base_dit_resolutions.py
4. `128x96` is below the stable floor and collapses into large color patches
5. the current recommended first training resolution is `320x240`
6. the next higher-quality fallback is `384x288`, with `512x384` reserved for
   later runs if memory allows

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

## Conditioning split

Before describing the backbone internals, it helps to define the two
conditioning streams that appear throughout the rest of this document:

1. the main cross-attention stream, passed as `encoder_hidden_states`
2. the VACE control stream, passed as `control_hidden_states`

### Cross-attention path

The repo now supports two cross-attention modes:

1. null conditioning for the current future-observation stage
2. action-token conditioning for a later stage

In action mode, conditioning enters the backbone through Wan's main
cross-attention path. Conceptually, this fills the text slot inside VACE's
Video Condition Unit. The action encoder no longer returns a single pooled
AdaLN vector. Instead, it returns an action-token sequence that is projected to
the embedding width expected by Wan VACE `encoder_hidden_states`.

The sequence length matches future latent time, not raw frame time. Each action
token corresponds to one future latent step. When the input actions come from a
frame-rate dataset window, the feature vector for that token is built from the
entire raw action chunk covered by the latent step rather than from one sampled
action.

### Current action-token encoder structure

The current repo does not use a DreamZero-style joint action diffusion head.
Instead, it builds action conditioning in two local stages before handing the
result to Wan cross-attention.

Stage 1: future action-plan construction in data preparation

1. the dataset loader requests an `action` window aligned with the sampled video
   window
2. `prepare_packed_batch(...)` converts that window into `a_plan`
3. when actions are sampled per frame, each future Wan latent step receives the
   full corresponding 4-frame action chunk
4. that chunk is flattened into one feature vector, so one latent step still
   maps to one action token, but that token can contain `4 * action_dim`
   scalar values

Stage 2: per-token projection into Wan text width

1. `ActionTokenEncoder` receives `a_plan` with shape `[B, T_future_latent, D_in]`
2. by default, it applies:
   - `LayerNorm(D_in)`
   - `Linear(D_in, D_wan_text)`
   - `Dropout`
3. the result has shape `[B, T_future_latent, D_wan_text]`
4. those projected tokens are then passed to Wan as `encoder_hidden_states`

An optional 2-layer MLP variant exists in the encoder implementation when
`mlp_dim` is set, but the current factory path uses the default shallow
projection. So the active action encoder is best described as:

1. exact future action chunking at the data level
2. flattening within each Wan 4-frame future block
3. independent per-block projection with `LayerNorm + Linear`
4. no temporal mixing between action blocks inside the local encoder

This is deliberate for the current world-model stage. It preserves short-range
control detail from datasets like ALOHA while keeping the Wan-compatible
conditioning interface simple.


Target contract:

1. queries Q come from noisy video tokens inside the Wan backbone
2. keys and values K,V come from action tokens passed as
   `encoder_hidden_states`

This directly follows the diffusers Wan VACE interface when action mode is
enabled, except that text tokens are replaced by action tokens. In other words,
the conditioning unit changes from `V = [T; F; M]` to `V_wm = [A; F; M]`, but
the implementation still uses VACE's original split between cross-attention
tokens and control hints. In the current null-conditioning stage, the
cross-attention tokens are explicit zeros with the same `[B, T, D]` contract.

At the backbone level, the vendored conditioning embedder expects this sequence
to already be projected to Wan's hidden width by the local action encoder. The
vendored `WanTimeTextImageEmbedding` module then applies a final built-in text-style
linear projection to the action tokens before they enter the cross-attention layers.

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

1. `control_hidden_states` are assembled from a combination of three feature blocks, forming a 96-channel tensor (`16 + 16 + 64`). The full tensor shape for the control stream is `[Batch, 96, Time_latent, Height_latent, Width_latent]`. The 16 channels simply represent the 3 RGB channels physically squashed by the VAE; the lengths of the context and generated windows are handled entirely by the `Time` dimension:
   - **Inactive (16 channels)**: This contains the actual clean, perfect latent frames from the past (the context frames and teacher-forced frames). For future unobserved frames, these latents are completely zero-filled.
   - **Reactive (16 channels)**: This is a secondary buffer that might hold extra data (like a rough sketch or an earlier guess) in other implementations, but is zero-filled in this repository.
   - **Mask (64 channels)**: Binary masks are usually 1 channel, but here the mask is duplicated 64 times so it becomes extremely "loud" mathematically. The mask is `0` for observed context and `1` for the future frames being generated.
2. these control tensors are processed by a parallel, specialized stack of
   `WanVACETransformerBlock` layers to generate "hints."
3. control injection happens at configured `vace_layers` (e.g., layers 0, 5, 10,
   etc.). At these specific depths in the main backbone, the corresponding hint
   is added directly (as a residual injection) to the main video representation.
   Past observations do not use cross-attention.

   *Note: The official `ali-vilab` VACE implementation injects hints every 2 layers by default (`[0, 2, 4...]`). This codebase inherits the `diffusers` configuration format which spaces injections every 5 layers (`[0, 5, 10...]`) to save VRAM and compute during inference. The difference between these injection schedules on temporal tracking should be explicitly tested.*

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

## Backbone mechanics from vendored Wan code

### Patchified video tokenization

The vendored Wan backbone is a true 3D video transformer keeping space and time together rather than a
framewise transformer wrapped around latent timesteps:

1. inputs enter as latent videos with shape `[B, C, T, H, W]`
2. `patch_embedding` is a `Conv3d` with `kernel_size=stride=patch_size`
3. the patch grid is flattened to token shape `[B, N_patch, D]`
4. output tokens are projected back to `out_channels * prod(patch_size)` and
   unpatchified back to `[B, C, T, H, W]`

This matters for the local adapter because all masks, chunk ids, and control
features must align with Wan's post-patch token grid, not only with latent
frames.

Using latent video $z \in \mathbb{R}^{B \times C \times T \times H \times W}$
and patch size $(p_t, p_h, p_w)$, the tokenization step is:

$$
z_{\text{tok}} = \operatorname{Flatten}\bigl(\operatorname{Conv3D}_{\text{patch}}(z)\bigr)
\in \mathbb{R}^{B \times N_{\text{patch}} \times D}
$$

with

$$
N_{\text{patch}} = \frac{T}{p_t}\frac{H}{p_h}\frac{W}{p_w}.
$$

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

Schematically, the timestep embedding produces:

$$
(\Delta_{\text{msa}}, \Gamma_{\text{msa}}, G_{\text{msa}},
\Delta_{\text{ff}}, \Gamma_{\text{ff}}, G_{\text{ff}}) = f_{\text{time}}(t),
$$

where each term has hidden width $D$. The normalized hidden state $h$ is
modulated as:

$$
\operatorname{AdaLN}(h; \Delta, \Gamma) = \operatorname{Norm}(h) \odot (1 + \Gamma) + \Delta.
$$

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

#### Mathematical view of one Wan block

Let $h^{(\ell)}$ be the hidden state entering block $\ell$, and let
$c$ denote the cross-attention conditioning tokens. In simplified form, the
main Wan block computes:

$$
\tilde{h}_{\text{msa}} =
\operatorname{AdaLN}\bigl(h^{(\ell)}; \Delta_{\text{msa}}, \Gamma_{\text{msa}}\bigr)
$$

$$
a_{\text{self}} =
\operatorname{SelfAttn}\bigl(\tilde{h}_{\text{msa}}; \text{RoPE}, M_{\text{self}}\bigr)
$$

$$
h' = h^{(\ell)} + G_{\text{msa}} \odot a_{\text{self}}
$$

$$
a_{\text{cross}} = \operatorname{CrossAttn}\bigl(\operatorname{Norm}(h'), c\bigr)
$$

$$
h'' = h' + a_{\text{cross}}
$$

$$
\tilde{h}_{\text{ff}} =
\operatorname{AdaLN}\bigl(h''; \Delta_{\text{ff}}, \Gamma_{\text{ff}}\bigr)
$$

$$
m = \operatorname{MLP}(\tilde{h}_{\text{ff}})
$$

$$
h^{(\ell+1)} = h'' + G_{\text{ff}} \odot m.
$$

This is the reason the document calls the backbone "DiT-like" rather than a
plain Transformer: each block is not only attention plus MLP, but
timestep-conditioned attention plus MLP with gated residual updates.

#### Mathematical view of VACE control injection

Let $u^{(k)}$ be the internal control-stream state and let
$\hat{c}^{(k)}$ be the projected control hint emitted by VACE block $k$.
If main block index $\ell$ is one of the configured `vace_layers`, the main
stream receives a residual hint injection:

$$
h^{(\ell+1)} \leftarrow h^{(\ell+1)} + \lambda_k \hat{c}^{(k)},
$$

where $\lambda_k$ is the runtime control scale from
`control_hidden_states_scale`.

### What one denoising step actually does

With the cross-attention stream and VACE control stream defined, we can now
describe one denoising step end to end.

In the smoke checks and prompt-style inference path, the progress bar often
shows `50` steps. Those are diffusion denoising steps, not video frames and not
training iterations.

At a high level, one denoising step does the following:

1. start from the current noisy latent video
2. run the Wan VACE transformer on that latent video plus its conditioning
3. predict a latent-space update direction
4. let the scheduler turn that prediction into a slightly less noisy latent
   video

The latent video stays in latent space for the whole denoising loop. The VAE is
not decoding and re-encoding on every step. Decode happens only after the full
sampling loop finishes.

Using $x_k$ for the current noisy latent sample at scheduler step $k$, the
high-level update is:

$$
\hat{v}_k = f_\theta(x_k, c_{\text{cross}}, c_{\text{vace}}, t_k),
$$

$$
x_{k+1} = \operatorname{SchedulerStep}(x_k, \hat{v}_k, t_k),
$$

where $f_\theta$ is the Wan VACE transformer and $\hat{v}_k$ is the
predicted latent-space flow or update direction used by the scheduler.

#### Step-by-step view inside one transformer pass

For one transformer evaluation inside one denoising step:

1. the latent video enters the model as `[B, C, T, H, W]`
2. a 3D patch embedder converts that latent video into a sequence of video
   tokens `[B, N_patch, D]`
3. the VACE control tensor goes through its own patch embedder and becomes a
   parallel control-token stream
4. the diffusion timestep is projected into a modulation embedding used to
   shift, scale, and gate the residual paths inside each block
5. rotary position embeddings are built for the patch-token grid over time,
   height, and width
6. the model runs the control stream through the smaller set of
   `WanVACETransformerBlock` layers to produce control hints
7. the main token stream runs through the full stack of `WanTransformerBlock`
   layers
8. at configured `vace_layers`, the corresponding control hint is added into
   the main hidden state
9. the final token sequence is normalized, projected back to latent channels,
   and unpatchified to `[B, C, T, H, W]`

That output is not yet a decoded video. It is the model's prediction in latent
space for the scheduler update.

#### What a single Wan block does

One main Wan transformer block is easiest to think of as:

1. pre-norm self-attention over video patch tokens
2. pre-norm cross-attention over the conditioning token sequence
3. pre-norm feed-forward MLP

Each sublayer is wrapped in a residual connection. The timestep embedding does
more than simply get added once at the input. It modulates the block through
AdaLN-style shift, scale, and gate terms, so the diffusion timestep can control
how strongly attention and the MLP update the hidden state at that noise level.

In more concrete terms, the block does:

1. normalize hidden states
2. apply timestep-dependent shift and scale
3. run self-attention and add the result back through a learned gate
4. normalize again and run cross-attention to action tokens or prompt tokens
5. normalize again, run the feed-forward network, and add that result back
   through another learned gate

So the intuition "the latents go through a DiT made of repeated attention and
MLP blocks" is correct. The important refinements are:

1. the latents are first patchified into video tokens
2. each denoising step runs many such blocks, not just one
3. the block is pre-norm and timestep-modulated rather than a plain
   post-norm Transformer block
4. VACE adds a second control path whose hints are injected at selected depths

#### Why one visible progress-bar step is expensive

One visible denoising step usually performs:

1. one conditional transformer pass using the real prompt or action
   conditioning
2. one unconditional transformer pass when classifier-free guidance is enabled
3. one scheduler update that combines those predictions into the next latent
   sample

This is why the wall-clock time per progress-bar step can be several seconds
even for a short generated clip. The expensive part is repeated full-video
latent denoising, not frame decoding.

When classifier-free guidance is enabled with guidance scale $w$, the
conditional and unconditional predictions are combined as:

$$
\hat{v}_{\text{cfg}} =
\hat{v}_{\text{uncond}} + w\bigl(\hat{v}_{\text{cond}} - \hat{v}_{\text{uncond}}\bigr).
$$

The scheduler then uses $\hat{v}_{\text{cfg}}$ rather than the raw
conditional prediction.

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

When a smoke-check script sets `num_inference_steps=50`, the `50` means:

1. construct a schedule of 50 noise levels
2. run 50 sequential latent denoising updates for the current chunk or full
   rollout
3. decode only after the 50th update has finished

This `50` also matches the current upstream VACE Wan inference default:
`ali-vilab/VACE` sets `args.sample_steps = 50` in
`vace/vace_wan_inference.py` when `--sample_steps` is omitted.

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
4. training/inference entrypoints and config schema/validation

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


## References

Primary references for this architecture:

1. https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B
2. https://github.com/Wan-Video/Wan2.1/tree/main/wan
3. https://github.com/ali-vilab/VACE
4. https://github.com/modelscope/DiffSynth-Studio
5. https://github.com/huggingface/diffusers
