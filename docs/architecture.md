# Architecture

## Overview

The model predicts future visual observations by operating entirely in the latent space of a frozen pretrained video VAE. The backbone is a pretrained diffusion transformer that is fine tuned to predict a flow matching velocity field for future latent chunks, conditioned on past latent context plus an action plan (for $a_{t:t+H-1}$) and optional proprio. Action/proprio information conditions the network only through AdaLN Zero (and not via attention over action/proprio tokens).

Core objective:

$$
\pi_\theta\!\left(o_{t:t+H}\mid o_{t-\ell:t}, a_{t:t+H-1}, q_t\right)
$$

Key constraints:

1. LIBERO streams are sampled at 10 Hz.
2. Context length in frames: $\ell = 10$.
3. Prediction horizon in frames: $H = 8$.
4. Chunking: $K+1$, defined in latent time.
5. Latent time is authoritative: the VAE may change effective timestep count, so all splits, masks, and chunk logic happen after encoding.

## Data flow

### Inputs

1. RGB frames: $o_{t-\ell:t}$ (context) and $o_{t:t+H}$ (future target), sampled at 10 Hz. Note the overlap at frame $t$.
2. Action plan: $a_{t:t+H-1}$, aligned to the future target window.
3. Proprio state: $q_t$ optional, aligned to the last context step.

### Latent encoding

A frozen Wan2.1 video VAE encodes frames to latents:

$$
z = \mathrm{VAEEnc}(o),\quad \hat o = \mathrm{VAEDec}(z)
$$

All training and masking operate on $z$, not on $o$.

### Latent time split

After encoding the full window, split by latent indices:

1. $z_{\text{past}}$ from the latent timesteps corresponding to the context
2. $z_{\text{future}}$ from the latent timesteps corresponding to the horizon

This split must be computed after encoding, not assumed from frame counts.

## Conditioning modules

### Action encoder

Maps an action plan to an AdaLN conditioning embedding:

$$
a_{t:t+H-1} \mapsto c_a \in \mathbb R^{d}
$$

Design option:

1. Pool the action plan over time and project to hidden size (d).
2. If per-timestep action features are computed, they must be pooled before AdaLN so they are not part of the transformer token sequence.

### Proprio encoder

Maps (q_t) to an AdaLN conditioning embedding:

$$
q_t \mapsto c_q \in \mathbb R^{d}
$$

Ablation:

1. With proprio: include $\mathrm{Tok}_q$.
2. Without proprio: drop or zero $\mathrm{Tok}_q$ using a config flag.

## Backbone and conditioning injection

### Diffusion transformer backbone

Start from Wan2.1 1.3B DiT pretrained weights. Fine tune to output a velocity field for noisy future latent chunks.

### AdaLN Zero injection

Each transformer block receives conditioning through AdaLN Zero, modulating normalized activations using scale and shift derived from conditioning embeddings, with zero initialization so the block is initially close to identity.

Conditioning sources:

1. timestep embedding for $t$
2. action conditioning embedding $c_a$
3. optional proprio conditioning embedding $c_q$

## Temporal Chunking Logic ($K+1$)

To prevent error drift over long horizons, the future latent window is split into $K+1$ segments and predicted sequentially (autoregressively):

1. **The Anchor (Chunk 0)**: Reconstructs the current observation (frame $t$) and begins the rollout. This ensures the prediction is perfectly "stitched" to the ground-truth history.
2. **Sequential Prediction**: The model loops through every chunk ($0, 1, \dots, K$).
3. **Teacher Forcing (Training)**: To learn stable transitions, the model uses clean ground-truth for all chunks *before* the current target.
4. **Causal Masking**: The model is blocked from seeing any chunks *after* the current target to prevent temporal leakage.
5. **The $+1$ Requirement**: We enforce $k \ge 1$ so there are always at least 2 chunks, forcing the model to learn autoregressive "handoffs" between segments of time.

## Attention masking

Requirement:

1. The current noisy chunk may attend to:

   * all past clean chunks
2. The current noisy chunk must not attend to any future chunk tokens.

Notes:

1. Actions and proprio do not participate in attention. They condition the network only through AdaLN Zero (and therefore do not appear in the attention mask).

Implement masking as a block causal mask defined by chunk ids, producing an attention mask tensor for the transformer.

Leakage tests must pass by construction:

1. Modify future tokens while holding past and current fixed.
2. Verify outputs for past and current tokens do not change under the mask.

## Training objective

### Flow matching setup

For each chunk $k$:

1. Start from clean latent chunk $z^{(k)}_1$.
2. Sample timestep $t_k \in [0,1]$.
3. Construct noisy latent $z^{(k)}_{t_k}$ and target velocity $v_k$ using the chosen flow matching path.
4. Predict velocity $u_\theta$ with conditioning and teacher forced context.

### Loss

$$
\mathcal L(\theta)=
\mathbb E_{z,a,q,{t_k}}
\left[
\frac{1}{K}\sum_{k=1}^{K} w(t_k)
\left|
u_\theta\Big(
z^{(k)}_{t_k}; z^{(1:k-1)}_{1}, a_{t:t+H-1}, q_t, t_k
\Big) - v_k
\right|_2^2
\right]
$$

Notes:

1. $w(t_k)$ is a timestep weighting function.
2. All variables are in latent space.
3. Actions and proprio are conditioning only.

## Inference

Open loop rollout:

1. Encode latest observed context frames into latents $z_{\text{past}}$.
2. Autoregressively generate future latent chunks by denoising from noisy latents to clean latents, one chunk at a time.
3. Decode predicted future latents to frames using the frozen VAE decoder.

Optional guidance:
Classifier free guidance can be applied by dropping conditioning tokens in the unconditional branch and combining predictions with scale $s$.

## Trainable versus frozen components

Frozen:

1. Wan video VAE encoder
2. Wan video VAE decoder

Trainable:

1. Wan DiT weights, full fine tune or LoRA first
2. Action encoder
3. Proprio encoder
4. AdaLN Zero modulation parameters and any adapters required to match hidden size

## Implementation notes

1. All causal logic is implemented in latent time after encoding.
2. Start with one camera stream for stability, then add the second camera stream as an additional conditioning stream if needed.
3. Prefer latent caching to remove video decode bottlenecks.
4. Keep all toggles in config: with proprio, without proprio, chunk schedule, and mask type.
