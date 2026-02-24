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

Structure:

1. Pool over time first (`mean`, `last`, or `flatten`) to get a single vector per batch item.
2. If `mlp_dim is None`: `nn.Sequential(LayerNorm(in_dim), Linear(in_dim, d), Dropout(p))`.
3. If `mlp_dim is set`: `nn.Sequential(LayerNorm(in_dim), Linear(in_dim, mlp_dim), GELU(), Dropout(p), Linear(mlp_dim, d), Dropout(p))`.

### Proprio encoder

Maps $q_t$ to an AdaLN conditioning embedding:

$$
q_t \mapsto c_q \in \mathbb R^{d}
$$

Structure:

1. If `mlp_dim is None`: `nn.Sequential(LayerNorm(Q), Linear(Q, d), Dropout(p))`.
2. If `mlp_dim is set`: `nn.Sequential(LayerNorm(Q), Linear(Q, mlp_dim), GELU(), Dropout(p), Linear(mlp_dim, d), Dropout(p))`.
3. If `enabled=False`, output zeros of shape `[B, d]` (no proprio signal).

## Backbone and conditioning injection

### Diffusion transformer backbone

Start from Wan2.1 1.3B DiT pretrained weights. Fine tune to output a velocity field for noisy future latent chunks.

The pretrained DiT already knows a lot about:
- spatial coherence: edges, textures, object like structure in latent space
- temporal coherence: smooth motion patterns, persistence of objects over time
- denoising dynamics: how to move from noisy latents back toward clean latents across timestep conditions

### DiT block architecture

Each transformer block follows the Peebles & Xie (2022) DiT design, using residual connections and AdaLN-Zero conditioning:

$$
\begin{aligned}
x_{attn} &\gets x + \alpha \cdot \text{Attn}(\text{AdaLN}(x \mid c)) \\
x_{out} &\gets x_{attn} + \beta \cdot \text{MLP}(\text{AdaLN}(x_{attn} \mid c))
\end{aligned}
$$

where:
- $c = \text{ActionProj}(c_a) + \text{TimestepProj}(t_{emb}) + \text{ProprioProj}(c_q)$
- $\alpha, \beta$ are per-block learnable scale parameters initialized to zero.
- AdaLN modulates LayerNorm using $(\text{scale}, \text{shift})$ vectors regressed from $c$.

### Timestep embedding

Timesteps $t \in [0, 1]$ are mapped to a sinusoidal embedding $t_{emb} \in \mathbb R^{256}$. The scalar $t$ is scaled by 1000 to match standard diffusion frequency ranges:

$$
\text{freq}_i = \exp\left( -\frac{i}{128} \ln(10000) \right)
$$
$$
t_{emb} = [\cos(1000t \cdot \text{freq}), \sin(1000t \cdot \text{freq})]
$$

## Temporal Chunking Logic ($K+1$)

To prevent error drift over long horizons, the future latent window is split into $K+1$ segments and predicted sequentially (autoregressively):

1. **The Anchor (Chunk 0)**: Reconstructs the current observation (frame $t$) and begins the rollout. This ensures the prediction is perfectly "stitched" to the ground-truth history.
2. **Sequential Prediction**: The model loops through every chunk ($0, 1, \dots, K$).
3. **Teacher Forcing (Training)**: To learn stable transitions, the model uses clean ground-truth for all chunks *before* the current target.
4. **Causal Masking**: The model is blocked from seeing any chunks *after* the current target to prevent temporal leakage.
5. **The $+1$ Requirement**: We enforce $k \ge 1$ so there are always at least 2 chunks, forcing the model to learn autoregressive "handoffs" between segments of time.

## Attention masking

## Attention mechanics

### Scaled Dot-Product Attention

The model uses standard multi-head attention over the context and noisy future tokens:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_h}} + \text{mask} \right) V
$$

where $Q, K, V$ are linear projections of the block input $h = \text{AdaLN}(x \mid c)$.

### Block-Causal Masking

The additive `mask` tensor $(0 / -\infty)$ ensures validity across the teacher-forced rollout:

1. The current noisy chunk may attend to:
   * **Teacher-forced history**: all past clean chunks ($k_{context} < k_{target}$).
   * **Self**: tokens within the current noisy chunk.
2. **Blocked Future**: Any chunk index $k > k_{target}$ is masked with $-\infty$.


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

### Optimization

The model is optimized using **AdamW**. It provides robust gradient updates and decouples weight decay, which is critical for stabilizing the training of large Diffusion Transformers (DiT).

Notes:

1. $w(t_k)$ is a timestep weighting function.
2. All variables are in latent space.
3. Actions and proprio are conditioning only.

## Inference

Open loop rollout:

1. Encode latest observed context frames into latents $z_{\text{past}}$.
2. Autoregressively generate future latent chunks from standard Gaussian noise, one chunk at a time, using Euler integration:
   * **Context**: Condition on $z_{\text{past}}$ and all previously generated future chunks.
   * **Steps**: Divide $t \in [0, 1]$ into $N$ steps of size $dt = 1/N$.
   * **Predict**: At step $i$, evaluate velocity $v = u_\theta(\text{noise}_i, \dots, t_{\text{mid}})$ where $t_{\text{mid}} = (i + 0.5) dt$.
   * **Update**: Remove noise via $z \gets z + dt \cdot v$.
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
