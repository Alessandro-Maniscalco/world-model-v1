# Scope and Roadmap

## Project goal

Build a latent-space, action-conditioned world model for LIBERO that predicts
future visual observations using a Wan VACE-compatible diffusion transformer,
following the repo's chunkwise teacher-forcing flow-matching style without
predicting actions.

Target conditional distribution:

$$
\pi_\theta\!\left(o_{t:t+H}\mid o_{t-\ell:t}, a_{t:t+H-1}\right)
$$

Key decisions:

1. Dataset: LeRobot, starting with `lerobot/libero` at 10 Hz.
2. Visual representation: Wan2.1 pretrained video VAE, frozen.
3. Backbone: Wan2.1 VACE 1.3B transformer, fine tuned through a local adapter.
4. Conditioning: actions enter as Wan cross-attention tokens and past observed
   latents enter through the VACE control stream.
5. Temporal definition: all causality, chunking, masking, and objectives are defined in latent time, because the VAE may change the effective timestep count.
6. Proprio conditioning is out of scope for the current Wan VACE runtime path.

## Non goals

1. No action prediction head, no action decoding.
2. No language conditioning in v1.
3. No simulator closed loop policy evaluation in v1, except optional sanity rollouts for world model open loop quality.
4. No large scale multi dataset training until single dataset training is stable.

## Success criteria

### Functional criteria

1. Deterministic data and latent pipeline: same input produces same cached latents.
2. No future leakage: mask invariance tests pass for the real model masking.
3. Overfit test: the full world model can overfit a tiny subset and produce
   visibly improved decoded future frames relative to a baseline.

### Modeling criteria

1. Stable optimization: flow matching loss decreases smoothly without collapse or divergence.
2. Open loop rollout quality improves with training, measured in latent error and decoded frame quality.

## Execution phases

The detailed implementation tasks, scripts, and definitions of done live in `/docs/roadmap_checklist.md`. This section describes only the phase level plan.

### Phase 0: Tooling and data correctness

Goal: deterministic dataset loading, VAE encode decode, latent caching, and leakage tests.

### Phase 1: Latent time chunking and masking

Goal: define all splits, chunking (K+1), and causal masking strictly in latent time.

### Phase 2: Conditioning pathway

Goal: action-plan conditioning through Wan cross-attention tokens and observed
latent conditioning through the VACE control stream.

### Phase 3: Backbone integration

Goal: wrap and fine tune Wan VACE so it predicts the flow-matching velocity
field for noisy future latent chunks.

### Phase 4: Training objective and stability

Goal: flow matching with teacher forcing across chunks trains stably and overfits a tiny subset as a pipeline validation.

### Phase 5: Evaluation and ablations

Goal: open loop rollout evaluation for the Wan VACE world model.

1. Use mixed precision.
2. Use gradient checkpointing in the Wan VACE backbone when needed.
3. Use gradient accumulation to reach an effective batch size.
4. Prefer latent caching for faster iteration.
5. Start with one camera stream, then add the second camera if needed.

## Risk register

1. Latent time mismatch: VAE temporal compression may make naive frame based splits incorrect.
   Mitigation: define all splits and masks in latent time after encoding.
2. Video decode bottlenecks: video decoding can be CPU bound.
   Mitigation: latent caching and smaller clips during development.
3. Leakage through masking: incorrect masks can silently inflate metrics.
   Mitigation: invariance tests that perturb future chunks and verify unchanged
   outputs on masked positions.
