# Scope and Roadmap

## Project goal

Build a latent space, action conditioned world model for LIBERO that predicts future visual observations using a diffusion transformer, following the overall training style of DreamZero but without predicting actions.

Target conditional distribution:

$$
\pi_\theta\!\left(o_{t:t+H}\mid o_{t-\ell:t}, a_{t:t+H-1}, q_t\right)
$$

Key decisions:

1. Dataset: LeRobot, starting with `lerobot/libero` at 10 Hz.
2. Visual representation: Wan2.1 pretrained video VAE, frozen.
3. Backbone: Wan2.1 1.3B diffusion transformer, fine tuned.
4. Conditioning: actions and optional proprio injected into every block via AdaLN Zero.
5. Temporal definition: all causality, chunking, masking, and objectives are defined in latent time, because the VAE may change the effective timestep count.

## Non goals

1. No action prediction head, no action decoding.
2. No language conditioning in v1.
3. No simulator closed loop policy evaluation in v1, except optional sanity rollouts for world model open loop quality.
4. No large scale multi dataset training until single dataset training is stable.

## Success criteria

### Functional criteria

1. Deterministic data and latent pipeline: same input produces same cached latents.
2. No future leakage: mask invariance tests pass for the real model masking.
3. Overfit test: the full world model can overfit a tiny subset and produce visibly improved decoded future frames relative to a baseline.

### Modeling criteria

1. Stable optimization: flow matching loss decreases smoothly without collapse or divergence.
2. Open loop rollout quality improves with training, measured in latent error and decoded frame quality.
3. Ablation: performance comparison for conditioning with proprio versus without proprio.

## Execution phases

The detailed implementation tasks, scripts, and definitions of done live in `/docs/roadmap_checklist.md`. This section describes only the phase level plan.

### Phase 0: Tooling and data correctness

Goal: deterministic dataset loading, VAE encode decode, latent caching, and leakage tests.

### Phase 1: Latent time chunking and masking

Goal: define all splits, chunking (K+1), and causal masking strictly in latent time.

### Phase 2: Conditioning pathway

Goal: action plan conditioning (future actions over the prediction horizon) and optional proprio conditioning injected into every block via AdaLN Zero, fully configurable.

### Phase 3: Backbone integration

Goal: wrap and fine tune Wan DiT so it predicts the flow matching velocity field for noisy future latent chunks.

### Phase 4: Training objective and stability

Goal: flow matching with teacher forcing across chunks trains stably and overfits a tiny subset as a pipeline validation.

### Phase 5: Evaluation and ablations

Goal: open loop rollout evaluation and a clear proprio versus no proprio comparison.

1. Use mixed precision.
2. Use gradient checkpointing in the DiT when needed.
3. Use gradient accumulation to reach an effective batch size.
4. Prefer latent caching for faster iteration.
5. Start with one camera stream, then add the second camera if needed.

## Risk register

1. Latent time mismatch: VAE temporal compression may make naive frame based splits incorrect.
   Mitigation: define all splits and masks in latent time after encoding.
2. Video decode bottlenecks: AV1 decoding can be CPU bound.
   Mitigation: latent caching and smaller clips during development.
3. Leakage through masking: incorrect masks can silently inflate metrics.
   Mitigation: invariance tests that perturb future tokens and verify unchanged outputs on past and current tokens.
