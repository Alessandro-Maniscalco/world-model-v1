# Building an Action-Conditioned World Model with Wan VACE on a 16 GB GPU

_Status: March 2026_

I am a sophmore majoring i nMechanical Engineering & Finance at the University of Pennsylvania. This summer
I did an amazing internship at Built Robtoics working on a powerful mecancial attachment.
recently I developed a passion for the predictive power of ML and have been studying it independently as
I am taking the limits of credits and cannot take a class. I have read a lot, taking notes on:
https://docs.google.com/document/d/1Q-M7ZCJGVbRLWiulOSyfcfaDw1_juOFTBJZOAF928NU. I was playing around with the
opensource Pi models and when I saw the beauty in DreamZero I was inspired to do my own.

The current goal is narrower and more concrete: learn an action-conditioned
latent video model that predicts future observations for robot manipulation.
More precisely, the training target is:

$$
\pi_\theta\!\left(o_{t:t+H}\mid o_{t-\ell:t}, a_{t:t+H-1}\right)
$$

So the model predicts future visual observations conditioned on past visual
context and future actions. It does not currently predict actions, so the DreamZero-style joint action diffusion head is a future project.


This project is:

- a latent-space world model for LeRobot-style robot data
- currently centered on Wan VACE-compatible training and evaluation
- currently optimized most heavily on `lerobot/aloha_static_fork_pick_up`
- built to run meaningful experiments on a single 16 GB RTX 3080 laptop


## Current architecture

As of March 2026, the default path in this repo is a Wan VACE-compatible world
model:

- Frozen visual codec: Wan2.1 video VAE encoder and decoder
- Prediction backbone: Wan2.1 VACE 1.3B transformer
- Training policy for the practical full-dataset path: LoRA adapters plus small
  local conditioning modules
- Conditioning split:
  - actions enter as Wan cross-attention tokens
  - past observed latents and masks enter through the VACE control stream
- Objective: chunkwise teacher-forced flow matching
- Temporal structure: K+1 chunking defined in latent time, not raw frame time

That latent-time detail matters. The Wan VAE compresses time, so chunking and
causality have to be defined after encoding rather than directly on raw frames.

## Implementation direction

The current codebase uses a hybrid approach:

- local code for dataset preparation, action-plan building, chunk scheduling,
  masking, training, inference, and evaluation
- a vendored Wan VACE backbone so the project stays aligned with the upstream
  model while still allowing local world-model changes

This ended up being much more maintainable than treating the entire backbone as
a fully custom implementation which I did initially.

## Current experimental picture

Many checkpoints now produce plausible videos, but the main failure mode is
still very clear: the model often moves too late, or moves in a way that is
visually misaligned with the commanded action sequence.

The strongest repo-backed findings so far are:

- Longer context helped stability. The `context_len=21`, `horizon_len=8` branch
  was a meaningful improvement over shorter-memory baselines.
- Ordered full-plan action conditioning was implemented and tested, but it did
  not fix the late-motion problem.
- Multi-chunk K+1 training recovered some useful temporal coverage, but the
  best `k=2` and `k=3` branches still trade off motion timing against held-out
  plausibility.
- Matching teacher forcing more closely to rollout inference improved some
  timing metrics in the `h16/k2` branch, but it still did not solve the
  held-out misalignment problem.

So the current situation is not "the model does nothing." It is more specific
than that:

- the system can generate plausible-looking rollouts
- changing the conditioning and rollout setup measurably changes behavior
- but early commitment, trajectory timing, and held-out consistency are still
  not where they need to be

## Automation and iteration

I also built a shared-session Codex controller around the training loop. Following `docs/controller_prompt.md`, its
job is to:

- propose bounded experiments
- edit local code when a structural hypothesis is worth testing
- run training and evaluation commands
- track decisions in `docs/training_optimizer.md`
- keep detailed chronology in `runs/training_optimizer/experiment_ledger.md`

That controller has been useful for speeding up iteration, but the problem remains.

## What I think is still unresolved

The open questions I care about most are:

1. Am I tackling the problem in the right way?
2. How do I know what is the right things to do?
3. How do I find what is exactly not working in such a complex system?

## Bottom line

The honest summary is: this project has moved beyond basic plumbing and can now
produce partially plausible action-conditioned predictions, but it still does
not reliably learn the task-relevant dynamics I want.
