# Historical DiT Input-Free Single-Step Check Design

This note is retained for history only. The referenced legacy DiT diagnostic
script and its tests were removed on March 2, 2026 when the repo dropped the
custom latent DiT wrapper path.

## Goal
Create a focused diagnostic that isolates DiT one-step latent updates from all conditioning inputs, then compares decoded output sharpness to baseline VAE reconstruction.

## What Changed
- Added a legacy input-free DiT diagnostic script to run:
  1. Image load and VAE encode.
  2. DiT velocity prediction with empty past context and zero action conditioning.
  3. One Euler latent update.
  4. Decode and save visual artifacts plus metrics.
- Added matching helper-level and mocked-model orchestration tests.

These files were later deleted together with the legacy wrapper path.

## Major Design Decisions
- **Strict no-input mode**: `past_clean_chunks` uses shape `[B,0,D]`, `action_conditioning` is all zeros, and `proprio_conditioning` is `None`.
  - Why: this directly matches the selected requirement to test DiT behavior without context or conditioning.
- **Single latent timestep (`T=1`)**:
  - Why: one still image maps cleanly to one future token for controlled diagnostics.
- **Triptych output (`input | recon | updated`) plus absolute diff image**:
  - Why: distinguishes VAE reconstruction blur from additional blur introduced by the DiT update.
- **Blur metric = Laplacian variance**:
  - Why: simple, fast, deterministic sharpness proxy suitable for local debugging.

## Validation Strategy
- Unit tests verify:
  - input-free tensor construction shapes and zero-conditioning semantics,
  - exact Euler update behavior,
  - blur metric response to synthetic blur,
  - mocked-model integration for one-step orchestration.
- Existing docstring policy tests ensure script/functions satisfy repository documentation requirements.
