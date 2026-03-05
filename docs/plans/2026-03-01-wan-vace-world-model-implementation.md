# Wan VACE World Model Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current local DiT wrapper with a Wan VACE-compatible inner model while preserving the repo's chunkwise teacher-forcing training structure.

**Architecture:** Vendor the upstream Wan/VACE backbone into `src/world_model/vendor/wan/`, add a local adapter that realizes the conceptual world-model conditioning unit `V_wm = [A; F; M]` using the same split VACE uses in diffusers (`encoder_hidden_states = action tokens`, `control_hidden_states = past latents + masks`), and patch the vendored self-attention path to accept the repo's block-causal mask expanded into Wan patch-token space. Keep the frozen Wan VAE and the current outer chunkwise flow-matching trainer, but migrate batch preparation and model/loss interfaces from flattened latent tokens to structured latent videos.

**Tech Stack:** Python, PyTorch, diffusers, vendored Wan VACE modules, pytest, YAML config.

---

### Task 1: Introduce structured latent-video training batches

**Files:**
- Modify: `src/world_model/data/schema.py`
- Modify: `src/world_model/data/pack.py`
- Modify: `src/world_model/data/prepare.py`
- Modify: `src/world_model/data/__init__.py`
- Test: `tests/test_data_prepare_wan_vace.py`

**Step 1: Write the failing test**

```python
def test_prepare_packed_batch_preserves_structured_latent_videos():
    prepared = prepare_packed_batch(...)
    assert prepared.z_past_video.ndim == 5
    assert prepared.z_future_video.ndim == 5
    assert prepared.a_plan.shape[1] == prepared.horizon_latent_steps
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_data_prepare_wan_vace.py -v`
Expected: FAIL because `PreparedPackedBatch` does not yet expose structured latent-video fields.

**Step 3: Write minimal implementation**

Add a new prepared-batch schema that stores:

```python
@dataclass(frozen=True)
class PreparedPackedBatch:
    z_past_video: torch.Tensor
    z_future_video: torch.Tensor
    a_plan: torch.Tensor
    q_last: torch.Tensor | None
    total_latent_steps: int
    context_latent_steps: int
    horizon_latent_steps: int
```

Update `prepare_packed_batch()` to split encoded latents along latent time while keeping `[B, C, T, H, W]`.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_data_prepare_wan_vace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/world_model/data/schema.py src/world_model/data/pack.py src/world_model/data/prepare.py src/world_model/data/__init__.py tests/test_data_prepare_wan_vace.py
git commit -m "refactor: preserve latent video structure in prepared batches"
```

### Task 2: Vendor the Wan/VACE backbone and expose a local import surface

**Files:**
- Create: `src/world_model/vendor/wan/__init__.py`
- Create: `src/world_model/vendor/wan/modules/...`
- Create: `src/world_model/vendor/wan/...`
- Modify: `src/world_model/models/__init__.py`
- Test: `tests/test_vendor_wan_vace_imports.py`

**Step 1: Write the failing test**

```python
def test_vendor_wan_vace_backbone_imports():
    from world_model.vendor.wan import WanVACETransformer3DModel
    assert WanVACETransformer3DModel is not None
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_vendor_wan_vace_imports.py -v`
Expected: FAIL because the vendored package does not exist yet.

**Step 3: Write minimal implementation**

Copy the upstream Wan code required for the VACE backbone and export the local model symbol:

```python
from world_model.vendor.wan.transformer_wan_vace import WanVACETransformer3DModel

__all__ = ["WanVACETransformer3DModel"]
```

Only vendor the files actually required by the Wan VACE model and its direct imports.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_vendor_wan_vace_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/world_model/vendor/wan src/world_model/models/__init__.py tests/test_vendor_wan_vace_imports.py
git commit -m "feat: vendor wan vace backbone"
```

### Task 3: Add Wan-compatible conditioning builders

**Files:**
- Create: `src/world_model/models/wan_vace_conditioning.py`
- Modify: `src/world_model/conditioning/action_encoder.py`
- Modify: `src/world_model/conditioning/__init__.py`
- Test: `tests/test_wan_vace_conditioning.py`

**Step 1: Write the failing test**

```python
def test_build_action_tokens_matches_wan_text_width():
    tokens = build_action_tokens(a_plan=torch.randn(2, 4, 7), hidden_dim=4096)
    assert tokens.shape == (2, 4, 4096)


def test_build_control_tensor_returns_vace_video_and_mask_channels():
    control = build_vace_control_tensor(
        z_observed=torch.randn(2, 16, 6, 8, 8),
        observed_mask=torch.ones(2, 1, 6, 8, 8),
    )
    assert control.ndim == 5
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_wan_vace_conditioning.py -v`
Expected: FAIL because the conditioning helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement a Wan-compatible conditioning helper:

```python
class ActionTokenEncoder(nn.Module):
    def forward(self, a_plan: torch.Tensor) -> torch.Tensor:
        return self.proj(a_plan)


def build_vace_control_tensor(observed_latents: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([observed_latents, observed_mask.expand_as(observed_latents)], dim=1)
```

Use the exact Wan VACE channel contract after confirming the vendored model configuration.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_wan_vace_conditioning.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/world_model/models/wan_vace_conditioning.py src/world_model/conditioning/action_encoder.py src/world_model/conditioning/__init__.py tests/test_wan_vace_conditioning.py
git commit -m "feat: add wan vace conditioning builders"
```

### Task 4: Build the local Wan VACE world-model adapter with mask threading

**Files:**
- Create: `src/world_model/models/wan_vace_world_model.py`
- Modify: `src/world_model/vendor/wan/...` (self-attention mask threading)
- Modify: `src/world_model/models/__init__.py`
- Test: `tests/test_wan_vace_world_model.py`

**Step 1: Write the failing test**

```python
def test_wan_vace_world_model_forwards_chunk_inputs():
    model = WanVACEWorldModel(...)
    out = model(
        noisy_future_video=torch.randn(2, 16, 4, 8, 8),
        observed_video=torch.randn(2, 16, 6, 8, 8),
        action_tokens=torch.randn(2, 4, 4096),
        timestep_t=torch.rand(2),
        block_causal_attention_mask=torch.zeros(2, 64, 64),
    )
    assert out.shape == (2, 16, 4, 8, 8)
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_wan_vace_world_model.py -v`
Expected: FAIL because the adapter and mask-threaded vendored backbone do not exist yet.

**Step 3: Write minimal implementation**

Create a local adapter that:

```python
class WanVACEWorldModel(nn.Module):
    def forward(...):
        control_hidden_states = build_vace_control_tensor(...)
        patch_mask = expand_block_causal_mask_to_patch_tokens(...)
        return self.backbone(
            hidden_states=noisy_future_video,
            timestep=timestep_t,
            encoder_hidden_states=action_tokens,
            control_hidden_states=control_hidden_states,
            attention_mask=patch_mask,
        )
```

Patch the vendored Wan/VACE attention path so self-attention accepts additive masks.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_wan_vace_world_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/world_model/models/wan_vace_world_model.py src/world_model/vendor/wan src/world_model/models/__init__.py tests/test_wan_vace_world_model.py
git commit -m "feat: add wan vace world model adapter"
```

### Task 5: Migrate chunkwise flow matching to structured latent videos

**Files:**
- Modify: `src/world_model/training/flow_matching.py`
- Modify: `src/world_model/training/chunkwise_training.py`
- Test: `tests/test_chunkwise_training_wan_vace.py`

**Step 1: Write the failing test**

```python
def test_chunkwise_teacher_forcing_loss_uses_only_active_chunk_for_supervision():
    loss = chunkwise_teacher_forcing_loss(
        model=fake_video_model,
        z_past_video=torch.randn(2, 16, 3, 8, 8),
        z_future_video=torch.randn(2, 16, 6, 8, 8),
        action_tokens=torch.randn(2, 6, 4096),
        k=1,
    )
    assert loss.ndim == 0
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_chunkwise_training_wan_vace.py -v`
Expected: FAIL because the loss path still expects flattened `[B, T, D]` tokens.

**Step 3: Write minimal implementation**

Update the flow-matching path to operate on structured video latents:

```python
clean_chunk = z_future_video[:, :, start:end]
noisy_chunk, target_chunk = make_noisy_and_target(clean_chunk, t)
```

Keep the outer chunk schedule and loss semantics unchanged while swapping the
model contract from token tensors to latent videos plus action-token sequences.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_chunkwise_training_wan_vace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/world_model/training/flow_matching.py src/world_model/training/chunkwise_training.py tests/test_chunkwise_training_wan_vace.py
git commit -m "refactor: use wan vace video latents in chunkwise training"
```

### Task 6: Switch the training entrypoint and config to the new model path

**Files:**
- Modify: `scripts/train/world_model.py`
- Modify: `src/world_model/config.py`
- Modify: `configs/train/world_model.yaml`
- Modify: `src/world_model/models/__init__.py`
- Test: `tests/test_train_world_model_wan_vace.py`

**Step 1: Write the failing test**

```python
def test_train_script_builds_wan_vace_world_model_from_config():
    cfg = load_train_config(...)
    model = build_model_from_config(cfg, prepared_batch)
    assert isinstance(model, WanVACEWorldModel)
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_train_world_model_wan_vace.py -v`
Expected: FAIL because the train script still constructs the legacy custom DiT wrapper.

**Step 3: Write minimal implementation**

Add config fields for Wan/VACE model ownership:

```python
wan_vace_model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
vace_layers: tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 35)
control_scale: float = 1.0
disable_proprio: bool = True
```

Update the training entrypoint to construct the new adapter and action-token
encoder.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_train_world_model_wan_vace.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/train/world_model.py src/world_model/config.py configs/train/world_model.yaml src/world_model/models/__init__.py tests/test_train_world_model_wan_vace.py
git commit -m "feat: switch training entrypoint to wan vace world model"
```

### Task 7: Reconcile docs and remove the legacy wrapper path

**Files:**
- Modify: `docs/architecture.md`
- Remove: the legacy custom latent DiT wrapper module
- Remove: the legacy wrapper-specific tests

**Step 1: Write the failing test**

```python
def test_legacy_custom_dit_wrapper_is_not_the_default_training_backbone():
    assert "legacy custom latent DiT wrapper" not in default_model_factory_name()
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_train_world_model_wan_vace.py -v`
Expected: FAIL until the legacy wrapper is removed from the runtime path.

**Step 3: Write minimal implementation**

Preferred outcome:

```python
assert build_model_from_config(...).__class__.__name__ == "WanVACEWorldModel"
```

Then delete the wrapper module and its dedicated tests/scripts.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_train_world_model_wan_vace.py tests/test_wan_vace_world_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/architecture.md src/world_model/models/__init__.py src/world_model/vendor/wan/__init__.py
git commit -m "chore: remove legacy wan dit wrapper"
```

### Final verification

Run:

```bash
source .venv/bin/activate && pytest tests/test_data_prepare_wan_vace.py tests/test_vendor_wan_vace_imports.py tests/test_wan_vace_conditioning.py tests/test_wan_vace_world_model.py tests/test_chunkwise_training_wan_vace.py tests/test_train_world_model_wan_vace.py -v
```

Expected:

1. All targeted tests PASS.
2. No import errors from the vendored Wan package.
3. No regressions in the chunkwise loss path.
