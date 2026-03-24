# Exact Steps for the Current Full-DiT Training Run

This document explains the current no-action full-backbone training path that
the investigation controller is using:

- Run family: `fullft_quality_probe_aloha_fork_pick_up_smoke_ep0_224x128_step100_runtime_dtype_adafactor`
- Conditioning: `conditioning_mode=none`
- Backbone policy: `trainable_backbone=full`
- Optimizer: `Adafactor`
- Memory levers: `bfloat16`, `gradient_checkpointing=true`, `batch_size=1`

I am intentionally not using action conditioning here.

## 1. The exact run this file is describing

The anchor run is the no-action quality probe from
`runs/training_optimizer/investigation_controller_state.json`.

The effective training command is:

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/train/world_model.py \
  --config configs/train/aloha_fork_pick_up_smoke.yaml \
  --output-dir runs/fullft_quality_probe_aloha_fork_pick_up_smoke_ep0_224x128_step100_runtime_dtype_adafactor \
  --repo-id lerobot/aloha_static_fork_pick_up \
  --episodes 0 \
  --video-key observation.images.cam_high \
  --frame-height 128 \
  --frame-width 224 \
  --context-len 9 \
  --horizon-len 8 \
  --dt 0.1 \
  --batch-size 1 \
  --k 1 \
  --chunk-schedule-mode k_chunks \
  --trainable-backbone full \
  --conditioning-mode none \
  --optimizer-name adafactor \
  --gradient-checkpointing \
  --max-steps 100 \
  --no-validation-enabled \
  --subset-size 1 \
  --overfit-one-batch \
  --checkpoint-every 50 \
  --checkpoint-early-every 25 \
  --checkpoint-early-until 100 \
  --log-every 10 \
  --seed 0
```

Important facts from the real run:

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 3080 Laptop GPU`
- Total VRAM: about `15.58 GiB`
- Runtime dtype: `torch.bfloat16`
- Latent window printed by training: `context=3 future=2 total=5`
- Trainable parameter count printed by training: `2,153,972,032`
- Step time from `metrics.jsonl`: about `1.43s` to `1.54s`
- Overfit loss: `0.986696 -> 0.625674`

## 2. What the model is actually trying to do

The model sees:

- `9` context frames
- `8` future frames
- no action conditioning

The Wan VAE changes frame-time into latent-time:

- total raw frames: `17 = 9 + 8`
- total latent steps: `5`
- context latent steps: `3`
- future latent steps: `2`

So this run is really training on:

- `z_past_video`: `[1, 16, 3, 16, 28]`
- `z_future_video`: `[1, 16, 2, 16, 28]`

Even with `conditioning_mode=none`, the training code still builds
`action_tokens`. They are just zeros from `NullConditioningEncoder`, so this is
not real action conditioning. It is only a zero placeholder because the Wan
backbone interface always expects cross-attention tokens.

## 3. Step by step, slowly

### Step 1: load the config

`scripts/train/world_model.py` loads `configs/train/aloha_fork_pick_up_smoke.yaml`
and then applies the CLI overrides above.

What matters most here:

- `frame_height=128`, `frame_width=224`
- `context_len=9`, `horizon_len=8`
- `k=1`
- `trainable_backbone=full`
- `conditioning_mode=none`
- `optimizer_name=adafactor`
- `gradient_checkpointing=true`
- `overfit_one_batch=true`

### Step 2: choose device and dtype

The script selects:

- device: CUDA
- runtime dtype: `bfloat16`


### Step 3: load the frozen Wan VAE

`WanVAE.from_pretrained(...)` loads the video VAE on the GPU in eval mode and
freezes it.

Measured VRAM right after VAE load:

- allocated: `0.236 GiB`
- reserved: `0.246 GiB`

### Step 4: build the first dataset batch

The LeRobot loader returns:

- raw video: `[1, 17, 3, 480, 640]`
- raw actions: `[1, 17, 14]`

Measured tensor sizes on GPU:

- raw video: `62,668,800` bytes, about `59.76 MiB`
- raw action: `952` bytes

Measured VRAM after moving the first batch onto CUDA:

- allocated: `0.295 GiB`
- reserved: `0.305 GiB`

### Step 5: resize and encode the batch with the VAE

`prepare_packed_batch(...)` does several things in order:

1. resize video to `128x224`
2. build a constant black video and encode it
3. build a constant gray video and encode it
4. encode the real video
5. split the latent video into past and future
6. flatten `4` raw action frames into each future latent-step action plan

The exact prepared tensors are:

| tensor | shape | bytes |
| --- | --- | ---: |
| `z_past_video` | `[1, 16, 3, 16, 28]` | `43,008` |
| `z_future_video` | `[1, 16, 2, 16, 28]` | `28,672` |
| `control_black_latents` | `[1, 16, 5, 16, 28]` | `71,680` |
| `control_gray_latents` | `[1, 16, 5, 16, 28]` | `71,680` |
| `a_plan` | `[1, 2, 56]` | `448` |

Why is `a_plan` shape `[1, 2, 56]`?

- the dataset action has `14` motor values per frame
- each future latent step covers `4` raw frames
- so each latent-step action token holds `4 * 14 = 56` values

Measured VRAM during batch preparation:

- allocated after prep finishes: `0.295 GiB`
- peak allocated during prep: `0.539 GiB`

That temporary peak happens because the VAE is still on the GPU and is used
three times: black control encode, gray control encode, and real video encode.

### Step 6: offload the VAE

Because `overfit_one_batch=true`, the script caches the prepared latent batch,
moves the VAE back to CPU, and calls `torch.cuda.empty_cache()`.

Measured VRAM after VAE offload:

- allocated: `0.059 GiB`
- reserved: `0.080 GiB`

This is one of the main reasons the run fits.

### Step 7: build the full trainable Wan VACE world model

The training script then:

1. loads the pretrained Wan VACE backbone
2. wraps it in `WanVACEWorldModel`
3. builds `NullConditioningEncoder`
4. moves train modules to CUDA in bf16
5. marks all backbone parameters trainable because `trainable_backbone=full`

Measured persistent model state right after this:

- model trainable params: `2,153,972,032`
- action encoder trainable params: `0`
- model trainable memory: `4.012 GiB`

Measured VRAM:

- allocated: `4.071 GiB`
- reserved: `4.088 GiB`

This is the first big memory jump. At this point, weights dominate memory.

### Step 8: build Adafactor

`Adafactor` is created, but before the first optimizer step it has no state
tensors yet.

Measured VRAM after optimizer construction:

- allocated: `4.071 GiB`
- optimizer state: `0.0 GiB`

This is why Adafactor helps. In this branch, AdamW previously failed at
`optimizer.step()` because its optimizer state was much larger.

### Step 9: compute the no-grad baseline loss

Because this is an overfit-one-batch run, the script first evaluates one
forward loss without gradients.

Measured values:

- baseline loss: `0.986696`
- peak allocated during this baseline forward: `4.131 GiB`

This step is useful because it tells us the pure forward path fits even before
we allocate gradients.

### Step 10: start the real train step

The script calls `optimizer.zero_grad(set_to_none=True)` and enters autocast.

Then it builds `action_tokens` from `NullConditioningEncoder`.

Exact action-token tensor:

- shape: `[1, 2, 4096]`
- bytes: `32,768`

Again, this is not action conditioning. These are zero tokens.

### Step 11: build the chunk schedule

This run uses:

- `future latent steps = 2`
- `k = 1`

So the schedule is only one future chunk:

- chunk `0`: latent steps `[0:2]`

That means the trainer does exactly one Wan forward pass per optimization step.
There is no multi-chunk teacher-forcing loop in this run.

### Step 12: sample a timestep and make the noisy target

For the one active chunk, the trainer:

1. samples `t` in `[0, 1]`
2. takes the clean future latents
3. mixes them with Gaussian noise
4. builds the flow-matching velocity target

Key tensors:

- `noisy_chunk`: `[1, 16, 2, 16, 28]`
- `target_chunk`: `[1, 16, 2, 16, 28]`

Each of those is only `28,672` bytes. The latent tensors are tiny compared to
the model weights.

### Step 13: build the observed prefix and masks

Because `k=1` and the active chunk starts at `0`, there is no earlier future
prefix to teacher-force in.

So:

- `observed_video = z_past_video`
- `observed_video` shape: `[1, 16, 3, 16, 28]`
- `observed_mask` shape: `[1, 1, 3, 16, 28]`

The chunk id sequence over the full latent timeline is:

- `[-1, -1, -1, 0, 0]`

That produces a `5 x 5` latent-frame block-causal mask before Wan expands it to
patch-token space.

At this resolution, each latent frame has:

- latent height `16`
- latent width `28`
- Wan patch size `(1, 2, 2)`
- patches per frame: `(16 / 2) * (28 / 2) = 112`

So the `5` latent frames become `560` patch tokens inside Wan, and the frame
mask becomes a `560 x 560` patch-token mask during the forward pass.

### Step 14: build the VACE control tensor

The control path uses:

- observed past latents for the context
- gray fill latents for the future slots
- a mask channel expansion with `mask_channels=64`

The control tensor shape is:

- `[1, 96, 5, 16, 28]`

Why `96` channels?

- inactive branch: `16`
- reactive branch: `16`
- mask features: `64`
- total: `16 + 16 + 64 = 96`

Measured control tensor size:

- `430,080` bytes, about `0.41 MiB`

### Step 15: run the Wan forward pass

The model forward receives:

- `noisy_future_video`
- `observed_video`
- zero `action_tokens`
- `control_hidden_states`
- block-causal attention mask

The backbone predicts:

- `pred_suffix`: `[1, 16, 2, 16, 28]`

Then the training code computes squared error against `target_chunk`.

The expanded patch-token attention mask is still small compared with weights and
gradients:

- `560 x 560 x 2` bytes in bf16
- about `0.60 MiB`

Measured VRAM after forward:

- allocated: `4.161 GiB`
- peak allocated: `4.180 GiB`

So the forward activations only add a modest amount on top of the `4.01 GiB`
weight footprint. Gradient checkpointing is helping here.

### Step 16: backward pass

This is the biggest jump.

When `loss.backward()` runs, the model allocates gradients for all
`2,153,972,032` trainable parameters.

Measured VRAM after backward:

- allocated: `8.100 GiB`
- peak allocated: `8.112 GiB`
- gradient memory: `4.012 GiB`

That matches the simple rule:

```text
bf16 gradient bytes ~= bf16 trainable weight bytes
```

This one fact explains most of the training memory.

### Step 17: clip gradients and run Adafactor

The script clips gradients and then runs `optimizer.step()`.

Measured VRAM after optimizer step:

- allocated: `8.117 GiB`
- peak allocated: `8.322 GiB`
- optimizer state: `0.0163 GiB`, about `16.7 MiB`

This is the final training peak we measured for this path.

### Step 18: save the checkpoint

The script writes:

- `runs/.../checkpoints/step_0000001.pt`

Measured VRAM stays basically unchanged:

- allocated: `8.117 GiB`
- peak allocated: `8.322 GiB`

Checkpointing is mostly a CPU and disk task here, not a new GPU spike.

## 4. Memory by task

### Persistent memory

| task | measured size |
| --- | ---: |
| full bf16 trainable model weights | `4.012 GiB` |
| full bf16 gradients after backward | `4.012 GiB` |
| Adafactor state after first step | `0.016 GiB` |

### Small batch tensors

| task | shape | size |
| --- | --- | ---: |
| raw video on GPU | `[1, 17, 3, 480, 640]` | `59.76 MiB` |
| raw actions | `[1, 17, 14]` | `952 B` |
| `z_past_video` | `[1, 16, 3, 16, 28]` | `43,008 B` |
| `z_future_video` | `[1, 16, 2, 16, 28]` | `28,672 B` |
| `control_black_latents` | `[1, 16, 5, 16, 28]` | `71,680 B` |
| `control_gray_latents` | `[1, 16, 5, 16, 28]` | `71,680 B` |
| `a_plan` | `[1, 2, 56]` | `448 B` |
| zero `action_tokens` | `[1, 2, 4096]` | `32,768 B` |
| `control_hidden_states` | `[1, 96, 5, 16, 28]` | `430,080 B` |

### Stage-level VRAM

| stage | allocated GiB | peak allocated GiB | reserved GiB |
| --- | ---: | ---: | ---: |
| device ready | `0.000` | `0.000` | `0.000` |
| VAE loaded | `0.236` | `0.236` | `0.246` |
| first batch on GPU | `0.295` | `0.295` | `0.305` |
| batch prepared | `0.295` | `0.539` | `0.592` |
| VAE offloaded | `0.059` | `0.539` | `0.080` |
| train modules ready | `4.071` | `4.071` | `4.088` |
| baseline loss done | `4.079` | `4.131` | `4.150` |
| chunk inputs ready | `4.080` | `4.131` | `4.150` |
| forward done | `4.161` | `4.180` | `4.191` |
| backward done | `8.100` | `8.112` | `8.260` |
| optimizer step done | `8.117` | `8.322` | `8.484` |
| checkpoint saved | `8.117` | `8.322` | `8.484` |

## 5. What actually makes this fit on a 15.58 GiB GPU

This run fits because several choices all point in the same direction:

1. The VAE is only needed for batch preparation, then it is moved back to CPU.
2. The trainable backbone lives in bf16 instead of fp32.
3. `gradient_checkpointing=true` keeps the forward activation overhead small.
4. `Adafactor` keeps optimizer state tiny compared with AdamW.
5. `batch_size=1` keeps the raw video and latent tensors small.
6. `224x128` is a low enough resolution that the latent grid is only `16x28`.
7. `k=1` means one future chunk and one Wan forward pass per train step.

The real measured peak in this setup is about `8.32 GiB`, which leaves a useful
margin under `15.58 GiB`.

## 6. The most important takeaway

For this exact no-action full-backbone run, memory is not dominated by the
latent video tensors. Those are tiny.

Memory is dominated by three things:

1. bf16 trainable weights: about `4.01 GiB`
2. bf16 gradients after backward: another `4.01 GiB`
3. everything else: comparatively small

So the training step is basically:

```text
VAE prep peak (~0.54 GiB) -> model weights (~4.07 GiB) ->
forward (~4.18 GiB) -> backward (~8.11 GiB) ->
optimizer step peak (~8.32 GiB)
```

That is the cleanest mental model for this current no-action training path.
