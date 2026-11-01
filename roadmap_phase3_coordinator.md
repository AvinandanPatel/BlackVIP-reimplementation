# Phase 3: Coordinator, Step-by-Step Build Guide

This guide expands Phase 3 of [roadmap.md](roadmap.md). It lists the steps only; you write the code. Each step says **what to build**, **the exact details** (shapes, channels, defaults), and **how to check it** before you move on.

Paper sections: §4.1, Supp. A.4, Supp. B. Reference code for cross-checking: [../BlackVIP/trainers/visual_prompters.py](../BlackVIP/trainers/visual_prompters.py) (`Coordinator`, `DecoderManual`, `EncoderManual`) and [../BlackVIP/trainers/utils.py](../BlackVIP/trainers/utils.py) (`clip_clipping`).

---

## What you are building

```
image x (N,3,224,224), CLIP-normalized
   │
   ├──► Encoder f (frozen, SSL ViT-MAE-B/16) ──► z (N,768)
   │                                              │
   │                       prompt trigger ϕt (1,800), learnable, shared
   │                                              │
   │                       concat ─► (N,1568) ─► reshape ─► (N,32,7,7)
   │                                              │
   │                       Decoder gϕd (learnable, ~9.1K params)
   │                       7 → 14 → 28 → 56 → 112 → 224
   │                                              │
   │                                    prompt h(x) (N,3,224,224)
   ▼                                              │
x̃ = clip_range( x + ε · h(x) )  ◄────────────────┘
```

**Learnable parameters ϕ = {ϕd, ϕt}, about 9.9K in total.** The zeroth-order optimizer will see them only as one flat vector.

---

## Target file layout

```
BlackVIP-reimplementation/
  blackvip/
    __init__.py
    constants.py              # Step 1
    config.py                 # Step 2
    prompting.py              # Step 3: clip_range, apply_prompt
    coordinator/
      __init__.py             # Step 11: build_coordinator, registry exports
      encoders.py             # Steps 4–6
      trigger.py              # Step 7
      decoder.py              # Step 8
      coordinator.py          # Steps 9–10
  tests/
    test_prompting.py
    test_encoders.py
    test_trigger.py
    test_decoder.py
    test_coordinator.py
  scripts/
    inspect_coordinator.py    # Step 13
```

Build the steps in order. Every step has a test, and you should not move on until it passes.

---

## Step 1: Constants (`blackvip/constants.py`)

- [ ] `CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)`
- [ ] `CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)`
- [ ] Derived per-channel bounds in normalized space, computed from the mean and std (don't hard-code them):
  - `CLIP_PIXEL_MIN[c] = (0 − mean[c]) / std[c]`
  - `CLIP_PIXEL_MAX[c] = (1 − mean[c]) / std[c]`
- [ ] Decoder spatial constants: `SEED_HW = 7`, `SEED_AREA = 49`, `IMAGE_SIZE = 224`
- [ ] Hugging Face checkpoint IDs:
  - `"vit-mae-base"` → `facebook/vit-mae-base`
  - `"vit-base"` → `google/vit-base-patch16-224-in21k`
  - `"dino-resnet-50"` → `Ramos-Ramos/dino-resnet-50`

**Check:** the bounds equal the reference values R ∈ [−1.79226253, 1.93033625], G ∈ [−1.75209713, 2.07488384], B ∈ [−1.48021977, 2.14589699] to about 1e-6.

---

## Step 2: Config (`blackvip/config.py`)

Use one dataclass, `CoordinatorConfig`, so that every later piece is built from it:

| field | default | meaning |
|---|---|---|
| `encoder` | `"vit-mae-base"` | one of `vit-mae-base`, `vit-base`, `dino-resnet-50`, `scratch` |
| `src_dim` | `1568` | total length after concatenating z and the trigger; must be divisible by 49 |
| `scratch_out_dim` | `0` | only used for the `scratch` encoder (64/128/256) |
| `trigger_init` | `(0.0, 0.1)` | uniform init range for ϕt |
| `act` | `"gelu"` | activation in the decoder |
| `p_eps` | `1.0` | prompt intensity ε (per dataset: 0.2–1.0) |
| `bn_mode` | `"train"` | how decoder BN behaves during ZOO queries; see Step 10 |
| `device` | `"cuda"` | |

- [ ] Add `validate()`, which raises clear errors when:
  - `src_dim % 49 != 0`
  - `src_dim <= z_dim` (no room for the trigger), for example `dino-resnet-50` with z_dim 2048 needs `src_dim` ≥ 2097, a multiple of 49
  - an unknown encoder name is given
- [ ] Add `from_dict` / `to_dict` so the config can be saved in checkpoints

**Check:** the default config validates; `src_dim=1500` raises; `encoder="dino-resnet-50", src_dim=1568` raises.

---

## Step 3: Prompt composition utilities (`blackvip/prompting.py`)

These live **outside** the Coordinator, because the trainer composes the prompt and the black-box CLIP only ever sees x̃.

- [ ] `clip_range(x: Tensor) -> Tensor`
  - Accepts `(N,3,H,W)` or `(3,H,W)`
  - Clamps each channel to `[CLIP_PIXEL_MIN[c], CLIP_PIXEL_MAX[c]]`
  - Implement it with broadcasting min/max tensors of shape `(1,3,1,1)` instead of per-channel `cat`. The result is the same and it is simpler.
  - Bounds must follow `x`'s device and dtype (fp16 under autocast)
- [ ] `apply_prompt(x, prompt, eps) -> Tensor` returns `clip_range(x + eps * prompt)`
- [ ] Assert `x.shape == prompt.shape`

**Tests (`test_prompting.py`):**
- [ ] A random tensor scaled by 10 is clamped inside the bounds per channel
- [ ] A tensor already inside the bounds comes back unchanged
- [ ] With `eps=0`, the output equals `clip_range(x)`
- [ ] A normalized all-black image maps exactly to `CLIP_PIXEL_MIN`, and a normalized all-white image to `CLIP_PIXEL_MAX`
- [ ] Works for 3-D and 4-D inputs, and for fp16 inputs

---

## Step 4: Encoder interface (`blackvip/coordinator/encoders.py`)

Define the contract before writing any concrete encoder.

- [ ] Abstract base class `FrozenEncoder(nn.Module)` with:
  - attribute `z_dim: int`
  - attribute `trainable: bool` (False for pre-trained encoders, True for scratch)
  - `forward(x: (N,3,224,224)) -> z: (N, z_dim)`
- [ ] For frozen encoders:
  - Call `requires_grad_(False)` on all parameters in `__init__`
  - Wrap `forward` in `torch.no_grad()`
  - Override `train(mode)` so the module **always stays in eval mode**. This keeps dropout off and the BN/LN statistics fixed, even when the parent calls `.train()`.
- [ ] Use a registry dict, `ENCODERS = {"vit-mae-base": ..., ...}`, plus `build_encoder(cfg)`

**Check:** calling `.train()` on a frozen encoder leaves `encoder.training == False` for every submodule.

---

## Step 5: MAE encoder, the default (`MAEEncoder`)

This is the most important part to get right, because Hugging Face's MAE has some pitfalls.

- [ ] Loading: there are two options. Pick one and write down which.
  - **A (matches the reference exactly):** `ViTForImageClassification.from_pretrained("facebook/vit-mae-base")`. This loads the MAE encoder weights into a plain ViT, which applies **no random masking**. Transformers will warn about a model-type mismatch and about newly initialized weights. Confirm that the newly initialized weights are only `classifier.*`.
  - **B (cleaner):** `ViTMAEModel.from_pretrained(..., mask_ratio=0.0)`. Note that ViTMAE still **shuffles patches randomly** even when the mask ratio is 0. The [CLS] output should be nearly unchanged, since attention doesn't care about token order, but check this against option A.
- [ ] Feature extraction (`z_dim = 768`):
  - Reference: `out = model(x, output_hidden_states=True)`, then `z = out.hidden_states[-1][:, 0, :]`
  - ⚠️ `hidden_states[-1]` is taken **before** the final LayerNorm. `last_hidden_state` is taken **after** it. The reference uses the pre-LN [CLS] for MAE, but uses the post-LN `last_hidden_state[:, 0]` for the supervised ViT. Copy this exactly.
- [ ] Input normalization: the reference passes the **CLIP-normalized** image straight in, without converting to ImageNet mean and std. Copy this, and put the alternative behind a flag such as `renormalize_input: bool = False`. If the flag is on: denormalize with the CLIP statistics, then normalize with the ImageNet ones.
- [ ] Input resolution: 224×224, so no position-embedding interpolation is needed. Assert the input size.
- [ ] dtype: the encoder runs in fp32 (autocast may downcast); return `z` as fp32 or as the input dtype, and stay consistent.

**Tests (`test_encoders.py`, marked slow/network):**
- [ ] Output shape is `(N, 768)` for N ∈ {1, 4}
- [ ] Deterministic: the same input twice gives an identical z
- [ ] No parameter has `requires_grad`, and `z.requires_grad == False`
- [ ] If you implemented both A and B, their [CLS] outputs have cosine similarity above 0.999
- [ ] The loading warning lists only `classifier.*` as newly initialized

---

## Step 6: Ablation encoders (stubs first, finish later)

Wire these into the registry now, even if only as stubs, so the interface is proven to generalize. They are needed for Tables 5 and 6.

- [ ] `SupervisedViTEncoder`: `ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")`, `z = last_hidden_state[:, 0]`, z_dim 768
- [ ] `DinoResNetEncoder`: `ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")`, `z = pooler_output.reshape(N, 2048)`, z_dim 2048
- [ ] `ScratchEncoder` (trainable, `trainable=True`, no `no_grad`, normal train/eval behavior): 5 downsampling stages, 224 → 7:
  - stage *i*: `Conv2d(c_in, c_in, 3, 1, 1)` → `Conv2d(c_in, c_out, 2, 2, 0, bias=False)` → `BN(c_out)` → `GELU`
  - channels: 3→32, 32→64, 64→64, 64→128, 128→`out_dim`
  - output `(N, out_dim, 7, 7)` is **already a 3-D map**, so the trigger is skipped (`z_dim = 0` path); see Step 7
  - ⚠️ Its parameters must then be included in the flat learnable vector (Step 10)

**Check:** each encoder returns the documented shape on a dummy batch.

---

## Step 7: Prompt trigger (`blackvip/coordinator/trigger.py`)

- [ ] `PromptTrigger(nn.Module)` with `__init__(z_dim, src_dim, init_range=(0.0, 0.1))`
  - `trigger_dim = src_dim − z_dim` (800 by default)
  - `self.trigger = nn.Parameter(torch.empty(1, trigger_dim))`, then `uniform_(0.0, 0.1)`
  - `self.out_channels = src_dim // 49` (32 by default)
- [ ] `forward(z: (N, z_dim)) -> (N, out_channels, 7, 7)`
  - `trigger.expand(N, -1)`; prefer `expand` to `repeat`, since it avoids copying
  - `torch.cat([z, trigger], dim=1)`, which gives (N, 1568). **The order matters: z first, then the trigger.**
  - `.reshape(N, -1, 7, 7)`
- [ ] `IdentityTrigger` for the scratch encoder path: passes an already 3-D map straight through and has no parameters
- [ ] Cast the trigger to z's dtype before concatenating (for autocast)

**Tests (`test_trigger.py`):**
- [ ] Output shape is `(N, 32, 7, 7)`
- [ ] Parameter count = 800, and all initial values are in [0, 0.1)
- [ ] The trigger part of the output is identical across the batch (the trigger is shared); the z part differs across the batch
- [ ] Round trip: flattening the output `(N, 1568)` gives back `[z | trigger]`

---

## Step 8: Decoder (`blackvip/coordinator/decoder.py`)

### 8.1 Building block
- [ ] Helper `up_block(in_c, out_c, groups, act) -> nn.Sequential`:
  1. `ConvTranspose2d(in_c, in_c_mid, kernel_size=2, stride=2, padding=0, groups=groups)` (depthwise ×2 upsample, **bias=True**, the PyTorch default)
  2. `ConvTranspose2d(in_c_mid, out_c, kernel_size=1, bias=False)` (pointwise)
  3. `BatchNorm2d(out_c)`
  4. `act()` (GELU)
- [ ] Careful: in stage 1, the depthwise layer also **expands** channels from 32 to 64 (groups=32, so each group maps 1→2 channels). Write the block so that `in_c_mid` can differ from `in_c`.

### 8.2 Architecture table for ViT encoders (src_c = 32)

| stage | depthwise ConvT (k2 s2) | pointwise ConvT (k1, no bias) | BN | spatial |
|---|---|---|---|---|
| 1 | 32 → 64, groups = min(src_c, 64) = 32 | 64 → 64 | 64 | 7 → 14 |
| 2 | 64 → 64, groups = 64 | 64 → 32 | 32 | 14 → 28 |
| 3 | 32 → 32, groups = 32 | 32 → 32 | 32 | 28 → 56 |
| 4 | 32 → 32, groups = 32 | 32 → 16 | 16 | 56 → 112 |
| head | `ConvT(16 → 3, k2 s2, bias=False)`, standard, **no BN, no activation** | | | 112 → 224 |

- [ ] Stage-1 group rule: `g = 64 if src_c >= 64 else src_c`. This lets you try `src_dim` 784 (16 ch) or 3136 (64 ch) later.
- [ ] DINO-RN50 variant: same table, but stage 1 uses `groups=32` (the input channel count depends on `src_dim`). Keep it as a second entry in a small `ARCH_SPECS` dict, not as copy-pasted code.
- [ ] Default PyTorch initialization; the reference applies no custom init.
- [ ] No output squashing (no tanh or sigmoid). Its output range is controlled only by ε and the final `clip_range`.

### 8.3 Expected parameter count (default config)

| layer | params |
|---|---|
| s1 depthwise (32·2·2·2 + 64 bias) | 320 |
| s1 pointwise (64·64) | 4,096 |
| s1 BN (γ, β) | 128 |
| s2 depthwise (64·1·4 + 64) | 320 |
| s2 pointwise (64·32) | 2,048 |
| s2 BN | 64 |
| s3 depthwise (32·1·4 + 32) | 160 |
| s3 pointwise (32·32) | 1,024 |
| s3 BN | 64 |
| s4 depthwise | 160 |
| s4 pointwise (32·16) | 512 |
| s4 BN | 32 |
| head (16·3·4) | 192 |
| **decoder total** | **9,120** |
| + trigger | 800 |
| **learnable total** | **9,920** |

(The paper says "9K". These counts come from reading the reference code by hand, so confirm them with `parameters_to_vector(...).numel()` on the reference model.)

**Tests (`test_decoder.py`):**
- [ ] Input `(N,32,7,7)` gives output `(N,3,224,224)`
- [ ] Trace the intermediate shapes with forward hooks: 14, 28, 56, 112, 224
- [ ] Parameter count = 9,120
- [ ] Gradient sanity, as a plain module test only (training itself never uses autograd): `out.sum().backward()` gives every parameter a non-None grad

---

## Step 9: Coordinator assembly (`blackvip/coordinator/coordinator.py`)

- [ ] `Coordinator(nn.Module).__init__(cfg)`:
  - `self.encoder = build_encoder(cfg)` (frozen)
  - `self.trigger = PromptTrigger(...)` or `IdentityTrigger`
  - `self.decoder = Decoder(...)`
  - Group the learnable parts: `self.learnable = nn.ModuleDict({"trigger": ..., "decoder": ...})`, plus the encoder when it is the scratch encoder. Every flat-vector operation then runs over `self.learnable` alone.
- [ ] `forward(x) -> (prompt, z)`
  1. `z = self.encoder(x)` (no_grad inside)
  2. `seed = self.trigger(z)` → `(N,32,7,7)`
  3. `prompt = self.decoder(seed)` → `(N,3,224,224)`
  4. Return both; z is useful for t-SNE and debugging
- [ ] Optional speed-up for later: `forward_from_z(z)`. During one SPSA step, the encoder output for a batch is **the same for all 2q perturbed queries**, because the encoder is frozen and doesn't depend on ϕ. So compute z once per batch and reuse it, which skips 9 of the 10 MAE forward passes per step. Add the method now, and use it in Phase 5.
- [ ] `extra_repr()` / `summary()`: print the encoder name, z_dim, trigger_dim and the learnable parameter count

**Check:** `Coordinator(cfg)(x)` returns a prompt of shape `(N,3,224,224)` and a z of shape `(N,768)`; `forward(x)` and `forward_from_z(encoder(x))` give identical prompts.

---

## Step 10: The ZOO-facing interface (also in `coordinator.py`)

This is the API that the SPSA-GC optimizer (Phase 4) and the trainer (Phase 5) use. Keep it small and well tested.

- [ ] `learnable_parameters() -> Iterator[nn.Parameter]` returns the parameters of `self.learnable`, in a **fixed, deterministic order**
- [ ] `num_learnable() -> int`
- [ ] `get_flat() -> Tensor[d]` via `torch.nn.utils.parameters_to_vector`, detached
- [ ] `set_flat(w: Tensor[d]) -> None` via `vector_to_parameters`, under `torch.no_grad()`; assert `w.numel() == d`, and handle device and dtype
- [ ] `bn_mode` policy (from the config), applied in `train()`:
  - `"train"` (reference behavior): decoder BN layers use batch statistics and **update their running statistics on every perturbed query**
  - `"eval"`: decoder BN layers always use their running statistics (ablation)
  - Document clearly that BN running stats are **buffers**: they are not in the flat vector, but they **are** in the `state_dict`
- [ ] `learnable_state_dict()` / `load_learnable_state_dict()`: saves the trigger, decoder weights and BN buffers, but **not** the frozen encoder (~330 MB), so checkpoints stay at a few KB
- [ ] Guard: an `assert_blackbox_safe()` helper that checks that no parameter has `.grad` populated after a step

**Tests (`test_coordinator.py`):**
- [ ] `num_learnable() == 9920` with the default config
- [ ] `set_flat(get_flat())` leaves the outputs exactly unchanged
- [ ] `set_flat(w + δ)` changes the prompt; `set_flat(w)` restores it exactly
- [ ] The flat vector does **not** contain encoder parameters: perturbing it leaves `z` unchanged
- [ ] Encoder parameters are bit-identical before and after 10 random `set_flat` calls plus forward passes
- [ ] With `bn_mode="train"`, the running mean changes after a forward pass in train mode; with `"eval"`, it does not
- [ ] Saving and loading the learnable state dict reproduces the prompts exactly, and the file is under 100 KB
- [ ] `parameters_to_vector` order is stable across two separately constructed models with the same seed

---

## Step 11: Factory and public API (`blackvip/coordinator/__init__.py`)

- [ ] `build_coordinator(cfg: CoordinatorConfig) -> Coordinator`, which runs `cfg.validate()` first
- [ ] Export only: `Coordinator`, `CoordinatorConfig`, `build_coordinator`, `clip_range`, `apply_prompt`
- [ ] Make sure no module under `coordinator/` imports CLIP. Keeping it independent of the target model is the point of the black-box setting.

**Check:** `from blackvip.coordinator import build_coordinator` works in a fresh shell.

---

## Step 12: Precision and device handling

- [ ] The whole Coordinator moves with one `.to(device)`; the `clip_range` bounds follow the input's device
- [ ] It runs under `torch.autocast("cuda", dtype=torch.float16)` without dtype errors: check the trigger concatenation, BN, and `clip_range` on fp16
- [ ] `set_flat` under autocast still writes fp32 parameters
- [ ] CPU-only fallback works (slow but correct), so the unit tests can run without a GPU

**Check:** the same input gives a max abs difference of about 1e-2 between the fp32 and autocast prompts.

---

## Step 13: Inspection script (`scripts/inspect_coordinator.py`)

This is a manual sanity tool, not a test.

- [ ] Load a few real images (e.g. EuroSAT) through the Phase 1 test transform
- [ ] Print the learnable parameter count and its per-module breakdown (should match the table in 8.3)
- [ ] Print prompt statistics at initialization: mean, std, min/max per channel. This tells you how strong `ε·h(x)` is relative to the image range, which helps when choosing ε.
- [ ] Save a grid of: original | prompt (min-max scaled to [0,1]) | prompted image (denormalized). It should look like low-amplitude, image-dependent texture at initialization.
- [ ] Check input dependence: show that prompts for two different images differ, and measure the mean pairwise L2 distance between prompts in a batch (it must be > 0)
- [ ] Timing and memory: forward time with a batch of 128, encoder-only vs decoder-only, and peak GPU memory. The decoder should be negligible; nearly all cost is the MAE.

---

## Step 14: Phase 3 sign-off checklist

- [ ] All unit tests pass (`pytest tests/ -m "not slow"` on CPU; the full suite, including the HF download, on GPU)
- [ ] Learnable params = **9,920** (decoder 9,120 + trigger 800)
- [ ] Prompt shape `(N,3,224,224)`; the prompted image stays within the CLIP pixel bounds
- [ ] Encoder is frozen, stays in eval mode, and gets no gradients
- [ ] `get_flat`/`set_flat` round trip is exact; checkpoints are only a few KB
- [ ] The decisions are recorded in a short `NOTES.md`: MAE loading option (A/B), pre-LN vs post-LN [CLS], input normalization, `bn_mode`
- [ ] Optional cross-check against the reference: copy the reference `DecoderManual` weights into your decoder (map them by parameter order) and confirm identical outputs on the same `z`

Once this list is complete, go to **Phase 4 (ZOO optimizers)**. They only need `get_flat`, `set_flat` and a loss closure.
