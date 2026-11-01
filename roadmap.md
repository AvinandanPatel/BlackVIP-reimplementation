# BlackVIP Reimplementation Roadmap

Reproducing **BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning** (Oh et al., CVPR 2023) from scratch.

- Paper: [docs/BlackVIP_paper.pdf](docs/BlackVIP_paper.pdf)
- Reference code (read-only, for cross-checking): [../BlackVIP/](../BlackVIP/)

**Core idea in one line:** a frozen CLIP is treated as a black box (logits only). A small **Coordinator** (frozen SSL encoder + ~9K-param conv decoder + prompt-trigger vector) produces an image-sized, input-dependent prompt that is added to the image. The Coordinator is trained with **SPSA-GC**, a zeroth-order optimizer: two forward queries per perturbation, plus Nesterov-style gradient correction.

Each phase ends with a **✅ Done when** check. Don't start the next phase until that check passes.

---

## Phase 0 — Project setup

- [ ] Choose the layout (suggested):
  ```
  blackvip/
    clip_wrapper.py      # black-box CLIP oracle
    coordinator.py       # encoder + decoder + trigger
    prompters.py         # VP padding prompter, BAR program
    zoo.py               # SPSA, SPSA-GC, RGF optimizers
    data/                # datasets, few-shot sampler, transforms, synthetic generators
    trainers/            # blackvip.py, vp_spsa.py, bar.py, vp_whitebox.py, zeroshot.py
    config.py            # dataclass/yaml configs
    train.py / eval.py
  configs/  scripts/  tests/  outputs/
  ```
- [ ] Set up the environment. Recent versions are fine, but record them, because the authors note that results vary with the PyTorch version and GPU.
  - [ ] `torch`, `torchvision`
  - [ ] OpenAI `clip` package (or `open_clip` with the OpenAI weights)
  - [ ] `transformers` for `facebook/vit-mae-base`, plus optionally `google/vit-base-patch16-224-in21k` and `Ramos-Ramos/dino-resnet-50`
  - [ ] `yacs` or `omegaconf`, `numpy`, `pillow`, `opencv-python` (Loc-MNIST), `tqdm`, optionally `wandb`/tensorboard
- [ ] Config system that supports: base yaml → dataset yaml → method yaml → CLI overrides
- [ ] Seed utility that seeds `random`, `numpy` and `torch` (CPU + CUDA). The seed must be reset right before the training loop, as the reference code does.
- [ ] Logger that writes to stdout and a file in the output dir
- [ ] Output directory naming that encodes dataset/method/hyperparameters/seed

✅ **Done when:** `python train.py --help` runs, and a config can be loaded, overridden from the CLI, and printed.

---

## Phase 1 — Data pipeline

### 1.1 Base abstractions
- [ ] `Datum(impath, label, classname)` record
- [ ] `DatasetBase` with `train_x`, `val`, `test`, `classnames`, `num_classes`, `lab2cname`
- [ ] `split_dataset_by_label`
- [ ] `generate_fewshot_dataset(data, num_shots)`: `random.sample` K items per class without repeats (use all items if a class has fewer than K)
- [ ] Few-shot cache: `split_fewshot/shot_{K}-seed_{seed}.pkl` (train = K-shot, val = min(K, 4)-shot)
- [ ] `subsample_classes(all|base|new)`: optional, only needed for base-to-new experiments

### 1.2 Transforms (CLIP preprocessing)
- [ ] CLIP mean `(0.48145466, 0.4578275, 0.40821073)`, std `(0.26862954, 0.26130258, 0.27577711)`
- [ ] Train: `RandomResizedCrop(224, bicubic)` → `RandomHorizontalFlip` → `ToTensor` → `Normalize`
- [ ] Test: `Resize(224, bicubic)` → `CenterCrop(224)` → `ToTensor` → `Normalize`
- [ ] "noaug" variant for the synthetic datasets (no crop or flip)
- [ ] DataLoader: train batch size 128 with shuffle, test batch size 100

### 1.3 The 14 benchmark datasets
Use the CoOp few-shot splits and file layouts (see CoOp `DATASETS.md`).
- [ ] Caltech101
- [ ] OxfordPets
- [ ] StanfordCars
- [ ] Flowers102
- [ ] Food101
- [ ] FGVCAircraft
- [ ] SUN397
- [ ] DTD
- [ ] EuroSAT
- [ ] UCF101 (middle frames)
- [ ] ImageNet
- [ ] SVHN (VP-style split `split_mlai_SVHN.json`)
- [ ] Resisc45 (`split_mlai_Resisc45.json`)
- [ ] CLEVR-count (`split_mlai_CLEVR.json`)

> Tip: start with **EuroSAT**. It is small (10 classes), it is the ablation dataset in Tables 5 and 6, and BlackVIP gains a lot on it (47.9 → 73.1).

### 1.4 Synthetic datasets
- [ ] **Loc-MNIST generator** (paper Supp. A.1, `mk_locmnist.py`):
  - [ ] 224×224 black canvas; the real 28×28 digit goes on a random edge strip (one coordinate ∈ {0, 196}, the other uniform in [0, 196]; randomly swap which axis)
  - [ ] Fake digit (random class, random MNIST image) centered, scaled by `f_size` ∈ {1, 4}
  - [ ] Output `locmnist_r1_f{1,4}/{train,test}/{digit}/*.jpg`, generated with a fixed seed
  - [ ] Train/val split 80/20 from train; the full test set
- [ ] **Biased (Colour) MNIST** (Bahng et al. 2020, ReBias):
  - [ ] Each digit has a pre-assigned background color; a fraction ρ of training images use it, the rest use one of the other colors uniformly
  - [ ] Test uses the reversed correlation (1 − ρ): easy = ρ 0.8/0.2, hard = ρ 0.9/0.1
  - [ ] Save as images; few-shot sample from train; use test as val
- [ ] Class-name text templates for both datasets: `"This is a photo of {}"`

✅ **Done when:** for EuroSAT, `len(train_x) == 16 * 10`, `len(val) == 4 * 10`, the test set is full size, the cached pkl reloads identically for the same seed, and a batch visualization looks correct. Loc-MNIST and Biased-MNIST sample grids match Fig. 8 of the paper.

---

## Phase 2 — Black-box target model (CLIP oracle)

- [ ] Load CLIP ViT-B/16 (also RN50, RN101, ViT-B/32 for the ablation) as **fp32**, in eval mode, with all `requires_grad=False`
- [ ] Per-dataset text templates (reference: `trainers/zsclip.py`, `CUSTOM_TEMPLATES`):
  - CoOp templates for 11 datasets (e.g. Pets `"a photo of a {}, a type of pet."`, EuroSAT `"a centered satellite photo of {}."`, DTD `"{} texture."`, UCF `"a photo of a person doing {}."`)
  - VP templates for SVHN/Resisc45 (`"This is a photo of a {}"`) and CLEVR (`"This is a photo of {} objects"`)
  - Replace `_` with a space in class names
- [ ] Precompute L2-normalized text features **once** and cache them
- [ ] `BlackBoxCLIP.query(images) -> logits` = `logit_scale.exp() * norm(img_feat) @ text_feat.T`, run under `torch.no_grad()`
- [ ] Enforce the black-box contract: trainers may call **only** `query()`. No gradients and no parameter access. Optionally count queries, for the query-efficiency plot (Fig. 7).
- [ ] Optional AMP autocast for speed (the reference uses it during SPSA forward passes)

✅ **Done when:** **zero-shot CLIP** accuracy on the test sets matches the ZS row of Table 3 within about ±0.5 (EuroSAT ≈ 47.9, Pets ≈ 89.1, Caltech ≈ 92.9, SVHN ≈ 18.1, CLEVR ≈ 14.5). This also validates the data pipeline and templates.

---

## Phase 3 — Coordinator (prompt generator)

> 📘 The detailed step-by-step build guide is in [roadmap_phase3_coordinator.md](roadmap_phase3_coordinator.md).

### 3.1 Frozen encoder `f(·)`
- [ ] `facebook/vit-mae-base` (load with `ViTForImageClassification`/`ViTModel`, **no masking**); z = `hidden_states[-1][:, 0, :]` ([CLS]) → shape `(N, 768)`
- [ ] Frozen, under `no_grad`, always in eval mode
- [ ] Ablation encoders (Table 6 and Table 5):
  - [ ] Supervised `google/vit-base-patch16-224-in21k`: `last_hidden_state[:, 0]` (768)
  - [ ] `dino-resnet-50`: `pooler_output` (2048)
  - [ ] "Scratch": a learnable 5-stage conv encoder (`EncoderManual`), trained together with the decoder
- [ ] Note: the reference code passes the **CLIP-normalized** image straight into MAE. Copy this to reproduce the paper exactly; renormalizing to ImageNet statistics is an optional experiment.

### 3.2 Prompt trigger `ϕ_t`
- [ ] Learnable vector of size `SRC_DIM − z_dim` = 1568 − 768 = **800**, init `U(0, 0.1)`
- [ ] Shared across all instances: `concat(z, ϕ_t)` → `(N, 1568)` → reshape `(N, 32, 7, 7)`
- [ ] Assert `SRC_DIM % 49 == 0`

### 3.3 Decoder `g_ϕd(·)` (depthwise-separable transposed convolutions, 7 → 224)
Each stage upsamples ×2 with a depthwise `ConvTranspose2d(k=2, s=2)` followed by a pointwise `ConvTranspose2d(k=1, bias=False)`, then `BatchNorm2d` and `GELU`:
- [ ] Stage 1: `ConvT(32→64, k2 s2, groups=32)` → `ConvT(64→64, k1)` → BN(64) → GELU  (7→14)
- [ ] Stage 2: `ConvT(64→64, k2 s2, groups=64)` → `ConvT(64→32, k1)` → BN(32) → GELU  (14→28)
- [ ] Stage 3: `ConvT(32→32, k2 s2, groups=32)` → `ConvT(32→32, k1)` → BN(32) → GELU  (28→56)
- [ ] Stage 4: `ConvT(32→32, k2 s2, groups=32)` → `ConvT(32→16, k1)` → BN(16) → GELU  (56→112)
- [ ] Head: `ConvT(16→3, k2 s2, bias=False)` (112→224). No output activation.
- [ ] Separate variant for dino-resnet-50 (z = 2048 → `SRC_DIM` must be a multiple of 49 and larger than 2048)

### 3.4 Prompted image
- [ ] `x̃ = clip_range(x + ε · h_ϕ(x))`, with ε = `P_EPS` (per dataset, 0.2–1.0)
- [ ] `clip_range` clamps each channel in **normalized** space to the valid RGB range `[(0−μ)/σ, (1−μ)/σ]`:
  R ∈ [−1.7923, 1.9303], G ∈ [−1.7521, 2.0749], B ∈ [−1.4802, 2.1459]

### 3.5 Flat-parameter interface (needed by ZOO)
- [ ] `get_flat() -> Tensor[d]` and `set_flat(w)` covering decoder params **+ trigger** (BN affine params included; BN running stats are buffers, not params)
- [ ] Decide on and document BN behavior. The reference keeps the decoder in **train mode** during SPSA queries, so running statistics update on every perturbed forward pass, and test uses eval mode with those stats. Copy this first.

✅ **Done when:** output shape is `(N, 3, 224, 224)`, the learnable parameter count is about 9.9K (the paper reports "9K"; check it against `parameters_to_vector(dec.parameters()).numel()` in the reference), `set_flat(get_flat())` is an identity, and the frozen encoder receives no updates.

---

## Phase 4 — Zeroth-order optimizers

> 📘 The detailed step-by-step build guide is in [roadmap_phase4_zoo.md](roadmap_phase4_zoo.md).

Implement them as standalone classes that only take a `loss_fn(w) -> scalar` closure. That keeps them testable without CLIP.

### 4.1 Perturbation distributions
- [ ] **Segmented Uniform** (default): magnitude `U[0.5, 1.0]`, random sign → `[−1, −0.5] ∪ [0.5, 1]`
- [ ] Rademacher `{−1, +1}` (alternative)

### 4.2 SPSA (two-sided)
- [ ] Gain schedules with global step `k` starting at **1**:
  `a_k = a / (k + o)^α`,  `c_k = c / k^γ`
  Config order is `SPSA_PARAMS = [o, c, a, α, γ]` (BlackVIP default: `[1.0, 0.01|0.005, 0.01, 0.4, 0.1|0.2]`)
- [ ] For q = `SP_AVG` = 5 repeats: sample Δ, then `ĝ = (L(w + c_kΔ) − L(w − c_kΔ)) / (2 c_k Δ)` (element-wise division by Δ); average the q estimates
- [ ] That is 2q = **10 black-box queries per step**, all on the same mini-batch
- [ ] Update: `w ← w − a_k ĝ`
- [ ] Report the training loss and accuracy as the mean of the ± evaluations

### 4.3 SPSA-GC (the paper's contribution)
- [ ] Paper (Alg. 1, Eq. 3), NAG form: `m_{k+1} = β m_k − a_k ĝ(w_k + β m_k)`, `w_{k+1} = w_k + m_{k+1}`
- [ ] Reference code (what produced the numbers), the "PyTorch-Nesterov" reformulation that evaluates ĝ at `w_k`, not at the look-ahead point:
  ```
  m = ĝ                 if k == 1
  m = β·m + ĝ           otherwise
  w ← w − a_k (ĝ + β·m)
  ```
- [ ] Implement the reference form as the default and the literal look-ahead form behind a flag. Compare them on Rosenbrock and EuroSAT and record the difference.
- [ ] `β` = `MOMS` (per dataset: 0.3–0.95)

### 4.4 RGF (for the BAR baseline)
- [ ] One pivot query `L(w)`, then for q = 5: `u ~ N(0, σ²I)`, normalized to unit L2 norm; `ĝ += (d/q) · u · (L(w + βu) − L(w)) / β`
- [ ] σ = 1.0, smoothing β = 0.01, d = number of params
- [ ] LR: poly decay `a_k = (lr0 − lr_min)(1 − k/K_decay)^0.9 + lr_min`, with `K_decay = 0.9 · total_steps`

### 4.5 First-order references (for the Rosenbrock comparison only)
- [ ] SGD with Nesterov momentum, using true gradients

✅ **Done when:** the Rosenbrock experiment (Fig. 3) is reproduced qualitatively:
- [ ] 100-D Rosenbrock; plot normalized loss `|L(θ*) − L(θ)| / |L(θ*) − L(θ0)|` over iterations for RGF, SPSA, SPSA-GC and SGD-NAG
- [ ] SPSA-GC converges faster and more stably than RGF and SPSA, and comes close to SGD-NAG
- [ ] Noise sensitivity: add `N(0, scale²)` to the loss for several scales; RGF degrades fastest and SPSA-GC is the most robust
- [ ] Unit test: on a quadratic, the SPSA gradient estimate averaged over many samples ≈ the true gradient (cosine similarity > 0.9)

---

## Phase 5 — BlackVIP trainer

- [ ] Build: data → BlackBoxCLIP (text features for this dataset's classnames) → Coordinator → flat params w
- [ ] Training step (all under `torch.no_grad()`, optionally with autocast):
  1. `k += 1`; compute `a_k` and `c_k`
  2. `w = coordinator.get_flat()`
  3. For each of the q repeats: `set_flat(w ± c_kΔ)` → `logits = clip.query(clip_range(x + ε·h(x)))` → CE loss
  4. Average ĝ, apply the SPSA-GC update, `set_flat(w_new)`
- [ ] Loss = cross-entropy (BlackVIP and VP), focal loss γ=2 (BAR)
- [ ] Epoch loop over the 16-shot loader. Epochs per dataset: 5000, except SUN397 (1000), StanfordCars (2500) and ImageNet (500).
- [ ] Logging: loss, acc, a_k, c_k, ‖ĝ‖, ‖w‖, and cumulative query count every N steps
- [ ] Checkpoint: save the flat params, trigger, BN buffers, momentum `m` and step `k` every 500 epochs and at the last epoch. Make sure resuming restores `m` and `k`.
- [ ] Evaluation: load the **last-epoch** checkpoint (not best-val), set the decoder to eval mode, report top-1 accuracy on the full test set
- [ ] Run 3 seeds (1, 2, 3); report mean ± std
- [ ] Note: the reference builds an SGD optimizer and LR scheduler from `OPTIM` but **only uses them for checkpoint registration**. The SPSA gains set the actual step size, so the "lr" in the logs is meaningless. Don't copy this.

✅ **Done when:** EuroSAT (ε = 0.4, β = 0.9, γ = 0.2, c = 0.005, a = 0.01, α = 0.4, o = 1) reaches about **73.1%** (±3, given the high variance the authors note), well above ZS (47.9).

---

## Phase 6 — Baselines

### 6.1 Zero-shot CLIP
- [ ] Already done in Phase 2; reuse it

### 6.2 VP (white-box, upper bound)
- [ ] Padding prompter: 4 learnable pads, 30 px wide, `randn` init, added to the 224 image (69K params); no clipping
- [ ] Backprop through CLIP; SGD lr 40 (per-dataset table), momentum 0.9, cosine schedule, 100 warmup epochs (linear from 0.1), 1000 epochs
- [ ] Target: Table 3 "VP (white-box)" row

### 6.3 VP w/ SPSA-GC (black-box)
- [ ] Same padding prompter, optimized with SPSA-GC from Phase 4
- [ ] Defaults `SPSA_PARAMS = [1.0, 0.01, 40.0, 0.5, 0.1]` (large a, because it optimizes pixels directly), β from the per-dataset table

### 6.4 BAR (Black-box Adversarial Reprogramming)
- [ ] Program `W ∈ R^{3×224×224}`, mask M = 0 in the center and 1 on the frame; the downstream image is resized to **194×194** and embedded in the center
- [ ] Pipeline: un-normalize to [0, 1] → place into the zero frame → add `sigmoid(W) ⊙ M` → re-normalize with CLIP statistics (~37K params)
- [ ] RGF optimizer, focal loss γ=2, poly LR `[init_lr, min_lr]` from the per-dataset table

✅ **Done when:** each baseline reproduces EuroSAT within a reasonable margin (VP-wb ≈ 90.8, VP-SPSA-GC ≈ 70.9, BAR ≈ 77.2) and the ordering holds on 2–3 other datasets.

---

## Phase 7 — Experiments

### 7.1 Synthetic robustness (Table 2)
- [ ] Biased MNIST ρ = 0.8 and 0.9, 16-shot and 32-shot, "noaug" config, 5000 epochs
- [ ] Loc-MNIST 1:1 and 1:4, 16-shot and 32-shot
- [ ] Hyperparameters from `../BlackVIP/docs/configuration.md` §2 (e.g. Loc-MNIST 1:1: β 0.9, α 0.5, a 0.01, c 0.005, ε 1.0)
- [ ] Targets (16-shot): BM 0.8 → 66.21, BM 0.9 → 62.47, LM 1:1 → 69.08, LM 1:4 → 60.86
- [ ] All 5 methods (VP-wb, ZS, BAR, VP-SPSA-GC, BlackVIP)

### 7.2 Few-shot transfer on 14 benchmarks (Table 3)
- [ ] Per-dataset hyperparameter table (epochs, β, γ, c, ε) from `configuration.md` §1, encoded as yaml/scripts
- [ ] Results table: 14 datasets × 5 methods, mean over 3 seeds, Avg and Win columns
- [ ] Target BlackVIP average: **64.0** (ZS 57.6, BAR 61.4, VP-SPSA-GC 58.8, VP-wb 71.1)
- [ ] Budget: ImageNet and SUN397 are expensive (10 CLIP forwards per step). Run them last.

### 7.3 Ablations
- [ ] **Table 5: target backbones** on EuroSAT, RN50 / RN101 / ViT-B/32 / ViT-B/16 × Coordinator encoders {MAE ViT-B/16, DINO RN50}; hyperparameters from `configuration.md` §3.1
- [ ] **Table 6: encoder × optimizer** on EuroSAT: {scratch, supervised ViT, SSL MAE} × {SPSA, SPSA-GC}; target SSL+SPSA 69.4 → SSL+SPSA-GC 73.1
- [ ] Optional hyperparameter sweep (Table 7 ranges): a ∈ {1, 0.1, 0.01, 0.005}, α ∈ {0.6, 0.5, 0.4, 0.3}, c ∈ {0.01, 0.005, 0.001}, γ ∈ {0.2, 0.1}, β ∈ {0.9, 0.7, 0.5, 0.3}; select on the 4-shot val set

### 7.4 Efficiency (Table 4, Fig. 7)
- [ ] Peak GPU memory (`torch.cuda.max_memory_allocated`) during training on ImageNet for FT, LP, VP-wb, BAR, VP-SPSA-GC and BlackVIP (ViT-B and ViT-L); target BlackVIP ≈ 2.4 GB vs VP-wb ≈ 11.9 GB on ViT-B
- [ ] Parameter counts: BlackVIP ≈ 9K, BAR ≈ 37K, VP ≈ 69K
- [ ] Query efficiency: test accuracy against the cumulative number of queries (and the $ cost at Clarifai pricing)

---

## Phase 8 — Analysis and visualization

- [ ] Prompt visualizations: original / prompt / prompted image for VP vs BlackVIP on Biased-MNIST and Loc-MNIST (Fig. 4); map prompts to [0, 1] for display
- [ ] t-SNE of the prompted-image CLIP embeddings plus the NMI score (Fig. 5)
- [ ] Grad-CAM on the CLIP image encoder's penultimate layer, original vs prompted (Fig. 6, Supp. C), on CLEVR, Pets, UCF101, SVHN, EuroSAT, Cars and the synthetic datasets. This step uses white-box access for **analysis only**.

---

## Phase 9 — Testing and reproducibility hygiene

- [ ] Unit tests: few-shot sampler counts; `clip_range` bounds; Coordinator shapes and param count; `get_flat`/`set_flat` round trip; SPSA estimator on a quadratic; SPSA-GC momentum bookkeeping (k = 1 special case); checkpoint save/load/resume
- [ ] Black-box guard test: assert that no CLIP parameter has `.grad` and that `requires_grad` stays False after a training step
- [ ] Smoke test: 2 epochs on EuroSAT with 1 shot in under 2 minutes
- [ ] Determinism: same seed and same GPU → identical loss curve for the first 100 steps
- [ ] Results registry (CSV/JSON): dataset, method, seed, hyperparameters, last-epoch acc, queries, wall-clock time
- [ ] A final `RESULTS.md` comparing reproduced and paper numbers for Tables 2–6, with notes on deviations

---

## Known paper ↔ code discrepancies to decide on

| # | Topic | Paper | Reference code | Suggested choice |
|---|---|---|---|---|
| 1 | SPSA-GC update | NAG, ĝ evaluated at look-ahead `w + βm` | ĝ at `w`; `m = βm + ĝ`; step `ĝ + βm` | Code version by default; paper version behind a flag |
| 2 | Decoder block order | "[NORM-ACT-CONV] × 5" | `ConvT → ConvT(1×1) → BN → GELU` ×4, then ConvT | Follow the code |
| 3 | Decoder conv type | "depthwise separable conv" | depthwise + pointwise **transposed** conv (upsampling) | Follow the code |
| 4 | Segmented uniform comment | `[−1, −0.5] ∪ [0.5, 1]` | comment says `[-1, 0.5]`, code samples `±U[0.5, 1]` | `±U[0.5, 1]` |
| 5 | Encoder input normalization | not specified | CLIP-normalized image fed to MAE | Follow the code; test ImageNet norm as an extra |
| 6 | Model selection | not specified | last-epoch checkpoint (best-val code commented out) | Last epoch |
| 7 | BN running stats | not specified | updated during every perturbed query (train mode) | Follow the code; consider freezing as an ablation |
| 8 | BAR embedded size | 194×194 | yaml says 64, script overrides to 194 | 194 |
| 9 | Param count | "9K" | ≈ 9.9K incl. trigger and BN affine | Report the exact count |

---

## Suggested order of work

1. Phases 0–2, until zero-shot EuroSAT is about 47.9
2. Phase 4 on Rosenbrock (fast, no GPU needed)
3. Phases 3 and 5, until EuroSAT BlackVIP is about 73
4. Phase 6 baselines on EuroSAT
5. Phase 7.1 synthetic, then 7.3 ablations (both EuroSAT/MNIST-scale)
6. Phase 7.2 on all 14 datasets (most expensive), then 7.4 and Phase 8
