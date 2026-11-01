# Phase 4: Zeroth-Order Optimization (ZOO): Step-by-Step Build Guide

Detailed guide for Phase 4 of [roadmap.md](roadmap.md). The goal is to build **SPSA**, **SPSA-GC** (the paper's optimizer), and **RGF** (the BAR baseline) as standalone, CLIP-free components, then validate them on toy problems (quadratic, Rosenbrock) before Phase 5 connects them to the Coordinator and the black-box CLIP.

Reference code (read-only):
- SPSA / SPSA-GC: [../BlackVIP/trainers/blackvip.py](../BlackVIP/trainers/blackvip.py), `forward_backward` and `spsa_grad_estimate_bi` (lines 112–173)
- RGF: [../BlackVIP/trainers/reprogramming.py](../BlackVIP/trainers/reprogramming.py), `forward_backward` (lines 118–161)

---

## What you are building

Black-box CLIP gives you **loss values only, no gradients**. A ZOO optimizer estimates the gradient from loss differences at perturbed parameters, then takes a step.

```
              ┌──────────── optimizer.step(w, closure) ────────────┐
 w (d,) ───►  │ k += 1; a_k, c_k = schedule(k)                     │
              │ for q repeats:                                     │
              │     Δ ~ perturbation(d)                            │
              │     L+ = closure(w + c_k Δ)    ◄── 1 query         │
              │     L- = closure(w - c_k Δ)    ◄── 1 query         │
              │     ĝ_i = (L+ - L-) / (2 c_k Δ)                    │
              │ ĝ = mean(ĝ_1..ĝ_q)                                 │
              │ update rule (SPSA | SPSA-GC | NAG look-ahead)      │
              └──► w_new (d,)                                      ┘
```

**Design rules:**
1. **Optimizers work on a flat vector `w` and a closure.** They never see the Coordinator, CLIP, or images. That's why they can be tested on a Rosenbrock function in milliseconds.
2. **The optimizer is functional with respect to `w`:** `w_new = opt.step(w, closure)`. The optimizer owns only its *state* (step `k`, momentum `m`, query count). The caller owns `w` and writes it back (`coordinator.set_flat(w_new)`).
3. **Each piece is swappable:** perturbation sampler × gain schedule × estimator × update rule. This makes the Table 6 ablation (SPSA vs SPSA-GC) and the RGF baseline into config choices.
4. **Bit-for-bit parity with the reference is an option, not an accident.** Wherever the reference has a specific behavior (RNG call order, ± query order, which loss is logged), there is a flag that reproduces it.

---

## Target file layout

```
blackvip/
  config.py               # + ZOOConfig (Step 2)
  zoo/
    __init__.py           # public API + build_optimizer(cfg)   (Step 10)
    perturbations.py      # segmented-uniform, Rademacher, Gaussian-sphere   (Step 3)
    schedules.py          # SPSA gains a_k, c_k; poly-decay LR   (Step 4)
    estimators.py         # spsa_two_sided, rgf   (Steps 5, 8)
    optimizers.py         # ZOOptimizer base, SPSA, SPSAGC, SPSAGCNAG, RGF   (Steps 6–9)
    benchmarks.py         # quadratic, rosenbrock, noisy wrapper   (Step 11)
scripts/
  rosenbrock_fig3.py      # Fig. 3 reproduction   (Step 12)
tests/
  test_zoo_perturbations.py
  test_zoo_schedules.py
  test_zoo_estimators.py
  test_zoo_optimizers.py
  test_zoo_benchmarks.py
```

---

## Step 0: Notation and reference facts (read once)

| Symbol | Meaning | Reference name | BlackVIP default |
|---|---|---|---|
| `d` | number of optimized parameters | `N_params` | 9,920 |
| `k` | global step, **starts at 1** | `self.step` (incremented *before* `forward_backward`) | |
| `a, o, α` | step-size gain: `a_k = a / (k + o)^α` | `a, o, alpha` | 0.01, 1.0, 0.4 |
| `c, γ` | perturbation gain: `c_k = c / k^γ` | `c, gamma` | 0.01 or 0.005, 0.1 or 0.2 |
| `q` | number of averaged estimates per step | `SP_AVG` | 5 |
| `β` | momentum coefficient | `MOMS` | 0.3–0.95 per dataset |
| `Δ` | perturbation vector, shape `(d,)` | `perturb` | segmented uniform `±U[0.5, 1]` |

- **Config order:** `SPSA_PARAMS = [o, c, a, α, γ]`. It's easy to swap `a` and `c` by mistake; your config should use named fields.
- **Queries per step:** SPSA, SPSA-GC: `2q` = 10. RGF: `q + 1` = 6.
- **The ± queries use the same mini-batch.** The closure is called with different `w` values, never different data.
- **Worked numbers** (`a=0.01, o=1, α=0.4, c=0.005, γ=0.2`):

  | k | a_k | c_k |
  |---|---|---|
  | 1 | 0.00758 | 0.00500 |
  | 10 | 0.00383 | 0.00315 |
  | 1000 | 0.00063 | 0.00126 |

  Use these as test fixtures in Step 4.

---

## Step 1: Understand the reference precisely

The reference loop, rewritten as pseudocode. Every line here is something your implementation must either copy or deliberately change behind a flag.

```
step += 1                                   # k = 1 on the first batch
a_k = a / (step + o) ** alpha
c_k = c / step ** gamma
w = parameters_to_vector(dec.parameters())  # p_trigger first, then body.*
for _ in range(q):
    Δ = segmented_uniform(d)                # CPU global RNG, then .cuda()
    set(w + c_k Δ); out1 = model(x)         # (+) query FIRST
    set(w - c_k Δ); out2 = model(x)         # (−) query SECOND
    ĝ_j = (CE(out1) − CE(out2)) / (2 c_k Δ)
ĝ = mean_j ĝ_j
if spsa-gc:  m = ĝ if step == 1 else β m + ĝ;   step_dir = ĝ + β m
if spsa:     step_dir = ĝ
w_new = w − a_k · step_dir;  set(w_new)
logged loss = (CE(out1) + CE(out2)) / 2     # from the LAST repeat only
logged acc  = mean of the two accuracies    # from the LAST repeat only
```

Observations to record in your notes:
- [ ] **`m = ĝ if step == 1` is redundant**, because `m` starts at 0 and `β·0 + ĝ = ĝ`. Initialize `m = 0` and drop the special case. Add a test proving the two forms are equivalent, and make sure resuming from a checkpoint restores `m` and `k`.
- [ ] **SPSA-GC in the code is exactly PyTorch's `SGD(nesterov=True)` rule** (`buf = μ·buf + g; p -= lr·(g + μ·buf)`), with ĝ evaluated at `w`, not at the look-ahead point. That is discrepancy #1 in [roadmap.md](roadmap.md#known-paper--code-discrepancies-to-decide-on).
- [ ] **The closure has side effects.** In the reference, each forward pass runs the decoder in train mode, so the BatchNorm running stats are updated on **every** query (10 times per step), in the order +, −, +, −, …. Your optimizer must not assume `closure` is a pure function, and it must keep the + then − order for parity.
- [ ] **The perturbation comes from the global torch RNG on CPU:** `torch.rand(d)`, then `torch.bernoulli(...)`. To reproduce reference runs bit for bit, replicate these exact calls in this order (Step 3).
- [ ] **The parameters are left at `w − c_k Δ` until the final `set(w_new)`.** The caller must always write back `w_new`, even when it's not logging.
- [ ] **The logged loss comes from the last repeat, not the average over all `q`.** Copy this in the logs for curve comparison, but also log the mean over all `2q` queries (it's free).
- [ ] **Autocast:** the reference runs the queries under `torch.cuda.amp.autocast()`. The loss difference `L+ − L−` is tiny (around `c_k · |∇L| · |Δ|`), so fp16 rounding in the logits can swamp it. Compute the CE loss in fp32 (autocast already does this for `cross_entropy`) and keep `ĝ` in fp32 (Step 5). Flag this as a risk area for Phase 5.

RGF (BAR baseline), from `reprogramming.py`:
```
a_k = (lr0 − lr_min)(1 − min(k, K)/K)^0.9 + lr_min,   K = 0.9 · total_steps
L0 = loss(w)                                  # pivot query
ĝ = 0
for _ in range(q):
    u ~ N(0, σ²I);  u /= ‖u‖₂                 # uniform direction on the unit sphere
    ĝ += (d / q) · u · (loss(w + β u) − L0) / β
w_new = w − a_k ĝ
logged loss = L0 (at w)
```
- [ ] Note: σ has no effect, because `u` is normalized. Keep the parameter for config parity only.

---

## Step 2: Config (`blackvip/config.py`): add `ZOOConfig`

- [ ] Dataclass `ZOOConfig`, pure Python like `CoordinatorConfig`:
  - `opt_type`: one of `("spsa", "spsa-gc", "spsa-gc-nag", "rgf")`
  - `o, c, a, alpha, gamma` as **named fields** (defaults `1.0, 0.005, 0.01, 0.4, 0.2`)
  - `moms` (β, default 0.9)
  - `sp_avg` (q, default 5)
  - `perturbation`: `("segmented_uniform", "rademacher")`, default `"segmented_uniform"`
  - `reference_rng: bool = True`: use the exact reference RNG call sequence (Step 3)
  - RGF only: `rgf_sigma = 1.0`, `rgf_smooth = 0.01`, `rgf_lrs = (lr0, lr_min)`, `total_steps: int` (needed for the poly decay)
- [ ] Classmethod `from_spsa_params([o, c, a, alpha, gamma], ...)`, so you can paste reference config values directly.
- [ ] `validate()`: `a, c > 0`; `0 ≤ moms < 1`; `sp_avg ≥ 1`; `alpha, gamma ≥ 0`; `total_steps > 0` when `opt_type == "rgf"`; known `opt_type` and `perturbation`.
- [ ] `to_dict` / `from_dict`, as in `CoordinatorConfig`.

**Tests (`test_config.py`, extend):**
- [ ] Defaults validate
- [ ] `from_spsa_params([1.0, 0.01, 0.01, 0.4, 0.1])` maps to `o=1.0, c=0.01, a=0.01, alpha=0.4, gamma=0.1`
- [ ] Each invalid field raises `ValueError`
- [ ] Round trip `from_dict(to_dict(cfg)) == cfg`

---

## Step 3: Perturbations (`blackvip/zoo/perturbations.py`)

Signature for every sampler: `sample(d, *, device, dtype=torch.float32, generator=None) -> Tensor[d]`.

- [ ] **`segmented_uniform`** (default): magnitudes `U[0.5, 1]`, random sign, so `Δ ∈ [−1, −0.5] ∪ [0.5, 1]`.
  - **Why not plain Rademacher?** Bounded away from 0 (so `1/Δ` is at most 2) and not all-equal magnitudes. The paper reports it works better.
  - **Reference-exact mode** (`reference_rng=True`), replicating these calls in this order on the **CPU global RNG**, then moving to `device`:
    ```
    p_side = (torch.rand(d).reshape(-1, 1) + 1) / 2          # U[0.5, 1]
    samples = cat([p_side, -p_side], dim=1)                   # (d, 2)
    idx = torch.bernoulli(torch.ones_like(p_side) / 2).long() # 0 → +, 1 → −
    Δ = gather(samples, 1, idx).reshape(-1)
    ```
  - **Clean mode:** `mag = 0.5 + 0.5·rand(d, generator=g)`, `sign = 2·randint(0, 2, (d,), generator=g) − 1`, `Δ = sign·mag`, sampled on `device`. It's faster and isolated from other RNG users, but it gives a different random stream from the reference.
- [ ] **`rademacher`**: `±1` with probability 1/2 each (the commented-out alternative in the reference).
- [ ] **`gaussian_sphere`** (for RGF): `u ~ N(0, σ²I)`, then `u / ‖u‖₂`.
- [ ] Registry dict `PERTURBATIONS = {"segmented_uniform": ..., "rademacher": ...}`.

**Tests (`test_zoo_perturbations.py`):**
- [ ] Shape `(d,)`, dtype, device
- [ ] Segmented uniform: all `|Δ| ∈ [0.5, 1]`, no value in `(−0.5, 0.5)`; sign fraction ≈ 0.5 (for `d = 100k`, within ±0.01)
- [ ] Moments (these drive the variance analysis in Step 5): `E[Δ] ≈ 0`, `E[Δ²] ≈ 7/12 ≈ 0.583`, `E[1/Δ²] ≈ 2`
- [ ] Rademacher: values ⊂ `{−1, 1}`
- [ ] Gaussian sphere: `‖u‖₂ = 1` (atol 1e-5)
- [ ] **Reproducibility:** same seed gives an identical Δ; different seeds give a different Δ
- [ ] **Reference parity:** with `torch.manual_seed(0)`, your reference-exact sampler equals a copy of the reference's three lines, pasted into the test

---

## Step 4: Gain schedules (`blackvip/zoo/schedules.py`)

- [ ] `spsa_gains(k, *, a, o, alpha, c, gamma) -> (a_k, c_k)`
  - `a_k = a / (k + o) ** alpha`, `c_k = c / k ** gamma`
  - **Raise `ValueError` if `k < 1`**: `c_k` at `k = 0` divides by zero (`0 ** γ = 0`). This is why the step counter starts at 1.
  - Return Python floats, not tensors (they're logged, and combined with tensors later).
- [ ] `poly_decay_lr(k, *, lr0, lr_min, total_steps, power=0.9, decay_frac=0.9) -> float`
  - `K = decay_frac · total_steps`; `k' = min(k, K)`; `(lr0 − lr_min)(1 − k'/K)^power + lr_min`
  - Check where the reference's `step_for_pdecay` starts and increments; match it (the first step gives exactly `lr0` if it starts at 0).
- [ ] Why the two gains decay at different rates (write this down in the docstring): the classic SPSA conditions (Spall) require `a_k → 0`, `c_k → 0`, `Σ a_k = ∞`, `Σ (a_k/c_k)² < ∞`. With `α = 0.4, γ = 0.2`, `a_k/c_k ∝ k^{−0.2}`. The squared sum diverges, so the reference violates the textbook condition. That's fine in practice for finite budgets, but don't expect asymptotic guarantees.

**Tests (`test_zoo_schedules.py`):**
- [ ] The worked-numbers table in Step 0 (rel tol 1e-3)
- [ ] `a_k`, `c_k` strictly decreasing in `k`
- [ ] `k = 0` raises
- [ ] Poly decay: `lr0` at the start, `lr_min` at and after `K`, monotone in between

---

## Step 5: SPSA gradient estimator (`blackvip/zoo/estimators.py`)

`spsa_two_sided(closure, w, c_k, q, sampler) -> EstimateResult`

- [ ] For `j = 1..q`:
  1. `Δ = sampler(d)`
  2. `L+ = closure(w + c_k·Δ)` **then** `L− = closure(w − c_k·Δ)` (this order, for parity)
  3. `ĝ_j = (L+ − L−) / (2·c_k·Δ)`, element-wise division, in **fp32**
- [ ] `ĝ = mean_j ĝ_j`. Stack-then-mean is fine, or use a running sum to save memory; d is small.
- [ ] **Closure contract:** `closure(w) -> (loss: 0-d tensor, aux: dict)`. `aux` carries anything the trainer wants to log (e.g. accuracy). The estimator treats it as opaque.
- [ ] Convert `loss` to fp32 **before** subtracting: `(L+.float() − L−.float())`.
- [ ] Return a small dataclass `EstimateResult`:
  - `ghat (d,)`
  - `loss_last` = `(L+ + L−)/2` of the last repeat (reference logging)
  - `loss_mean` over all `2q` queries
  - `aux_last` (the aux of the last ± pair, e.g. both accuracies)
  - `n_queries = 2q`
- [ ] Never call `.backward()`; wrap in `torch.no_grad()`.

### Theory to verify with tests (understand it before you test it)
- **Unbiased for quadratics.** For `L(w) = ½ wᵀAw + bᵀw`, the two-sided difference is exact: `L+ − L− = 2c_k Δᵀ∇L`. So `ĝ_i = ∇_i + Σ_{j≠i} ∇_j Δ_j/Δ_i`, and with independent symmetric `Δ`, `E[ĝ] = ∇L` exactly, for any `c_k`.
- **Per-sample variance:** `Var(ĝ_i) = Σ_{j≠i} ∇_j² · E[Δ²] · E[1/Δ²]` ≈ `‖∇‖² · 7/6` (segmented uniform) or `‖∇‖²` (Rademacher).
- **So a single estimate is a very noisy direction:** `cos(ĝ, ∇) ≈ 1/√(1 + (7/6)·d)` per sample. With `d = 9,920` that's about **0.009**, and about **0.02 with q = 5**. Each step is nearly orthogonal to the true gradient, and progress comes from averaging over many steps. This is exactly why momentum (the "GC" in SPSA-GC) helps: it accumulates the consistent component of ĝ across steps.
- **Bias for non-quadratics** is `O(c_k²)` (third-derivative terms). A larger `c` lowers the variance relative to noise in the loss but adds bias; a smaller `c` amplifies loss noise (`/(2c_k)`). That trade-off is what Table 7 sweeps.

**Tests (`test_zoo_estimators.py`), all on CPU, using the quadratic from Step 11:**
- [ ] Shape `(d,)`, fp32, `n_queries == 2q`, closure called exactly `2q` times
- [ ] **Query order:** a recording closure sees `w + cΔ₁, w − cΔ₁, w + cΔ₂, …`
- [ ] **Exactness on a 1-D quadratic:** `d = 1` gives `ĝ == ∇` exactly (no other coordinates to leak)
- [ ] **Unbiasedness:** `d = 50`, average 20,000 estimates; cosine to the true gradient > 0.99 and relative error < 5%
- [ ] **Variance formula:** the empirical `Var(ĝ_i)` matches `(7/6)·Σ_{j≠i} ∇_j²` within 10%
- [ ] **Single-sample cosine:** at `d = 1000`, the mean cosine ≈ `1/√(1 + 7d/6)` (within 30%), showing that you understand the noise level
- [ ] **Averaging:** variance with `q = 5` ≈ variance with `q = 1` / 5
- [ ] **Constant loss gives `ĝ = 0`**; a linear loss `bᵀw` gives `E[ĝ] = b`

---

## Step 6: Optimizer base class (`blackvip/zoo/optimizers.py`)

`class ZOOptimizer` (plain Python class, not `nn.Module`):

- [ ] `__init__(cfg: ZOOConfig, *, generator=None)`: stores the config, `self.k = 0`, `self.m = None`, `self.n_queries = 0`
- [ ] `step(w, closure) -> StepResult`, a template method (same idea as `FrozenEncoder.forward`):
  1. `self.k += 1`
  2. compute the gains (subclass hook `_gains()`)
  3. `est = self._estimate(w, closure, ...)` (subclass hook)
  4. `w_new = self._update(w, est.ghat, a_k)` (subclass hook)
  5. `self.n_queries += est.n_queries`
  6. return `StepResult(w_new, loss_last, loss_mean, aux_last, a_k, c_k, ghat_norm, step_norm, k, n_queries)`
- [ ] Validate `w`: 1-D, floating point, finite. Raise on NaN/Inf in `w` **and** in `ĝ`. A NaN loss poisons `w` forever otherwise.
- [ ] `state_dict() -> dict` / `load_state_dict(d)`: `k`, `m` (tensor or None), `n_queries`, and the generator state if you use a private generator. **Phase 5 checkpoints must include this**, or resuming restarts the gain schedule at `k = 1` (large steps), with no momentum.
- [ ] `reset()`, for tests

---

## Step 7: SPSA and SPSA-GC update rules

- [ ] **`SPSA`**: `w_new = w − a_k·ĝ`
- [ ] **`SPSAGC`** (reference form, the **default**):
  ```
  m ← β·m + ĝ          (m initialized to zeros_like(w))
  w_new = w − a_k·(ĝ + β·m)
  ```
  - This is PyTorch's Nesterov SGD buffer rule. Momentum `m` is stored *without* `a_k`, so when `a_k` decays, the old momentum is scaled by the *current* `a_k`.
- [ ] **`SPSAGCNAG`** (the paper's Algorithm 1, behind `opt_type="spsa-gc-nag"`):
  ```
  w_look = w + β·v                       (v initialized to 0; v includes the step size)
  ĝ = estimate at w_look                 (the closure is queried around the look-ahead point!)
  v ← β·v − a_k·ĝ
  w_new = w + v
  ```
  - Override `_estimate` to call the estimator at `w_look`, not `w`
  - The logged loss is then around `w_look`, not `w`; document this
  - `v` and the reference's `m` have different units (`v` already includes `a_k`); don't load one optimizer's state into the other
- [ ] Why both: the paper's numbers come from the code (reference form). The NAG form is what the paper *describes*. Step 12 measures whether the difference matters.

**Tests (`test_zoo_optimizers.py`):**
- [ ] **Manual trace:** with a fixed `ĝ` sequence (stub the estimator to return `ĝ₁, ĝ₂, ĝ₃`), check that `m` and `w` match hand-computed values for 3 steps, for each optimizer
- [ ] **`k == 1` equivalence:** reference "`m = ĝ` at step 1" == "`m = 0` then `β·m + ĝ`"
- [ ] **β = 0:** SPSA-GC ≡ SPSA (same RNG seed → identical `w` sequence)
- [ ] **Equivalence to PyTorch:** with the *true* gradient fed as ĝ and a constant `a_k`, `SPSAGC` reproduces `torch.optim.SGD(lr=a, momentum=β, nesterov=True)` to 1e-6 over 50 steps on a quadratic. This proves the claim from Step 1.
- [ ] **Convergence:** on a 10-D quadratic, all three optimizers reduce the loss by > 90% within 2,000 steps (tune `a` for the test; it's a smoke test, not a benchmark)
- [ ] **State round trip:** run 10 steps, save the state, run 10 more → `w_A`; reload the state at step 10 into a new optimizer, restore the RNG, run 10 → `w_B`; `torch.equal(w_A, w_B)`
- [ ] Query count after `n` steps == `2qn`
- [ ] NaN loss raises

---

## Step 8: RGF estimator + optimizer (BAR baseline)

- [ ] `rgf(closure, w, smooth, q, sigma, generator)` in `estimators.py`:
  1. `L0 = closure(w)` (pivot, 1 query)
  2. for `j = 1..q`: `u = gaussian_sphere(d)`; `Lj = closure(w + smooth·u)`; `ĝ += (d/q)·u·(Lj − L0)/smooth`
  3. return `EstimateResult(ghat, loss_last=L0, loss_mean=mean over all q+1, aux_last = aux of the pivot, n_queries = q + 1)`
- [ ] `class RGF(ZOOptimizer)`: gains from `poly_decay_lr` (there's no `c_k`; log `smooth` in its place); update `w_new = w − a_k·ĝ`
- [ ] Theory: `E[d·uuᵀ] = I` for `u` uniform on the sphere, so `E[ĝ] ≈ ∇L` (first order, one-sided, with `O(smooth)` bias). It's **one-sided**, so its bias is `O(smooth)`, versus `O(c²)` for two-sided SPSA, and its variance scales with `d`. That's why it's the weakest optimizer in Fig. 3.

**Tests:**
- [ ] `n_queries == q + 1`; the first closure call is at `w` itself
- [ ] Unbiasedness on a quadratic (average many estimates, cosine > 0.95; it's looser because of the one-sided bias)
- [ ] The poly-decay LR is used, not the SPSA gains

---

## Step 9: First-order reference (Rosenbrock comparison only)

- [ ] `SGDNAG`: a thin wrapper with the same `step(w, closure)` interface, but the closure must also return the **true** gradient. Implement by hand (`buf = β·buf + g; w −= lr·(g + β·buf)`) or wrap `torch.optim.SGD(nesterov=True)` on a leaf tensor.
- [ ] Clearly mark it as **white-box**; Phase 5 must never use it with CLIP.

---

## Step 10: Factory and public API (`blackvip/zoo/__init__.py`)

- [ ] `build_optimizer(cfg: ZOOConfig, *, generator=None) -> ZOOptimizer`, via a registry `{"spsa": SPSA, "spsa-gc": SPSAGC, "spsa-gc-nag": SPSAGCNAG, "rgf": RGF}`
- [ ] `__all__` exports: `ZOOConfig`, `build_optimizer`, `StepResult`, the optimizer classes, `spsa_gains`
- [ ] **The Phase 5 interface contract** (write it in the module docstring):
  ```
  w = coordinator.get_flat()
  def closure(w_):
      coordinator.set_flat(w_)
      logits = clip.query(apply_prompt(x, coordinator(x), eps))
      return F.cross_entropy(logits.float(), y), {"acc": accuracy(logits, y)}
  res = opt.step(w, closure)
  coordinator.set_flat(res.w_new)        # ALWAYS: the params are left at w − c_kΔ otherwise
  ```

**Tests:**
- [ ] Each `opt_type` builds the right class
- [ ] An unknown `opt_type` raises

---

## Step 11: Benchmark functions (`blackvip/zoo/benchmarks.py`)

All functions take a flat `w (d,)` and return `(loss, aux)`, the same closure contract. They optionally also return the true gradient, for SGD-NAG and the cosine tests.

- [ ] `quadratic(A, b)` → closure for `½wᵀAw + bᵀw`; true gradient `Aw + b`; optimum `−A⁻¹b`. Use a diagonal `A` with a condition number you control (e.g. eigenvalues log-spaced in `[1, 100]`).
- [ ] `rosenbrock(d)` → `Σ_{i=1}^{d−1} [100(w_{i+1} − w_i²)² + (1 − w_i)²]`; minimum 0 at `w = 1`; the true gradient by hand or with autograd (benchmarks only)
- [ ] `noisy(closure, scale, generator)` → adds `N(0, scale²)` to the loss on **every call**, so `L+` and `L−` get independent noise (that's the realistic setting: mini-batch noise and API noise)
- [ ] `normalized_loss(L, L_star, L0) = |L* − L| / |L* − L0|` (the Fig. 3 y-axis)

**Tests (`test_zoo_benchmarks.py`):**
- [ ] Rosenbrock(1⃗) == 0, and its gradient is 0 at 1⃗
- [ ] The analytic gradient equals the autograd gradient at random points
- [ ] Noisy wrapper: mean ≈ the clean loss, std ≈ the scale, two calls at the same `w` differ

---

## Step 12: Reproduce Fig. 3 (`scripts/rosenbrock_fig3.py`)

The goal is a **qualitative** reproduction: the ranking and the shape of the curves, not exact numbers.

- [ ] Before coding, re-read the paper's Fig. 3 caption and text, and record the exact settings (dimension, initial point, iteration budget, noise scales, hyperparameters). If anything isn't specified, choose it, and **write the choice at the top of the script** (the roadmap assumes 100-D).
- [ ] Methods: RGF, SPSA, SPSA-GC (reference form), SPSA-GC-NAG (paper form), SGD-NAG (white-box)
- [ ] **Match the budget by queries, not steps:**
  - per step, SPSA/GC use `2q` queries, RGF uses `q + 1`, and SGD uses 1 gradient
  - plot against the number of queries (and also against steps)
- [ ] **Tune each method fairly:** a small grid over `a` (and `c` for SPSA) per method; pick the best final loss on one seed, then report 5 seeds (mean ± std band)
- [ ] Plot the normalized loss, on a log scale
- [ ] **Noise experiment:** repeat with `noisy(·, scale)` for 3–4 scales; plot the final normalized loss against the scale
- [ ] Save the figure and a JSON of the settings and final numbers to `outputs/phase4/`

**Expected outcome (qualitative, from the paper):**
- [ ] SPSA-GC converges faster and more stably than SPSA, and much better than RGF
- [ ] SPSA-GC gets closest to SGD-NAG
- [ ] As the noise grows, RGF degrades fastest; SPSA-GC is the most robust
- [ ] Record whether the reference form and the NAG form differ noticeably. This answers discrepancy #1 empirically.

---

## Step 13: Diagnostics to carry into Phase 5

Add these to `StepResult` now, so Phase 5 can log them for free:
- [ ] `a_k`, `c_k`
- [ ] `‖ĝ‖`, `‖step‖ = a_k‖ĝ + βm‖`, `‖w‖`, `‖m‖`
- [ ] `L+ − L−` magnitude per repeat. If this is ~0 relative to the fp16 resolution of the loss, the estimate is pure rounding noise (the autocast risk from Step 1).
- [ ] Cumulative query count (Fig. 7's x-axis)
- [ ] Optional, on toy problems only: the cosine of ĝ to the true gradient

---

## Step 14: Phase 4 sign-off checklist

- [ ] `python -m pytest tests -k zoo -v`: all green, CPU only, under 1 minute
- [ ] Reference parity: the perturbation sampler matches the reference's RNG calls; SPSA-GC matches `torch.optim.SGD(nesterov=True)` given the same gradients
- [ ] Estimator unbiasedness and variance tests pass, and you can explain the `≈ 1/√d` cosine
- [ ] Optimizer state save/load gives bit-identical continuation
- [ ] Fig. 3 reproduced qualitatively (ranking + noise robustness), and the figure is saved
- [ ] Decision recorded: reference form vs NAG form (default = reference)
- [ ] The Phase 5 interface contract (Step 10) is written down, including "always `set_flat(w_new)`" and the BN side effect

✅ **Phase 4 done when:** all of the above are checked. Phase 5 then only needs to write the closure.
