# Independent audit of the cubic Fokker–Planck PINN (heat-flux G0 stage) and design of the G1 stage

**Scope.** `FP_PINN/pinn/cubic_homogeneous_3d` at commit `97f9ea8a` (branch
`work/fp-pinn-heatflux-g0-refine`), checkpoint `epoch-012500` of job 63178434
(commit `a2d928f7`), and the ten continuation runs in the handoff bundle.
HYQMOM / Stage 54–56 work is out of scope.  All numbers below were
re-computed in this audit with the tools shipped in `audit_tools/` and the
new `g1/` package; nothing is quoted from the runs' own `metrics.json` unless
labelled so.

**Bottom line.** The G0 checkpoint is better than its own evaluator said
(true analytic-Q_x L2 = **2.33 %**, not 3.57 % or 6.38 %), but it is not a
defensible solution: it violates mass by 3.1 % and energy by 4.5 %, it has
grown a spurious stress anisotropy p_xx − p_yy = −0.105 by t = 1, its
f-weighted residual is O(1), and every one of its gate numbers was a single
draw from a Monte-Carlo distribution whose one-sigma width (3.6–4.5 pp) exceeds
every threshold it was asked to decide.  The evaluator, not the physics, drove
the last four weeks of continuation runs.  The FP-PINN line should **continue
for exactly one more stage** (G1, delivered here), because the exact solution
is representable (required correction ≤ 2.2 in log f against a cap of 12),
the failure is structural/optimisation rather than representational, and a
deterministic reference and evaluator now exist.  If G1 does not pass its
(deterministic, frozen) gates on three seeds, the line should be stopped.

---

## 1. Physics and implementation audit of the training code

### 1.1 What is correct

* **Operator.** `cubic_operator.py` / `closure_tf` build the 9×9 system for
  (C_ij, Γ_i) with λ = −ν |dev P|² / DM2^3.5, nubol = 2ν, and the RHS
  q-rate term (3ν − 4ν/3) q = (5/3)ν q.  The residual operator in
  `make_train_step` is div a + a·∇h + ∂_t h − D(∆h + |∇h|²) with
  D = ν·DM2/3.  I re-implemented the same operator in numpy (finite
  differences, float64) and applied it to an independent solution of the
  equation: f-weighted RMS 0.019, core max 0.52 (discretisation error), i.e.
  the operator is right and O(1) residuals of the PINN are real (§3).
* **Heat-flux law.** The deterministic axisymmetric finite-volume solver
  (`g1/axisym_fp_reference.py`, §4) re-solves the closure from grid moments
  every RK stage and reproduces dQ_x/dt = −(4/3)νQ_x: fitted rate 1.3325 at
  400×200, Q_x-vs-exact L2 = 3.2e-4, converging second order.  The frozen
  analytic target Q_x(t) = 0.25 e^{−4t/3} and Pr_eff = 2/3 are therefore an
  independent consequence of the repository closure, not an assumption.
* **Exact initial condition and axisymmetry.** `DensityModel.log_density`
  forces f(0,c) = f₀(c) exactly (verified: L∞ = 2e-8 in f) and, with
  `axisymmetric_heat_flux`, depends on (t, c_x, |c|²) only, so f is exactly
  rotationally symmetric about x.  Q_y = Q_z = 0 **structurally**.
* **Float64 moments / float32 network** split is appropriate.

### 1.2 Findings (file / function, severity)

| # | Where | Finding | Effect |
|---|---|---|---|
| F1 | `train_stage2.evaluate`, `_evaluation_proposal`, `sample_tf_proposal` (antithetic branch) | The 4-fold antithetic reflections (c_y, c_z → ±) add **nothing** for Q_x, mass, energy or any axisymmetric moment because the model is exactly axisymmetric; they only zero Q_y, Q_z, which are already structurally zero. The effective sample size of every "N-sample" evaluation is N/4. | "131072 samples" = 32768 independent points. |
| F2 | `train_stage2.evaluate` (MC moments), `evaluate_stage2_checkpoints.py` (selection score) | The gate metric is a Monte-Carlo estimate with no uncertainty attached. Re-evaluating the *same* checkpoint with 16 independent seeds at N = 131072: mean 0.060, std **0.045**, range 0.014–0.138; at N = 524288 (8 seeds): mean 0.049, std **0.036**, range 0.013–0.118. The deterministic value is 0.0233. | 3.57 % (CONTINUATION_PASS) and 6.38 % (NO_GO) are two draws of the same checkpoint; all continuation decisions and checkpoint selections were made on noise. |
| F3 | same | The relative-L2 metric is a norm of a noisy vector, so sampling noise biases it **upward** (E‖q̂−q‖ > ‖E q̂ − q‖). The coarser evaluation happened to draw low. | Explains why the 4-panel/131072 number looked better than the 8-panel/524288 one. |
| F4 | `reference_particle.py` vs job configs | Two different particle references were used (250 k particles / dt 0.005 in job 63178434; 1 M / dt 0.0025 in 63374997). Their Q_x(1) are 0.0618 and 0.0717 vs exact 0.0659; `particle_analytic_relative_l2` = 0.026 and 0.020. | The particle reference is itself noisier than the 2 % gate; it must not decide anything. |
| F5 | `train_stage2.make_train_step` (mass/momentum/energy losses) | Invariants are soft penalties added to a pointwise loss. The optimiser lowered the total by violating them: true max |mass−1| = 0.031, max |DM2−3| = 0.045, oscillatory in t. Projecting the PINN's PDE residual f·R on (1, c_x, |c|²) gives d(mass)/dt swings of ±0.17 and d(energy)/dt of ±0.37 per unit time (numerically identical to d/dt of the deterministic moments, §3). | The model buys pointwise fit with conservation. Raising the weights (EXACT runs: 200/100/200) did not cure it — it only froze the optimiser (all three selected epoch 0). |
| F6 | `DensityModel.log_density` | The correction network has components along the collision invariants; nothing in the ansatz makes f's mass/momentum/energy independent of the weights. | Root cause of F5. |
| F7 | `train_stage2.evaluate` (`stress_history_relative_l2`) | Deviatoric reference stress is ≈ 0 for this case, so the metric divides by ~1e-4 and reports 12–66. It hides a real problem: the model's p_xx − p_yy reaches **−0.105** at t = 1 (FV reference: +1.8e-4). The dev-stress projection of the residual shows the model generating anisotropy at a rate −0.05 → −0.31, consistent with d/dt(dev P) + 2ν dev P. | A metric that cannot be read and a physical error that was never gated. |
| F8 | `analytic_heat_flux_rate_loss` (CRN finite difference, step 0.01) and `analytic_heat_flux_history_loss` | Both re-use the training cloud; the history loss trains on the quantity under test. The RATE_STRONG / EXACT variants therefore optimise the gate metric directly while making the residual and conservation worse. | Circular; delete from the qualification protocol (kept only as labelled ablations in G1). |
| F9 | `heat_flux_g0.fit_decay_rate` | Pins Q_x(0) = 0.25 (exact by the IC), fine; but the G0 reported "decay rate 1.414" was fitted to a noisy Q_x(t). Deterministic value for epoch-12500: 1.376. | Same MC noise. |
| F10 | `evaluate_stage2_checkpoints.py` | Selects the checkpoint with the lowest noisy score; smoke-admissible first. With std 4 pp the selection is a lottery (visible in the EXACT runs: epoch 0 selected three times). | — |
| F11 | `tf_initial_logpdf`, `_tf_normal_logpdf`, `tf_equilibrium_logpdf` | `tf.cast(python_float, float64)` rounds through float32, so the "float64" IC/Maxwellian normalisations are off by 1.2e-8 relative. | Harmless for training; spoils 1e-9 tests (use `tf.constant(x, dtype)`). |
| F12 | `train_stage2.moment_tensors` (general) | Subtracts a mean in every coordinate; applied to axisymmetric (c_x, ρ, 0) nodes it returns Var(ρ) instead of ⟨ρ²⟩. Not a G0 bug, but a trap for anyone reusing it with a reduced quadrature (I fell into it once in G1 and added a dedicated routine + test). | — |

No bug was found that would invalidate the operator or the closure.  The
defects are all in **how the problem is posed to the optimiser and how the
result is measured**.

---

## 2. Evaluator audit: why 3.57 % became 6.38 %

1. Reproduced the repository evaluator bit-for-bit in numpy
   (`audit_tools/np_model.py`): N = 131072 → 0.03555, N = 524288 → 0.06368
   (reported 0.03568 / 0.06384; the residual is float32 op ordering).
2. Swapped only the random seed of the evaluation cloud (table in F2).  Both
   reported values sit inside the one-sigma band of the *same* checkpoint.
3. Removed the antithetic duplication: base-only and 4× antithetic give
   identical Q_x to the third decimal (F1) — the reflections are inert.
4. Replaced sampling by a deterministic 2-D quadrature exploiting the exact
   axisymmetry (trapezoid 721×361, L = 9; and Gauss–Legendre 257×64, L = 9):
   grid-converged to 5 decimals, **Q_x L2 = 0.0233**.

So the discrepancy is **Monte-Carlo sampling noise of the evaluation cloud**,
biased upward (F3), with effective sample sizes a quarter of the nominal
ones (F1).  Nothing in the code path or the panel count changed the
checkpoint's quality between the two evaluations.

Corollary: the "quadrature-converged uncertainty < 0.5 pp" gate requested for
the next stage cannot be met by *any* affordable Monte-Carlo protocol (it would
need n_eff ≈ 10⁷ per time slice); it is met trivially by the deterministic
evaluator (measured 0.07 pp between the 129×32 and 257×64 grids for this
checkpoint).

---

## 3. What is actually wrong with the checkpoint (deterministic diagnosis)

All values for `epoch-012500`, computed with `g1/evaluate_g1.py` on the
257×64 held-out grid against the FV reference (`audit_tools/eval_g0ckpt/`):

| quantity | value | comment |
|---|---|---|
| analytic-Q_x L2 (raw network) | **2.33 %** | just above the 2 % primary gate |
| analytic-Q_x L2 after exact tilt (§5.2) applied post hoc | **1.51 %** | restoring the invariants alone brings it under the gate |
| max |mass − 1| / max |DM2 − 3| | 3.1 % / 4.5 % | oscillatory; residual projections ±0.17 / ±0.37 per unit time |
| p_xx − p_yy at t = 1 | −0.105 | FV reference +1.8e-4; pure artefact |
| stacked marginals vs FV (x, y, z at t = 0, ½, 1) | 3.00 % | marginal_x over all 21 times 4.6 % |
| full-field relative L2 vs FV | 7–8 % for t ∈ [0.2, 1] | max |log f error| 0.8–1.15 where f > 1e-6 |
| f-weighted RMS log residual | 0.65–1.03 (core max 3–5.5) | FV reference through the same operator: 0.019 |
| required correction log f_ref − log bridge | ≤ 2.2 | cap is 12: capacity is **not** the limit |
| fitted decay rate / Pr_eff | 1.376 / 0.69 | exact 1.333 / 0.667 |

Interpretation.  The residual's projections onto the invariants are exactly
the time derivatives of the model's mass and energy (checked numerically to
round-off), so the PDE loss *is* being violated in the invariant directions,
and the dev-stress projection shows the violation generating anisotropy.  The
optimiser found a density that fits the third moment well enough while
leaking mass, energy and isotropy — because nothing in the ansatz forbids it
and the soft penalties were cheaper to violate than the pointwise residual.
Increasing the penalties (BALANCED / EXACT variants) removed the escape route
without removing the conflict, so the optimiser stalled (epoch 0 selected).
STOPGRAD removed the closure gradient, which is not where the problem is.

This is a **structural** failure (wrong search space) compounded by a
**measurement** failure (noisy gate), not a representational one.

---

## 4. Deterministic reference (new)

`g1/axisym_fp_reference.py`: conservative second-order finite-volume
discretisation of the cubic-FP equation in (c_x, ρ) with zero-flux boundaries,
RK4 in time, closure re-solved every stage, D = ν·DM2/3.

| grid / dt | Q_x vs exact L2 | fitted rate | mass error | energy drift (grid diffusion) | wall |
|---|---|---|---|---|---|
| 200×100 / 8e-4 | 1.3e-3 | 1.3300 | 2e-16 | 1.3e-2 | 12 s (1 core) |
| 400×200 / 2e-4 | 3.2e-4 | 1.3325 | 2e-16 | 3.3e-3 | ≈ 4 min |

Marginals change by 5.5e-4 (relative L2) between the two grids, so the
400×200 reference is accurate to ≈ 2e-4 in the marginals — two orders below
the 3 % gate.  The particle references agree with it to 0.7–1.4 % (their own
noise).  The FV reference is shipped (`reference_fv_400x200/reference.npz`)
and regenerated inside every Slurm task.

---

## 5. G1 design (delivered in `g1/`, no existing file modified)

### 5.1 Deterministic quadrature instead of sampling
Tensor product of a composite trapezoid rule in c_x (129 nodes on [−8, 8],
spectrally accurate for Gaussian-like integrands) and Gauss–Legendre in ρ
(32 nodes on [0, 8], includes the 2πρ Jacobian).  The exact IC moments are
reproduced to 1e-12 (mass, momentum, energy, Q_x, DM4).  Each epoch the c_x
grid is shifted by a random fraction of a cell (a continuum of "panels");
evaluation uses an unshifted, finer and wider grid (257×64, L = 9).  Because
the network is axisymmetric, the 3-D autodiff Laplacian at (c_x, ρ, 0) equals
the axisymmetric Laplacian — no change to the G0 differentiation code
(unit-tested against finite differences).

### 5.2 Exact invariants by exponential tilt (replaces F5/F6)
log f = log f̃ + β(t)·ψ, ψ = (1, c_x, |c|²−3).  For every time slice β ∈ ℝ³
is the Newton solution of ∫ f̃ e^{β·ψ} ψ dc = (1, 0, 0) on the quadrature
(4 steps, unrolled, gradients flow through).  Mass, momentum and energy are
exact to ~1e-15 at every time for *any* weights; β is quadrature-independent
to 1e-7 (tested on 129×32 vs 257×64), i.e. a property of the continuum
ansatz.  dβ/dt enters ∂_t log f through implicit differentiation of the
constraint (tested against finite differences); ∇(β·ψ) and ∆(β·ψ) are
analytic.  This is the minimum-KL projection onto the invariant manifold —
the "conservation-orthogonal correction" requested in the brief, realised as
a per-time projection rather than a basis restriction (which would have
required the raw network to be orthogonal under a weight that itself
depends on the weights).

A gauge freedom appears (log f̃ → log f̃ + γ·ψ, β → β − γ leaves f unchanged);
it is fixed by a penalty w|β|² (w = 10) that constrains nothing about f.

### 5.3 Explicit heat-flux mode
b(t)·c_x(|c|²−5) (third Hermite mode, orthogonal to the invariants under the
Maxwellian, tested), b(t) = t·b_cap·tanh(head(t)/b_cap) from a 2×16 network of
t alone; b(0) = 0 keeps the hard IC; |b| ≤ 0.02 t.

### 5.4 Loss
f-weighted mean square of the log residual on the quadrature (the Fisher
norm ∫(∂_t f − Lf)²/f dc, weight detached) + gauge penalty.  **Nothing else**:
no invariant penalties (exact), no weak third-moment loss, no finite-difference
rate loss, no analytic history, no anchor.  The (4/3)ν law never enters
training; `--heat-flux-rate-weight` (operator's own weak identity) exists only
for a labelled ablation and is 0 in the qualification run.

### 5.5 Evaluator and selection
`g1/evaluate_g1.py` computes every metric deterministically on both grids and
reports their difference as the quadrature uncertainty; compares marginals,
fields and stress anisotropy with the FV reference; checks the structural
properties (IC, axisymmetry, positivity, portable reload); sweeps checkpoints
and selects by a deterministic score among structurally admissible ones.
Three from-scratch seeds + one warm start from the G0 checkpoint run as one
Slurm array; `aggregate_g1_seeds.py` applies the seed-agreement gate.

### 5.6 Gates (frozen; equality passes)
Blocking: Q_x L2 ≤ 2 % on the held-out grid (1 % = publication);
|L2_held-out − L2_train| ≤ 0.5 pp; tilted mass/momentum/energy ≤ 0.5/0.1/0.5 %
(structural); marginals vs FV ≤ 3 %; anisotropy error ≤ 0.02; transverse
Q = 0 and positivity (structural); IC L∞ ≤ 2e-6; decay rate within 5 % of
4/3; portable reload ≤ 1e-7; all three seeds PASS with spread ≤ 1 pp.
Diagnostic (non-blocking, in the selection score): pre-tilt mass/energy
drift ≤ 5 %, residual RMS by time, full-field L2.

Applied to the G0 checkpoint these gates give NO_GO for the right reasons
(anisotropy 0.105, marginals 3.00 %) while showing Q_x at 1.5 % after the
tilt — the gates discriminate, which the old ones did not.

### 5.7 Verification done in this audit (CPU)
12 unit tests pass; a 60-epoch smoke run on a 65×16 grid already reaches
Q_x L2 = 1.7 %, field L2 ≤ 4.6 %, with exact invariants; the full Slurm
pipeline (tests → FV reference → training → sweep → ZIP + SHA-256 →
seed aggregation) was executed locally end to end with tiny settings.
No GPU run has been performed: the numbers for G1 proper will come from
Unity.

---

## 6. Unity launch, resources, packaging

```bash
# once: a clean checkout containing g1/ (see README)
bash /project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g1/FP_PINN/pinn/cubic_homogeneous_3d/RUN_HEAT_FLUX_G1_UNITY.sh
```

* Pins: `--constraint=sm_75&vram12`, excluded gypsum nodes, `dsmc-gpu`
  environment, commit recorded and optionally enforced
  (`FP_EXPECTED_COMMIT`), no `gh` CLI.
* Array 0–2: seeds 20260901/02/03 from scratch; 3: warm start (G0 weights
  extracted automatically from the G0 archive).  Dependent CPU job writes
  `G1_SEED_SUMMARY.md` (overall PASS / NO_GO).
* Per task: `FP_PINN_G1_JOB<ID>_HEAT_FLUX_<VARIANT>_COMPLETE.zip` +
  `.sha256`, root-level `STATUS`, `run_metadata.json` with commit.
* Resources: 66 k residual points per step (G0: 49 k), same network and
  Hessian → ~1.3× G0 GPU memory (fits 12 GB; `FP_G1_N_TIME_BATCH=12` if not);
  ≈ 0.15–0.3 s/step on a ≥ sm_75 GPU → 25 k epochs ≈ 1–2 h, FV reference
  4 min, evaluation + sweep 10–15 min; 6 h walltime requested; ≈ 4–8
  GPU-hours for the whole array.

---

## 7. Continue / kill recommendation

**Continue — one stage, then decide.**  Evidence for continuing: the true
error of the existing checkpoint is already 2.3 % (1.5 % with exact
invariants), the required correction is far inside the network's range, the
operator and closure are verified, and the two things that made the last
four runs uninformative (noisy evaluator, soft invariants) are now removed
by construction.  Evidence against: every "improvement" run so far went the
wrong way, which is exactly what a noisy gate plus a conflicted loss
produce, so it is not evidence about the method.

Decision rule after G1 (pre-registered here):

* all three seeds PASS → homogeneous gate credible; move to the stress case
  with the same machinery, then spatial problems;
* NO_GO with small residual (RMS ≲ 0.05) but Q_x > 2 % → the closure, not the
  network, limits; stop the PINN line for this case and publish the FV
  reference as the closure's own answer;
* NO_GO with large residual → one optimisation-only stage at most (longer
  schedule, 24 slices, L-BFGS polish); then kill.

No shock or spatial runs before the homogeneous gate passes deterministically.

---

## 8. Deliverables in this package

```
FP_PINN_G1_PACKAGE/
  AUDIT_REPORT.md                       this document
  patch/FP_PINN/pinn/cubic_homogeneous_3d/
      g1/ (8 modules + tests)           new stage, pure addition
      slurm/run_heat_flux_g1_array.sbatch
      RUN_HEAT_FLUX_G1_UNITY.sh
  reference_fv_400x200/reference.npz    deterministic FV reference + metrics
  audit_tools/                          numpy re-implementation of the G0 model,
                                        deterministic evaluator, residual audit,
                                        MC-uncertainty data, tilted-G0 evaluation
  README_FA.md                          Persian summary
```
