# Heat-flux G1 stage — structure-preserving cubic-FP PINN on a deterministic quadrature

This directory is a **pure addition** to `FP_PINN/pinn/cubic_homogeneous_3d`
(commit `97f9ea8`, branch `work/fp-pinn-heatflux-g0-refine`).  No existing
file is modified; `train_stage2.py`, `cubic_operator.py`, `heat_flux_g0.py`
and the G0 tests are imported unchanged.  Scope is the homogeneous heat-flux
qualification case only; nothing here touches HYQMOM or any spatial problem.

## Why G1 exists (one paragraph)

The deterministic audit of the G0 checkpoint `epoch-012500` (job 63178434)
showed that (i) the G0 evaluator's Monte-Carlo metric had a one-sigma
uncertainty of 3.6–4.5 percentage points, so "3.57 %" and "6.38 %" were two
draws of the same ~2.3 % checkpoint; (ii) the true analytic-Qx L2 error was
2.33 %, but mass drifted by 3.1 %, energy by 4.5 %, a spurious stress
anisotropy p_xx − p_yy = −0.105 had grown by t = 1, and the f-weighted RMS log
residual was O(1); (iii) the soft invariant penalties were being traded against
the pointwise residual by the optimiser.  G1 removes every one of those
failure modes *by construction* rather than by re-weighting losses.

## What changes

| G0 (`train_stage2.py`) | G1 (`g1/`) |
|---|---|
| importance-sampled Monte-Carlo clouds (4× antithetic, n_eff = N/4) | deterministic tensor-product quadrature: trapezoid in c_x (129 nodes, spectral for Gaussians) × Gauss–Legendre in ρ (32 nodes); random per-epoch shift of the c_x grid |
| mass/momentum/energy as soft penalties (weights 30–200) | **exact** at every time: exponential tilt `f = f̃ exp(β(t)·ψ)`, ψ = (1, c_x, |c|²−3), β solved by Newton on the quadrature; dβ/dt by implicit differentiation |
| weak third-moment loss, CRN finite-difference dQ_x/dt loss, analytic-history loss, resume-anchor loss | **PDE residual only** (f-weighted mean square of the log residual) + a gauge-fixing penalty |β|² that constrains nothing about f |
| network correction only | + explicit third-Hermite heat-flux mode b(t)·c_x(|c|²−5), b(0)=0, bounded |
| Monte-Carlo evaluator with particle reference (250 k vs 1 M particles disagree by 1 % in Q_x) | deterministic evaluator on a finer held-out grid (257×64, L = 9) + deterministic axisymmetric finite-volume reference of the same equation (`axisym_fp_reference.py`, Q_x vs exact 3.2e-4, mass to 1e-16) |
| gates decided by one noisy number | gates decided by quadrature-converged numbers with the quadrature error *measured* (train grid vs held-out grid) |

The bridge ansatz `(1−α)f₀ + αM`, the exact initial condition, the tanh-capped
correction, the 9×9 closure (`closure_tf`) and the residual operator are those
of G0.  A G0 weight file loads into `model.base` unchanged (unit-tested), so the
G0 checkpoint can be used as a warm start.

## Files

```
g1/axisym_quadrature.py     (cx, ρ) quadrature, shifted panels, invariant / Hermite features
g1/structure_model.py       StructuredDensityModel, exact tilt, axisymmetric moments, residual
g1/train_g1.py              training script (deterministic, PDE-residual-only loss)
g1/evaluate_g1.py           deterministic evaluator, gates, checkpoint sweep + selection
g1/axisym_fp_reference.py   finite-volume RK4 reference of the same cubic-FP equation
g1/package_g1.py            atomic ZIP + SHA-256 sidecar + STATUS (PASS / NO_GO)
g1/aggregate_g1_seeds.py    seed-agreement verdict across array tasks
g1/tests/test_g1_structure.py  12 structural unit tests (run in the Slurm job)
g1/tests/test_g1_aggregation.py  4 fail-closed seed-verdict regression tests
g1/audit/AUDIT_REPORT.md    deterministic G0 audit and G1 design rationale
g1/audit/README_FA.md       Persian audit summary
slurm/run_heat_flux_g1_array.sbatch   array: 3 seeds from scratch + 1 warm start
RUN_HEAT_FLUX_G1_UNITY.sh             one-line submission (pins checkout, commit, GPUs)
```

The integration review also makes partial arrays fail closed, checks every
held-out log-density value for finiteness, fully re-evaluates the selected
checkpoint (residual, field and reload diagnostics), rejects non-finite
training gradients, and supports the flat layout of the packaged G0 archive.

## Gates (frozen before the run; equality passes)

Blocking (all must hold for PASS):

* analytic-Q_x L2 on the held-out 257×64 grid ≤ **2 %** (primary); ≤ 1 % = publication level
* |L2(held-out grid) − L2(training grid)| ≤ **0.5 pp** (quadrature-converged)
* tilted mass / momentum / energy drift ≤ 0.5 % / 0.1 % / 0.5 % (structural; observed ~1e-15)
* stacked marginals (x, y, z at t = 0, 0.5, 1) vs FV reference ≤ **3 %**
* max |(p_xx − p_yy) − (p_xx − p_yy)_ref| ≤ **0.02** (no spurious anisotropy)
* transverse heat flux = 0 (structural), positivity (structural), exact IC (L∞ ≤ 2e-6), exact axisymmetry
* fitted decay rate within 5 % of 4/3
* portable reload L∞ ≤ 1e-7
* **seed agreement**: all three from-scratch seeds PASS and their Q_x L2 spread ≤ 1 pp

Diagnostic (reported, enter the checkpoint-selection score, do not block):
pre-tilt mass / energy drift ≤ 5 % (how much of the solution the three tilt
parameters carry), f-weighted residual RMS by time, full-field L2 vs FV.

The analytic (4/3)ν law is **never used in training** (`--heat-flux-rate-weight 0`
is the default and the qualification setting).  The only flag that would feed
the operator's own weak heat-flux identity into the loss exists for a labelled
ablation.

## Running on Unity

One line (from a login node):

```bash
bash /project/pi_roohie_umass_edu/github_sync/FPCode-pinn-g1/FP_PINN/pinn/cubic_homogeneous_3d/RUN_HEAT_FLUX_G1_UNITY.sh
```

Before that, once:

```bash
cd /project/pi_roohie_umass_edu/github_sync
git clone --single-branch --branch work/fp-pinn-heatflux-g1 https://github.com/Ehsan-Roohi/FPCode FPCode-pinn-g1
```

The launcher refuses to run if `FP_EXPECTED_COMMIT` is set and does not match
`git rev-parse HEAD`; it always records the commit in `run_metadata.json`.
It submits the GPU array (`--constraint=sm_75&vram12`, the gypsum nodes that
fail the TF 2.21 build excluded) and a dependent CPU job that writes
`G1_SEED_SUMMARY.md` with the overall verdict.  The warm-start checkpoint is
extracted automatically from
`FPCode-pinn-g0/FP_PINN_STAGE2_JOB63178434_HEAT_FLUX_COMPLETE.zip`.

Each task writes `outputs/g1-<JOBID>/<variant>/` (config, loss history,
checkpoints every 1000 epochs, `final_evaluation/`, `checkpoint_sweep/` with
`selected.weights.h5`) and packages it as
`FP_PINN_G1_JOB<JOBID>_HEAT_FLUX_<VARIANT>_COMPLETE.zip` + `.sha256` with a
root-level `STATUS` file.

Useful overrides: `FP_G1_EPOCHS` (25000), `FP_G1_N_TIME_BATCH` (16; reduce to 12
on a 12 GB GPU if the Hessian batch does not fit), `FP_G1_LEARNING_RATE`
(2e-4), `FP_G1_TILT_PENALTY_WEIGHT` (10), `FP_G1_HEAT_FLUX_MODE_CAP` (0.02),
`FP_G1_USE_HEAT_FLUX_MODE=0`, `FP_G1_STOP_GRADIENT_CLOSURE=1`,
`FP_G1_SHIFT_CX_GRID=0`, `FP_FV_NX/NR/DT` (400 / 200 / 2e-4), `FP_ARRAY`.

## Resource estimate

* Residual points per step: 16 slices × 4128 nodes = 66 k (G0: 12 × 4096 = 49 k),
  same network (5 × 128 tanh) and the same pfor Hessian, so GPU memory is
  ~1.3× G0 — fits a 12 GB card; host memory 24 GB requested.
* Measured on one CPU core: 0.26 s/step at 4.2 k points.  Scaling to 66 k points
  and a ≥20× GPU speed-up gives ≈0.15–0.3 s/step → 25 k epochs ≈ **1–2 h**;
  FV reference ≈ 4 min (CPU); final evaluation + 26-checkpoint sweep ≈ 10–15 min.
  The 6 h walltime is conservative.
* The whole G1 array (4 tasks) is therefore ≈ 4–8 GPU-hours.

## Reading the result

* `G1_SEED_SUMMARY.md`: PASS only if every from-scratch seed passes every
  blocking gate and the seeds agree to 1 pp.
* `<variant>/checkpoint_sweep/checkpoint_sweep.json`: per-checkpoint
  deterministic metrics and the selected checkpoint; its `selected_metrics`
  are recomputed with the full residual, field and portable-reload checks.
* `<variant>/checkpoint_sweep/selected_evaluation/g1_validation.png`: decisive
  selected-checkpoint plot of Q_x vs analytic and FV,
  invariants before/after tilt, marginals vs FV, residual RMS by time, stress
  anisotropy.

## Stop / go after the G1 run

* **PASS** (3 seeds ≤ 2 %, anisotropy ≤ 0.02, marginals ≤ 3 %): the homogeneous
  heat-flux gate is credible and reproducible; proceed to the stress case with
  the same machinery, then to spatial problems.
* **NO_GO with small residual (RMS ≲ 0.05) but Q_x > 2 %**: the cubic closure,
  not the network, is limiting — stop the PINN line for this case and report
  the FV reference as the closure's own answer.
* **NO_GO with large residual**: optimisation, not representation (the exact
  correction is ≤ 2.2 in log f, cap 12) — one more stage at most (longer
  schedule, `n_time_batch` 24, L-BFGS polish); if that fails, kill the line.

## Local checks (CPU, < 3 min)

```bash
cd FP_PINN/pinn/cubic_homogeneous_3d
python -m unittest discover -s g1/tests -t . -v
python g1/axisym_fp_reference.py --output-dir /tmp/ref --nx 200 --nr 100 --dt 8e-4   # 12 s
python g1/train_g1.py --output-dir /tmp/g1smoke --reference /tmp/ref/reference.npz \
    --epochs 40 --n-time-batch 4 --n-cx 65 --n-rho 16 --eval-n-cx 129 --eval-n-rho 32 \
    --checkpoint-every 20 --print-every 10
python g1/evaluate_g1.py --config /tmp/g1smoke/config.json --sweep-dir /tmp/g1smoke/checkpoints_h5 \
    --reference /tmp/ref/reference.npz --output /tmp/g1smoke/checkpoint_sweep
```

`numpy ≥ 2` is required by `axisym_fp_reference.py` (`np.trapezoid`); the
`dsmc-gpu` environment on Unity satisfies this.
