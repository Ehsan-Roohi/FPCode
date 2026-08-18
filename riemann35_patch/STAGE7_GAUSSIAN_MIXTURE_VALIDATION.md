# Stage 7: finite-width Gaussian-mixture M5/M6 closure

Stage 7 replaces the failed discrete-node sixth-order tail with a smooth
equal-variance Gaussian-mixture reconstruction.  The cubic Fokker--Planck law,
the 35 evolved moments, the speed limiter, and the collision invariants are
unchanged.

The update is applied in increment form,

```text
M^(n+1) = M^n + M(F_dt[Q(M^n)]) - M(Q(M^n)),
```

where `Q` is the smooth Gaussian-mixture reconstruction and `F_dt` is the
finite particle-structure FP map. Subtracting the reconstructed input moments
cancels the small reconstruction residual, so it cannot accumulate as a false
source in cross moments or heat flux. Conservation and the degree-two
realizability margin are checked after every step.

The local workstation command is

```bash
python3 riemann35_patch/stage7/run_stage7.py \
  --case symmetric \
  --particles 100000 \
  --seeds 16 \
  --workers 8 \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --output-dir results/riemann35_stage7_symmetric
```

The second homogeneous gate changes only the initial distribution:

```bash
python3 riemann35_patch/stage7/run_stage7.py \
  --case asymmetric \
  --particles 100000 \
  --seeds 16 \
  --workers 8 \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --output-dir results/riemann35_stage7_asymmetric
```

The stronger multidimensional gate rotates an anisotropic asymmetric mixture,
thereby activating laboratory-frame covariance and higher-order cross moments:

```bash
python3 riemann35_patch/stage7/run_stage7.py \
  --case correlated \
  --particles 100000 \
  --seeds 16 \
  --workers 8 \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --output-dir results/riemann35_stage7_correlated
```

The structural leptokurtic gate has zero skewness but positive fourth
cumulant. It exercises the unequal-variance location--scale branch:

```bash
python3 riemann35_patch/stage7/run_stage7.py \
  --case leptokurtic \
  --particles 100000 \
  --seeds 16 \
  --workers 8 \
  --steps 200 \
  --dt 2.5e-4 \
  --tau 1.0 \
  --output-dir results/riemann35_stage7_leptokurtic
```

The symmetric acceptance quantities are the histories of `M200`, `M400`, and
the stress norm.  The asymmetric gate additionally exercises `M300` and `qx`.
The correlated gate also checks `M110` and `M210`.
The leptokurtic gate checks that positive fourth cumulant is matched without
the divergent low-weight satellite produced by equal-variance EQMOM near zero
skewness.
Every active history must have relative L2 error below 3%, the moment-matrix
margin must remain nonnegative, and mass, momentum, and energy must be
conserved to roundoff.
