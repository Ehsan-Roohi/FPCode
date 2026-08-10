# Stage 7 results: preliminary short-time Gaussian-mixture FP closure

## Outcome

The Stage-7 closure passed four short-time homogeneous validation gates against an
independent 16-seed, 1.6-million-particle ensemble. Each seed used 100,000
particles, 200 collision steps, `dt/tau = 2.5e-4`, `Pr = 2/3`, and
`gamma_scale = 0.05`. The prescribed acceptance threshold was 3% history-wise
relative L2 error for every active physical diagnostic.

| Case | Diagnostic | Mixture error | Single-Gaussian error |
|---|---:|---:|---:|
| Symmetric | M200 | 0.025% | 0.545% |
| Symmetric | M400 | 0.118% | 5.800% |
| Symmetric | stress norm | 0.163% | 2.125% |
| Asymmetric | M200 | 0.018% | 0.682% |
| Asymmetric | M400 | 0.157% | 5.899% |
| Asymmetric | stress norm | 0.089% | 2.709% |
| Asymmetric | M300 | 0.433% | 0.178% |
| Asymmetric | qx | 0.987% | 0.567% |
| Correlated | M200 | 0.046% | 0.532% |
| Correlated | M400 | 0.364% | 5.529% |
| Correlated | stress norm | 0.325% | 2.660% |
| Correlated | M300 | 0.750% | 0.803% |
| Correlated | qx | 2.003% | 3.816% |
| Correlated | M110 | 0.465% | 2.396% |
| Correlated | M210 | 1.677% | 2.030% |
| Leptokurtic | M200 | 0.332% | 0.409% |
| Leptokurtic | M400 | 0.630% | 3.589% |
| Leptokurtic | stress norm | 0.675% | 1.347% |

Mass and momentum drift were zero to printed precision. The largest final
energy drift among the three closure runs was `2.22e-15`. The minimum
degree-two moment-matrix realizability margins were 0.3233, 0.2838, and 0.2106
for the symmetric, asymmetric, and correlated cases, respectively.

## What changed after Stages 2--6

The cubic Fokker--Planck law and the 35 evolved moments remain unchanged. The
failed point-node M5/M6 tail was replaced by a smooth, finite-width
equal-variance Gaussian-mixture reconstruction. A fourth-order tensor
Gauss--Hermite rule represents the reconstructed velocity distribution, and
the same finite collision map used by the particle calculation is applied to
those nodes.

The moment update uses increment form,

```text
M^(n+1) = M^n + M(F_dt[Q(M^n)]) - M(Q(M^n)).
```

This cancels the small input reconstruction residual rather than allowing it
to accumulate as a false source. It was decisive for rotated heat flux: the
correlated-case qx error decreased from 3.090% to 2.003% without relaxing the
acceptance criterion. All 200 macro steps were accepted with no subcycling or
rejection.

## Interpretation and next gate

The mixture closure is not uniformly better for every noisy odd-moment
diagnostic: in the asymmetric axis-aligned test the continuous
single-Gaussian baseline is slightly closer for M300 and qx. The important
gain is the non-Gaussian fourth-order tail: the mixture reduces M400 error from
about 5.5--5.9% to 0.12--0.36%. In the rotated correlated test it also reduces
qx, M110, and M210 errors while keeping every active history below 3%.

These results do **not** close homogeneous validation.  The original time
window ends at only `t/tau = 0.05`, and the symmetric initial distribution is
inside the equal-variance two-Gaussian ansatz. Stage 8 therefore adds
adversarial and long-time gates before any spatial deployment.

The Stage-8 audit found that the unequal-variance location--scale extension
does remove the positive-kurtosis singularity: at fixed `m2=1`, `m4=4.5`, its
sixth moment remains 37.5 as `kappa3 -> 0`, whereas equal-variance EQMOM grows
rapidly and becomes degenerate. However, the current principal-axis tensor
closure is still not general. A nonseparable vector-mean/full-covariance
mixture has 16.062% residual already in its known `M0--M4` moments and 32.202%
in `M5--M6`. Long-time closure-only trajectories also leave the realizability
cone at `t/tau = 2.635`, 2.375, and 2.5025 for the symmetric, asymmetric, and
leptokurtic cases, while the correlated reconstruction fails at 0.8975.

Accordingly, the present deployment status is **NOT READY FOR SPATIAL**. The
next method must use a genuinely multivariate Gaussian-EQMOM/GQMOM-type fit
with vector means and full covariance matrices, followed by a deterministic
velocity-space reference through `t/tau = 2--3` and head-to-head comparisons
with Gaussian-GQMOM, Grad-HyQMOM, and a single-Gaussian floor. The published
HyQMOM transport flux should remain unchanged; only the cubic FP collision
source requires the new sixth-order tail.
