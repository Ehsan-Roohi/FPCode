# Riemann35 cubic-FP Stage 3: bounded homogeneous validation

Stage 2 established that the unclipped analytical M5 + CHyQMOM-M6 source is
not merely limited by the original 256-substep cap. Near `t=0.025` its source
norm reached `5.18e10`, a realizable adaptive trajectory required
`h/dt < 2^-24`, and the integration stopped before the next macro sample. The
five collision invariants remained at roundoff and the accepted realizability
margin remained positive, identifying a high-moment runaway in the raw source.

Stage 3 retains that raw source as a diagnostic control and validates a separate
bounded source. The new source:

1. evaluates the nonlinear drift directly on the non-negative CHyQMOM nodes;
2. applies FPCode's `min(|c|^2, 25*theta)` limiter;
3. subtracts the mean nonlinear drift to preserve momentum; and
4. applies `-lambda*c`, where `lambda=<c dot N>/(3*theta)`, the continuous-time
   limit of FPCode's finite-step alpha energy rescaling.

Direct quadrature is intentional: the clipped radial terms are non-polynomial
and cannot be represented exactly by a finite M5/M6 raw-moment lookup. The
retained/evolved state remains the same 35 moments through total degree four.

The Unity job first runs the focused Riemann35 source tests, then repeats the
same seeded 100,000-particle, 200-step comparison used in Stage 2. It writes
aligned histories, adaptive-step metrics, and `summary.json` under
`results/riemann35_stage3_JOBID/`. Completion, realizability, time alignment,
and mass/momentum/energy conservation are hard gates; moment-history accuracy
against the particle reference remains a reported diagnostic.

Unity job `62746158` passed all 36 focused Julia tests but rejected the
continuous bounded trajectory during the first macro interval. The equivalent
source norm was `1.38e5`; eight microsteps were accepted, 25 rejected, and the
next required `h/dt` fell below `2^-24`. The accepted margin remained positive
and all invariants were exact. Thus the bounded continuous function is retained
as a negative control, while Stage 4 tests FPCode's actual finite-step map
rather than forcing that map into an explicit-Euler source interpretation.

Submit from the FPCode branch root on Unity:

```bash
sbatch --export=ALL,JULIA_MODULE=julia/1.10.5 \
  riemann35_patch/run_unity_stage3.sbatch
```
