# Riemann35 cubic-FP Stage 5: residual-cancelled finite map

Stage 4 solved the stiffness problem: all 48 focused Julia tests passed and
the finite map reached `t=0.05` in 200 full steps with no rejection, positive
realizability margin, and mass, momentum, and energy conserved to roundoff.
It did not pass the scientific comparison. The final `M400` error was 97.087%,
versus the 9.702% Gaussian-tail baseline, and the largest equivalent source
norm was `1.91e4`.

The failure was a closure-consistency defect in the map, not finite-step FP
stiffness. `chyqmom_nodes_3d` deliberately truncates six retained high-order
cross moments. Stage 4 replaced the exact incoming 35-vector by moments of
that quadrature for every positive timestep. Consequently the map contained
an order-one reconstruction jump and was discontinuous at `dt = 0`.

Stage 5 evaluates both the unmapped and mapped moments on the same CHyQMOM
quadrature and applies only their difference to the exact incoming state:

```text
M_new = M_exact + (Q_mapped - Q_unmapped).
```

The common quadrature residual cancels, so the collision increment is
`O(dt)` and all 35 retained moments approach the identity continuously. The
focused Julia tests include a timestep-halving continuity check. The Unity job
also records the quadrature residual and the actual collision increment
separately. Operational completion and collision invariants remain hard gates;
the job additionally requires the final `M400` error to beat the 9.702%
Gaussian-tail baseline.

Submit from a checkout of the latest GitHub branch:

```bash
sbatch --export=ALL,JULIA_MODULE=julia/1.10.5 \
  riemann35_patch/run_unity_stage5.sbatch
```

## Unity result

Job `62748157` passed all 52 focused Julia tests but failed at the first
homogeneous macro step.  The residual-cancelled finite candidate for
`dt=2.5e-4` left the realizability cone before any step could be accepted.
The failure is therefore a finite-timestep admissibility failure, not a unit
test, package, scheduler, memory, or conservation failure.  The Stage-5 log
values `alpha=Inf`, zero increment, and zero source norm are only uninitialized
post-failure summaries; the map failed before those diagnostics were recorded.

Stage 6 keeps the residual-cancelled map and retries each complete macro
interval with powers-of-two subcycling.  This tests whether Stage 5 is a
recoverable timestep problem before replacing the moment closure.
