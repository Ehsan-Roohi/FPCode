# Riemann35 cubic-FP Stage 4: finite-map validation

Stage 3 established that the bounded continuous source is still not suitable
for explicit Euler integration. All 36 focused Julia tests passed, but the
homogeneous trajectory failed during the first macro interval with an initial
source norm of `1.38e5` and a requested `h/dt < 2^-24`.

This does not reproduce the algorithm used by FPCode. The particle kernel uses
a finite-step map containing the exact Ornstein--Uhlenbeck factor, capped
nonlinear drift, full finite-step alpha correction, and stochastic variance.

Stage 4 applies that same map to the CHyQMOM quadrature. Each mapped node is
treated as the mean of an isotropic Gaussian carrying the exact noise variance;
the Gaussian raw moments through degree four are accumulated analytically. An
isotropic affine recenter/rescale then enforces the same collision invariants as
the homogeneous particle validator. No explicit source microstepping is used.

The job repeats the common seeded 100,000-particle, 200-step comparison and
records alpha, the largest node `|c|^2/theta`, realizability, invariants, history
errors, and the final M400 difference. The raw and continuous-bounded functions
remain in the patch chain as documented negative controls.

Unity job `62747589` passed all 48 focused Julia tests and reached final time
in 200 full finite steps with no rejection. The minimum accepted realizability
margin was `6.48e-4`; mass, momentum, and energy drift were zero to roundoff.
However, the final `M400` error was 97.087% and the largest equivalent source
norm was `1.91e4`. The finite map was therefore operationally stable but
scientifically rejected. The cause was an order-one CHyQMOM reconstruction
jump in six retained cross moments for every positive timestep. Stage 5
cancels that common reconstruction residual before applying the collision
increment.

Submit from a checkout of the latest GitHub branch:

```bash
sbatch --export=ALL,JULIA_MODULE=julia/1.10.5 \
  riemann35_patch/run_unity_stage4.sbatch
```
