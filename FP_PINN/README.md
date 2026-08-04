# Physics-Informed Fokker-Planck Solver

This directory hosts the next-generation physics-informed extension of the GPU-native cubic Fokker-Planck solver published in the *Journal of Computational Physics*.

## Planned scope

- Positive kinetic representation for the velocity distribution function
- Strong and weak Fokker-Planck residuals
- Differentiable cubic-FP moment closure
- Conservation and kinetic boundary-condition enforcement
- Validation using homogeneous relaxation, Couette/Fourier flow, normal shocks, and a lid-driven cavity
- GPU and SLURM workflows for UMass Unity

## Repository policy

Only source code, lightweight configurations, small verification inputs, and scripts belong in this repository. Large training datasets, particle dumps, flow-field outputs, checkpoints, and cluster logs remain outside GitHub and are referenced through reproducible generation or download instructions.

## Unity source

The legacy working directory is:

```text
/project/pi_roohie_umass_edu/fokkerplanckDeeponet
```

Selected source files will be synchronized into `FP_PINN/legacy_source/` after review.