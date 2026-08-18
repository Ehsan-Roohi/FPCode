# Stage-2 source and configuration audit for heat-flux G0

Audit date: 2026-08-18

## Recovered production lineage

The historical Stage-2 production was Slurm array job `62716918`, tasks
`0=equilibrium`, `1=stress`, and `2=heat_flux`.  All three tasks completed.
The heat-flux task ran 30,000 optimizer epochs; it was not the earlier
500-epoch plumbing pilot.

The preserved submission transcript gives the exact command-line overrides:

```text
--array=0-2%3
--time=02:00:00
FP_STAGE2_EPOCHS=30000
FP_REFERENCE_PARTICLES=500000
FP_EVALUATION_SAMPLES=65536
```

Every other value came from the Stage-2 source defaults.  The recovered source
is under `FP_PINN/pinn/cubic_homogeneous_3d/` on branch
`codex/fp-pinn-cubic-stage2`; commit
`418c82a26fa6d9644938a784500cb156f455476b` contains the production trainer and
Slurm script used for this reconstruction.  The source defaults and explicit
job overrides are recorded machine-readably in
`stage2_job62716918_reconstructed_config.json`.

The historical heat-flux metric was `0.12210977614349633` (12.21%) against the
particle history.  This is the 30,000-epoch result.  It must not be replaced by
or described as the earlier 500-epoch pilot result.

## Evidence strength

- Direct evidence: the saved submission command, job ID, completed `sacct`
  records, epoch-30,000 log lines, and final metrics.
- Preserved evidence-log SHA-256:
  `ce8e9f554c0395d958bebbe0db0473848d86d1a465b6b039aa31cf6d18cd2f1f`.
- Source evidence: the Stage-2 branch and commit above reproduce every default
  that was not overridden by the saved command.
- Limitation: job `62716918` predates automatic `run_metadata.json`; its log did
  not print the runtime Git hash.  Therefore the commit attribution is a
  high-confidence reconstruction, not a cryptographic runtime proof.

## G0 implementation lineage

G0 is implemented on the separate local branch `work/fp-pinn-heatflux-g0`,
based on `origin/agent/fp-pinn-heatflux-v2` at commit
`c35cc32`.  The main branch is not modified.

Relative to the historical production code, this line adds:

1. the analytic primary target
   `Qx(t)=0.25 exp[-(4/3) nu t]`;
2. a particle-free weak PDE projection onto the third central moment;
3. a particle-free moment-rate residual
   `dQx/dt + (4/3) nu Qx = 0`;
4. an exactly reweighted broad-tail sampling mixture;
5. axisymmetric/antithetic heat-flux quadrature and rotating fixed panels;
6. independent continuation, primary, and publication gates of 5%, 2%, and 1%.

Particle data remain an independent cross-check and marginal-distribution
reference.  They are not used as targets in either heat-flux training loss.
