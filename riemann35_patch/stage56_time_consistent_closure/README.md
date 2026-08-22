# Stage 56: time-consistent degree-six qualification

## Decision being tested

Stage 55 transported the 35 retained moments plus all 49 raw M5/M6 values,
but halving the time step increased the full third-tensor error from about
6% to 10% and the trace-free error from about 11% to 18%.  It also requested
a five-node refinement while the post-step projection remained hard-coded to
four nodes.

Stage 56 is a frozen stop gate for that 84-scalar branch.  It does not tune a
relaxation time and it does not launch a spatial or shock calculation.

## Corrections

- the requested node count is used by both the M7/M8 source quadrature and the
  positive M5/M6 projection target;
- every raw moment through total degree six receives the exact isotropic
  Ornstein--Uhlenbeck map;
- the nonlinear cubic-FP source uses SSPRK2;
- positive-tail relaxation is applied with symmetric Strang splitting;
- the 20x20 degree-three moment matrix is checked after every substep.

Positive semidefiniteness of H3 is necessary but not sufficient for a full
degree-six truncated-moment realizability proof.  The documentation and plots
therefore report it only as a necessary condition.

## Frozen array

Six CPU tasks compare four time steps and two node-refinement pairs:

| Method | dt/tau | marginal nodes |
|---|---:|---:|
| q4_dt2500 | 0.0025 | 4 |
| q5_dt2500 | 0.0025 | 5 |
| q5_dt1250 | 0.00125 | 5 |
| q5_dt0625 | 0.000625 | 5 |
| q5_dt03125 | 0.0003125 | 5 |
| q6_dt0625 | 0.000625 | 6 |

The converged Stage-55 QMC archive is copied into the result bundle with its
SHA-256 provenance.  It is not recomputed.

## Qualification gates

- all six runs complete;
- QMC scramble spread below 3%;
- positive projection weights and collision invariants;
- H3 margin at least `-1e-10`;
- no material H3 limiter (`minimum limiter >= 0.999`);
- finest time and node changes below 1% for both the full and trace-free
  third-order tensors;
- selected full third-tensor error below 3%;
- selected trace-free error below 5%;
- every normalized component RMSE below 3%.

The collector always returns a complete ZIP.  If any gate fails, the 35+49
branch stops and the next admissible model is a positive compressed cubature.

## Local test

```bash
python -m riemann35_patch.stage56_time_consistent_closure.test_time_consistent
```

## Unity

```bash
FP_STAGE56_COMMIT=<40-character-commit> bash riemann35_patch/stage56_time_consistent_closure/submit_unity_stage56.sh
```
