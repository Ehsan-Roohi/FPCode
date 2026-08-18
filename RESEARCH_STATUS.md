# Research status

Last repository audit: 2026-08-18.

This page separates verified software checks from scientific qualification.
Passing unit tests means the implementation contracts below are reproducible;
it does not turn a development or held physical case into a validated result.

## Active conclusions

- The positive 1-D Ornstein–Uhlenbeck PINN is the exact-solution baseline.
- The homogeneous 3-D cubic-FP track includes heat-flux, anisotropic-stress,
  combined, and OOD initial states.  Its NumPy/reference contracts are tested;
  full TensorFlow training remains a GPU/Unity workflow.
- In the HyQMOM-35 track, moments through degree four do not uniquely identify
  the fifth- and sixth-order tail required by the cubic-FP source.  The selected
  research direction therefore retains a causal positive kinetic microstate
  when the macro closure is insufficient.
- Stage 30 is the last fully passed local spatial qualification in the
  Riemann35 sequence.
- Stage 31 is a `WORKSTATION_HOLD` and Stage 32 is a `DEVELOPMENT_HOLD`; neither
  is a passed blind validation.  Mach 2.5 remains reserved.
- The Stage-34 two-peak material is an audit/model-comparison record, not an
  independent production validation.

## Publication gates still open

1. Reproduce relevant GPU runs from a clean Unity environment and archive
   machine-readable configuration/provenance.
2. Add an independently discretized positive DVM or spectral reference that
   does not share the particle/QMC finite collision map.
3. Validate spatial profiles against independent DSMC/DVM evidence over a
   declared Mach/Knudsen test matrix.
4. Report accuracy, conservation, positivity, active-cell fraction, and cost
   against pure HyQMOM and BGK/FP baselines.
5. Freeze a held-out protocol before running the reserved blind case.

## Repository contracts

- Maintained Python modules compile without the historical `legacy_source`
  snapshots.
- The lightweight suite runs with NumPy, SciPy, Matplotlib, and pytest.
- TensorFlow-specific tests skip, rather than silently passing, when TensorFlow
  is unavailable.
- Large checkpoints, particle dumps, flow fields, and scheduler logs remain
  external and must be connected through generation or retrieval instructions.
