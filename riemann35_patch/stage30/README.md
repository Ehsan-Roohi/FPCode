# Stage 30: complete front-following lifecycle

Stage 30 doubles the Stage-29 moving-pocket horizon.  It samples inactive-cell
activation every eight steps and audits only active-cell release every four
steps.  It requires both directions of the localized lifecycle in one run: a
causal positive kinetic birth ahead of the
advecting pocket and a release behind it that remains inactive for at least
four subsequent steps.

The split cadence avoids paying for irrelevant inactive-cell closure sensors
when only release needs a denser audit.  The expensive activation closure is
also restricted to cells that possess a causal inflow or active neighbour;
cells without a donor cannot be born and are reported as audited skips.  No
activation or release threshold is retuned.  The front detector retains the
frozen `tail_on = 0.40` value, and release uses the frozen Stage-25 off
thresholds, eight safe observations, and twenty-step minimum kinetic dwell.

Run from the repository root:

```bash
python -m riemann35_patch.stage30.run_front_lifecycle --mode workstation --output results/riemann35_stage30/local
```

Qualification also retains the Stage-29 requirements: causal provenance,
positive and conservative evolution, final and space-time `M400`/`M420`
errors below 3%, peak kinetic support below 50%, and measured speedup over the
same-grid coarse Full-DVM reference.  This is numerical validation of the
implemented cubic FP operator, not independent MD/DSMC physical validation.
