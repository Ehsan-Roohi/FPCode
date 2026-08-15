# Stage 31: held-out Mach-2 normal shock

Stage 31 applies the complete frozen Stage-30 causal kinetic lifecycle to a
normal shock that was not used in the Stage-25A Mach-3 threshold campaign.
The upstream Mach number is 2; sensor thresholds, activation/release holds,
minimum dwell, sensor cadences, and the 3% profile limit are unchanged.
The initial interface keeps the Stage-25A frozen two-cell half-width.

Unlike the uniform-background moving-pocket tests, a shock has distinct
known upstream and downstream Maxwellians.  Each potential birth therefore
uses the positive initial carrier belonging to its target cell.  The carrier
field is fixed at the initial time and cannot import future information.

The workstation gate compares density, streamwise velocity, temperature,
normal stress, heat flux, `M300`, retained `M400`, and predictive `M420`
against independently evolved positive coarse/refined Full-DVM solutions. It
also requires causal provenance, positive masses, conservation,
realizability, localized kinetic support, no four-step release/rebirth
chatter, and measured speedup over the same-grid coarse Full-DVM.

Run from the repository root:

```bash
python -m riemann35_patch.stage31.run_heldout_shock --mode workstation --output results/riemann35_stage31/local
```

This is numerical cross-case validation of the implemented cubic FP model,
not independent DSMC, direct-Boltzmann, or experimental validation.
