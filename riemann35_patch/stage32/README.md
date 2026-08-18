# Stage 32: direction-aware causal precursor

Stage 32 treats the Stage-31 Mach-2 shock only as a development case.  It
keeps the frozen Stage-25/30 threshold, sensor cadences, initial support,
release holds, minimum dwell, positive carrier birth, and all numerical
contracts.  Stage 31 remains `WORKSTATION_HOLD`.

The new precursor evaluates four normalized incoming half-range discrepancies
from the same already-active positive DVM donor and the target cell's fixed
positive carrier: mass, normal-stress weight, heat-flux weight, and predictive
`M420` weight.  Each signal is bounded by two and uses the unchanged `0.40`
front threshold.  The added weighted signals are admitted only along the
active donor's mean-flow direction.  Their one-sided causal growth is
extrapolated over the already-frozen 20-step minimum kinetic dwell; this adds
no fitted threshold or independent horizon.  No inactive-cell tail is
reconstructed from its 35 moments.

A development pass requires at least one downstream (`left_neighbor`) front
birth and at least one birth for which a new weighted signal crosses `0.40`
while the legacy mass signal remains below it.  All Stage-31 physical-profile,
causality, positivity, conservation, realizability, localization, chatter, and
measured-performance gates also remain active.

Run from the repository root:

```bash
python -m riemann35_patch.stage32.run_direction_aware_precursor --mode workstation --output results/riemann35_stage32/local
```

Mach 2.5 is reserved for Stage 33 and is not executed here.  The Stage-32 rule
must be frozen before that blind cross-case test.

The selected local run is archived under
`reference_results/local_hold_20260815`.  It achieved two causal downstream
births, no chatter, a 1.503x measured speedup, and sub-3% stress and shock-core
errors, but full-domain heat flux remained 4.41%.  Its decision is therefore
`DEVELOPMENT_HOLD`; the rule is not frozen and Stage 33 remains untouched.
