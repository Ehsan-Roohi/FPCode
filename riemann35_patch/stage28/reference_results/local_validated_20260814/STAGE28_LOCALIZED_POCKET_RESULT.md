# Stage 28 localized kinetic-pocket result

- Decision: **QUALIFICATION_PASS**
- Mean/peak kinetic fraction: 16.667% / 16.667%
- Final adaptive M400 error vs refined DVM: 0.209240%
- Final adaptive M420 error vs refined DVM: 0.497421%
- Space-time adaptive M400/M420 errors: 0.129364% / 0.468946%
- Sensor evaluations: 144/2304 (6.250%)
- Exact/near-exact Maxwellian collision shortcuts: 380
- Adaptive/coarse-DVM wall-time ratio: 0.876x
- Measured speedup: 1.142x
- Minimum positive DVM mass: 1.491e-13
- Maximum finite-volume balance residual: 7.412e-14
- Maximum micro/macro sync residual: 3.842e-15

The initially retained pocket and buffer come from known positive DVM data.
Skipped sensor intervals do not advance release counters or invent births.
This qualifies numerical behavior for the implemented cubic FP operator;
independent MD/DSMC validation is still required for physical fidelity.
