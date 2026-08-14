# Stage 29 advecting causal kinetic-front result

- Decision: **WORKSTATION_PASS**
- Causal births / kinetic-front births: 4 / 4
- Mean/peak/final kinetic fraction: 16.833% / 20.833% / 20.833%
- Final adaptive M400 error vs refined DVM: 0.229996%
- Final adaptive M420 error vs refined DVM: 0.428031%
- Space-time adaptive M400/M420 errors: 0.109924% / 0.640520%
- Expensive/front sensor evaluations: 144 / 48
- Adaptive/coarse-DVM wall-time ratio: 0.801x
- Measured speedup: 1.248x
- Maximum finite-volume balance residual: 3.451e-15
- Maximum micro/macro sync residual: 1.841e-13

Every new front cell inherited a positive upwind carrier--donor proposal
from a neighbour active at the start of the step.  No tail was invented
from the 35 retained moments.  Independent MD/DSMC validation remains
necessary before making a physical-fidelity claim.
