# Stage 19: predictive kinetic-activation sensor calibration

The online sensor compares the already-available Stage-9 and Grad/GQMOM cubic-FP sources. The offline label uses the exact tail of each generating Gaussian mixture and marks a state unsafe only when both algebraic closures exceed the 3% fourth-order source gate. The generating mixture is a controlled audit truth, not a claim of unique identifiability.

The ensemble contains 292 states, of which 263 are unsafe for both algebraic closures.

| Rule | Threshold(s) | Recall | Precision | False-positive rate | Active fraction |
|---|---|---:|---:|---:|---:|
| fourth-source disagreement | d >= 0.098028 | 95.1% | 97.7% | 20.7% | 87.7% |
| source OR tail disagreement | d >= 0.10124 or t >= 0.41005 | 95.1% | 98.0% | 17.2% | 87.3% |

These are in-sample thresholds on the 292-state synthetic Gaussian-mixture ensemble. They are suitable for selecting the sensor form, but not yet publication-level operating thresholds. Held-out DVM/kinetic states and a spatial false-positive/cost study remain required.

Microstate activation is causal: the kinetic state must be inherited from an active neighbor or kinetic inflow, or initialized while a known physical population decomposition is still available. Reconstructing it from the same 35 moments after the alarm would reproduce the non-identifiability exposed by Stage 17.
