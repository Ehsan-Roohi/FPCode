# Stage 27 spatial causal-donor result

- Decision: **WORKSTATION_PASS**
- Full causal activation reached at step: 3
- Safe blocked attempts before donor arrival: 8
- Initial adaptive M420 ambiguity/error: 0.585792%
- Final adaptive M400 error vs refined DVM: 0.157269%
- Final adaptive M420 error vs refined DVM: 0.293717%
- Post-activation space-time M400 error: 0.127506%
- Post-activation space-time M420 error: 0.275062%
- Coarse/refined final M420 difference: 0.321873%
- Adaptive/coarse-DVM wall-time ratio: 7.525x
- Minimum positive DVM mass: 7.862e-35
- Maximum finite-volume balance residual: 1.056e-13
- Maximum micro/macro sync residual: 4.461e-12
- Workstation wall time: 72.581 s

Blocked attempts are expected and safe: an alarm is denied until a causal donor
arrives.  A just-born cell cannot relay memory again during the same step.
This strongly non-equilibrium case becomes 100% kinetic, so the current
unoptimized adaptive path provides accuracy but no wall-time saving.
The result qualifies this implementation numerically; it is not independent
MD/DSMC evidence for physical fidelity.
