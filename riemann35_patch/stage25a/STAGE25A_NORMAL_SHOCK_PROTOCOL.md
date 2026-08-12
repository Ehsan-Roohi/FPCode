# Stage 25A - Frozen one-dimensional normal-shock protocol

Protocol frozen before the qualification run. Development smoke results do
not define or modify any threshold below.

## Scientific question

Can the Stage-24 cubic Fokker-Planck collision method transport unresolved
tail memory through a spatial Mach-3 normal shock while remaining causal,
positive, conservative, more economical than a full DVM, and accurate against
that full kinetic reference?

## Physical problem

- Shock-fixed, one-dimensional physical domain; three-dimensional velocity.
- Monatomic gas, `gamma=5/3`, upstream `rho=1`, `theta=1`, `Ma=3`.
- Downstream Maxwellian follows the exact Rankine-Hugoniot density, momentum,
  and total-enthalpy relations.
- Fixed Maxwellian inflow distributions at both ends. The initial condition is
  the corresponding piecewise-Maxwellian discontinuity.
- Cubic FP collision controls remain `tau=1` and `Pr=2/3`.

## Compared methods

1. 35-moment macro solver with positive kinetic (BFL-like) face fluxes and the
   retained Stage-9 finite-mixture collision map.
2. Adaptive causal macro-micro solver. Active cells transport positive DVM
   masses using shared conservative upwind fluxes.
3. Full positive DVM reference in every spatial cell, with the manufactured-
   qualified Stage-24 Scharfetter-Gummel collision step and minimum-KL target
   projection.

The adaptive solver pre-activates only the known initial shock interface.
Subsequent macro-to-micro birth requires an active neighbour or physical
inflow donor. A temporary positive representation may supply an incoming
flux from a sensor-safe macro cell, but it is never retained as kinetic memory
and cannot satisfy a birth request.

## Frozen sensor and lifecycle

- Source on/off: `0.10124 / 0.05062`.
- Tail on/off: `0.40 / 0.205025`.
- Skew on/off: `0.001 / 0.0005`.
- Activation hold: one sensor evaluation.
- Release hold: eight safe evaluations.
- Minimum active duration: 20 steps.

No threshold may be changed after any spatial history is inspected.

## Qualification discretization

- Physical domain: `-20 <= x/lambda_1 <= 20`, 160 uniform cells.
- Velocity domain: `[-12,14] x [-10,10] x [-10,10]`.
- Base velocity grid: `61 x 33 x 33` cells.
- Transport CFL: `0.35`.
- 2400 base steps. The shock-location drift and profile residual must be
  reported; a longer run may only extend this frozen trajectory.

Reference qualification additionally requires a physical-grid refinement, a
velocity-grid refinement, a time-step refinement, and a velocity-domain
widening. The final successive change for every gating profile must be below
1%. The smoke grid is not one of these reference levels.

## Frozen observables and decisions

Primary profiles are density, streamwise velocity, temperature, normal
stress, heat flux, `M300`, `M400`, and the complete 15-component degree-four
block. The directed adaptive-to-DVM profile discrepancy must be below 3% for
every primary profile and the combined degree-four block.

The spatial gate also requires:

- nonnegative DVM cell masses and realizable 35-moment states throughout;
- mass, momentum, and energy balance residuals below `2e-8`;
- micro/macro face-flux synchronization residual below `2e-8`;
- zero unphysical births and zero persistent representation chatter;
- no sensor or threshold retuning on the Mach-3 history;
- mean kinetic active fraction below 50%; and
- at least a 2x wall-time reduction relative to the like-for-like full DVM.

Accuracy with activation nearly everywhere is reported as an accurate kinetic
method but fails the hybrid-economy gate. Passing this Stage 25A gate permits,
but does not replace, the subsequent three-dimensional crossing-jet and
external DSMC/direct-Boltzmann validation.
