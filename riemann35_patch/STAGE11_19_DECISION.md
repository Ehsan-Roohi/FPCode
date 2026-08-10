# Cubic-FP / HyQMOM-35 decision record: Stages 11--20

## Outcome

The rare-beam discrepancy is not primarily a time-step, limiter, clipping, or realizability problem. It is a structural closure-identifiability problem: moments through degree four do not uniquely determine the fifth- and sixth-order tail required by the cubic Fokker--Planck source.

No instantaneous algebraic closure tested here meets the 3% `M400` history gate. A positive kinetic microstate with memory does meet the gate. Stage 20 now adds causal positive projection and a non-chattering activation/release lifecycle, so this is the selected direction for the spatial coupling stage. It is not yet a spatial production method.

## Evidence

| Stage | Test | Main quantitative result | Decision |
|---|---|---|---|
| 11 | 16 independent seeds x 100k particles to `t/tau=1` | Stage 9 / Grad rare-beam `M400` errors: 16.59% / 12.67%; halving `dt` does not remove the bias | accuracy gap confirmed |
| 12 | two-population, residual, persistent, dynamic M6/M8 | best candidate is dynamic M8 at 15.19% | reject all |
| 13 | finite Hermite truncations K=6,8 | full 35-moment realizability becomes negative near `t/tau=0.034`, also at half `dt` | reject as physical reference |
| 14 | positive weighted QMC kinetic reference | fine node/time differences about 0.6%; four-scramble spread 0.64% for rare-beam | accept as one side of reference envelope |
| 15 | 8x200k particle refinement | particle--QMC rare-beam history difference remains about 1% | keep a conservative two-reference envelope |
| 16 | positive discrete maximum entropy | stable and realizable, but `M400` error 19.6--22.2% | reject |
| 17 | constructive identifiability LP | on an explicitly documented compact support, identical 35 moments permit `M600` to vary 8.83% and self-consistent `dM400/dt` from -127.5 to +24.2 | instantaneous 35-to-tail map is non-unique |
| 18 | positive persistent micro-solver sizing | 2048 nodes: 1.67% vs particles, 1.10% vs QMC, 2.32% scramble spread | proof-of-concept passes 3% gate |
| 19 | predictive closure-disagreement sensor | 95.1% recall, 98.0% precision, but 87.3% of the deliberately broad 292-state ensemble activates | useful alarm; not yet an economical production sensor |
| 20 | causal positive projection and hysteretic lifecycle on all six Stage-11 histories | rare-beam `M400`: 1.66% vs particles, 1.12% vs QMC, 2.33% scramble spread; no blocked activations or chatter; equal-case micro fraction 16.7% | homogeneous lifecycle gate passes |

## Interpretation

The Stage-17 witnesses are nonnegative distributions on a common velocity support. Their 35 retained moments agree to approximately `1e-12` or better, yet their cubic-FP fourth-moment production differs even in sign. This proves that choosing a two-Gaussian, Grad/GQMOM, or maximum-entropy ansatz is an additional modeling assumption; exactness cannot follow from the 35 numbers alone.

The LP range is an explicit inner witness range on 1024 fixed support centers, not a claim of a global extremum. The standardized support box and maximum speed are recorded in the Stage-17 JSON. On the unrestricted velocity domain the sixth-moment supremum is generally unbounded with moments only through degree four fixed. Replacing every atom by a common Maxwellian kernel of variance `1e-3 theta` and re-solving changes the `M600` range only from 8.83% to 8.67%, so the result is not an atomic-measure artifact.

Freezing the 9x9 FP coefficients separates direct tail ambiguity from coefficient feedback. For the `M600` witness pair, the frozen source span is 100.9 and the self-consistent span is 151.7. Coefficient feedback therefore contributes 33.5% of the self-consistent span and causes the sign reversal, but a large direct-tail ambiguity remains. Regularizing the coefficient solve is useful for robustness; it cannot restore identifiability.

The useful method contribution is therefore not another universal algebraic `M6` formula. The defensible direction is a realizability-aware adaptive HyQMOM--kinetic method:

1. HyQMOM-35 remains the inexpensive transport model in regular cells.
2. A disagreement/high-skew sensor activates a positive persistent velocity microstate before tail information is lost.
3. The kinetic microstate evaluates the cubic-FP source and retains the missing history.
4. Projection to 35 moments is conservative and realizable by construction.
5. Hysteresis controls activation/deactivation so that cells do not chatter between representations.

Stage 18 establishes accuracy only for homogeneous rare-beam relaxation initialized from a known two-population VDF. It does not solve microstate initialization from an arbitrary 35-moment state, nor spatial transport of the microstate.

The microstate must therefore be causal and inherited, not reconstructed after the alarm from the same 35 moments. Admissible birth paths are: inheritance from an already active neighboring cell, injection from a kinetic inflow boundary, or activation while a physically known population decomposition is still available. If none is available, the method must mark the state as tail-ambiguous and use a documented conservative prior; it must not describe that prior as recovered information.

Stage 19 tests the cheapest online proxy for the Stage-17 range: disagreement between the Stage-9 and Grad/GQMOM fourth-order sources. On the deliberately difficult 292-state Gaussian-mixture ensemble, the best source-or-tail rule reaches 95.1% recall and 98.0% precision, but activates 87.3% of states. This is scientifically useful because it demonstrates predictive value, but computationally it says the broad random-mixture family lies mostly outside both algebraic models. Production thresholds must instead be calibrated on held-out kinetic/DVM trajectories representative of the target flow, then assessed by missed-activation rate, false-positive rate, and active-cell fraction.

Stage 20 makes the sensor and lifecycle solver-facing. The initial source-disagreement rule produced a delayed false alarm on the exactly symmetric counter-stream trajectories even though their Stage-9 `M400` errors are about 0.2%. Requiring nonzero standardized skewness for the source-disagreement branch removes that target-family false alarm; the tail-disagreement branch remains independent so symmetric high-kurtosis states are not categorically suppressed. Retrospectively, this gate changes the Stage-19 synthetic-ensemble recall from 95.1% to 94.7%, precision from 98.0% to 98.4%, false-positive rate from 17.2% to 13.8%, and active fraction from 87.3% to 86.6%. This disclosed refinement is calibrated on the six homogeneous trajectories and must not be presented as a universal production threshold.

The macro-to-micro map positively reweights a causal velocity support by a discrete entropy projection and matches all 35 transported moments. The maximum relative projection residual in the four-replicate rare-beam audit is `5.46e-11`; the reverse map is direct positive quadrature. Rare-beam error falls from 16.59% to 1.66% against particles and 1.12% against the positive QMC reference, while the four-scramble spread is 2.33%. Only rare-beam is active over `t/tau <= 1`, giving an equal-case active fraction of 16.7%. A separate safe causal-birth control executes exactly one micro-to-macro release at `t/tau=0.2`, so both sides of the state machine are exercised.

The anisotropic rare-hot trajectory remains macro: its Stage-9 discrepancy is 4.45% against particles but only 1.64% against the positive QMC reference, consistent with the Stage-11 warning that the particle statistic has large rare-population variance. This case should not be used to retune the activation threshold until the independent DVM reference resolves that reference conflict.

## Steelman scope of the eliminated candidates

The negative table is evidence only because each candidate and its limitation are stated precisely.

| Stage | Strong version actually tested | What the rejection does and does not establish |
|---|---|---|
| 12 | exact-at-initial-time rank-one two-population reconstruction; algebraic, residual-corrected, persistent, dynamic-`M6`, and dynamic-`M8` variants; full `t/tau=1` histories | rejects these five finite-memory constructions for the rare-beam gate; does not reject all multivariate mixture dynamics |
| 13 | zero-tail Hermite moment systems at `K=6` and `K=8`, SSP-RK2, including half time step | rejects this naive signed finite truncation as a physical reference; does not reject a well-resolved positive-filtered spectral/DVM solver |
| 14 | positive Sobol/QMC kinetics with exact component weights, corrected mean/covariance, seeded permutation, node/time refinement, and four scrambles | supplies one positive reference family, but shares the same finite collision map as the particles |
| 15 | 16x100k and 8x200k independent particle ensembles | finds no evidence that the particle--QMC gap is removed by doubling particles per seed; does not bound common finite-map bias |
| 16 | positive discrete maximum entropy on adaptive supports of 128, 432, and 1024 nodes; all 35 constraints matched and full-time realizability retained | rejects these documented support/regularization choices at 19.6--22.2%; it does not imply that maximum entropy is mathematically invalid or universally inaccurate |

The particle and QMC references quantify sampling and velocity-discretization variation within the same finite collision-map family. They do not bound a shared time-map bias. A conservative DVM or independently discretized positive spectral solver is therefore still a publication gate, especially because the Stage-18 margin below 3% is not wide compared with its scramble spread.

## Literature position

Stage 17 is a computational truncated-moment construction, not a new invention of the moment problem. Its mathematical context is the flat-extension theory of Curto--Fialkow, compact semialgebraic moment results of Schmuedgen, and moment/SOS bounds of Lasserre and Bertsimas--Popescu. Junk's analysis of the domain of Levermore moment systems and the realizability criterion of Levermore--Morokoff--Nadiga connect that mathematics to kinetic closures.

The hybrid direction also has established predecessors: dynamically localized kinetic--fluid methods, moment-guided Monte Carlo, and realizability-based hybrid hierarchies. The proposed contribution must therefore be stated narrowly: a rare, positive tail-memory correction on top of HyQMOM-35; cubic-FP source coupling; a constructive reason why memory is necessary; and quantitative sensor/activation evidence. It is not the generic idea of kinetic--fluid domain decomposition.

Primary references used for this positioning:

1. R. Curto and L. Fialkow, *Flat extensions of positive moment matrices*, Memoirs AMS 136(648), 1998, DOI [10.1090/memo/0648](https://doi.org/10.1090/memo/0648).
2. K. Schmuedgen, *The K-moment problem for compact semi-algebraic sets*, Math. Ann. 289 (1991), 203--206.
3. J. B. Lasserre, *Global optimization with polynomials and the problem of moments*, SIAM J. Optim. 11 (2001), 796--817, DOI [10.1137/S1052623400366802](https://doi.org/10.1137/S1052623400366802).
4. D. Bertsimas and I. Popescu, *Optimal inequalities in probability theory: A convex optimization approach*, SIAM J. Optim. 15 (2005), 780--804, DOI [10.1137/S1052623401399903](https://doi.org/10.1137/S1052623401399903).
5. M. Junk, *Domain of definition of Levermore's five-moment system*, J. Stat. Phys. 93 (1998), 1143--1167, DOI [10.1023/B:JOSS.0000033155.07331.d9](https://doi.org/10.1023/B:JOSS.0000033155.07331.d9).
6. C. D. Levermore, W. J. Morokoff, and B. T. Nadiga, *Moment realizability and the validity of the Navier--Stokes equations for rarefied gas dynamics*, Phys. Fluids 10 (1998), 3214--3226, DOI [10.1063/1.869849](https://doi.org/10.1063/1.869849).
7. P. Degond, G. Dimarco, and L. Mieussens, *A multiscale kinetic--fluid solver with dynamic localization of kinetic effects*, JCP 229 (2010), 4907--4933, DOI [10.1016/j.jcp.2010.03.009](https://doi.org/10.1016/j.jcp.2010.03.009).
8. P. Degond, G. Dimarco, and L. Pareschi, *The moment-guided Monte Carlo method*, IJNMF 67 (2011), 189--213, DOI [10.1002/fld.2345](https://doi.org/10.1002/fld.2345).
9. F. Filbet and T. Rey, *A hierarchy of hybrid numerical methods for multiscale kinetic equations*, SIAM J. Sci. Comput. 37 (2015), DOI [10.1137/140958773](https://doi.org/10.1137/140958773).

## Next implementation gate

The homogeneous hysteresis/projection gate has passed. The next stage is a one-dimensional shock/relaxation coupling with transported microstates and explicit donor selection. A JCP claim should wait for:

- confirmation of the documented sensor/hysteresis thresholds on held-out spatial trajectories;
- conservative face transport and neighbor/inflow donor selection in addition to the homogeneous bidirectional projection;
- accuracy below 3% on the homogeneous target family;
- a spatial comparison against particle/DVM data;
- an independently discretized positive DVM/spectral reference that does not share the particle/QMC finite map;
- cost and active-cell-fraction measurements against pure HyQMOM and BGK/FP baselines.

The publication-level novelty would then be the adaptive tail-memory coupling for cubic FP inside HyQMOM-35, supported by the constructive non-identifiability result—not the Gaussian-mixture ansatz by itself.
