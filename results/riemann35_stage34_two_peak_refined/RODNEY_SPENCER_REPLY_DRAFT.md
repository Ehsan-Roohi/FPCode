# Joint reply draft — prepared, not sent

Subject: Re: Spatial validation update: causal HyQMOM-FP normal shocks at Mach 2 and 2.5

Rodney and Spencer,

Thank you. I agree that transport and collisions must be tested separately:
loss of realizability in a homogeneous calculation is a collision-closure
failure, not a transport issue. I have therefore completed the collision-only
audit before returning to spatial cases.

I also withdraw the four-delta case as a fair test of a 35-moment closure
truncated at total degree four; resolving four independent streams requires
information through approximately order eight. The replacement is the
official equal-weight, genuinely two-peak `counterstream_ma20` case, evolved
to `t/tau=1`.

Stage 34 passes for that case. Relative L2 errors against the time-refined
positive QMC cubic-FP reference are 0.311% (degree 2), 0.596% (degree 4),
0.911% (degree 6), and 2.703% (degree 8). The aggregate retained M0--M4 error
is 0.573%, and the worst active retained component is M004 at 0.911%. The
8x100,000-particle cross-check differs from QMC by 0.276%, 0.401%, 0.769%, and
2.126% over the same degree blocks. The independent QMC degree-8 node and
time changes are 3.275% and 3.197%, so I regard degree 8 as a diagnostic, not
a calibrated accuracy claim.

The retained H2 margin stays positive at 4.2482e-5 without a limiter, the
necessary H4-PSD margin is 3.19739e-5, and the maximum invariant drift is
2.7e-15. A positive H4 margin is necessary but is not, by itself, proof of
the full multivariate truncated moment problem.

The comparison also shows why the positive Stage-9 reconstruction matters.
The Grad comparator has degree-6 and degree-8 errors of 3.602% and 9.816%.
For the crossing-Ma20 initialization, 4.606% of its quadrature mass is
negative and its necessary H4-PSD margin is -8.131e-3, whereas Stage 9 is
positive and reconstructs the initial M0--M8 sequence to roundoff.

The causal/microstate layer is not a replacement for the hyperbolic moment
solver. HyQMOM remains the transport solver; the microstate is only a
possible collision-side diagnostic or adaptive fallback for unresolved
tail/path information. The modest speedup reported earlier included kinetic
reconstruction/projection overhead and should not be presented as the speed
of the pure 35-moment hyperbolic solver.

For the Mach-2 prototype, the code used tau=1. The mapping is

Kn = rho tau theta^(1-omega).

With omega=0.5 this gives 1 upstream and 3.295 downstream in the code
normalization. Because the physical reference-length convention was not
frozen, I do not regard those as defensible physical Knudsen numbers; that
convention must be fixed before the shock result is used quantitatively.

For the model comparison, BGK and ES-BGK use
`tau_BGK=tau_sigma,ES=tau_FP/2` to match the cubic-FP stress-relaxation clock.
At Pr=2/3, ES-BGK also matches the cubic-FP linear heat-flux rate. The models
then agree on second-moment relaxation but differ materially at high order;
for example, their final M800 values are 102.19 (BGK), 43.43 (ES-BGK), and
175.37 for the positive cubic-FP QMC reference. These are model differences,
not errors against physical truth. This symmetric case has zero heat flux, so
an asymmetric two-peak test is still needed to exercise the Prandtl response.

The roles are therefore clear: BGK is the conservative cost floor; ES-BGK is
the established production comparator; the 35-moment cubic FP is justified
only where it improves nonequilibrium relaxation against independent
physical evidence; particle/QMC is the positive reference for the same cubic
law; and a DVM or causal microstate is a diagnostic/fallback, not the default
solver.

Best,
Ehsan
