# Stage 32 development notes

The Stage-31 Mach-2 case was used only for development.  No Stage-33 Mach-2.5
run was performed.

## Selected candidate B

Candidate B keeps the legacy mass front signal instantaneous and
bidirectional.  Stress-, heat-flux-, and `M420`-weighted signals are eligible
only in the direction of the active donor's mean velocity.  Their one-sided
growth is projected over the already-frozen 20-step minimum kinetic dwell;
the activation threshold remains `0.40`.

The full run produced two causal downstream births and removed the Stage-31
shock-core defect.  Stress, core heat flux, core `M420`, and full `M420` fell
to 2.55%, 2.32%, 0.73%, and 0.44%, respectively.  Full-domain heat flux
remained 4.41%, so the result is `DEVELOPMENT_HOLD`.

## Rejected diagnostics

- Candidate A applied all weighted signals instantaneously in both directions.
  Its full run had zero downstream births, one chatter event, 5.83% stress,
  9.28% heat flux, and 3.95% core `M420` error.
- Extending the lookahead to the 32-step release-observation horizon propagated
  support farther downstream but caused immediate release/rebirth chatter.
- Reusing `source_off=0.05062` as a collision shortcut worsened full/core heat
  flux to 4.71% / 3.49%.
- Exploratory causal carrier-flux diagnostics reduced full heat flux to 3.34%,
  3.19%, and finally 3.08% for different predeclared halo interpretations,
  but none passed every 3% gate.  No fourth halo was tried because that would
  amount to tuning directly against the acceptance boundary.  These were
  adaptive-only diagnostics against the frozen refined final reference, not
  qualification runs, and their experimental code was not retained.

The next method change must address conservative macro far-field transport on
an independently specified basis, not by lowering the profile limit or adding
another halo until Mach 2 passes.
