# Stage 30 frozen local result

This directory archives the accepted 48-cell, 48-step workstation
qualification produced on 2026-08-14.

- `STAGE30_FRONT_LIFECYCLE_RESULT.md` is the compact report.
- `stage30_front_lifecycle_summary.json` contains configuration, gates,
  timing, birth provenance, and release provenance.
- `stage30_front_lifecycle_histories.npz` contains float32 `M400`/`M420`
  profile histories, the active mask, and birth counts.  The solver and every
  reported gate were evaluated using the full 35-moment float64 state, with
  exact qualification metrics retained in the JSON file.
- `stage30_front_lifecycle.png` shows the physical profiles and the complete
  front-following support history, including release gaps behind the front.

The run passed all causal, lifecycle, numerical, localization, accuracy, and
measured-performance gates.  It is local numerical evidence for the
implemented cubic FP operator, not independent physical validation or a
portable cluster benchmark.
