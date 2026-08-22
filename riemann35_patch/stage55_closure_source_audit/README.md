# Stage 55: closure-source audit and projected M5/M6 memory

## Why this stage exists

Stage 54 established a converged positive Full-FP QMC reference for Rodney's
oblique nonzero-heat-flux state.  The base positive HyQMOM-35 finite-mixture
map remained realizable and conservative, but its full third-order history
error was about 25%.  A full positive microstate removed that error, but it
never released the microstate and therefore did not qualify a 35-moment
closure.

Stage 55 determines whether the error originates in the instantaneous M5/M6
closure or in the finite-time map, then tests a bounded replacement.  The new
candidate transports the original 35 moments plus 49 raw moments of degrees
five and six.  The M5/M6 memory is relaxed toward a positive two-population
source quadrature after each step.  No velocity microstate is retained.
Positivity of that projection target and H2 realizability of the retained 35
moments are checked separately; this stage does **not** claim that the blended
35+49 sequence is a fully realizable degree-six moment sequence.

This is a qualification experiment, not a predeclared success.  A failed
scientific gate still produces a complete result bundle and a successful
collector job.

## Frozen methods

Six jobs run concurrently:

1. positive Full-FP QMC, 32768 nodes per initial Gaussian component, four
   independent Sobol scramblings, `dt/tau=0.00125`;
2. the Stage-54 positive finite-mixture HyQMOM-35 baseline;
3. unprojected dynamic M5/M6 memory (failure/control branch);
4. projected 35+49 memory with a four-node marginal source quadrature;
5. the same projected memory with five-node marginal source quadrature; and
6. the node-refined candidate at half the time step.

The initial M5/M6 values are evaluated analytically from Rodney's known
regularized Gaussian-mixture construction.  Subsequent updates retain only 49
tail scalars.  The source-local two-population quadrature is rebuilt and
discarded at each step.

## Diagnostics and gates

At every saved QMC state the collector evaluates both the Gaussian
finite-mixture and positive two-population M5/M6 closures on exactly the same
35 moments.  It reports:

- instantaneous error in all ten central third-order source components;
- full M5/M6 reconstruction error;
- full, contracted, and trace-free third-order history errors;
- node and time-step refinement of the projected candidate;
- H2 realizability, positivity, limiter use, and collision invariants.

The qualification objectives remain the Stage-54 objectives: below 3% for the
full third-order tensor, below 5% for its trace-free part, and below 3% for
every normalized component RMSE.  No gate is relaxed to manufacture a pass.

## Outputs

- `stage55_closure_source_summary.json`
- `stage55_history_errors.csv`
- `stage55_source_errors.csv`
- `stage55_third_order_components.png`
- `stage55_source_audit.png`
- `STAGE55_RESULTS.md`
- `STAGE55_CLOSURE_SOURCE_RESULTS_<timestamp>.zip`

Both figures use curve panels only; there are no bar charts and no annotations
over data curves.

## Local test

```bash
python riemann35_patch/stage55_closure_source_audit/test_closure_source_audit.py
```

## Unity submission

Use the exact 40-character commit printed for this stage:

```bash
FP_STAGE55_COMMIT=<commit> bash riemann35_patch/stage55_closure_source_audit/submit_unity_stage55.sh
```
