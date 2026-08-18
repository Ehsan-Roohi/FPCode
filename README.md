# FPCode

Research software and reproducible evidence for Fokker–Planck models in
rarefied-gas dynamics.  The repository keeps three related workstreams in one
place while their interfaces and validation gates are still evolving.

## Start here

| Workstream | Location | Current status |
| --- | --- | --- |
| Physics-informed FP solvers | [`FP_PINN/`](FP_PINN/) | 1-D Ornstein–Uhlenbeck baseline, 3-D homogeneous cubic-FP heat-flux/OOD studies, and a Mach-2 shock audit |
| HyQMOM–cubic-FP coupling | [`hyqmom_fp/`](hyqmom_fp/), [`HYQMOM_FP_README.md`](HYQMOM_FP_README.md) | Homogeneous adaptive macro–micro gate passed; the spatial method remains research-stage |
| Riemann35 integration record | [`riemann35_patch/`](riemann35_patch/), [`RIEMANN35_FP_INTEGRATION.md`](RIEMANN35_FP_INTEGRATION.md) | Stages 1–34, with passed, held, and exploratory cases kept explicitly distinct |

The concise decision ledger is [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).
Detailed stage-by-stage evidence is recorded in
[`riemann35_patch/FP_PROJECT_STATE.md`](riemann35_patch/FP_PROJECT_STATE.md).

## Reproduce the lightweight checks

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
python -m compileall -q FP_PINN/pinn hyqmom_fp examples tests
```

TensorFlow-dependent PINN tests skip automatically when TensorFlow is absent.
The production GPU path is exercised in the documented Unity environment;
large checkpoints, scheduler logs, and generated particle fields are not
stored in GitHub.

## Data and provenance

Small reference arrays, summaries, and figures needed to audit reported stage
decisions are versioned alongside the generating scripts.  Large or
regenerable artifacts belong outside the repository.  Result directories use
labels such as `validated`, `hold`, or `development` deliberately; a hold must
not be presented as a passed validation.

## Historical material

The notebooks at the repository root and `FP_PINN/legacy_source/` are retained
for provenance.  They are not part of the tested package, and some snapshots
depend on their original HPC environment or contain incomplete historical
edits.  Use the maintained modules and documented entry points above for new
work.

## Citation and license

Citation metadata and a repository license have not yet been declared.  Add
them only after the authorship, release scope, and licensing choice are
confirmed.
