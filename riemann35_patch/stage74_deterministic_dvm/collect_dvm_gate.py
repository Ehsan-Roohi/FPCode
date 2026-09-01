#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (
    central_third_components,
    irreducible_decomposition,
    relative_history_error,
    symmetric_tensor,
)
from riemann35_patch.stage71_harder_unseen.hard_cases import hard_case

CASES = ("rare_beam_3d", "dense_hot_extreme", "dilute_broad")


def _derived(history: np.ndarray):
    components = central_third_components(history)
    q, _, tf = irreducible_decomposition(components)
    return {"components": components, "heat_flux": q, "trace_free": tf, "full_tensor": symmetric_tensor(components)}


def _invmax(d: dict) -> float:
    return max(float(d["maximum_mass_drift"]), float(d["maximum_momentum_drift"]), float(d["maximum_energy_trace_drift"]))


def _evaluate(root: Path, name: str):
    npz = np.load(root / f"stage74_{name}.npz")
    summary = json.loads((root / f"stage74_{name}_summary.json").read_text())
    times = np.asarray(npz["times"], float)
    coarse = _derived(np.asarray(npz["dvm_coarse_histories"], float))
    fine = _derived(np.asarray(npz["dvm_fine_histories"], float))
    closure = _derived(np.asarray(npz["closure_histories"], float))
    pr = 2.0 / 3.0
    exact_q = fine["heat_flux"][0][None, :] * np.exp(-2.0 * pr * times[:, None])
    errors = {
        "dvm_refinement_third": relative_history_error(coarse["full_tensor"], fine["full_tensor"]),
        "dvm_refinement_heat_flux": relative_history_error(coarse["heat_flux"], fine["heat_flux"]),
        "closure_third": relative_history_error(closure["full_tensor"], fine["full_tensor"]),
        "closure_trace_free": relative_history_error(closure["trace_free"], fine["trace_free"]),
        "closure_heat_flux": relative_history_error(closure["heat_flux"], fine["heat_flux"]),
        "dvm_exact_heat_flux": relative_history_error(fine["heat_flux"], exact_q),
        "closure_exact_heat_flux": relative_history_error(closure["heat_flux"], exact_q),
    }
    case = hard_case(name)
    controls = summary["controls"]
    dc = summary["dvm_coarse"]
    df = summary["dvm_fine"]
    ci = summary["closure"]["invariants"]
    gates = {
        "frozen_case_fingerprint": summary["stage71_fingerprint"] == case.fingerprint,
        "no_qmc_reference": controls.get("qmc_used") is False,
        "density_jacobian_fix_declared": controls.get("density_jacobian_fix") is True,
        "no_closure_refit": controls.get("closure_parameter_refit") is False,
        "dvm_positive": min(float(dc["minimum_cell_mass"]), float(df["minimum_cell_mass"])) > 0.0,
        "dvm_initial_projection": max(float(dc["initial_projection_relative_residual"]), float(df["initial_projection_relative_residual"])) < 2.0e-10,
        "dvm_step_projection": max(float(dc["maximum_step_projection_relative_residual"]), float(df["maximum_step_projection_relative_residual"])) < 2.0e-10,
        "dvm_invariants": max(_invmax(dc), _invmax(df)) < 2.0e-8,
        "closure_invariants": _invmax(ci) < 2.0e-8,
        "dvm_refinement_third": errors["dvm_refinement_third"] < 0.03,
        "closure_third": errors["closure_third"] < 0.03,
        "closure_trace_free": errors["closure_trace_free"] < 0.05,
        "closure_heat_flux": errors["closure_heat_flux"] < 0.03,
        "dvm_exact_heat_flux": errors["dvm_exact_heat_flux"] < 0.03,
        "closure_exact_heat_flux": errors["closure_exact_heat_flux"] < 1.0e-6,
    }
    return {"case": name, "pass": all(gates.values()), "gates": gates, "errors": errors, "fine_grid": df["grid"], "fine_seconds": df["elapsed_seconds"], "closure_seconds": summary["closure"]["diagnostics"]["elapsed_seconds"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    a = p.parse_args()
    missing = [n for n in CASES if not (a.root / f"stage74_{n}.npz").is_file()]
    results = {n: _evaluate(a.root, n) for n in CASES if n not in missing}
    passed = not missing and all(r["pass"] for r in results.values())
    summary = {
        "schema": "riemann35-stage74-deterministic-dvm-gate-v1",
        "scientific_scope": "deterministic clean-room DVM validation of the density-consistent persistent four-Gaussian homogeneous cubic-FP closure",
        "reference": "guided positive Scharfetter-Gummel DVM; no QMC",
        "frozen_cases": list(CASES),
        "missing_cases": missing,
        "qualification_pass": passed,
        "prospective_thresholds": {
            "dvm_refinement_third": 0.03,
            "closure_third": 0.03,
            "closure_trace_free": 0.05,
            "closure_heat_flux": 0.03,
            "dvm_exact_heat_flux": 0.03,
            "closure_exact_heat_flux": 1.0e-6,
            "invariants": 2.0e-8,
            "projection": 2.0e-10,
        },
        "case_results": results,
    }
    (a.root / "stage74_dvm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (a.root / "stage74_case_metrics.csv").open("w", newline="") as f:
        fields = ["case", "pass", "dvm_refinement_third", "dvm_refinement_heat_flux", "closure_third", "closure_trace_free", "closure_heat_flux", "dvm_exact_heat_flux", "closure_exact_heat_flux"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for n in CASES:
            if n not in results: continue
            r = results[n]
            w.writerow({"case": n, "pass": r["pass"], **r["errors"]})
    lines = [
        "# Stage 74 — deterministic DVM validation",
        "",
        f"Qualification objective: **{'PASS' if passed else 'FAIL'}**",
        "",
        "No QMC reference is used. Cases and thresholds were fixed before the DVM runs.",
        "",
        "| case | DVM refine third | closure vs DVM third | closure trace-free | closure q | DVM q vs exact | closure q vs exact | result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for n in CASES:
        r = results.get(n)
        if r is None:
            lines.append(f"| {n} | -- | -- | -- | -- | -- | -- | FAIL |")
            continue
        e = r["errors"]
        lines.append(f"| {n} | {100*e['dvm_refinement_third']:.3f}% | {100*e['closure_third']:.3f}% | {100*e['closure_trace_free']:.3f}% | {100*e['closure_heat_flux']:.3f}% | {100*e['dvm_exact_heat_flux']:.3f}% | {100*e['closure_exact_heat_flux']:.6f}% | {'PASS' if r['pass'] else 'FAIL'} |")
    (a.root / "STAGE74_RESULTS.md").write_text("\n".join(lines) + "\n")
    a.bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.root.rglob("*")):
            if path.is_file() and path.resolve() != a.bundle.resolve():
                z.write(path, arcname=path.relative_to(a.root))
    digest = hashlib.sha256(a.bundle.read_bytes()).hexdigest()
    a.bundle.with_name(a.bundle.name + ".sha256.txt").write_text(f"{digest}  {a.bundle.name}\n")
    print(f"[stage74] qualification_pass={passed} bundle={a.bundle}", flush=True)

if __name__ == "__main__":
    main()
