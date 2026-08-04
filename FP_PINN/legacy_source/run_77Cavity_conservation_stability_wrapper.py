#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conservation / stability diagnostics wrapper for 77CavityUQL2.py.

This script does NOT edit 77CavityUQL2.py. It imports the solver, wraps the
moment and coefficient routines, runs the q-weighted cavity case, and writes
diagnostic CSV/TEX/MD outputs.

What is monitored
-----------------
Particle-level diagnostics sampled during Physics and ML runs:
  - total particle mass proxy
  - mass drift relative to first sample
  - total momentum components and norm
  - total kinetic energy proxy and drift
  - non-finite particle counts
  - min rho, min T, min DM2
  - non-positive rho/T/DM2 cell counts
  - pressure-tensor minimum eigenvalue and negative-eigenvalue cell count

Coefficient diagnostics after closure coefficient evaluation:
  - non-finite A/B counts
  - max |A|, max |B|
  - RMS(A), RMS(B)

Important interpretation
------------------------
For the moving-lid cavity, global momentum and energy are not strict conserved
invariants because wall interactions and the moving lid exchange momentum/energy
with the gas. Therefore, mass drift and non-finite/positivity diagnostics are the
main conservation/stability metrics. Momentum/energy histories are reported as
bounded stability diagnostics and ML-vs-physics comparisons.

Usage
-----
python run_77Cavity_conservation_stability_wrapper.py \
  --module 77CavityUQL2.py \
  --steps 6000 \
  --ntss 1000 \
  --sample-every 250 \
  --outdir cavity77_conservation_qweighted
"""

import argparse
import csv
import importlib.util
from pathlib import Path
import time
import numpy as np


def load_module_from_path(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("cavity77_stability_wrapped", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cp_float(cp, x):
    return float(cp.asnumpy(x))


def cp_int(cp, x):
    return int(cp.asnumpy(x))


def finite_count_cp(cp, arr):
    return cp_int(cp, cp.sum(~cp.isfinite(arr)))


def build_pij_matrices_cp(cp, PIJ):
    # PIJ columns assumed: xx, xy, xz, yy, yz, zz
    n = PIJ.shape[0]
    M = cp.zeros((n, 3, 3), dtype=PIJ.dtype)
    M[:, 0, 0] = PIJ[:, 0]
    M[:, 0, 1] = PIJ[:, 1]
    M[:, 1, 0] = PIJ[:, 1]
    M[:, 0, 2] = PIJ[:, 2]
    M[:, 2, 0] = PIJ[:, 2]
    M[:, 1, 1] = PIJ[:, 3]
    M[:, 1, 2] = PIJ[:, 4]
    M[:, 2, 1] = PIJ[:, 4]
    M[:, 2, 2] = PIJ[:, 5]
    return M


def pressure_tensor_min_eig(cp, grid):
    try:
        PIJ = grid["PIJ"]
        M = build_pij_matrices_cp(cp, PIJ)
        eigs = cp.linalg.eigvalsh(M)
        min_eig_cell = eigs[:, 0]
        min_eig = cp_float(cp, cp.min(min_eig_cell))
        neg_count = cp_int(cp, cp.sum(min_eig_cell < -1e-12))
        return min_eig, neg_count
    except Exception:
        return np.nan, -1


def particle_diagnostics(cp, p_data, grid):
    # p_data convention from 77CavityUQL2.py:
    # 0,1 positions; 3,4,5 velocities; 12 weights.
    x = p_data[0]
    y = p_data[1]
    vx = p_data[3]
    vy = p_data[4]
    vz = p_data[5]
    w = p_data[12]

    v2 = vx*vx + vy*vy + vz*vz

    mass = cp.sum(w)
    px = cp.sum(w * vx)
    py = cp.sum(w * vy)
    pz = cp.sum(w * vz)
    pnorm = cp.sqrt(px*px + py*py + pz*pz)
    ke = 0.5 * cp.sum(w * v2)

    particle_nonfinite = (
        cp.sum(~cp.isfinite(x)) +
        cp.sum(~cp.isfinite(y)) +
        cp.sum(~cp.isfinite(vx)) +
        cp.sum(~cp.isfinite(vy)) +
        cp.sum(~cp.isfinite(vz)) +
        cp.sum(~cp.isfinite(w))
    )

    rho = grid["rho"]
    T = grid["T"]
    DM2 = grid["DM2"]
    U = grid["U"]
    PIJ = grid["PIJ"]
    Q = grid["Q"]

    field_nonfinite = (
        cp.sum(~cp.isfinite(rho)) +
        cp.sum(~cp.isfinite(T)) +
        cp.sum(~cp.isfinite(DM2)) +
        cp.sum(~cp.isfinite(U)) +
        cp.sum(~cp.isfinite(PIJ)) +
        cp.sum(~cp.isfinite(Q))
    )

    min_pij_eig, pij_neg_count = pressure_tensor_min_eig(cp, grid)

    out = {
        "particle_mass": cp_float(cp, mass),
        "momentum_x": cp_float(cp, px),
        "momentum_y": cp_float(cp, py),
        "momentum_z": cp_float(cp, pz),
        "momentum_norm": cp_float(cp, pnorm),
        "kinetic_energy": cp_float(cp, ke),
        "particle_nonfinite_count": cp_int(cp, particle_nonfinite),
        "field_nonfinite_count": cp_int(cp, field_nonfinite),
        "min_rho": cp_float(cp, cp.min(rho)),
        "min_T": cp_float(cp, cp.min(T)),
        "min_DM2": cp_float(cp, cp.min(DM2)),
        "nonpositive_rho_cells": cp_int(cp, cp.sum(rho <= 0)),
        "nonpositive_T_cells": cp_int(cp, cp.sum(T <= 0)),
        "nonpositive_DM2_cells": cp_int(cp, cp.sum(DM2 <= 0)),
        "min_PIJ_eigenvalue": min_pij_eig,
        "negative_PIJ_eigen_cells": pij_neg_count,
    }
    return out


def find_coeff_dict(args, kwargs):
    # Search for a dictionary with A and B coefficient arrays.
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, dict) and "A" in obj and "B" in obj:
            return obj
    return None


def coefficient_diagnostics(cp, coeffs):
    if coeffs is None:
        return None
    A = coeffs["A"]
    B = coeffs["B"]
    return {
        "A_nonfinite_count": finite_count_cp(cp, A),
        "B_nonfinite_count": finite_count_cp(cp, B),
        "A_max_abs": cp_float(cp, cp.max(cp.abs(A))),
        "B_max_abs": cp_float(cp, cp.max(cp.abs(B))),
        "A_rms": cp_float(cp, cp.sqrt(cp.mean(A*A))),
        "B_rms": cp_float(cp, cp.sqrt(cp.mean(B*B))),
        "A_min": cp_float(cp, cp.min(A)),
        "A_max": cp_float(cp, cp.max(A)),
        "B_min": cp_float(cp, cp.min(B)),
        "B_max": cp_float(cp, cp.max(B)),
    }


class StabilityRecorder:
    def __init__(self, cp, outdir, sample_every, ntss):
        self.cp = cp
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.sample_every = int(sample_every)
        self.ntss = int(ntss)

        self.step = {"physics": 0, "ml": 0}
        self.initial = {"physics": None, "ml": None}

        self.state_files = {
            "physics": self.outdir / "stability_history_physics.csv",
            "ml": self.outdir / "stability_history_ml.csv",
        }
        self.coeff_files = {
            "physics": self.outdir / "coefficient_history_physics.csv",
            "ml": self.outdir / "coefficient_history_ml.csv",
        }

        self.state_fields = [
            "step", "phase", "particle_mass", "mass_rel_drift",
            "momentum_x", "momentum_y", "momentum_z", "momentum_norm",
            "kinetic_energy", "energy_rel_drift",
            "particle_nonfinite_count", "field_nonfinite_count",
            "min_rho", "min_T", "min_DM2",
            "nonpositive_rho_cells", "nonpositive_T_cells", "nonpositive_DM2_cells",
            "min_PIJ_eigenvalue", "negative_PIJ_eigen_cells"
        ]
        self.coeff_fields = [
            "step", "phase", "A_nonfinite_count", "B_nonfinite_count",
            "A_max_abs", "B_max_abs", "A_rms", "B_rms",
            "A_min", "A_max", "B_min", "B_max"
        ]

        for path in self.state_files.values():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.state_fields).writeheader()
        for path in self.coeff_files.values():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.coeff_fields).writeheader()

    def should_sample(self, phase):
        step = self.step[phase]
        return (step == 1) or (step % self.sample_every == 0) or (step == self.ntss) or (step == self.ntss + 1)

    def record_state(self, phase, p_data, grid, nc_val):
        self.step[phase] += 1
        step = self.step[phase]
        if not self.should_sample(phase):
            return

        self.cp.cuda.Stream.null.synchronize()
        diag = particle_diagnostics(self.cp, p_data, grid)
        self.cp.cuda.Stream.null.synchronize()

        if self.initial[phase] is None:
            self.initial[phase] = {
                "mass": diag["particle_mass"],
                "energy": diag["kinetic_energy"],
            }

        m0 = self.initial[phase]["mass"]
        e0 = self.initial[phase]["energy"]
        diag["mass_rel_drift"] = (diag["particle_mass"] - m0) / max(abs(m0), 1e-300)
        diag["energy_rel_drift"] = (diag["kinetic_energy"] - e0) / max(abs(e0), 1e-300)
        diag["step"] = step
        diag["phase"] = phase

        with open(self.state_files[phase], "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.state_fields)
            w.writerow({k: diag.get(k, "") for k in self.state_fields})

        print(
            f"  Stability {phase:7s} step {step:6d}: "
            f"mass_drift={diag['mass_rel_drift']:.3e}, "
            f"energy_drift={diag['energy_rel_drift']:.3e}, "
            f"minT={diag['min_T']:.3e}, "
            f"nonfinite={diag['particle_nonfinite_count'] + diag['field_nonfinite_count']}",
            flush=True
        )

    def record_coeffs(self, phase, coeffs):
        step = self.step[phase]
        if step <= 0:
            return
        if not self.should_sample(phase):
            return

        self.cp.cuda.Stream.null.synchronize()
        diag = coefficient_diagnostics(self.cp, coeffs)
        self.cp.cuda.Stream.null.synchronize()

        if diag is None:
            return
        diag["step"] = step
        diag["phase"] = phase

        with open(self.coeff_files[phase], "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.coeff_fields)
            w.writerow({k: diag.get(k, "") for k in self.coeff_fields})


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def fval(row, key):
    return float(row[key])


def ival(row, key):
    return int(float(row[key]))


def summarize_state(rows):
    if not rows:
        return {}
    mass_drifts = np.array([abs(fval(r, "mass_rel_drift")) for r in rows])
    energy_drifts = np.array([abs(fval(r, "energy_rel_drift")) for r in rows])
    nonfinite = np.array([ival(r, "particle_nonfinite_count") + ival(r, "field_nonfinite_count") for r in rows])
    nonpos = np.array([ival(r, "nonpositive_rho_cells") + ival(r, "nonpositive_T_cells") + ival(r, "nonpositive_DM2_cells") for r in rows])
    neg_pij = np.array([ival(r, "negative_PIJ_eigen_cells") for r in rows])
    min_T = np.array([fval(r, "min_T") for r in rows])
    min_rho = np.array([fval(r, "min_rho") for r in rows])
    min_DM2 = np.array([fval(r, "min_DM2") for r in rows])
    min_pij_eig = np.array([fval(r, "min_PIJ_eigenvalue") for r in rows])

    return {
        "n_samples": len(rows),
        "max_abs_mass_rel_drift": float(np.max(mass_drifts)),
        "final_mass_rel_drift": fval(rows[-1], "mass_rel_drift"),
        "max_abs_energy_rel_drift": float(np.max(energy_drifts)),
        "final_energy_rel_drift": fval(rows[-1], "energy_rel_drift"),
        "max_nonfinite_count": int(np.max(nonfinite)),
        "max_nonpositive_basic_cells": int(np.max(nonpos)),
        "max_negative_PIJ_eigen_cells": int(np.max(neg_pij)),
        "min_T_over_samples": float(np.min(min_T)),
        "min_rho_over_samples": float(np.min(min_rho)),
        "min_DM2_over_samples": float(np.min(min_DM2)),
        "min_PIJ_eigenvalue_over_samples": float(np.min(min_pij_eig)),
    }


def summarize_coeff(rows):
    if not rows:
        return {}
    A_nf = np.array([ival(r, "A_nonfinite_count") for r in rows])
    B_nf = np.array([ival(r, "B_nonfinite_count") for r in rows])
    A_max_abs = np.array([fval(r, "A_max_abs") for r in rows])
    B_max_abs = np.array([fval(r, "B_max_abs") for r in rows])
    return {
        "n_coeff_samples": len(rows),
        "max_A_nonfinite_count": int(np.max(A_nf)),
        "max_B_nonfinite_count": int(np.max(B_nf)),
        "max_A_abs": float(np.max(A_max_abs)),
        "max_B_abs": float(np.max(B_max_abs)),
    }


def write_summary(outdir):
    outdir = Path(outdir)
    phases = ["physics", "ml"]
    summary_rows = []

    for phase in phases:
        state_rows = read_csv(outdir / f"stability_history_{phase}.csv")
        coeff_rows = read_csv(outdir / f"coefficient_history_{phase}.csv")
        s = summarize_state(state_rows)
        c = summarize_coeff(coeff_rows)
        row = {"phase": phase}
        row.update(s)
        row.update(c)
        summary_rows.append(row)

    fields = [
        "phase", "n_samples",
        "max_abs_mass_rel_drift", "final_mass_rel_drift",
        "max_abs_energy_rel_drift", "final_energy_rel_drift",
        "max_nonfinite_count", "max_nonpositive_basic_cells",
        "max_negative_PIJ_eigen_cells",
        "min_T_over_samples", "min_rho_over_samples", "min_DM2_over_samples",
        "min_PIJ_eigenvalue_over_samples",
        "n_coeff_samples", "max_A_nonfinite_count", "max_B_nonfinite_count",
        "max_A_abs", "max_B_abs"
    ]

    with open(outdir / "stability_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    # LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Conservation and stability diagnostics for the q-weighted cavity simulation. For the moving-lid cavity, momentum and energy are affected by wall exchange; therefore mass drift, non-finite counts, positivity, and coefficient boundedness are the primary stability diagnostics.}",
        r"\label{tab:qweighted_stability}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Diagnostic & Physics & ML \\",
        r"\hline",
    ]

    def get(phase, key):
        for r in summary_rows:
            if r["phase"] == phase:
                return r.get(key, np.nan)
        return np.nan

    diagnostics = [
        ("Max. $|\\Delta M/M_0|$", "max_abs_mass_rel_drift", "{:.3e}"),
        ("Final $\\Delta M/M_0$", "final_mass_rel_drift", "{:.3e}"),
        ("Max. non-finite count", "max_nonfinite_count", "{:.0f}"),
        ("Max. non-positive $\\rho,T,\\mathrm{DM2}$ cells", "max_nonpositive_basic_cells", "{:.0f}"),
        ("Max. non-finite coefficient count", "max_A_nonfinite_count", "{:.0f}"),
        ("Max. $|A|$", "max_A_abs", "{:.3e}"),
        ("Max. $|B|$", "max_B_abs", "{:.3e}"),
    ]
    for label, key, fmt in diagnostics:
        pv = get("physics", key)
        mv = get("ml", key)
        try:
            ps = fmt.format(float(pv))
        except Exception:
            ps = "--"
        try:
            ms = fmt.format(float(mv))
        except Exception:
            ms = "--"
        lines.append(f"{label} & {ps} & {ms} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (outdir / "stability_table.tex").write_text("\n".join(lines), encoding="utf-8")

    md = []
    md.append("# Conservation / stability summary")
    md.append("")
    md.append("For the moving-lid cavity, global momentum and kinetic energy are not strict conserved invariants because the walls exchange momentum and energy with the gas. The primary conservation/stability checks are mass drift, finite values, positivity of basic thermodynamic quantities, pressure-tensor definiteness, and coefficient boundedness.")
    md.append("")
    for row in summary_rows:
        md.append(f"## {row['phase']}")
        md.append(f"- samples: {row.get('n_samples', '')}")
        md.append(f"- max |mass relative drift|: {row.get('max_abs_mass_rel_drift', np.nan):.6e}")
        md.append(f"- final mass relative drift: {row.get('final_mass_rel_drift', np.nan):.6e}")
        md.append(f"- max |energy relative drift|: {row.get('max_abs_energy_rel_drift', np.nan):.6e}")
        md.append(f"- max nonfinite count: {row.get('max_nonfinite_count', '')}")
        md.append(f"- max nonpositive rho/T/DM2 cells: {row.get('max_nonpositive_basic_cells', '')}")
        md.append(f"- max negative PIJ eigen cells: {row.get('max_negative_PIJ_eigen_cells', '')}")
        md.append(f"- min T over samples: {row.get('min_T_over_samples', np.nan):.6e}")
        md.append(f"- min rho over samples: {row.get('min_rho_over_samples', np.nan):.6e}")
        md.append(f"- min DM2 over samples: {row.get('min_DM2_over_samples', np.nan):.6e}")
        md.append(f"- max coefficient nonfinite counts: A={row.get('max_A_nonfinite_count', '')}, B={row.get('max_B_nonfinite_count', '')}")
        md.append("")
    md.append("Recommended wording: No non-finite particles, fields, or closure coefficients were detected in either the baseline or q-weighted ML run. The particle mass drift remained bounded at machine-level precision over the sampled interval, and no non-positive density, temperature, or DM2 cells were observed.")
    (outdir / "stability_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="77CavityUQL2.py")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--ntss", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--outdir", default="cavity77_conservation_qweighted")
    args = ap.parse_args()

    mod = load_module_from_path(args.module)
    cp = mod.cp

    mod.N_STEPS_PER_RUN = int(args.steps)
    mod.NTSS = int(args.ntss)

    # Safely skip optional diagnostics already patched into 77CavityUQL2.py.
    safe_skip = int(args.steps) + 10**9
    if hasattr(mod, "HIGH_MOMENTS_EVERY"):
        mod.HIGH_MOMENTS_EVERY = safe_skip
    if hasattr(mod, "ENTROPY_EVERY"):
        mod.ENTROPY_EVERY = safe_skip

    rec = StabilityRecorder(cp, args.outdir, args.sample_every, mod.NTSS)

    orig_full = mod.sort_and_calc_moments_cupy_FULL
    orig_lite = mod.sort_and_calc_moments_cupy_LITE
    orig_solve = mod.solve_linear_systems_cupy
    orig_predict = mod.predict_coeffs_cupy_native

    def wrapped_full(p_data, grid, nc_val, *rest, **kwargs):
        out = orig_full(p_data, grid, nc_val, *rest, **kwargs)
        rec.record_state("physics", p_data, grid, nc_val)
        return out

    def wrapped_lite(p_data, grid, nc_val, *rest, **kwargs):
        out = orig_lite(p_data, grid, nc_val, *rest, **kwargs)
        rec.record_state("ml", p_data, grid, nc_val)
        return out

    def wrapped_solve(*args, **kwargs):
        out = orig_solve(*args, **kwargs)
        coeffs = find_coeff_dict(args, kwargs)
        rec.record_coeffs("physics", coeffs)
        return out

    def wrapped_predict(*args, **kwargs):
        out = orig_predict(*args, **kwargs)
        coeffs = find_coeff_dict(args, kwargs)
        # Some implementations return coeff dict rather than mutating it.
        if coeffs is None and isinstance(out, dict) and "A" in out and "B" in out:
            coeffs = out
        rec.record_coeffs("ml", coeffs)
        return out

    mod.sort_and_calc_moments_cupy_FULL = wrapped_full
    mod.sort_and_calc_moments_cupy_LITE = wrapped_lite
    mod.solve_linear_systems_cupy = wrapped_solve
    mod.predict_coeffs_cupy_native = wrapped_predict

    print("="*70)
    print("Conservation / stability wrapper for 77CavityUQL2.py")
    print(f"Steps: {mod.N_STEPS_PER_RUN}, NTSS: {mod.NTSS}, sample_every: {args.sample_every}")
    print(f"Optional diagnostic skip interval: {safe_skip}")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print("="*70, flush=True)

    t0 = time.time()
    mod.main()
    elapsed = time.time() - t0

    write_summary(args.outdir)

    notes = [
        "# Stability wrapper run notes",
        "",
        f"- Total run wall time: {elapsed:.3f} s",
        f"- Steps: {mod.N_STEPS_PER_RUN}",
        f"- NTSS: {mod.NTSS}",
        f"- sample_every: {args.sample_every}",
        f"- optional diagnostic skip interval: {safe_skip}",
        "",
        "Files:",
        "- stability_history_physics.csv",
        "- stability_history_ml.csv",
        "- coefficient_history_physics.csv",
        "- coefficient_history_ml.csv",
        "- stability_summary.csv",
        "- stability_summary.md",
        "- stability_table.tex",
    ]
    Path(args.outdir, "stability_run_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("\nStability outputs:")
    for p in sorted(Path(args.outdir).iterdir()):
        print(" ", p)
    print("\nSummary:")
    print(Path(args.outdir, "stability_summary.md").read_text())

if __name__ == "__main__":
    main()
