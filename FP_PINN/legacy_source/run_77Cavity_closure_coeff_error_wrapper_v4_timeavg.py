#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct closure-coefficient error time-history audit for 77CavityUQL2.py.

Purpose
-------
This script measures the DNN closure-coefficient error directly, in time, for the
2D cavity. At sampled post-transient steps along the PHYSICS trajectory, it computes:

    Physics coefficients:
        FULL moments -> 9x9 cubic-FP system -> solve -> [A_ij, B_i]

    ML coefficients on the exact same cell state:
        low-order features from the same grid -> GPU DNN forward pass -> [A_ij, B_i]

This isolates the closure-map error itself. It is stronger and cleaner than comparing
only final macroscopic fields, because it directly answers:
    "What is the time evolution of closure coefficient errors?"

Outputs
-------
outdir/
  closure_coeff_error_history.csv
  closure_coeff_error_components_long.csv
  closure_coeff_error_summary.csv
  closure_coeff_error_summary.tex
  closure_coeff_error_components.tex
  closure_coeff_error_time_history.pdf/png
  closure_coeff_scaled_rmse_time_history.pdf/png
  closure_coeff_component_relL2_time_history.pdf/png
  coefficient_audit_metadata.json
  coefficient_audit_notes.md

Typical run
-----------
python -u run_77Cavity_closure_coeff_error_wrapper.py \
  --module 77CavityUQL2.py \
  --model model_params_robust_cavity_qweighted_for_cupy.npz \
  --target-kn 1.0 \
  --base-kn 0.15 \
  --steps 12000 \
  --ntss 2000 \
  --ppc 1000 \
  --audit-every 50 \
  --outdir cavity77_coeff_error_Kn1p0
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np


COEFF_NAMES = [
    "A11", "A12", "A13", "A22", "A23", "A33",
    "B1", "B2", "B3",
]


def load_module_from_path(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("cavity77_coeff_audit", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def cp_to_float(x):
    try:
        return float(x.get())
    except Exception:
        return float(x)


def robust_rel_l2(cp, pred, ref, eps=1e-300):
    num = cp.linalg.norm(pred - ref)
    den = cp.linalg.norm(ref)
    return 100.0 * cp_to_float(num / cp.maximum(den, eps))


def robust_symmetric_rel_l2(cp, a, b, eps=1e-300):
    """Symmetric relative L2, useful for split-half noise-floor estimates."""
    num = cp.linalg.norm(a - b)
    den = cp.linalg.norm(0.5 * (a + b))
    return 100.0 * cp_to_float(num / cp.maximum(den, eps))


def robust_rmse(cp, x):
    return cp_to_float(cp.sqrt(cp.mean(x * x)))


def robust_mean_abs(cp, x):
    return cp_to_float(cp.mean(cp.abs(x)))


def robust_max_abs(cp, x):
    return cp_to_float(cp.max(cp.abs(x)))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def make_latex_summary(outdir, summary_rows, comp_rows):
    outdir = Path(outdir)

    # Compact block summary table.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Time-resolved closure-coefficient audit for the cavity. At each sampled post-transient step, the ML closure coefficients are compared against the cubic-FP coefficients obtained from the full moment system on the same physics-trajectory cell state.}")
    lines.append(r"\label{tab:cavity_closure_coeff_time_error}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Coefficient block & Mean $\varepsilon_{L_2}$ (\%) & Max. $\varepsilon_{L_2}$ (\%) & Final $\varepsilon_{L_2}$ (\%) \\")
    lines.append(r"\midrule")

    labels = {
        "total": r"All coefficients $[A_{ij},B_i]$",
        "A": r"Drift tensor block $A_{ij}$",
        "B": r"Heat-flux block $B_i$",
    }
    for r in summary_rows:
        block = r["block"]
        lines.append(
            f"{labels.get(block, block)} & "
            f"{float(r['mean_relL2_percent']):.3f} & "
            f"{float(r['max_relL2_percent']):.3f} & "
            f"{float(r['final_relL2_percent']):.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (outdir / "closure_coeff_error_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per-coefficient component table.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-component closure-coefficient errors for the cavity coefficient audit. The scaled RMSE is computed in the standardized output space used by the trained network.}")
    lines.append(r"\label{tab:cavity_closure_coeff_component_errors}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Coefficient & Mean $\varepsilon_{L_2}$ (\%) & Max. $\varepsilon_{L_2}$ (\%) & Mean scaled RMSE \\")
    lines.append(r"\midrule")

    latex_names = {
        "A11": r"$A_{11}$",
        "A12": r"$A_{12}$",
        "A13": r"$A_{13}$",
        "A22": r"$A_{22}$",
        "A23": r"$A_{23}$",
        "A33": r"$A_{33}$",
        "B1": r"$B_{1}$",
        "B2": r"$B_{2}$",
        "B3": r"$B_{3}$",
    }

    # comp_rows is already summarized one row per coefficient.
    for r in comp_rows:
        c = r["coefficient"]
        lines.append(
            f"{latex_names.get(c, c)} & "
            f"{float(r['mean_relL2_percent']):.3f} & "
            f"{float(r['max_relL2_percent']):.3f} & "
            f"{float(r['mean_scaled_rmse']):.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (outdir / "closure_coeff_error_components.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(outdir, history_rows, comp_long_rows):
    outdir = Path(outdir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (outdir / "plot_warning.txt").write_text(
            f"Matplotlib unavailable; CSV and TeX tables were still written.\n{e}\n",
            encoding="utf-8"
        )
        return

    steps = np.array([int(r["step"]) for r in history_rows], dtype=float)
    total = np.array([float(r["relL2_total_percent"]) for r in history_rows], dtype=float)
    A = np.array([float(r["relL2_A_percent"]) for r in history_rows], dtype=float)
    B = np.array([float(r["relL2_B_percent"]) for r in history_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(steps, total, label=r"all $[A_{ij},B_i]$")
    ax.plot(steps, A, label=r"$A_{ij}$ block")
    ax.plot(steps, B, label=r"$B_i$ block")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Relative $L_2$ coefficient error (%)")
    ax.set_title("Closure-coefficient error time history")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "closure_coeff_error_time_history.png", dpi=300)
    fig.savefig(outdir / "closure_coeff_error_time_history.pdf")
    plt.close(fig)

    scaled = np.array([float(r["scaled_rmse_all"]) for r in history_rows], dtype=float)
    scaled_max = np.array([float(r["scaled_max_abs_all"]) for r in history_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(steps, scaled, label="scaled RMSE")
    ax.plot(steps, scaled_max, label="max scaled abs.")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Error in standardized coefficient space")
    ax.set_title("Standardized closure-coefficient error")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "closure_coeff_scaled_rmse_time_history.png", dpi=300)
    fig.savefig(outdir / "closure_coeff_scaled_rmse_time_history.pdf")
    plt.close(fig)

    # Per-component relL2 history.
    by_coeff = {c: [] for c in COEFF_NAMES}
    for r in comp_long_rows:
        by_coeff[r["coefficient"]].append((int(r["step"]), float(r["relL2_percent"])))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for c in COEFF_NAMES:
        vals = sorted(by_coeff[c])
        if not vals:
            continue
        xs = np.array([v[0] for v in vals], dtype=float)
        ys = np.array([v[1] for v in vals], dtype=float)
        ax.plot(xs, ys, label=c)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Per-component relative $L_2$ error (%)")
    ax.set_title("Per-component closure-coefficient error")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "closure_coeff_component_relL2_time_history.png", dpi=300)
    fig.savefig(outdir / "closure_coeff_component_relL2_time_history.pdf")
    plt.close(fig)


def make_cumulative_plots(outdir, avg_history_rows):
    outdir = Path(outdir)
    if not avg_history_rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (outdir / "cumulative_plot_warning.txt").write_text(
            f"Matplotlib unavailable; cumulative CSV and TeX tables were still written.\n{e}\n",
            encoding="utf-8"
        )
        return

    steps = np.array([int(r["step"]) for r in avg_history_rows], dtype=float)

    total = np.array([float(r["scaled_cumulative_relL2_total_percent"]) for r in avg_history_rows], dtype=float)
    A = np.array([float(r["scaled_cumulative_relL2_A_percent"]) for r in avg_history_rows], dtype=float)
    B = np.array([float(r["scaled_cumulative_relL2_B_percent"]) for r in avg_history_rows], dtype=float)
    noise = np.array([float(r["physics_split_half_total_percent"]) for r in avg_history_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(steps, total, label=r"ML vs physics, all $[A_{ij},B_i]$")
    ax.plot(steps, A, label=r"ML vs physics, $A_{ij}$")
    ax.plot(steps, B, label=r"ML vs physics, $B_i$")
    if np.any(np.isfinite(noise)):
        ax.plot(steps, noise, "--", label="physics split-half noise floor")
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Cumulative-average $\varepsilon_{L_2}^{(s)}$ (%)")
    ax.set_title("Time-averaged closure-coefficient audit")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "closure_coeff_cumulative_average_relL2_time_history.png", dpi=300)
    fig.savefig(outdir / "closure_coeff_cumulative_average_relL2_time_history.pdf")
    plt.close(fig)

    rmse = np.array([float(r["scaled_cumulative_rmse_all"]) for r in avg_history_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(steps, rmse, label="cumulative-average scaled RMSE")
    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE in standardized coefficient space")
    ax.set_title("Cumulative-average standardized coefficient error")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "closure_coeff_cumulative_average_scaled_rmse.png", dpi=300)
    fig.savefig(outdir / "closure_coeff_cumulative_average_scaled_rmse.pdf")
    plt.close(fig)


def compute_summaries(history_rows, comp_long_rows):
    def summarize_block(block, key):
        vals = [float(r[key]) for r in history_rows]
        return {
            "block": block,
            "mean_relL2_percent": float(np.mean(vals)) if vals else np.nan,
            "max_relL2_percent": float(np.max(vals)) if vals else np.nan,
            "final_relL2_percent": float(vals[-1]) if vals else np.nan,
        }

    summary_rows = [
        summarize_block("total", "relL2_total_percent"),
        summarize_block("A", "relL2_A_percent"),
        summarize_block("B", "relL2_B_percent"),
    ]

    comp_rows = []
    for c in COEFF_NAMES:
        rows = [r for r in comp_long_rows if r["coefficient"] == c]
        rel = [float(r["relL2_percent"]) for r in rows]
        scaled = [float(r["scaled_rmse"]) for r in rows]
        comp_rows.append({
            "coefficient": c,
            "mean_relL2_percent": float(np.mean(rel)) if rel else np.nan,
            "max_relL2_percent": float(np.max(rel)) if rel else np.nan,
            "final_relL2_percent": float(rel[-1]) if rel else np.nan,
            "mean_scaled_rmse": float(np.mean(scaled)) if scaled else np.nan,
        })

    return summary_rows, comp_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="77CavityUQL2.py")
    ap.add_argument("--model", default="model_params_robust_cavity_qweighted_for_cupy.npz")
    ap.add_argument("--target-kn", type=float, default=1.0)
    ap.add_argument("--base-kn", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--ntss", type=int, default=2000)
    ap.add_argument("--ppc", type=int, default=1000)
    ap.add_argument("--uw", type=float, default=800.0)
    ap.add_argument("--audit-every", type=int, default=50)
    ap.add_argument("--min-count", type=float, default=0.0, help="Optional raw particle-count threshold. Default 0 disables this filter because grid['N'] in 77Cavity is a weighted cell mass, not a raw count.")
    ap.add_argument("--outdir", default="cavity77_closure_coeff_audit")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mod = load_module_from_path(args.module)
    cp = mod.cp

    # Use q-weighted model for both the audit and, if supported, the module's own ML run.
    if hasattr(mod, "MODEL_PARAMS_FILE"):
        mod.MODEL_PARAMS_FILE = args.model
    os.environ["CAVITY_MODEL_PARAMS"] = args.model

    # Override nominal Kn via density scaling, exactly as in the Kn-sweep wrapper.
    rho_original = float(mod.RHO_IN_BASE)
    scale = float(args.base_kn) / float(args.target_kn)
    rho_target = rho_original * scale
    mod.RHO_IN_BASE = rho_target
    mod.RHO_IN = rho_target

    # Override run length and particles.
    mod.N_STEPS_PER_RUN = int(args.steps)
    mod.NTSS = int(args.ntss)
    mod.PARTICLES_PER_CELL_TARGET = int(args.ppc)
    mod.NP = int(mod.PARTICLES_PER_CELL_TARGET * mod.NC)

    # Override lid speed.
    if args.uw is not None:
        mod.UW_LID = float(args.uw)
        mod.DT = 0.2 * min(mod.DX, mod.DY) / max(mod.UW_LID, mod.THETA_IN)

    # Consistent particle weight at target density.
    mod.W_PARTICLE = (mod.LX * mod.LY * mod.RHO_IN_BASE) / float(mod.NP)

    # Disable any optional patched diagnostics to avoid extra overhead / modulo-zero issues.
    safe_skip = int(args.steps) + 10**9
    if hasattr(mod, "HIGH_MOMENTS_EVERY"):
        mod.HIGH_MOMENTS_EVERY = safe_skip
    if hasattr(mod, "ENTROPY_EVERY"):
        mod.ENTROPY_EVERY = safe_skip

    # Load ML parameters once for the audit.
    params_npz = np.load(args.model)
    ml_params = {k: cp.asarray(v) for k, v in params_npz.items()}
    params_npz.close()

    y_mean = ml_params.get("y_mean", None)
    y_scale = ml_params.get("y_scale", None)
    if y_mean is None or y_scale is None:
        raise RuntimeError("Model npz must contain y_mean and y_scale for scaled coefficient diagnostics.")
    y_scale_safe = cp.where(cp.abs(y_scale) > 0, y_scale, 1.0)

    history_rows = []
    comp_long_rows = []
    avg_history_rows = []

    # Cumulative time-averaged coefficient audit.
    # Instantaneous full-moment coefficients are noisy because they are computed
    # from high-order particle moments. The cumulative audit compares the time-
    # averaged ML and physics coefficient fields and also estimates a physics
    # split-half noise floor from odd/even sampled physics coefficients.
    avg_state = {
        "count": 0,
        "Zp_sum": None,
        "Zm_sum": None,
        "Zp_odd_sum": None,
        "Zp_even_sum": None,
        "n_odd": 0,
        "n_even": 0,
    }

    state = {
        "last_grid": None,
        "solve_call": 0,
    }

    orig_build = mod.build_linear_systems_cupy
    orig_solve = mod.solve_linear_systems_cupy

    def wrapped_build(grid, linsys, *rest, **kwargs):
        state["last_grid"] = grid
        return orig_build(grid, linsys, *rest, **kwargs)

    def wrapped_solve(linsys, coeffs, *rest, **kwargs):
        out = orig_solve(linsys, coeffs, *rest, **kwargs)

        state["solve_call"] += 1
        step = state["solve_call"]

        if step > mod.NTSS and (step % int(args.audit_every) == 0):
            grid = state["last_grid"]
            if grid is None:
                return out

            # Physics coefficients from the just-solved 9x9 system.
            A_phys = coeffs["A"].copy()
            B_phys = coeffs["B"].copy()

            # ML coefficients on the SAME grid state.
            coeffs_ml = {
                "A": cp.empty_like(coeffs["A"]),
                "B": cp.empty_like(coeffs["B"]),
                "C": cp.zeros_like(coeffs["C"]) if isinstance(coeffs, dict) and "C" in coeffs else cp.zeros((coeffs["A"].shape[0], 1), dtype=coeffs["A"].dtype),
            }
            mod.predict_coeffs_cupy_native(grid, coeffs_ml, ml_params)
            A_ml = coeffs_ml["A"]
            B_ml = coeffs_ml["B"]

            Y_phys = cp.concatenate([A_phys, B_phys], axis=1)
            Y_ml = cp.concatenate([A_ml, B_ml], axis=1)

            finite = cp.all(cp.isfinite(Y_phys), axis=1) & cp.all(cp.isfinite(Y_ml), axis=1)

            # Important: in the 77Cavity code, grid["N"] is not a raw particle
            # count. It is a weighted cell population / mass-like quantity
            # of order 1e-14--1e-13 in the saved NPZ files. The previous audit
            # mistakenly applied grid["N"] > 10 and therefore rejected every cell.
            # We only apply a count filter if the selected field looks like an
            # actual count array. By default --min-count=0 disables this filter.
            count_filter_applied = False
            count_key_used = "none"
            if float(args.min_count) > 0:
                for candidate_key in ["count", "counts", "particle_count", "particle_counts", "N_count", "raw_count"]:
                    if candidate_key in grid:
                        cc = grid[candidate_key]
                        if cp_to_float(cp.nanmax(cc)) > float(args.min_count):
                            finite = finite & (cc > float(args.min_count))
                            count_filter_applied = True
                            count_key_used = candidate_key
                            break
                # Do NOT fall back to grid["N"] unless it looks like a raw count.
                if (not count_filter_applied) and ("N" in grid):
                    nn = grid["N"]
                    if cp_to_float(cp.nanmax(nn)) > float(args.min_count):
                        finite = finite & (nn > float(args.min_count))
                        count_filter_applied = True
                        count_key_used = "N"

            n_valid = int(cp_to_float(cp.sum(finite.astype(cp.int32))))
            n_total = int(Y_phys.shape[0])

            if n_valid <= 0:
                row = {
                    "step": step,
                    "n_valid_cells": 0,
                    "n_total_cells": n_total,
                    "relL2_total_percent": np.nan,
                    "relL2_A_percent": np.nan,
                    "relL2_B_percent": np.nan,
                    "scaled_relL2_total_percent": np.nan,
                    "scaled_relL2_A_percent": np.nan,
                    "scaled_relL2_B_percent": np.nan,
                    "scaled_rmse_all": np.nan,
                    "scaled_mean_abs_all": np.nan,
                    "scaled_max_abs_all": np.nan,
                    "raw_max_abs_all": np.nan,
                    "nonfinite_cell_count": n_total,
                    "count_filter_applied": count_filter_applied,
                    "count_key_used": count_key_used,
                }
                history_rows.append(row)
                return out

            Yp = Y_phys[finite]
            Ym = Y_ml[finite]
            Ap = A_phys[finite]
            Am = A_ml[finite]
            Bp = B_phys[finite]
            Bm = B_ml[finite]

            Dy_scaled = (Ym - Yp) / y_scale_safe
            Zp = (Yp - y_mean) / y_scale_safe
            Zm = (Ym - y_mean) / y_scale_safe
            Zp_A, Zm_A = Zp[:, :6], Zm[:, :6]
            Zp_B, Zm_B = Zp[:, 6:], Zm[:, 6:]
            row = {
                "step": step,
                "n_valid_cells": n_valid,
                "n_total_cells": n_total,
                "relL2_total_percent": robust_rel_l2(cp, Ym, Yp),
                "relL2_A_percent": robust_rel_l2(cp, Am, Ap),
                "relL2_B_percent": robust_rel_l2(cp, Bm, Bp),
                "scaled_relL2_total_percent": robust_rel_l2(cp, Zm, Zp),
                "scaled_relL2_A_percent": robust_rel_l2(cp, Zm_A, Zp_A),
                "scaled_relL2_B_percent": robust_rel_l2(cp, Zm_B, Zp_B),
                "scaled_rmse_all": robust_rmse(cp, Dy_scaled),
                "scaled_mean_abs_all": robust_mean_abs(cp, Dy_scaled),
                "scaled_max_abs_all": robust_max_abs(cp, Dy_scaled),
                "raw_max_abs_all": robust_max_abs(cp, Ym - Yp),
                "nonfinite_cell_count": n_total - n_valid,
                "count_filter_applied": count_filter_applied,
                "count_key_used": count_key_used,
            }

            # Per-component diagnostics.
            for j, name in enumerate(COEFF_NAMES):
                pj = Yp[:, j]
                mj = Ym[:, j]
                dscaled_j = (mj - pj) / y_scale_safe[j]
                zpj = (pj - y_mean[j]) / y_scale_safe[j]
                zmj = (mj - y_mean[j]) / y_scale_safe[j]
                rel_j = robust_rel_l2(cp, mj, pj)
                scaled_rel_j = robust_rel_l2(cp, zmj, zpj)
                rmse_j = robust_rmse(cp, dscaled_j)

                row[f"relL2_{name}_percent"] = rel_j
                row[f"scaled_relL2_{name}_percent"] = scaled_rel_j
                row[f"scaled_rmse_{name}"] = rmse_j

                comp_long_rows.append({
                    "step": step,
                    "coefficient": name,
                    "relL2_percent": rel_j,
                    "scaled_relL2_percent": scaled_rel_j,
                    "scaled_rmse": rmse_j,
                    "scaled_mean_abs": robust_mean_abs(cp, dscaled_j),
                    "scaled_max_abs": robust_max_abs(cp, dscaled_j),
                    "raw_max_abs": robust_max_abs(cp, mj - pj),
                })

            # Cumulative time-averaged scale-aware audit.
            # Only update this metric when every cell is valid, so the cell
            # ordering is fixed across samples. With COEFF_MINCOUNT=0 this is
            # normally the case for the cavity runs.
            if n_valid == n_total:
                if avg_state["Zp_sum"] is None:
                    avg_state["Zp_sum"] = cp.zeros_like(Zp)
                    avg_state["Zm_sum"] = cp.zeros_like(Zm)
                    avg_state["Zp_odd_sum"] = cp.zeros_like(Zp)
                    avg_state["Zp_even_sum"] = cp.zeros_like(Zp)

                avg_state["count"] += 1
                sample_index = avg_state["count"]

                avg_state["Zp_sum"] += Zp
                avg_state["Zm_sum"] += Zm

                if sample_index % 2 == 1:
                    avg_state["Zp_odd_sum"] += Zp
                    avg_state["n_odd"] += 1
                else:
                    avg_state["Zp_even_sum"] += Zp
                    avg_state["n_even"] += 1

                Zp_bar = avg_state["Zp_sum"] / sample_index
                Zm_bar = avg_state["Zm_sum"] / sample_index

                Zp_bar_A, Zm_bar_A = Zp_bar[:, :6], Zm_bar[:, :6]
                Zp_bar_B, Zm_bar_B = Zp_bar[:, 6:], Zm_bar[:, 6:]

                cum_row = {
                    "step": step,
                    "sample_index": sample_index,
                    "n_valid_cells": n_valid,
                    "scaled_cumulative_relL2_total_percent": robust_rel_l2(cp, Zm_bar, Zp_bar),
                    "scaled_cumulative_relL2_A_percent": robust_rel_l2(cp, Zm_bar_A, Zp_bar_A),
                    "scaled_cumulative_relL2_B_percent": robust_rel_l2(cp, Zm_bar_B, Zp_bar_B),
                    "scaled_cumulative_rmse_all": robust_rmse(cp, Zm_bar - Zp_bar),
                    "physics_split_half_total_percent": np.nan,
                    "physics_split_half_A_percent": np.nan,
                    "physics_split_half_B_percent": np.nan,
                    "model_error_over_noise_total": np.nan,
                }

                if avg_state["n_odd"] > 0 and avg_state["n_even"] > 0:
                    Zp_odd_bar = avg_state["Zp_odd_sum"] / avg_state["n_odd"]
                    Zp_even_bar = avg_state["Zp_even_sum"] / avg_state["n_even"]
                    noise_total = robust_symmetric_rel_l2(cp, Zp_odd_bar, Zp_even_bar)
                    noise_A = robust_symmetric_rel_l2(cp, Zp_odd_bar[:, :6], Zp_even_bar[:, :6])
                    noise_B = robust_symmetric_rel_l2(cp, Zp_odd_bar[:, 6:], Zp_even_bar[:, 6:])
                    cum_row["physics_split_half_total_percent"] = noise_total
                    cum_row["physics_split_half_A_percent"] = noise_A
                    cum_row["physics_split_half_B_percent"] = noise_B
                    if np.isfinite(noise_total) and noise_total > 1e-14:
                        cum_row["model_error_over_noise_total"] = cum_row["scaled_cumulative_relL2_total_percent"] / noise_total

                avg_history_rows.append(cum_row)

            history_rows.append(row)

        return out

    mod.build_linear_systems_cupy = wrapped_build
    mod.solve_linear_systems_cupy = wrapped_solve

    meta = {
        "module": str(args.module),
        "model": str(args.model),
        "target_kn": float(args.target_kn),
        "base_kn_assumed": float(args.base_kn),
        "density_scale": float(scale),
        "rho_original": float(rho_original),
        "rho_target": float(rho_target),
        "steps": int(mod.N_STEPS_PER_RUN),
        "ntss": int(mod.NTSS),
        "ppc": int(mod.PARTICLES_PER_CELL_TARGET),
        "np": int(mod.NP),
        "nx": int(mod.NX),
        "ny": int(mod.NY),
        "nc": int(mod.NC),
        "uw_lid": float(mod.UW_LID),
        "dt": float(mod.DT),
        "audit_every": int(args.audit_every),
        "min_count": float(args.min_count),
        "audit_version": "v4_time_averaged_noise_floor",
        "audit_definition": "ML and physics closure coefficients compared on the same physics-trajectory cell state after the full cubic-FP solve. Cumulative-average errors and a split-half physics noise floor are also reported.",
    }
    (outdir / "coefficient_audit_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=" * 78)
    print("Cavity direct closure-coefficient error audit")
    print(f"Module:       {args.module}")
    print(f"Model:        {args.model}")
    print(f"Target Kn:    {args.target_kn}  (density scale={scale:.6e})")
    print(f"Steps/NTSS:   {mod.N_STEPS_PER_RUN}/{mod.NTSS}")
    print(f"PPC/NP:       {mod.PARTICLES_PER_CELL_TARGET}/{mod.NP}")
    print(f"Audit every:  {args.audit_every} steps after NTSS")
    print(f"Outdir:       {outdir}")
    print(f"GPU:          {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print("=" * 78, flush=True)

    t0 = time.time()
    mod.main()
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - t0

    # Write outputs.
    history_fields = [
        "step", "n_valid_cells", "n_total_cells",
        "relL2_total_percent", "relL2_A_percent", "relL2_B_percent",
        "scaled_relL2_total_percent", "scaled_relL2_A_percent", "scaled_relL2_B_percent",
        "scaled_rmse_all", "scaled_mean_abs_all", "scaled_max_abs_all",
        "raw_max_abs_all", "nonfinite_cell_count", "count_filter_applied", "count_key_used",
    ]
    for name in COEFF_NAMES:
        history_fields.append(f"relL2_{name}_percent")
    for name in COEFF_NAMES:
        history_fields.append(f"scaled_relL2_{name}_percent")
    for name in COEFF_NAMES:
        history_fields.append(f"scaled_rmse_{name}")

    write_csv(outdir / "closure_coeff_error_history.csv", history_rows, history_fields)

    cumulative_fields = [
        "step", "sample_index", "n_valid_cells",
        "scaled_cumulative_relL2_total_percent",
        "scaled_cumulative_relL2_A_percent",
        "scaled_cumulative_relL2_B_percent",
        "scaled_cumulative_rmse_all",
        "physics_split_half_total_percent",
        "physics_split_half_A_percent",
        "physics_split_half_B_percent",
        "model_error_over_noise_total",
    ]
    write_csv(outdir / "closure_coeff_cumulative_average_error.csv", avg_history_rows, cumulative_fields)

    # Final cumulative-average audit table. This is usually the most useful
    # coefficient-level metric for the paper because it filters high-order
    # particle-moment noise in the instantaneous cubic-FP target coefficients.
    cumulative_summary_rows = []
    if avg_history_rows:
        last = avg_history_rows[-1]
        for block, key, nkey in [
            ("total", "scaled_cumulative_relL2_total_percent", "physics_split_half_total_percent"),
            ("A", "scaled_cumulative_relL2_A_percent", "physics_split_half_A_percent"),
            ("B", "scaled_cumulative_relL2_B_percent", "physics_split_half_B_percent"),
        ]:
            vals = [float(r[key]) for r in avg_history_rows]
            cumulative_summary_rows.append({
                "block": block,
                "final_cumulative_scaled_relL2_percent": float(last[key]),
                "mean_cumulative_scaled_relL2_percent": float(np.mean(vals)),
                "final_physics_split_half_percent": float(last[nkey]) if np.isfinite(float(last[nkey])) else np.nan,
                "final_scaled_cumulative_rmse": float(last["scaled_cumulative_rmse_all"]),
            })

    write_csv(
        outdir / "closure_coeff_cumulative_average_summary.csv",
        cumulative_summary_rows,
        ["block", "final_cumulative_scaled_relL2_percent", "mean_cumulative_scaled_relL2_percent", "final_physics_split_half_percent", "final_scaled_cumulative_rmse"]
    )

    if cumulative_summary_rows:
        labels_cum = {
            "total": r"All coefficients $[A_{ij},B_i]$",
            "A": r"Drift block $A_{ij}$",
            "B": r"Heat-flux block $B_i$",
        }
        tex_lines_cum = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Time-averaged closure-coefficient audit for the cavity. The error is computed in the standardized output space used during DNN training. The split-half column estimates the statistical noise floor of the cubic-FP coefficient target by comparing odd and even sampled physics coefficients.}",
            r"\label{tab:cavity_closure_coeff_cumulative_average}",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Coefficient block & Final cumulative $\varepsilon_{L_2}^{(s)}$ (\%) & Physics split-half noise (\%) & Final cumulative RMSE \\",
            r"\midrule",
        ]
        for rr in cumulative_summary_rows:
            noise = rr["final_physics_split_half_percent"]
            noise_str = f"{noise:.3f}" if np.isfinite(noise) else "--"
            tex_lines_cum.append(
                f"{labels_cum.get(rr['block'], rr['block'])} & "
                f"{rr['final_cumulative_scaled_relL2_percent']:.3f} & "
                f"{noise_str} & "
                f"{rr['final_scaled_cumulative_rmse']:.4f} " + r"\\"
            )
        tex_lines_cum += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (outdir / "closure_coeff_cumulative_average_summary.tex").write_text("\n".join(tex_lines_cum) + "\n", encoding="utf-8")

    comp_fields = [
        "step", "coefficient", "relL2_percent", "scaled_relL2_percent", "scaled_rmse",
        "scaled_mean_abs", "scaled_max_abs", "raw_max_abs",
    ]
    write_csv(outdir / "closure_coeff_error_components_long.csv", comp_long_rows, comp_fields)

    summary_rows, comp_summary_rows = compute_summaries(history_rows, comp_long_rows)
    write_csv(
        outdir / "closure_coeff_error_summary.csv",
        summary_rows,
        ["block", "mean_relL2_percent", "max_relL2_percent", "final_relL2_percent"]
    )

    # Scale-aware coefficient-error summary in the standardized output space used by the DNN.
    scale_rows = []
    for block, key in [
        ("total", "scaled_relL2_total_percent"),
        ("A", "scaled_relL2_A_percent"),
        ("B", "scaled_relL2_B_percent"),
    ]:
        vals = [float(r[key]) for r in history_rows]
        scale_rows.append({
            "block": block,
            "mean_scaled_relL2_percent": float(np.mean(vals)) if vals else np.nan,
            "max_scaled_relL2_percent": float(np.max(vals)) if vals else np.nan,
            "final_scaled_relL2_percent": float(vals[-1]) if vals else np.nan,
            "mean_scaled_rmse": float(np.mean([float(r["scaled_rmse_all"]) for r in history_rows])) if vals else np.nan,
        })
    write_csv(
        outdir / "closure_coeff_error_summary_scaleaware.csv",
        scale_rows,
        ["block", "mean_scaled_relL2_percent", "max_scaled_relL2_percent", "final_scaled_relL2_percent", "mean_scaled_rmse"]
    )
    write_csv(
        outdir / "closure_coeff_error_component_summary.csv",
        comp_summary_rows,
        ["coefficient", "mean_relL2_percent", "max_relL2_percent", "final_relL2_percent", "mean_scaled_rmse"]
    )

    comp_scale_rows = []
    for c in COEFF_NAMES:
        rows_c = [r for r in comp_long_rows if r["coefficient"] == c]
        vals = [float(r["scaled_relL2_percent"]) for r in rows_c]
        rms = [float(r["scaled_rmse"]) for r in rows_c]
        comp_scale_rows.append({
            "coefficient": c,
            "mean_scaled_relL2_percent": float(np.mean(vals)) if vals else np.nan,
            "max_scaled_relL2_percent": float(np.max(vals)) if vals else np.nan,
            "final_scaled_relL2_percent": float(vals[-1]) if vals else np.nan,
            "mean_scaled_rmse": float(np.mean(rms)) if rms else np.nan,
        })
    write_csv(
        outdir / "closure_coeff_error_component_summary_scaleaware.csv",
        comp_scale_rows,
        ["coefficient", "mean_scaled_relL2_percent", "max_scaled_relL2_percent", "final_scaled_relL2_percent", "mean_scaled_rmse"]
    )

    # A compact LaTeX table for the manuscript/response, using the scale-aware metric.
    labels = {"total": r"All coefficients $[A_{ij},B_i]$", "A": r"Drift block $A_{ij}$", "B": r"Heat-flux block $B_i$"}
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scale-aware time-resolved closure-coefficient audit for the cavity. Errors are computed in the standardized coefficient space used during DNN training.}",
        r"\label{tab:cavity_closure_coeff_scaleaware}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Coefficient block & Mean $\\varepsilon_{L_2}^{(s)}$ (\\%) & Max. $\\varepsilon_{L_2}^{(s)}$ (\\%) & Final $\\varepsilon_{L_2}^{(s)}$ (\\%) & Mean scaled RMSE " + r"\\",
        r"\midrule",
    ]
    for rr in scale_rows:
        tex_lines.append(f"{labels.get(rr['block'], rr['block'])} & {rr['mean_scaled_relL2_percent']:.3f} & {rr['max_scaled_relL2_percent']:.3f} & {rr['final_scaled_relL2_percent']:.3f} & {rr['mean_scaled_rmse']:.4f} " + r"\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "closure_coeff_error_summary_scaleaware.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    make_latex_summary(outdir, summary_rows, comp_summary_rows)
    make_plots(outdir, history_rows, comp_long_rows)
    make_cumulative_plots(outdir, avg_history_rows)

    # Copy standard outputs into the audit folder, if produced.
    for src, dst in [
        ("cavity_data_PHYSICS.npz", "cavity_data_PHYSICS_coeffaudit.npz"),
        ("cavity_data_FAST_ML.npz", "cavity_data_FAST_ML_coeffaudit.npz"),
        (f"cavity_comparison_robust_{int(mod.UW_LID)}ms.jpg", f"cavity_comparison_robust_{int(mod.UW_LID)}ms.jpg"),
        ("cavity_comparison_robust_800ms.jpg", "cavity_comparison_robust_800ms.jpg"),
    ]:
        p = Path(src)
        if p.exists():
            try:
                shutil.copy2(p, outdir / dst)
            except Exception:
                pass

    notes = [
        "# Direct closure-coefficient error audit",
        "",
        "Audit wrapper version: v4-time-averaged-noise-floor",
        "Definition:",
        "- At each sampled post-transient step, the physics coefficients are obtained from FULL moments and the batched 9x9 cubic-FP solve.",
        "- The ML coefficients are then predicted from the same grid state using the GPU-native DNN forward pass.",
        "- Therefore the reported error is a direct closure-map error, not an indirect trajectory error.",
        "",
        f"Wall time: {elapsed:.3f} s",
        f"Samples written: {len(history_rows)}",
        "",
        "Main files:",
        "- closure_coeff_error_history.csv",
        "- closure_coeff_error_components_long.csv",
        "- closure_coeff_error_summary.tex",
        "- closure_coeff_error_summary_scaleaware.tex",
        "- closure_coeff_cumulative_average_summary.tex",
        "- closure_coeff_cumulative_average_error.csv",
        "- closure_coeff_cumulative_average_relL2_time_history.pdf",
        "- closure_coeff_error_components.tex",
        "- closure_coeff_error_time_history.pdf",
        "- closure_coeff_scaled_rmse_time_history.pdf",
        "",
        "Suggested manuscript wording:",
        "At sampled post-transient steps along the physics trajectory, we recomputed the ML closure coefficients from the same low-order cell moments used by the DNN and compared them with the cubic-FP coefficients obtained from the full high-order moment system. Instantaneous coefficients are noisy because they depend on high-order particle moments, so the audit also reports cumulative time-averaged coefficient errors and a split-half physics noise floor.",
    ]
    (outdir / "coefficient_audit_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("")
    print("=" * 78)
    print("Coefficient audit complete.")
    print(f"Output directory: {outdir}")
    for p in sorted(outdir.iterdir()):
        print(" ", p.name)
    print("=" * 78)


if __name__ == "__main__":
    main()
