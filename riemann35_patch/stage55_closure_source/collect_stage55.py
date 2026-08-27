#!/usr/bin/env python3
"""Collect, classify, plot, and bundle the Stage-55 source audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage55")
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from riemann35_patch.stage54_heat_flux_third_order.collect_heat_flux import (
    irreducible_decomposition,
    symmetric_tensor,
)

METHODS = (
    "qmc_base", "qmc_refined", "exact_coeff_gaussian_projection",
    "gaussian_coeff_exact_projection", "gaussian_both", "compact_positive",
)
LABELS = {
    "qmc_refined": "positive kinetic reference",
    "exact_coeff_gaussian_projection": "Gaussian projection tail only",
    "gaussian_coeff_exact_projection": "Gaussian coefficient M5 only",
    "gaussian_both": "both Gaussian-tail substitutions",
    "compact_positive": "positive compact reconstruction",
}
COLORS = {
    "qmc_refined": "#111111", "exact_coeff_gaussian_projection": "#d97706",
    "gaussian_coeff_exact_projection": "#7c3aed", "gaussian_both": "#c2410c",
    "compact_positive": "#1677a3",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-gate", type=float, default=0.03)
    parser.add_argument("--diagnosis-fraction", type=float, default=0.70)
    parser.add_argument("--compact-third-gate", type=float, default=0.03)
    parser.add_argument("--compact-tracefree-gate", type=float, default=0.05)
    parser.add_argument("--residual-gate", type=float, default=2e-8)
    parser.add_argument("--support-reduction-gate", type=float, default=100.0)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def load(root: Path, method: str):
    path = root / f"stage55_{method}.npz"
    summary = root / f"stage55_{method}_summary.json"
    if not path.is_file() or not summary.is_file():
        return None
    archive = np.load(path)
    return {key: np.asarray(archive[key]) for key in archive.files} | {"summary": json.loads(summary.read_text())}


def rel(candidate, reference):
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-14))


def derived(third):
    q, _, tf = irreducible_decomposition(third)
    return symmetric_tensor(third), q, tf


def main():
    args = arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    loaded = {method: load(args.root, method) for method in METHODS}
    missing = [method for method, item in loaded.items() if item is None]
    if missing:
        raise FileNotFoundError("missing Stage-55 branches: " + ", ".join(missing))
    positions = loaded["qmc_refined"]["third_positions"].astype(int)
    means = {method: np.mean(item["sources"], axis=0)[:, positions] for method, item in loaded.items()}
    reference = means["qmc_refined"]
    reference_node_change = rel(means["qmc_base"], reference)
    reference_spread = rel(
        np.std(loaded["qmc_refined"]["sources"][:, :, positions], axis=0, ddof=1), reference
    )
    errors = {}
    time_errors = {}
    for method in METHODS[2:]:
        full, q, tf = derived(means[method])
        reference_full, reference_q, reference_tf = derived(reference)
        errors[method] = {"third_source": rel(full, reference_full), "heat_flux_source": rel(q, reference_q), "trace_free_source": rel(tf, reference_tf)}
        time_errors[method] = np.linalg.norm((full - reference_full).reshape(len(full), -1), axis=1) / np.maximum(np.linalg.norm(reference_full.reshape(len(full), -1), axis=1), 1e-14)
    full_error = errors["gaussian_both"]["third_source"]
    coefficient_error = errors["gaussian_coeff_exact_projection"]["third_source"]
    projection_error = errors["exact_coeff_gaussian_projection"]["third_source"]
    dominant_fraction = max(coefficient_error, projection_error) / max(full_error, 1e-14)
    dominant_mechanism = "coefficient M5" if coefficient_error >= projection_error else "source-projection tail"
    compact = loaded["compact_positive"]["summary"]
    refined_support = int(np.max(loaded["qmc_refined"]["support"]))
    compact_support = int(np.max(loaded["compact_positive"]["support"]))
    support_reduction = refined_support / max(compact_support, 1)
    gates = {
        "reference_node_converged": reference_node_change < args.reference_gate,
        "reference_scramble_converged": reference_spread < args.reference_gate,
        "dominant_source_identified": dominant_fraction >= args.diagnosis_fraction,
        "compact_positive": float(compact["minimum_weight"]) >= -1e-14,
        "compact_retained_residual": float(compact["maximum_retained_moment_residual"]) < args.residual_gate,
        "compact_support_reduced": support_reduction >= args.support_reduction_gate,
        "compact_third_source": errors["compact_positive"]["third_source"] < args.compact_third_gate,
        "compact_trace_free_source": errors["compact_positive"]["trace_free_source"] < args.compact_tracefree_gate,
    }
    diagnosis_pass = all(gates[key] for key in ("reference_node_converged", "reference_scramble_converged", "dominant_source_identified"))
    compact_pass = diagnosis_pass and all(gates[key] for key in gates if key.startswith("compact_"))
    classification = "COMPACT_PASS" if compact_pass else ("DIAGNOSIS_PASS" if diagnosis_pass else "HOLD")
    summary = {
        "schema": "riemann35-stage55-closure-source-summary-v1", "classification": classification,
        "diagnosis_pass": diagnosis_pass, "compact_pass": compact_pass, "gates": gates,
        "reference_node_change": reference_node_change, "reference_scramble_spread": reference_spread,
        "errors": errors, "dominant_mechanism": dominant_mechanism, "dominant_fraction": dominant_fraction,
        "refined_support": refined_support, "compact_support": compact_support, "support_reduction": support_reduction,
    }
    (args.root / "stage55_closure_source_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.root / "stage55_closure_source_errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("method", "third_source", "heat_flux_source", "trace_free_source"))
        for method, values in errors.items(): writer.writerow((method, values["third_source"], values["heat_flux_source"], values["trace_free_source"]))

    import matplotlib.pyplot as plt
    times = loaded["qmc_refined"]["times"]
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    for method in METHODS[2:]:
        axes[0].plot(times, 100 * time_errors[method], marker="o", linewidth=1.8, color=COLORS[method], label=LABELS[method])
    axes[0].axhline(3, color="0.45", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel(r"Third-source error (\%)")
    axes[0].set_title("Instantaneous third-order FP source: causal closure audit")
    component = int(np.argmax(np.linalg.norm(reference, axis=0)))
    axes[1].plot(times, reference[:, component], color=COLORS["qmc_refined"], linewidth=2.2, label=LABELS["qmc_refined"])
    for method in METHODS[2:]:
        axes[1].plot(times, means[method][:, component], marker="o", linewidth=1.6, color=COLORS[method], label=LABELS[method])
    axes[1].set_xlabel(r"Time, $t/\tau$"); axes[1].set_ylabel("Dominant component source")
    for axis in axes: axis.grid(alpha=.22); axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.tight_layout(); figure.savefig(args.root / "stage55_source_diagnosis.png", dpi=300, facecolor="white"); plt.close(figure)

    report = ["# Stage 55 result", "", f"Classification: **{classification}**", "", f"Dominant diagnosed mechanism: **{dominant_mechanism}** ({dominant_fraction:.1%} of the full Gaussian-tail source error).", "", f"Reference node change: {reference_node_change:.2%}; scramble spread: {reference_spread:.2%}.", "", f"Compact source errors: T3 {errors['compact_positive']['third_source']:.2%}, trace-free {errors['compact_positive']['trace_free_source']:.2%}; support reduction {support_reduction:.1f}x.", "", "`DIAGNOSIS_PASS` does not imply that the compact closure passed."]
    (args.root / "STAGE55_RESULTS.md").write_text("\n".join(report) + "\n")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.bundle, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(args.root.glob("stage55_*")):
            if path.is_file(): bundle.write(path, path.name)
        bundle.write(args.root / "STAGE55_RESULTS.md", "STAGE55_RESULTS.md")
    print(json.dumps(summary, indent=2), flush=True)
    if classification == "HOLD" and not args.no_fail:
        raise SystemExit(3)


if __name__ == "__main__": main()
