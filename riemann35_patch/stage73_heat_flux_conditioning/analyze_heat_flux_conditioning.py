#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
from riemann35_patch.stage71_harder_unseen.hard_cases import CASE_NAMES


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage72-dir", type=Path, required=True)
    return parser.parse_args()


def _derived(histories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    components = np.asarray(
        [central_third_components(history) for history in histories], dtype=float
    )
    heat_flux, _, _ = irreducible_decomposition(components)
    full_tensor = symmetric_tensor(components)
    return heat_flux, full_tensor


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value))


def _case_metrics(root: Path, name: str) -> dict[str, object]:
    archive = np.load(root / f"stage71_{name}.npz")
    summary = json.loads((root / f"stage71_{name}_summary.json").read_text())
    times = np.asarray(archive["times"], dtype=float)
    qmc_histories = np.asarray(archive["qmc_histories"], dtype=float)
    fine_histories = np.asarray(archive["persistent_fine_histories"], dtype=float)

    qmc_q, qmc_tensor = _derived(qmc_histories)
    fine_q, fine_tensor = _derived(fine_histories)
    ref_q = np.mean(qmc_q, axis=0)
    ref_tensor = np.mean(qmc_tensor, axis=0)
    cand_q = np.mean(fine_q, axis=0)
    cand_tensor = np.mean(fine_tensor, axis=0)

    q_std = np.std(qmc_q, axis=0, ddof=1)
    q_sem = q_std / np.sqrt(qmc_q.shape[0])
    q_signal = _norm(ref_q)
    tensor_signal = _norm(ref_tensor)
    q_difference = _norm(cand_q - ref_q)
    two_sem = 2.0 * _norm(q_sem)

    prandtl = float(summary["controls"]["prandtl"])
    analytic = ref_q[0][None, :] * np.exp(-2.0 * prandtl * times)[:, None]
    qmc_vs_analytic = relative_history_error(ref_q, analytic)
    candidate_vs_analytic = relative_history_error(cand_q, analytic)

    ref_q_norm = np.linalg.norm(ref_q, axis=-1)
    analytic_norm = np.linalg.norm(analytic, axis=-1)
    candidate_norm = np.linalg.norm(cand_q, axis=-1)

    return {
        "case": name,
        "density": float(summary["configuration"]["density"]),
        "existing_heat_flux_relative_error": relative_history_error(cand_q, ref_q),
        "existing_full_third_relative_error": relative_history_error(cand_tensor, ref_tensor),
        "q_signal_over_full_tensor_signal": q_signal / max(tensor_signal, 1.0e-14),
        "q_error_on_full_tensor_scale": q_difference / max(tensor_signal, 1.0e-14),
        "qmc_heat_flux_relative_spread": _norm(q_std) / max(q_signal, 1.0e-14),
        "qmc_heat_flux_relative_sem": _norm(q_sem) / max(q_signal, 1.0e-14),
        "candidate_error_over_two_qmc_sem": q_difference / max(two_sem, 1.0e-14),
        "qmc_vs_exact_decay_relative_error": qmc_vs_analytic,
        "candidate_vs_exact_decay_relative_error": candidate_vs_analytic,
        "reference_q_norm_min": float(np.min(ref_q_norm)),
        "reference_q_norm_max": float(np.max(ref_q_norm)),
        "reference_q_norm_final": float(ref_q_norm[-1]),
        "analytic_q_norm_final": float(analytic_norm[-1]),
        "candidate_q_norm_final": float(candidate_norm[-1]),
        "exact_decay_rate": 2.0 * prandtl,
    }


def main() -> None:
    args = arguments()
    root = args.stage72_dir
    rows = [_case_metrics(root, name) for name in CASE_NAMES]
    summary = {
        "schema": "riemann35-stage73-heat-flux-conditioning-v1",
        "purpose": (
            "Diagnostic-only audit of the two remaining Stage71 heat-flux gate failures. "
            "No Stage71 threshold or closure parameter is changed."
        ),
        "exact_identity": "q(t)=q(0)*exp(-2*Pr*t/tau) for the homogeneous cubic-FP heat-flux mode",
        "cases": rows,
    }
    (root / "stage73_heat_flux_conditioning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 73 — heat-flux conditioning audit",
        "",
        "Diagnostic only: Stage71 gates are unchanged.",
        "",
        "| case | old q err | full-third err | q/full signal | q err/full scale | QMC q spread | err/(2 SEM) | QMC vs exact decay | closure vs exact decay |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {100*row['existing_heat_flux_relative_error']:.3f}% | "
            f"{100*row['existing_full_third_relative_error']:.3f}% | "
            f"{row['q_signal_over_full_tensor_signal']:.3e} | "
            f"{100*row['q_error_on_full_tensor_scale']:.3f}% | "
            f"{100*row['qmc_heat_flux_relative_spread']:.3f}% | "
            f"{row['candidate_error_over_two_qmc_sem']:.3f} | "
            f"{100*row['qmc_vs_exact_decay_relative_error']:.3f}% | "
            f"{100*row['candidate_vs_exact_decay_relative_error']:.3f}% |"
        )
    lines += [
        "",
        "Interpretation:",
        "- A large heat-flux relative error with small q/full signal and small q-error/full-scale indicates contraction/cancellation conditioning rather than a large third-tensor defect.",
        "- err/(2 SEM) <= 1 means the closure/reference heat-flux difference is within two QMC standard errors.",
        "- The exact-decay columns distinguish QMC sampling/time-discretization error from closure error without changing the frozen Stage71 gate.",
    ]
    (root / "STAGE73_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
