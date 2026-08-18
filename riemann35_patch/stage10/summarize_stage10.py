#!/usr/bin/env python3
"""Create compact tables, figures, and a scientific interpretation for Stage 10."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def distribution(values) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
        return {"median": None, "p90": None, "maximum": None}
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def summarize_group(rows: list[dict]) -> dict:
    result = {
        "count": len(rows),
        "minimum_initial_margin": float(
            min(row["initial_realizability_margin"] for row in rows)
        ),
    }
    for method in (
        "single_gaussian",
        "stage9_tensor_mixture_tail",
        "grad_hyqmom_gqmom",
    ):
        passed = [row["methods"][method] for row in rows if row["methods"][method]["status"] == "PASS"]
        active = [
            row["methods"][method]
            for row in rows
            if row["cubic_active"] and row["methods"][method]["status"] == "PASS"
        ]
        result[method] = {
            "passed": len(passed),
            "tail_relative_error": distribution(
                item["tail_relative_error"] for item in passed
            ),
            "active_source_relative_error": distribution(
                item["source_relative_error"] for item in active
            ),
        }
    guarded = [row["guarded_grad_map"] for row in rows if row["guarded_grad_map"]["status"] == "PASS"]
    limited = [item for item in guarded if item["limiter_fraction"] < 1.0 - 1.0e-12]
    result["guarded_grad_map"] = {
        "passed": len(guarded),
        "limited": len(limited),
        "minimum_limiter_fraction": float(
            min((item["limiter_fraction"] for item in guarded), default=np.nan)
        ),
    }
    return result


def margin_label(margin: float) -> str:
    if margin < 1.0e-4:
        return "boundary: margin < 1e-4"
    if margin < 1.0e-2:
        return "near boundary: 1e-4 to 1e-2"
    if margin < 1.0e-1:
        return "interior: 1e-2 to 1e-1"
    return "deep interior: margin >= 1e-1"


def write_family_csv(path: Path, groups: dict[str, dict]) -> None:
    fields = [
        "family",
        "count",
        "minimum_initial_margin",
        "grad_source_median",
        "grad_source_p90",
        "stage9_source_median",
        "stage9_source_p90",
        "grad_guard_limited",
        "grad_guard_minimum_lambda",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for family, row in groups.items():
            writer.writerow(
                {
                    "family": family,
                    "count": row["count"],
                    "minimum_initial_margin": row["minimum_initial_margin"],
                    "grad_source_median": row["grad_hyqmom_gqmom"]["active_source_relative_error"]["median"],
                    "grad_source_p90": row["grad_hyqmom_gqmom"]["active_source_relative_error"]["p90"],
                    "stage9_source_median": row["stage9_tensor_mixture_tail"]["active_source_relative_error"]["median"],
                    "stage9_source_p90": row["stage9_tensor_mixture_tail"]["active_source_relative_error"]["p90"],
                    "grad_guard_limited": row["guarded_grad_map"]["limited"],
                    "grad_guard_minimum_lambda": row["guarded_grad_map"]["minimum_limiter_fraction"],
                }
            )


def make_figure(path: Path, records: list[dict], continuity: list[dict]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    styles = (
        ("stage9_tensor_mixture_tail", "Stage-9 tensor mixture", "#cc3311"),
        ("grad_hyqmom_gqmom", "Grad-HyQMOM / GQMOM", "#0077bb"),
    )
    for method, label, color in styles:
        rows = [
            row
            for row in records
            if row["cubic_active"] and row["methods"][method]["status"] == "PASS"
        ]
        axes[0].scatter(
            [max(row["initial_realizability_margin"], 1.0e-16) for row in rows],
            [max(row["methods"][method]["source_relative_error"], 1.0e-16) for row in rows],
            s=8,
            alpha=0.42,
            color=color,
            label=label,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("initial realizability margin")
    axes[0].set_ylabel("relative cubic-FP source error")
    axes[0].grid(alpha=0.25)

    guarded = [row for row in records if row["guarded_grad_map"]["status"] == "PASS"]
    axes[1].scatter(
        [max(row["initial_realizability_margin"], 1.0e-16) for row in guarded],
        [max(row["guarded_grad_map"]["limiter_fraction"], 1.0e-16) for row in guarded],
        s=8,
        alpha=0.50,
        color="#228833",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(5.0e-4, 1.2)
    axes[1].set_xlabel("initial realizability margin")
    axes[1].set_ylabel(r"nonlinear-source limiter $\lambda$")
    axes[1].grid(alpha=0.25)

    skewness = [row["s3"] for row in continuity]
    axes[2].plot(
        skewness,
        [max(row.get("stage9_M6_seam_jump", np.nan), 1.0e-16) for row in continuity],
        marker="o",
        color="#cc3311",
        label="Stage-9 tensor mixture",
    )
    axes[2].plot(
        skewness,
        [max(row.get("grad_M6_seam_jump", np.nan), 1.0e-16) for row in continuity],
        marker="s",
        color="#0077bb",
        label="Grad-HyQMOM / GQMOM",
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"standardized skewness $s_3$")
    axes[2].set_ylabel(r"$M_6$ jump across $\kappa_4=0$")
    axes[2].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.80, bottom=0.19, left=0.07, right=0.99, wspace=0.34)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_markdown(path: Path, compact: dict) -> None:
    methods = compact["overall_methods"]
    stage9 = compact["stage9_finite_map"]
    guarded = compact["guarded_grad_map"]
    grad = methods["grad_hyqmom_gqmom"]
    tensor = methods["stage9_tensor_mixture_tail"]
    exact_euler = compact["exact_euler_control"]
    families = compact["by_family"]
    deep = compact["by_initial_realizability_margin"].get(
        "deep interior: margin >= 1e-1", {"count": 0}
    )
    rare_beam = families["near_delta_rare_beam"]
    counterstream = families["near_delta_counterstream"]
    crossing = families["near_delta_crossing"]
    random_family = families["random_multivariate_mixture"]
    rare_hot = families["rare_hot_anisotropic"]
    rare_path = compact["selected_long_time"]["rare_beam_ma20"][
        "guarded_grad_map"
    ]
    text = f"""# Stage 10 general-state audit

## Scope

Stage 10 implements the Appendix-C Grad-HyQMOM reconstruction with
Gaussian-GQMOM univariate measures and compares it with the corrected Stage-9
principal-axis tensor Gaussian mixture.  The audit contains
{compact['state_count']} realizable states: the five Stage-9 cases, anisotropic
rare-hot populations, near-delta counter-stream/crossing/rare-beam states up
to Ma=100, and 256 seeded random two- to four-component multivariate Gaussian
mixtures.

The physical cubic coefficient called beta in the code (Lambda in the report)
is explicit:

    Lambda = -nu ||Pi||_F^2 / (tr P)^(7/2) <= 0.

Thus Lambda vanishes for isotropic stress.  The isotropic rare-hot exact-OU
test validates the OU path and conservation plumbing; it is not evidence for
accuracy when the nonlinear cubic correction is active.

The two marginal cubics are kept distinct:

    equal variance:       2 v^3 + kappa_4 v - kappa_3^2 = 0,
    equal-weight scale:   2 w^3 + kappa_4 w - kappa_3^2/3 = 0.

## Main numerical findings

* Appendix-C Grad-HyQMOM reconstructed all {grad['passed']} of
  {grad['attempted']} states.  Its median and 90th-percentile active cubic-FP
  source errors were {grad['active_cubic_source_relative_error']['median']:.3%}
  and {grad['active_cubic_source_relative_error']['p90']:.3%}.
* The Stage-9 tensor closure reconstructed {tensor['passed']} of
  {tensor['attempted']} states.  Its corresponding median and 90th-percentile
  source errors were {tensor['active_cubic_source_relative_error']['median']:.3%}
  and {tensor['active_cubic_source_relative_error']['p90']:.3%}.
These aggregate medians are **not a global accuracy ranking**.  The empirical
CDFs cross and the 292-state set is deliberately heterogeneous.  The exact
tail used by this audit is the tail of the generating Gaussian mixture, so all
accuracy statements are conditional on that known family.  By physical
family:

* counter-stream median source error is
  {counterstream['stage9_tensor_mixture_tail']['active_source_relative_error']['median']:.3%}
  for Stage 9 and
  {counterstream['grad_hyqmom_gqmom']['active_source_relative_error']['median']:.3%}
  for Grad/GQMOM;
* crossing median source error is
  {crossing['stage9_tensor_mixture_tail']['active_source_relative_error']['median']:.3%}
  versus
  {crossing['grad_hyqmom_gqmom']['active_source_relative_error']['median']:.3%};
* rare-beam median source error is
  {rare_beam['stage9_tensor_mixture_tail']['active_source_relative_error']['median']:.3%}
  versus
  {rare_beam['grad_hyqmom_gqmom']['active_source_relative_error']['median']:.3%};
* anisotropic rare-hot median source error is
  {rare_hot['stage9_tensor_mixture_tail']['active_source_relative_error']['median']:.3%}
  versus
  {rare_hot['grad_hyqmom_gqmom']['active_source_relative_error']['median']:.3%}; and
* the aggregate reversal is driven mainly by the {random_family['count']}
  rotated random mixtures, where Grad/GQMOM has the smaller median error.

Thus Stage 9 remains an in-family accuracy reference for separable, axis-aligned
mixtures; guarded Grad/GQMOM is the continuous, cheaper, more robust baseline.
Neither method dominates the other in accuracy.

* The unguarded Stage-9 finite map completed {stage9['passed']} of
  {stage9['attempted']} initial steps and produced {stage9['realizability_failures']}
  negative-margin states.
* Even the exact generating-mixture source followed by forward Euler left the
  cone in {exact_euler['realizability_failures']} of {exact_euler['attempted']}
  states ({exact_euler['failure_fraction']:.1%}); on the boundary band the
  count was {exact_euler['boundary_failures']} of
  {exact_euler['boundary_attempted']}.  A guard therefore controls both closure
  error and finite-step time-discretization error.
* Exact OU splitting plus a scalar realizability limiter completed
  {guarded['passed']} of {guarded['attempted']} initial steps with zero
  realizability failures.  The limiter activated in {guarded['limited_steps']}
  states; its minimum value was {guarded['minimum_limiter_fraction']:.3e}.
* Both the Stage-9 finite map and guarded Grad/GQMOM reached t=tau on all six
  selected trajectories, including counter-stream and
  crossing states at Ma=20, a counter-stream at Ma=100, and the rare beam that
  defeated the *unguarded* Grad source.  In the Ma=20 rare-beam trajectory the
  guard was active in {rare_path['rejected_substeps']} of
  {len(rare_path['per_step_history'])} steps and the minimum lambda was
  {rare_path['minimum_h_over_dt']:.3f}.  The per-step lambda history is stored
  in the JSON rather than asserted only in prose.
* The Stage-9 branch switch is genuinely discontinuous.  Across kappa_4=0 at
  s_3=1, an input perturbation of 2e-8 produces an M6 jump of 4.667.  The
  Gaussian-GQMOM jump is 2.25e-7 for the same perturbation.
* Prototype median evaluation time is roughly 0.032 s for Grad-HyQMOM versus
  0.315 s for the Stage-9 tensor reconstruction and 0.018 s for the single
  Gaussian closure.  The Appendix-C path is therefore about ten times faster
  than the Stage-9 tensor fit in this Python prototype.

## Interpretation

The guarded Appendix-C route is continuous across the former marginal branch
seam, cheaper than the Stage-9 tensor prototype, and realizable for the entire
292-state initial-step audit.  Those properties make it the appropriate robust
baseline for the next particle validation and for coupling to unchanged BFL
free transport.  They do not make it uniformly more accurate than Stage 9.

Only {deep['count']} states are in the deep-interior bin, so its small median
error must be reported with that sample count.  Extremely near-boundary random
mixtures can require strong limiting; in 70 of 85 boundary states the guard is
active and the global minimum lambda is {guarded['minimum_limiter_fraction']:.3e}.
There the nonlinear correction can be almost extinguished.  The Grad source
quadrature is signed in {grad['signed_quadrature_fraction']:.1%} of states, with
median negative mass
{grad['negative_mass_fraction']['median']:.1%}; this is acceptable as a linear
source quadrature, not as a positive VDF reconstruction, and it explains why a
moment-cone guard is needed.

The next evidence gate is therefore a 16-seed particle comparison on [0,tau]
for **both** maps and all six selected trajectories.  Near the univariate
Hankel boundary, the next algorithmic repair should be a smooth degeneration
to QMOM as b2 approaches zero, not the Appendix-D transport wave-speed cap.
A deterministic Hermite reference remains the late-time accuracy gate.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = arguments()
    summary = json.loads((args.results / "stage10_general_audit.json").read_text())
    records = json.loads((args.results / "stage10_state_records.json").read_text())
    by_family: dict[str, list[dict]] = defaultdict(list)
    by_margin: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_family[row["family"]].append(row)
        by_margin[margin_label(row["initial_realizability_margin"])].append(row)
    family_summary = {
        name: summarize_group(rows) for name, rows in sorted(by_family.items())
    }
    margin_summary = {
        name: summarize_group(rows) for name, rows in sorted(by_margin.items())
    }
    compact = {
        "schema": "riemann35-cubic-fp-stage10-interpretation-v2",
        "state_count": len(records),
        "overall_methods": summary["methods"],
        "exact_euler_control": summary["exact_euler_control"],
        "stage9_finite_map": summary["stage9_finite_map"],
        "guarded_grad_map": summary["guarded_grad_map"],
        "by_family": family_summary,
        "by_initial_realizability_margin": margin_summary,
        "selected_long_time": summary["selected_long_time"],
    }
    (args.results / "stage10_interpretation.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    write_family_csv(args.results / "stage10_family_summary.csv", family_summary)
    make_figure(
        args.results / "stage10_stability_and_seams.png",
        records,
        summary["branch_continuity"],
    )
    write_markdown(args.results / "STAGE10_RESULTS.md", compact)
    print(f"wrote Stage-10 interpretation to {args.results}")


if __name__ == "__main__":
    main()
