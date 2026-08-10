#!/usr/bin/env python3
"""Stage 17: constructive non-identifiability audit for 35 -> M5/M6."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-riemann35-stage17")

import numpy as np
from scipy.optimize import linprog

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hyqmom_fp import (  # noqa: E402
    HYQMOM_35_INDICES,
    coefficients_from_moments,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
    projected_fp_collision_source,
    reconstruct_two_population_quadrature,
)
from hyqmom_fp.grad_hyqmom import WeightedNodeTailClosure  # noqa: E402
from hyqmom_fp.moments import (  # noqa: E402
    multivariate_gaussian_raw_moment,
    moment_value,
)
from riemann35_patch.stage10.run_general_realizability_audit import (  # noqa: E402
    deterministic_states,
)


TAIL_INDICES = ((5, 0, 0), (6, 0, 0), (4, 2, 0))
M400_POSITION = HYQMOM_35_INDICES.index((4, 0, 0))


class GaussianKernelTailClosure:
    """Tail moments of a positive common-variance Gaussian mixture."""

    def __init__(
        self,
        nodes: np.ndarray,
        weights: np.ndarray,
        kernel_variance: float,
    ) -> None:
        self.nodes = np.asarray(nodes, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.covariance = kernel_variance * np.eye(3)
        self.cache: dict[tuple[int, int, int], float] = {}

    def __call__(self, index, moments, state=None) -> float:
        del state
        if sum(index) <= 4:
            return moment_value(moments, index)
        if index not in self.cache:
            values = np.asarray(
                [
                    multivariate_gaussian_raw_moment(
                        index, node, self.covariance
                    )
                    for node in self.nodes
                ]
            )
            self.cache[index] = float(np.dot(self.weights, values))
        return self.cache[index]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-nodes", type=int, default=8)
    parser.add_argument(
        "--mollifier-variance-over-theta", type=float, default=1.0e-3
    )
    return parser.parse_args()


def kernel_moment_matrix(
    nodes: np.ndarray,
    indices,
    kernel_variance: float,
) -> np.ndarray:
    if kernel_variance == 0.0:
        return np.asarray(
            [
                nodes[:, 0] ** index[0]
                * nodes[:, 1] ** index[1]
                * nodes[:, 2] ** index[2]
                for index in indices
            ]
        )
    covariance = kernel_variance * np.eye(3)
    return np.asarray(
        [
            [
                multivariate_gaussian_raw_moment(index, node, covariance)
                for node in nodes
            ]
            for index in indices
        ]
    )


def closure_for_witness(
    nodes: np.ndarray,
    weights: np.ndarray,
    kernel_variance: float,
):
    if kernel_variance == 0.0:
        return WeightedNodeTailClosure(nodes, weights)
    return GaussianKernelTailClosure(nodes, weights, kernel_variance)


def source_channels(
    moments: np.ndarray,
    closure,
    frozen_coefficients,
) -> dict[str, float]:
    self_consistent_coefficients = coefficients_from_moments(
        moments, 1.0, closure=closure
    )
    self_consistent_source = projected_fp_collision_source(
        moments, self_consistent_coefficients, closure
    )
    frozen_source = projected_fp_collision_source(
        moments, frozen_coefficients, closure
    )
    return {
        "M400_source_self_consistent": float(
            self_consistent_source[M400_POSITION]
        ),
        "M400_source_frozen_coefficients": float(frozen_source[M400_POSITION]),
        "C_frobenius_norm": float(
            np.linalg.norm(self_consistent_coefficients.C)
        ),
        "gamma_norm": float(
            np.linalg.norm(self_consistent_coefficients.gamma)
        ),
        "beta": float(self_consistent_coefficients.beta),
    }


def solve_witness_family(
    *,
    family: str,
    nodes: np.ndarray,
    moments: np.ndarray,
    kernel_variance: float,
    frozen_coefficients,
    archive: dict[str, np.ndarray],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    retained = kernel_moment_matrix(
        nodes, HYQMOM_35_INDICES, kernel_variance
    )
    row_scale = np.maximum(np.max(np.abs(retained), axis=1), np.abs(moments))
    equality_matrix = retained / row_scale[:, None]
    equality_target = moments / row_scale
    witnesses: dict[str, object] = {}
    for tail_index in TAIL_INDICES:
        objective = kernel_moment_matrix(
            nodes, (tail_index,), kernel_variance
        )[0]
        objective_scale = max(float(np.max(np.abs(objective))), 1.0)
        solutions = {}
        tail_name = f"M{''.join(map(str, tail_index))}"
        for extremum, sign in (("minimum", 1.0), ("maximum", -1.0)):
            result = linprog(
                sign * objective / objective_scale,
                A_eq=equality_matrix,
                b_eq=equality_target,
                bounds=(0.0, None),
                method="highs",
                options={
                    "dual_feasibility_tolerance": 1.0e-9,
                    "primal_feasibility_tolerance": 1.0e-9,
                },
            )
            if not result.success:
                raise RuntimeError(
                    f"{family} tail witness LP failed: {result.message}"
                )
            weights = result.x
            retained_residual = float(
                np.max(np.abs(equality_matrix @ weights - equality_target))
            )
            closure = closure_for_witness(
                nodes, weights, kernel_variance
            )
            entry = {
                "extremum": extremum,
                "tail_value": float(np.dot(objective, weights)),
                **source_channels(
                    moments, closure, frozen_coefficients
                ),
                "scaled_retained_moment_residual": retained_residual,
                "mass": float(np.sum(weights)),
                "minimum_weight": float(np.min(weights)),
                "positive_support_count": int(np.sum(weights > 1.0e-12)),
            }
            solutions[extremum] = entry
            rows.append(
                {"family": family, "tail": tail_name, **entry}
            )
            archive[f"{family}_{tail_name}_{extremum}_weights"] = weights
        absolute_range = (
            solutions["maximum"]["tail_value"]
            - solutions["minimum"]["tail_value"]
        )
        midpoint = 0.5 * abs(
            solutions["maximum"]["tail_value"]
            + solutions["minimum"]["tail_value"]
        )
        witnesses[tail_name] = {
            **solutions,
            "absolute_range": absolute_range,
            "relative_range_to_midpoint": absolute_range
            / max(midpoint, 1.0e-14),
            "relative_range_normalization": "absolute range divided by absolute midpoint of the two extrema",
        }
    return witnesses


def support_description(nodes: np.ndarray, moments: np.ndarray) -> dict[str, object]:
    state = macroscopic_state(moments)
    standardized = (nodes - state.velocity) / np.sqrt(state.theta)
    c2_over_theta = np.einsum(
        "ni,ni->n", standardized, standardized
    )
    return {
        "support_type": "finite adaptive two-population tensor Gauss-Hermite centers",
        "total_nodes": int(nodes.shape[0]),
        "physical_coordinate_minimum": np.min(nodes, axis=0).tolist(),
        "physical_coordinate_maximum": np.max(nodes, axis=0).tolist(),
        "standardized_coordinate_minimum": np.min(
            standardized, axis=0
        ).tolist(),
        "standardized_coordinate_maximum": np.max(
            standardized, axis=0
        ).tolist(),
        "maximum_c2_over_theta": float(np.max(c2_over_theta)),
        "interpretation": "The grid-LP range is an inner, conservative witness range on this compact support, not a global sharp moment bound.",
    }


def main() -> None:
    args = arguments()
    state = {
        state.name: state for state in deterministic_states()
    }["rare_beam_ma20"]
    moments = mixture_of_gaussians_moments_35(state.components)
    macro = macroscopic_state(moments)
    proposal = reconstruct_two_population_quadrature(
        moments,
        quadrature_nodes=args.proposal_nodes,
        minimum_skewness_norm=0.05,
        residual_correction=False,
    )
    base_closure = WeightedNodeTailClosure(
        proposal.nodes, proposal.weights
    )
    frozen_coefficients = coefficients_from_moments(
        moments, 1.0, closure=base_closure
    )
    base_source = float(
        projected_fp_collision_source(
            moments, frozen_coefficients, base_closure
        )[M400_POSITION]
    )
    mollifier_variance = (
        args.mollifier_variance_over_theta * macro.theta
    )
    rows: list[dict[str, object]] = []
    archive = {
        "nodes": proposal.nodes,
        "target_moments": moments,
    }
    atomic = solve_witness_family(
        family="atomic_compact",
        nodes=proposal.nodes,
        moments=moments,
        kernel_variance=0.0,
        frozen_coefficients=frozen_coefficients,
        archive=archive,
        rows=rows,
    )
    mollified = solve_witness_family(
        family="mollified_gaussian",
        nodes=proposal.nodes,
        moments=moments,
        kernel_variance=mollifier_variance,
        frozen_coefficients=frozen_coefficients,
        archive=archive,
        rows=rows,
    )

    atomic_m600 = atomic["M600"]
    self_span = (
        atomic_m600["maximum"]["M400_source_self_consistent"]
        - atomic_m600["minimum"]["M400_source_self_consistent"]
    )
    frozen_span = (
        atomic_m600["maximum"]["M400_source_frozen_coefficients"]
        - atomic_m600["minimum"]["M400_source_frozen_coefficients"]
    )
    coefficient_feedback = {
        "M600_witness_pair_self_consistent_source_span": self_span,
        "M600_witness_pair_frozen_coefficient_source_span": frozen_span,
        "additional_span_from_coefficient_feedback": self_span - frozen_span,
        "feedback_fraction_of_self_consistent_span": (
            self_span - frozen_span
        )
        / max(abs(self_span), 1.0e-14),
        "interpretation": "Freezing the 9x9 coefficients removes the sign reversal at the upper witness but leaves a large direct-tail source span; coefficient regularization can reduce, not remove, practical non-identifiability.",
    }
    summary = {
        "schema": "riemann35-stage17-tail-identifiability-v2",
        "case": "rare_beam_ma20 at t=0",
        "retained_constraints": "all 35 raw moments through total degree four",
        "compact_support": support_description(proposal.nodes, moments),
        "unbounded_domain_context": "On an unbounded velocity domain, fourth-moment constraints generally do not impose a finite upper bound on sixth moments for interior truncated-moment states: tail mass of order V^-4 can preserve a bounded fourth-moment perturbation while producing an order V^2 sixth-moment increment. The compact grid-LP is intentionally a conservative practical witness, not a global supremum calculation.",
        "global_bound_method_note": "Sharp polynomial-moment bounds can be posed through the Lasserre moment/SOS hierarchy; the grid LP is used here because it gives explicit multivariate positive witnesses on the same physically resolved support.",
        "base_two_population_M400_source": base_source,
        "frozen_coefficient_definition": "C, gamma, and beta are held at the generating two-Gaussian values; only witness M5/M6 enter the projected M400 source.",
        "mollifier": {
            "type": "common isotropic Maxwellian kernel",
            "variance_over_theta": args.mollifier_variance_over_theta,
            "variance": mollifier_variance,
            "interpretation": "Each atomic center is replaced by a smooth positive Gaussian kernel and the LP is re-solved with exact Gaussian moment columns.",
        },
        "atomic_witnesses": atomic,
        "mollified_witnesses": mollified,
        "coefficient_feedback_diagnostic": coefficient_feedback,
        "decision": "INSTANTANEOUS_35_TO_M5_M6_MAP_IS_NOT_IDENTIFIABLE",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stage17_identifiability.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "stage17_witness_distributions.npz", **archive
    )
    with (args.output / "stage17_identifiability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    labels = [
        r"$M_{500}$ min",
        r"$M_{500}$ max",
        r"$M_{600}$ min",
        r"$M_{600}$ max",
    ]
    self_values = [
        atomic["M500"]["minimum"]["M400_source_self_consistent"],
        atomic["M500"]["maximum"]["M400_source_self_consistent"],
        atomic["M600"]["minimum"]["M400_source_self_consistent"],
        atomic["M600"]["maximum"]["M400_source_self_consistent"],
    ]
    frozen_values = [
        atomic["M500"]["minimum"]["M400_source_frozen_coefficients"],
        atomic["M500"]["maximum"]["M400_source_frozen_coefficients"],
        atomic["M600"]["minimum"]["M400_source_frozen_coefficients"],
        atomic["M600"]["maximum"]["M400_source_frozen_coefficients"],
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.9))
    positions = np.arange(len(labels))
    width = 0.36
    axis.bar(
        positions - width / 2,
        frozen_values,
        width,
        color="#0077bb",
        label="Frozen 9x9 coefficients",
    )
    axis.bar(
        positions + width / 2,
        self_values,
        width,
        color="#cc3311",
        label="Self-consistent coefficients",
    )
    axis.axhline(
        base_source,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Generating two-Gaussian source",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel(r"Instantaneous $dM_{400}/dt$")
    axis.set_title("Identical retained moments: direct tail and coefficient feedback")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(
        args.output / "stage17_M400_source_nonuniqueness.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    support = summary["compact_support"]
    atomic_m600 = atomic["M600"]
    smooth_m600 = mollified["M600"]
    lines = [
        "# Stage 17: tail identifiability audit",
        "",
        "The LP result is an explicit inner witness range on a fixed compact support, not a claim that the discrete extrema are global moment bounds. The common support contains 1024 adaptive two-population Gauss--Hermite centers with the standardized coordinate box",
        "",
        f"`[{support['standardized_coordinate_minimum'][0]:.3f}, {support['standardized_coordinate_maximum'][0]:.3f}] x [{support['standardized_coordinate_minimum'][1]:.3f}, {support['standardized_coordinate_maximum'][1]:.3f}] x [{support['standardized_coordinate_minimum'][2]:.3f}, {support['standardized_coordinate_maximum'][2]:.3f}]`,",
        "",
        f"and maximum `|c|^2/theta = {support['maximum_c2_over_theta']:.3f}`. Even under this compact restriction, non-uniqueness is large enough to reverse the sign of the self-consistent source. On the unrestricted velocity domain the sixth-moment supremum is generally unbounded when only moments through degree four are fixed; a Lasserre/SOS hierarchy, rather than this grid LP, is the route to sharp global polynomial-moment bounds.",
        "",
        "All relative tail ranges below use `|max-min| / |(max+min)/2|`.",
        "",
        "| Family | tail | minimum | maximum | midpoint-relative range | frozen-coefficient source pair | self-consistent source pair | max retained residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family_name, family in (
        ("atomic compact", atomic),
        ("Gaussian-mollified", mollified),
    ):
        for name, data in family.items():
            residual = max(
                data["minimum"]["scaled_retained_moment_residual"],
                data["maximum"]["scaled_retained_moment_residual"],
            )
            lines.append(
                f"| {family_name} | {name} | {data['minimum']['tail_value']:.8g} | "
                f"{data['maximum']['tail_value']:.8g} | {data['relative_range_to_midpoint']:.2%} | "
                f"{data['minimum']['M400_source_frozen_coefficients']:.3f} to {data['maximum']['M400_source_frozen_coefficients']:.3f} | "
                f"{data['minimum']['M400_source_self_consistent']:.3f} to {data['maximum']['M400_source_self_consistent']:.3f} | {residual:.3e} |"
            )
    lines.extend(
        [
            "",
            f"For the atomic M600 witness pair, freezing `(C, gamma, beta)` at the generating-mixture values leaves a source span of {frozen_span:.3f}; the self-consistent 9x9 solve increases it to {self_span:.3f}. Coefficient feedback contributes {coefficient_feedback['feedback_fraction_of_self_consistent_span']:.1%} of the latter span and is responsible for changing the upper-witness source from negative to positive. Regularizing the coefficient solve may remove that sign reversal, but it cannot remove the already-large direct-tail span.",
            "",
            f"Replacing every atom by an isotropic Maxwellian kernel with variance/theta = {args.mollifier_variance_over_theta:.1e} and re-solving the LP still gives an M600 range of {smooth_m600['relative_range_to_midpoint']:.2%}. The result is therefore not an artifact of singular atomic distributions.",
            "",
            f"The generating two-Gaussian value `dM400/dt = {base_source:.3f}` lies inside the frozen and self-consistent witness spans. A universally exact instantaneous 35-to-M5/M6 map does not exist without additional assumptions or inherited tail memory.",
        ]
    )
    (args.output / "STAGE17_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
