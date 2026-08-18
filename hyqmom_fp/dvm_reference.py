"""Clean-room Scharfetter--Gummel reference for the cubic FP operator.

This module is an independently assembled clean-room implementation, not a
reconstruction of the missing Stage-23/24 source.  Its scientific qualification
is the absolute manufactured-operator contract in Stage 24E-R, grid/time/domain
sensitivity, and direct comparison to independently generated references.  The
retained historical percentages are corroboration, not a bitwise acceptance lock.

Cell masses are evolved by a positive, no-flux Scharfetter--Gummel proposal.
Raw moments are exact integrals of the piecewise-constant cell density.  The
guided path then matches a trapezoidal weak 35-moment target with a strictly
positive minimum-KL projection on the unchanged velocity support.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb
from time import perf_counter
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import logsumexp, ndtr

from .collision import (
    CubicFPCoefficients,
    coefficients_from_moments,
    projected_fp_collision_source,
)
from .moments import HYQMOM_35_INDICES, MacroscopicState, MultiIndex, macroscopic_state


def _indices_through(order: int) -> tuple[MultiIndex, ...]:
    return tuple(
        (i, j, degree - i - j)
        for degree in range(order + 1)
        for i in range(degree, -1, -1)
        for j in range(degree - i, -1, -1)
    )


MOMENT_INDICES_6 = _indices_through(6)


@dataclass(frozen=True)
class DVMGrid:
    """Uniform Cartesian finite-volume grid in velocity space."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        shape = np.asarray(self.shape, dtype=int)
        if lower.shape != (3,) or upper.shape != (3,) or shape.shape != (3,):
            raise ValueError("DVM grid requires three lower/upper bounds and sizes")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("DVM bounds must be finite")
        if np.any(upper <= lower) or np.any(shape < 3):
            raise ValueError("DVM grid requires upper>lower and at least 3 cells/axis")

    @property
    def widths(self) -> np.ndarray:
        return (np.asarray(self.upper) - np.asarray(self.lower)) / np.asarray(self.shape)

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.widths))

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def edges(self, axis: int) -> np.ndarray:
        return np.linspace(self.lower[axis], self.upper[axis], self.shape[axis] + 1)

    def centers_1d(self, axis: int) -> np.ndarray:
        edges = self.edges(axis)
        return 0.5 * (edges[:-1] + edges[1:])

    def centers(self) -> np.ndarray:
        mesh = np.meshgrid(*(self.centers_1d(axis) for axis in range(3)), indexing="ij")
        return np.column_stack([item.ravel() for item in mesh])

    def cell_average_powers(
        self,
        axis: int,
        maximum_order: int,
        *,
        shift: float = 0.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Exact cell averages of ``((v-shift)/scale)**p``."""

        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("cell-moment scale must be finite and positive")
        edges = (self.edges(axis) - shift) / scale
        widths = edges[1:] - edges[:-1]
        values = np.ones((self.shape[axis], maximum_order + 1))
        for order in range(1, maximum_order + 1):
            values[:, order] = (
                edges[1:] ** (order + 1) - edges[:-1] ** (order + 1)
            ) / ((order + 1) * widths)
        return values

    def feature_matrix(
        self,
        indices: Sequence[MultiIndex],
        *,
        shift: Sequence[float] = (0.0, 0.0, 0.0),
        scale: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Return exact piecewise-constant cell averages of monomials."""

        maximum = max(max(index) for index in indices)
        tables = [
            self.cell_average_powers(
                axis, maximum, shift=float(shift[axis]), scale=float(scale[axis])
            )
            for axis in range(3)
        ]
        features = np.empty((self.size, len(indices)))
        for column, index in enumerate(indices):
            values = (
                tables[0][:, index[0], None, None]
                * tables[1][None, :, index[1], None]
                * tables[2][None, None, :, index[2]]
            )
            features[:, column] = values.ravel()
        return features


@dataclass(frozen=True)
class DVMState:
    grid: DVMGrid
    masses: np.ndarray

    def __post_init__(self) -> None:
        masses = np.asarray(self.masses, dtype=float)
        if masses.shape != (self.grid.size,):
            raise ValueError("DVM masses do not match the grid")
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("DVM cell masses must be finite and nonnegative")
        if float(np.sum(masses)) <= 0.0:
            raise ValueError("DVM mass must be positive")
        object.__setattr__(self, "masses", masses)

    def moments(self, indices: Sequence[MultiIndex] = HYQMOM_35_INDICES) -> np.ndarray:
        return _raw_feature_matrix(self.grid, tuple(indices)).T @ self.masses


@lru_cache(maxsize=12)
def _raw_feature_matrix(
    grid: DVMGrid, indices: tuple[MultiIndex, ...]
) -> np.ndarray:
    """Cache immutable exact raw cell features shared by every time step."""

    features = grid.feature_matrix(indices)
    features.setflags(write=False)
    return features


@dataclass(frozen=True)
class DVMProjectionDiagnostics:
    line_search_mode: str
    iterations: int
    line_search_activations: int
    line_search_backtracks: int
    relative_moment_residual: float
    scaled_constraint_residual: float
    minimum_probability: float
    probabilities_at_numerical_floor: int
    accepted_on_physical_residual: bool
    near_failure: bool


@dataclass(frozen=True)
class DVMStepDiagnostics:
    guided: bool
    proposal_minimum_mass: float
    final_minimum_mass: float
    mass_drift: float
    momentum_drift: float
    energy_drift: float
    weak_target_relative_increment: float
    projection: DVMProjectionDiagnostics | None
    initial_source_seconds: float
    sg_proposal_seconds: float
    proposal_source_seconds: float
    target_assembly_seconds: float
    projection_seconds: float
    final_diagnostics_seconds: float
    total_seconds: float


class CellMassMomentClosure:
    """Tail closure supplied directly by exact DVM cell integrals."""

    def __init__(self, state: DVMState):
        self.state = state
        values = state.moments(MOMENT_INDICES_6)
        self._cache = {
            index: float(value) for index, value in zip(MOMENT_INDICES_6, values)
        }

    def __call__(
        self,
        index: MultiIndex,
        moments: Sequence[float],
        macro: MacroscopicState,
    ) -> float:
        del moments, macro
        try:
            return self._cache[index]
        except KeyError as error:
            raise ValueError("DVM cell closure supports moments through degree six") from error


def bernoulli_function(value: np.ndarray | float) -> np.ndarray:
    """Stable Bernoulli function ``B(x)=x/(exp(x)-1)``."""

    x = np.asarray(value, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1.0e-5
    xs = x[small]
    result[small] = 1.0 - xs / 2.0 + xs**2 / 12.0 - xs**4 / 720.0
    regular = ~small
    result[regular] = x[regular] / np.expm1(x[regular])
    return result


def _raw_to_axis_standardized_targets(
    moments: np.ndarray,
    indices: Sequence[MultiIndex],
    shift: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    position = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}
    values = []
    for index in indices:
        total = 0.0
        for px, py, pz in product(
            range(index[0] + 1), range(index[1] + 1), range(index[2] + 1)
        ):
            raw = (px, py, pz)
            coefficient = 1.0
            for axis, exponent in enumerate((px, py, pz)):
                coefficient *= (
                    comb(index[axis], exponent)
                    * (-shift[axis]) ** (index[axis] - exponent)
                    / scale[axis] ** index[axis]
                )
            total += coefficient * moments[position[raw]]
        values.append(total)
    return np.asarray(values)


def project_cell_masses_minimum_kl(
    proposal: DVMState,
    target_moments: Sequence[float],
    *,
    tolerance: float = 1.0e-9,
    physical_relative_tolerance: float = 2.0e-10,
    maximum_iterations: int = 100,
    line_search_mode: Literal["residual", "objective"] = "residual",
) -> tuple[DVMState, DVMProjectionDiagnostics]:
    """Strictly positively reweight cells to an exact 35-moment target."""

    target = np.asarray(target_moments, dtype=float)
    if target.shape != (35,) or not np.all(np.isfinite(target)) or target[0] <= 0.0:
        raise ValueError("expected a finite positive 35-moment target")
    if line_search_mode not in ("residual", "objective"):
        raise ValueError("line_search_mode must be 'residual' or 'objective'")
    macro = macroscopic_state(target)
    scale = np.sqrt(np.maximum(np.diag(macro.covariance), 1.0e-14 * macro.theta))
    indices = HYQMOM_35_INDICES[1:]
    raw_features = proposal.grid.feature_matrix(
        indices, shift=macro.velocity, scale=scale
    )
    physical_features = _raw_feature_matrix(proposal.grid, HYQMOM_35_INDICES)
    standardized_target = _raw_to_axis_standardized_targets(
        target, indices, macro.velocity, scale
    ) / target[0]

    tiny = np.finfo(float).tiny
    prior = np.maximum(proposal.masses, tiny)
    prior /= np.sum(prior)
    feature_scale = np.sqrt(
        np.maximum(np.sum(prior[:, None] * raw_features**2, axis=0), 1.0e-24)
    )
    features = raw_features / feature_scale[None, :]
    scaled_target = standardized_target / feature_scale
    parameters = np.zeros(features.shape[1])
    log_prior = np.log(prior)

    def evaluate(candidate: np.ndarray):
        log_weights = log_prior + features @ candidate
        normalization = logsumexp(log_weights)
        log_probabilities = log_weights - normalization
        probabilities = np.exp(
            np.maximum(log_probabilities, np.log(np.finfo(float).tiny))
        )
        probabilities /= np.sum(probabilities)
        mean = probabilities @ features
        residual = mean - scaled_target
        objective = float(normalization - np.dot(candidate, scaled_target))
        return objective, probabilities, mean, residual

    objective, probabilities, mean, residual = evaluate(parameters)
    line_search_activations = 0
    line_search_backtracks = 0
    iterations = 0
    accepted_on_physical_residual = False

    def physical_relative_residual(candidate_probabilities: np.ndarray) -> float:
        reconstructed = physical_features.T @ (target[0] * candidate_probabilities)
        return float(
            np.linalg.norm(reconstructed - target)
            / max(np.linalg.norm(target), 1.0e-30)
        )

    def line_search_accepts(
        trial: tuple[float, np.ndarray, np.ndarray, np.ndarray],
        step: float,
        descent: float,
        residual_norm: float,
    ) -> bool:
        objective_decrease = bool(
            np.isfinite(trial[0])
            and trial[0] <= objective - 1.0e-4 * step * descent
        )
        if line_search_mode == "objective":
            return objective_decrease
        residual_decrease = bool(
            np.isfinite(trial[0])
            and np.linalg.norm(trial[3])
            <= (1.0 - 1.0e-4 * step) * residual_norm
        )
        return residual_decrease or objective_decrease

    for iterations in range(1, maximum_iterations + 1):
        residual_norm = float(np.linalg.norm(residual))
        if (
            residual_norm <= tolerance
            and physical_relative_residual(probabilities)
            <= physical_relative_tolerance
        ):
            break
        centered = features - mean[None, :]
        hessian = centered.T @ (probabilities[:, None] * centered)
        matrix_scale = max(float(np.linalg.norm(hessian, ord=np.inf)), 1.0)
        regularized = hessian + 1.0e-12 * matrix_scale * np.eye(hessian.shape[0])
        try:
            direction = np.linalg.solve(regularized, residual)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(regularized, residual, rcond=1.0e-12)[0]
        descent = float(np.dot(residual, direction))
        step = 1.0
        accepted = False
        local_backtracks = 0
        for _ in range(35):
            trial_parameters = parameters - step * direction
            trial = evaluate(trial_parameters)
            if line_search_accepts(trial, step, descent, residual_norm):
                parameters = trial_parameters
                objective, probabilities, mean, residual = trial
                accepted = True
                break
            step *= 0.5
            local_backtracks += 1
        if local_backtracks:
            line_search_activations += 1
            line_search_backtracks += local_backtracks
        if not accepted:
            if physical_relative_residual(probabilities) <= physical_relative_tolerance:
                accepted_on_physical_residual = True
                break
            # A nearly singular covariance Hessian can make the regularized
            # Newton direction cease to be useful before the physical moment
            # residual is resolved.  Fall back to the true steepest-descent
            # direction for the same convex dual objective; this changes no
            # target, support, or acceptance tolerance.
            direction = residual.copy()
            descent = float(np.dot(residual, residual))
            step = 1.0
            fallback_backtracks = 0
            for _ in range(50):
                trial_parameters = parameters - step * direction
                trial = evaluate(trial_parameters)
                if line_search_accepts(trial, step, descent, residual_norm):
                    parameters = trial_parameters
                    objective, probabilities, mean, residual = trial
                    accepted = True
                    break
                step *= 0.5
                fallback_backtracks += 1
            line_search_activations += 1
            line_search_backtracks += fallback_backtracks
        if not accepted:
            raise FloatingPointError(
                "DVM minimum-KL line search stalled; target may be outside the "
                "positive cell-feature convex hull; "
                f"scaled residual={residual_norm:.3e}, "
                "physical all-35 relative residual="
                f"{physical_relative_residual(probabilities):.3e}"
            )

    residual_norm = float(np.linalg.norm(residual))
    physical_residual = physical_relative_residual(probabilities)
    if (
        residual_norm > 10.0 * tolerance
        and physical_residual > physical_relative_tolerance
    ):
        raise FloatingPointError(
            "DVM target is unresolved on the fixed support: "
            f"iterations={iterations}, scaled residual={residual_norm:.3e}, "
            f"physical relative residual={physical_residual:.3e}"
        )
    masses = target[0] * probabilities
    projected = DVMState(proposal.grid, masses)
    relative_residual = physical_residual
    if relative_residual > physical_relative_tolerance:
        raise FloatingPointError(
            "DVM projection missed the physical target: "
            f"iterations={iterations}, scaled residual={residual_norm:.3e}, "
            f"relative residual={relative_residual:.3e}"
        )
    floor_count = int(np.sum(probabilities <= 1.01 * tiny))
    near_failure = bool(
        iterations >= int(0.8 * maximum_iterations)
        or line_search_backtracks >= 20
        or residual_norm > 5.0 * tolerance
        or accepted_on_physical_residual
    )
    return projected, DVMProjectionDiagnostics(
        line_search_mode=line_search_mode,
        iterations=iterations,
        line_search_activations=line_search_activations,
        line_search_backtracks=line_search_backtracks,
        relative_moment_residual=relative_residual,
        scaled_constraint_residual=residual_norm,
        minimum_probability=float(np.min(probabilities)),
        probabilities_at_numerical_floor=floor_count,
        accepted_on_physical_residual=accepted_on_physical_residual,
        near_failure=near_failure,
    )


def initialize_diagonal_gaussian_mixture(
    grid: DVMGrid,
    components: Iterable[tuple[float, Sequence[float], Sequence[Sequence[float]]]],
    *,
    match_exact_moments: bool = True,
) -> tuple[DVMState, DVMProjectionDiagnostics | None]:
    """Integrate a diagonal Gaussian mixture over cells and optionally project."""

    component_list = list(components)
    masses = np.zeros(grid.shape)
    for weight, mean, covariance in component_list:
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if not np.allclose(covariance, np.diag(np.diag(covariance)), atol=1.0e-13):
            raise ValueError("clean-room initializer currently requires diagonal covariance")
        if weight <= 0.0 or np.any(np.diag(covariance) <= 0.0):
            raise ValueError("mixture weights and variances must be positive")
        axis_masses = []
        for axis in range(3):
            z = (grid.edges(axis) - mean[axis]) / np.sqrt(covariance[axis, axis])
            probabilities = np.diff(ndtr(z))
            axis_masses.append(np.maximum(probabilities, 0.0))
        masses += float(weight) * (
            axis_masses[0][:, None, None]
            * axis_masses[1][None, :, None]
            * axis_masses[2][None, None, :]
        )
    masses = np.maximum(masses.ravel(), np.finfo(float).tiny)
    masses /= np.sum(masses)
    proposal = DVMState(grid, masses)
    if not match_exact_moments:
        return proposal, None

    # Local import avoids coupling the DVM evolution to an algebraic closure.
    from .moments import mixture_of_gaussians_moments_35

    target = mixture_of_gaussians_moments_35(component_list)
    return project_cell_masses_minimum_kl(proposal, target)


def _drift_at_centers(
    grid: DVMGrid,
    moments: np.ndarray,
    coefficients: CubicFPCoefficients,
) -> np.ndarray:
    macro = macroscopic_state(moments)
    c = grid.centers() - macro.velocity[None, :]
    c2 = np.einsum("ni,ni->n", c, c)
    theta = macro.theta if coefficients.theta is None else coefficients.theta
    drift = -c / coefficients.tau + c @ coefficients.C.T
    drift += (c2 - 3.0 * theta)[:, None] * coefficients.gamma[None, :]
    drift += coefficients.beta * (
        c2[:, None] * c - 2.0 * macro.heat_flux[None, :] / macro.rho
    )
    return drift.reshape(grid.shape + (3,))


def _implicit_sg_axis(
    density: np.ndarray,
    drift: np.ndarray,
    diffusion: float,
    width: float,
    dt: float,
    axis: int,
) -> np.ndarray:
    if dt == 0.0:
        return density
    moved_density = np.moveaxis(density, axis, -1)
    moved_drift = np.moveaxis(drift, axis, -1)
    line_shape = moved_density.shape
    values = moved_density.reshape(-1, line_shape[-1])
    drift_values = moved_drift.reshape(-1, line_shape[-1])
    face_drift = 0.5 * (drift_values[:, :-1] + drift_values[:, 1:])
    peclet = face_drift * width / diffusion
    b_forward = bernoulli_function(peclet)
    b_backward = bernoulli_function(-peclet)
    factor = diffusion / width**2
    output = np.empty_like(values)
    size = values.shape[1]
    for line in range(values.shape[0]):
        diagonal = np.ones(size)
        diagonal[:-1] += dt * factor * b_backward[line]
        diagonal[1:] += dt * factor * b_forward[line]
        upper = -dt * factor * b_forward[line]
        lower = -dt * factor * b_backward[line]
        matrix = np.zeros((3, size))
        matrix[0, 1:] = upper
        matrix[1] = diagonal
        matrix[2, :-1] = lower
        output[line] = solve_banded((1, 1), matrix, values[line], check_finite=False)
    result = output.reshape(line_shape)
    return np.moveaxis(result, -1, axis)


def scharfetter_gummel_proposal(
    state: DVMState,
    dt: float,
    coefficients: CubicFPCoefficients,
) -> DVMState:
    """Positive Strang-split implicit SG proposal with no-flux boundaries."""

    if dt <= 0.0:
        raise ValueError("DVM step size must be positive")
    moments = state.moments()
    macro = macroscopic_state(moments)
    theta = macro.theta if coefficients.theta is None else coefficients.theta
    diffusion = theta / coefficients.tau
    if diffusion <= 0.0:
        raise ValueError("DVM diffusion must be positive")
    drift = _drift_at_centers(state.grid, moments, coefficients)
    return scharfetter_gummel_prescribed_proposal(
        state, dt, drift=drift, diffusion=diffusion
    )


def scharfetter_gummel_prescribed_proposal(
    state: DVMState,
    dt: float,
    *,
    drift: np.ndarray,
    diffusion: float,
) -> DVMState:
    """Advance a prescribed drift-diffusion field with the SG composition.

    This manufactured-audit path isolates the finite-volume operator from the
    cubic coefficient solve. ``drift`` is frozen at cell centres for the
    interval and face values use the same arithmetic average as the physical
    cubic-FP proposal.
    """

    if dt <= 0.0:
        raise ValueError("DVM step size must be positive")
    drift_array = np.asarray(drift, dtype=float)
    if drift_array.shape != state.grid.shape + (3,):
        raise ValueError("prescribed drift must have shape grid.shape + (3,)")
    if not np.all(np.isfinite(drift_array)):
        raise ValueError("prescribed drift must be finite")
    if not np.isfinite(diffusion) or diffusion <= 0.0:
        raise ValueError("prescribed diffusion must be finite and positive")
    density = (state.masses / state.grid.cell_volume).reshape(state.grid.shape)
    for axis, fraction in ((0, 0.5), (1, 0.5), (2, 1.0), (1, 0.5), (0, 0.5)):
        density = _implicit_sg_axis(
            density,
            drift_array[..., axis],
            diffusion,
            float(state.grid.widths[axis]),
            fraction * dt,
            axis,
        )
    masses = density.ravel() * state.grid.cell_volume
    roundoff_floor = -1.0e-13 * max(float(np.sum(state.masses)), 1.0)
    if float(np.min(masses)) < roundoff_floor:
        raise FloatingPointError("implicit SG proposal produced negative cell mass")
    masses = np.maximum(masses, np.finfo(float).tiny)
    masses *= float(np.sum(state.masses)) / float(np.sum(masses))
    return DVMState(state.grid, masses)


def _invariant_defects(before: np.ndarray, after: np.ndarray) -> tuple[float, float, float]:
    position = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}
    mass = abs(after[position[(0, 0, 0)]] - before[position[(0, 0, 0)]])
    momentum = np.linalg.norm(
        [
            after[position[index]] - before[position[index]]
            for index in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ]
    )
    energy = abs(
        sum(after[position[index]] - before[position[index]] for index in (
            (2, 0, 0), (0, 2, 0), (0, 0, 2)
        ))
    )
    return float(mass), float(momentum), float(energy)


def dvm_cubic_fp_step(
    state: DVMState,
    dt: float,
    tau: float,
    *,
    prandtl: float = 2.0 / 3.0,
    guided: bool = True,
    projection_tolerance: float = 1.0e-9,
) -> tuple[DVMState, DVMStepDiagnostics]:
    """Advance one independently assembled cubic-FP DVM interval."""

    total_start = perf_counter()
    phase_start = perf_counter()
    initial_moments = state.moments()
    initial_closure = CellMassMomentClosure(state)
    initial_coefficients = coefficients_from_moments(
        initial_moments, tau=tau, prandtl=prandtl, closure=initial_closure
    )
    initial_source = projected_fp_collision_source(
        initial_moments, initial_coefficients, closure=initial_closure
    )
    initial_source_seconds = perf_counter() - phase_start

    phase_start = perf_counter()
    proposal = scharfetter_gummel_proposal(state, dt, initial_coefficients)
    sg_proposal_seconds = perf_counter() - phase_start
    projection = None
    weak_increment = 0.0
    proposal_source_seconds = 0.0
    target_assembly_seconds = 0.0
    projection_seconds = 0.0
    if guided:
        phase_start = perf_counter()
        proposal_moments = proposal.moments()
        proposal_closure = CellMassMomentClosure(proposal)
        proposal_coefficients = coefficients_from_moments(
            proposal_moments, tau=tau, prandtl=prandtl, closure=proposal_closure
        )
        proposal_source = projected_fp_collision_source(
            proposal_moments, proposal_coefficients, closure=proposal_closure
        )
        proposal_source_seconds = perf_counter() - phase_start

        phase_start = perf_counter()
        target = initial_moments + 0.5 * dt * (initial_source + proposal_source)
        weak_increment = float(
            np.linalg.norm(target - initial_moments)
            / max(np.linalg.norm(initial_moments), 1.0e-30)
        )
        target_assembly_seconds = perf_counter() - phase_start

        phase_start = perf_counter()
        final, projection = project_cell_masses_minimum_kl(
            proposal, target, tolerance=projection_tolerance
        )
        projection_seconds = perf_counter() - phase_start
    else:
        final = proposal

    phase_start = perf_counter()
    final_moments = final.moments()
    mass_drift, momentum_drift, energy_drift = _invariant_defects(
        initial_moments, final_moments
    )
    final_diagnostics_seconds = perf_counter() - phase_start
    total_seconds = perf_counter() - total_start
    return final, DVMStepDiagnostics(
        guided=guided,
        proposal_minimum_mass=float(np.min(proposal.masses)),
        final_minimum_mass=float(np.min(final.masses)),
        mass_drift=mass_drift,
        momentum_drift=momentum_drift,
        energy_drift=energy_drift,
        weak_target_relative_increment=weak_increment,
        projection=projection,
        initial_source_seconds=initial_source_seconds,
        sg_proposal_seconds=sg_proposal_seconds,
        proposal_source_seconds=proposal_source_seconds,
        target_assembly_seconds=target_assembly_seconds,
        projection_seconds=projection_seconds,
        final_diagnostics_seconds=final_diagnostics_seconds,
        total_seconds=total_seconds,
    )
