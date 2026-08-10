"""Causal positive tail memory for the cubic-FP HyQMOM-35 model.

Moments through total degree four do not identify the fifth- and sixth-order
tail used by the cubic Fokker--Planck source.  This module therefore does not
construct another instantaneous tail closure.  It supplies the reusable
pieces of an adaptive macro--micro collision update:

* an online disagreement sensor evaluated only from the retained 35 moments;
* a two-threshold, dwell-time hysteresis policy;
* positive entropy projection of a known/inherited velocity microstate onto
  the current 35 moments; and
* a lifecycle that can discard a microstate conservatively, but refuses to
  invent one after the tail information has been lost.

The current implementation is homogeneous.  Spatial transport and neighbor
selection belong to the subsequent coupling stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
from scipy.special import logsumexp

from .collision import coefficients_from_moments, projected_fp_collision_source
from .grad_hyqmom import (
    WeightedNodeTailClosure,
    reconstruct_grad_hyqmom_quadrature,
)
from .mixture_closure import (
    finite_gaussian_mixture_fp_step,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
)
from .moments import (
    HYQMOM_35_INDICES,
    central_moment,
    macroscopic_state,
    mixture_of_gaussians_moments_35,
)
from .particle_reference import moments_35_from_particles
from .qmc_reference import qmc_cubic_fp_step, sample_gaussian_mixture_qmc


TAIL_INDICES = tuple(
    (i, j, order - i - j)
    for order in (5, 6)
    for i in range(order + 1)
    for j in range(order - i + 1)
)
FOURTH_POSITIONS = np.asarray(
    [sum(index) == 4 for index in HYQMOM_35_INDICES], dtype=bool
)


@dataclass(frozen=True)
class KineticSensorReading:
    """Online closure-disagreement indicators for one 35-moment state."""

    fourth_source_disagreement: float
    tail_disagreement: float
    standardized_skewness_norm: float
    reconstruction_failure: bool
    stage9_status: str
    grad_status: str


@dataclass(frozen=True)
class ActivationHysteresis:
    """Two-threshold activation policy with dwell and release holds."""

    source_on: float = 0.10124
    source_off: float = 0.05062
    tail_on: float = 0.41005
    tail_off: float = 0.205025
    skew_on: float = 1.0e-3
    skew_off: float = 5.0e-4
    activation_hold_steps: int = 1
    release_hold_steps: int = 8
    minimum_active_steps: int = 20

    def __post_init__(self) -> None:
        values = (
            self.source_on,
            self.source_off,
            self.tail_on,
            self.tail_off,
            self.skew_on,
            self.skew_off,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("hysteresis thresholds must be finite and nonnegative")
        if (
            self.source_off >= self.source_on
            or self.tail_off >= self.tail_on
            or self.skew_off >= self.skew_on
        ):
            raise ValueError("each off threshold must be below its on threshold")
        if self.activation_hold_steps < 1 or self.release_hold_steps < 1:
            raise ValueError("hysteresis hold counts must be positive")
        if self.minimum_active_steps < 0:
            raise ValueError("minimum_active_steps must be nonnegative")

    def requests_activation(self, reading: KineticSensorReading) -> bool:
        return bool(
            reading.reconstruction_failure
            or (
                reading.fourth_source_disagreement >= self.source_on
                and reading.standardized_skewness_norm >= self.skew_on
            )
            or reading.tail_disagreement >= self.tail_on
        )

    def permits_release(self, reading: KineticSensorReading) -> bool:
        return bool(
            not reading.reconstruction_failure
            and (
                reading.fourth_source_disagreement <= self.source_off
                or reading.standardized_skewness_norm <= self.skew_off
            )
            and reading.tail_disagreement <= self.tail_off
        )


@dataclass(frozen=True)
class PositiveMicrostate:
    """A positive weighted velocity state with explicit causal provenance."""

    velocities: np.ndarray
    weights: np.ndarray
    rho: float
    provenance: str


@dataclass(frozen=True)
class MicroProjectionDiagnostics:
    """Accuracy and conditioning of a positive entropy projection."""

    relative_moment_residual: float
    scaled_constraint_residual: float
    iterations: int
    minimum_probability: float


@dataclass(frozen=True)
class AdaptiveTailMemoryState:
    """Homogeneous macro--micro lifecycle state."""

    moments: np.ndarray
    microstate: PositiveMicrostate | None
    mode: str
    global_step: int
    steps_in_mode: int
    activation_counter: int
    release_counter: int
    transition_count: int
    tail_ambiguous: bool
    noise_seed: int
    sensor_reading: KineticSensorReading


@dataclass(frozen=True)
class AdaptiveStepDiagnostics:
    """One adaptive collision step and its representation transition."""

    mode_before: str
    mode_after: str
    transition: str
    used_micro_step: bool
    activation_requested: bool
    activation_blocked: bool
    tail_ambiguous: bool
    sensor_before: KineticSensorReading
    sensor_after: KineticSensorReading
    sensor_evaluated: bool
    realizability_margin: float
    projection_relative_residual: float


def _symmetric_relative_difference(
    left: np.ndarray, right: np.ndarray, floor: float
) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(0.5 * (np.linalg.norm(left) + np.linalg.norm(right)), floor)
    )


def _standardized_skewness_norm(moments: np.ndarray) -> float:
    state = macroscopic_state(moments)
    eigenvalues, eigenvectors = np.linalg.eigh(state.covariance)
    scale = max(float(np.max(eigenvalues)), state.theta, 1.0)
    if np.min(eigenvalues) <= 1.0e-13 * scale:
        return float("inf")
    inverse_root = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    third = np.empty((3, 3, 3), dtype=float)
    for directions in product(range(3), repeat=3):
        index = [0, 0, 0]
        for direction in directions:
            index[direction] += 1
        third[directions] = central_moment(moments, tuple(index)) / state.rho
    standardized = np.einsum(
        "ia,jb,kc,abc->ijk",
        inverse_root,
        inverse_root,
        inverse_root,
        third,
        optimize=True,
    )
    return float(np.linalg.norm(standardized))


def kinetic_activation_sensor(
    moments: Sequence[float],
    *,
    tau: float = 1.0,
    prandtl: float = 2.0 / 3.0,
) -> KineticSensorReading:
    """Evaluate the Stage-19 source/tail disagreement sensor online."""

    vector = np.asarray(moments, dtype=float)
    if vector.shape != (35,) or not np.all(np.isfinite(vector)):
        raise ValueError("expected a finite 35-moment vector")
    state = macroscopic_state(vector)
    closures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    statuses: dict[str, str] = {}
    for name, builder in (
        ("stage9", reconstruct_gaussian_mixture_quadrature),
        ("grad", reconstruct_grad_hyqmom_quadrature),
    ):
        try:
            quadrature = builder(vector)
            closure = WeightedNodeTailClosure(
                quadrature.nodes, quadrature.weights
            )
            coefficients = coefficients_from_moments(
                vector, tau=tau, prandtl=prandtl, closure=closure
            )
            source = projected_fp_collision_source(
                vector, coefficients, closure=closure
            )
            tail = np.asarray(
                [closure(index, vector, state) for index in TAIL_INDICES],
                dtype=float,
            )
            if not np.all(np.isfinite(source)) or not np.all(np.isfinite(tail)):
                raise FloatingPointError("closure produced NaN or infinity")
            closures[name] = (source, tail)
            statuses[name] = "PASS"
        except Exception as error:
            statuses[name] = f"{type(error).__name__}: {error}"

    failed = len(closures) != 2
    if failed:
        source_disagreement = float("inf")
        tail_disagreement = float("inf")
    else:
        source_scale = state.rho * state.theta**2
        source_disagreement = _symmetric_relative_difference(
            closures["stage9"][0][FOURTH_POSITIONS],
            closures["grad"][0][FOURTH_POSITIONS],
            1.0e-8 * source_scale,
        )
        tail_scales = np.asarray(
            [
                state.rho * state.theta ** (sum(index) / 2.0)
                for index in TAIL_INDICES
            ]
        )
        tail_disagreement = _symmetric_relative_difference(
            closures["stage9"][1] / tail_scales,
            closures["grad"][1] / tail_scales,
            1.0e-8,
        )
    return KineticSensorReading(
        fourth_source_disagreement=source_disagreement,
        tail_disagreement=tail_disagreement,
        standardized_skewness_norm=_standardized_skewness_norm(vector),
        reconstruction_failure=failed,
        stage9_status=statuses["stage9"],
        grad_status=statuses["grad"],
    )


def _standardized_targets(moments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = macroscopic_state(moments)
    eigenvalues, eigenvectors = np.linalg.eigh(state.covariance)
    scale = max(float(np.max(eigenvalues)), state.theta, 1.0)
    if np.min(eigenvalues) <= 1.0e-13 * scale:
        raise ValueError("target covariance is singular")
    inverse_root = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    tensors: dict[int, np.ndarray] = {}
    for order in (3, 4):
        central = np.empty((3,) * order, dtype=float)
        for directions in product(range(3), repeat=order):
            index = [0, 0, 0]
            for direction in directions:
                index[direction] += 1
            central[directions] = central_moment(moments, tuple(index)) / state.rho
        if order == 3:
            tensors[order] = np.einsum(
                "ia,jb,kc,abc->ijk",
                inverse_root,
                inverse_root,
                inverse_root,
                central,
                optimize=True,
            )
        else:
            tensors[order] = np.einsum(
                "ia,jb,kc,ld,abcd->ijkl",
                inverse_root,
                inverse_root,
                inverse_root,
                inverse_root,
                central,
                optimize=True,
            )
    targets = []
    for index in HYQMOM_35_INDICES:
        order = sum(index)
        directions = tuple(
            direction
            for direction, exponent in enumerate(index)
            for _ in range(exponent)
        )
        if order == 0:
            value = 1.0
        elif order == 1:
            value = 0.0
        elif order == 2:
            value = 1.0 if directions[0] == directions[1] else 0.0
        else:
            value = float(tensors[order][directions])
        targets.append(value)
    return np.asarray(targets), inverse_root


def _monomial_features(nodes: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            nodes[:, 0] ** i * nodes[:, 1] ** j * nodes[:, 2] ** k
            for i, j, k in HYQMOM_35_INDICES[1:]
        ]
    )


def project_positive_microstate(
    target_moments: Sequence[float],
    candidate: PositiveMicrostate,
    *,
    tolerance: float = 1.0e-9,
    maximum_iterations: int = 100,
) -> tuple[PositiveMicrostate, MicroProjectionDiagnostics]:
    """Positively reweight a causal candidate to match all 35 target moments.

    The velocity support is inherited unchanged.  If the target lies outside
    its discrete convex hull, the function fails explicitly instead of
    fabricating a tail or silently changing the transported moments.
    """

    target = np.asarray(target_moments, dtype=float)
    nodes = np.asarray(candidate.velocities, dtype=float)
    prior_weights = np.asarray(candidate.weights, dtype=float)
    if target.shape != (35,) or not np.all(np.isfinite(target)):
        raise ValueError("expected a finite 35-moment target")
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 35:
        raise ValueError("microstate support must have shape (n>=35, 3)")
    if prior_weights.shape != (nodes.shape[0],) or np.any(prior_weights <= 0.0):
        raise ValueError("microstate projection requires strictly positive weights")
    state = macroscopic_state(target)
    targets, inverse_root = _standardized_targets(target)
    standardized_nodes = (nodes - state.velocity) @ inverse_root.T
    raw_features = _monomial_features(standardized_nodes)
    prior = prior_weights / np.sum(prior_weights)
    raw_target = targets[1:]
    feature_scale = np.sqrt(
        np.maximum(np.sum(prior[:, None] * raw_features**2, axis=0), 1.0e-24)
    )
    features = raw_features / feature_scale[None, :]
    target_scaled = raw_target / feature_scale
    parameters = np.zeros(features.shape[1])
    log_prior = np.log(prior)

    def evaluate(candidate_parameters: np.ndarray):
        log_weights = log_prior + features @ candidate_parameters
        normalization = logsumexp(log_weights)
        probabilities = np.exp(log_weights - normalization)
        mean = probabilities @ features
        residual = mean - target_scaled
        objective = float(
            normalization - np.dot(candidate_parameters, target_scaled)
        )
        return objective, probabilities, mean, residual

    objective, probabilities, mean, residual = evaluate(parameters)
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            break
        centered = features - mean[None, :]
        hessian = centered.T @ (probabilities[:, None] * centered)
        matrix_scale = max(float(np.linalg.norm(hessian, ord=np.inf)), 1.0)
        regularized = hessian + 1.0e-12 * matrix_scale * np.eye(
            hessian.shape[0]
        )
        try:
            direction = np.linalg.solve(regularized, residual)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(
                regularized, residual, rcond=1.0e-12
            )[0]
        descent = float(np.dot(residual, direction))
        step = 1.0
        accepted = False
        for _ in range(35):
            trial_parameters = parameters - step * direction
            trial = evaluate(trial_parameters)
            if (
                np.isfinite(trial[0])
                and trial[0] <= objective - 1.0e-4 * step * descent
            ):
                parameters = trial_parameters
                objective, probabilities, mean, residual = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if residual_norm <= 10.0 * tolerance:
                break
            raise FloatingPointError(
                "positive microstate projection line search stalled"
            )
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > 10.0 * tolerance:
        raise FloatingPointError(
            "target moments are not resolved on the inherited positive support: "
            f"scaled residual={residual_norm:.3e}"
        )
    weights = state.rho * probabilities
    reconstructed = moments_35_from_particles(
        nodes, weights, rho=state.rho
    )
    relative_residual = float(
        np.linalg.norm(reconstructed - target)
        / max(np.linalg.norm(target), 1.0e-15)
    )
    if relative_residual > 2.0e-8:
        raise FloatingPointError(
            "positive projection did not reproduce the physical moments: "
            f"relative residual={relative_residual:.3e}"
        )
    projected = PositiveMicrostate(
        velocities=nodes.copy(),
        weights=weights,
        rho=state.rho,
        provenance=candidate.provenance,
    )
    return projected, MicroProjectionDiagnostics(
        relative_moment_residual=relative_residual,
        scaled_constraint_residual=residual_norm,
        iterations=iterations,
        minimum_probability=float(np.min(probabilities)),
    )


def positive_microstate_from_components(
    components: Iterable[
        tuple[float, Sequence[float], Sequence[Sequence[float]]]
    ],
    *,
    points_per_component: int = 1024,
    seed: int = 20_260_810,
    provenance: str = "known-mixture-initialization",
) -> tuple[PositiveMicrostate, np.ndarray, MicroProjectionDiagnostics]:
    """Build and conservatively project a positive state from known physics."""

    component_list = list(components)
    target = mixture_of_gaussians_moments_35(component_list)
    nodes, weights = sample_gaussian_mixture_qmc(
        component_list,
        points_per_component=points_per_component,
        seed=seed,
    )
    candidate = PositiveMicrostate(
        velocities=nodes,
        weights=weights,
        rho=float(target[0]),
        provenance=provenance,
    )
    projected, diagnostics = project_positive_microstate(target, candidate)
    return projected, target, diagnostics


def positive_microstate_moments(microstate: PositiveMicrostate) -> np.ndarray:
    """Project a positive microstate to the transported 35 moments."""

    return moments_35_from_particles(
        microstate.velocities,
        microstate.weights,
        rho=microstate.rho,
    )


def initialize_adaptive_tail_memory(
    moments: Sequence[float],
    *,
    candidate_microstate: PositiveMicrostate | None,
    hysteresis: ActivationHysteresis = ActivationHysteresis(),
    force_causal_birth: bool = False,
    noise_seed: int = 20_260_810,
    tau: float = 1.0,
    prandtl: float = 2.0 / 3.0,
) -> AdaptiveTailMemoryState:
    """Initialize the lifecycle without reconstructing a missing tail."""

    vector = np.asarray(moments, dtype=float)
    reading = kinetic_activation_sensor(
        vector, tau=tau, prandtl=prandtl
    )
    requested = hysteresis.requests_activation(reading)
    microstate = None
    tail_ambiguous = False
    mode = "macro"
    if candidate_microstate is not None and (requested or force_causal_birth):
        microstate, _ = project_positive_microstate(vector, candidate_microstate)
        mode = "micro"
    elif requested:
        tail_ambiguous = True
    return AdaptiveTailMemoryState(
        moments=vector.copy(),
        microstate=microstate,
        mode=mode,
        global_step=0,
        steps_in_mode=0,
        activation_counter=int(requested and mode == "macro"),
        release_counter=0,
        transition_count=0,
        tail_ambiguous=tail_ambiguous,
        noise_seed=int(noise_seed),
        sensor_reading=reading,
    )


def adaptive_tail_memory_fp_step(
    adaptive: AdaptiveTailMemoryState,
    dt: float,
    tau: float,
    *,
    hysteresis: ActivationHysteresis = ActivationHysteresis(),
    prandtl: float = 2.0 / 3.0,
    incoming_microstate: PositiveMicrostate | None = None,
    sensor_interval_steps: int = 10,
) -> tuple[AdaptiveTailMemoryState, AdaptiveStepDiagnostics]:
    """Advance one causal adaptive macro--micro collision interval."""

    if dt <= 0.0 or tau <= 0.0:
        raise ValueError("dt and tau must be positive")
    if sensor_interval_steps < 1:
        raise ValueError("sensor_interval_steps must be positive")
    if adaptive.mode not in ("macro", "micro"):
        raise ValueError("adaptive mode must be 'macro' or 'micro'")
    if (adaptive.mode == "micro") != (adaptive.microstate is not None):
        raise ValueError("micro mode and microstate presence are inconsistent")

    before = np.asarray(adaptive.moments, dtype=float)
    sensor_before = adaptive.sensor_reading
    activation_requested = hysteresis.requests_activation(sensor_before)
    activation_counter = adaptive.activation_counter
    mode_before = adaptive.mode
    microstate = adaptive.microstate
    projection_residual = 0.0
    activation_blocked = False
    used_micro_step = mode_before == "micro"
    transition = f"{mode_before}->{mode_before}"
    tail_ambiguous = adaptive.tail_ambiguous
    transition_count = adaptive.transition_count
    steps_in_mode = adaptive.steps_in_mode
    release_counter = adaptive.release_counter

    if (
        mode_before == "macro"
        and activation_counter >= hysteresis.activation_hold_steps
    ):
        if incoming_microstate is None:
            activation_blocked = True
            tail_ambiguous = True
        else:
            microstate, projection = project_positive_microstate(
                before, incoming_microstate
            )
            projection_residual = projection.relative_moment_residual
            used_micro_step = True
            transition = "macro->micro"
            transition_count += 1
            steps_in_mode = 0
            activation_counter = 0
            release_counter = 0

    if used_micro_step:
        if microstate is None:  # pragma: no cover - protected by checks above
            raise RuntimeError("micro step requested without a microstate")
        updated_velocities, _ = qmc_cubic_fp_step(
            microstate.velocities,
            microstate.weights,
            dt=dt,
            tau=tau,
            seed=adaptive.noise_seed + 104_729 * (adaptive.global_step + 1),
            prandtl=prandtl,
        )
        microstate = PositiveMicrostate(
            velocities=updated_velocities,
            weights=microstate.weights.copy(),
            rho=microstate.rho,
            provenance=microstate.provenance,
        )
        updated = positive_microstate_moments(microstate)
    else:
        updated, _ = finite_gaussian_mixture_fp_step(
            before,
            dt,
            tau,
            prandtl=prandtl,
            speed_cap=np.inf,
        )

    sensor_evaluated = (
        (adaptive.global_step + 1) % sensor_interval_steps == 0
    )
    sensor_after = (
        kinetic_activation_sensor(updated, tau=tau, prandtl=prandtl)
        if sensor_evaluated
        else sensor_before
    )
    mode_after = "micro" if used_micro_step else "macro"
    if mode_after == "micro":
        steps_in_mode += 1
        if sensor_evaluated:
            release_counter = (
                release_counter + 1
                if hysteresis.permits_release(sensor_after)
                else 0
            )
        if (
            steps_in_mode >= hysteresis.minimum_active_steps
            and release_counter >= hysteresis.release_hold_steps
        ):
            mode_after = "macro"
            microstate = None
            transition = "micro->macro"
            transition_count += 1
            steps_in_mode = 0
            activation_counter = 0
            release_counter = 0
            tail_ambiguous = False
        elif transition != "macro->micro":
            transition = "micro->micro"
    else:
        steps_in_mode += 1
        microstate = None
        release_counter = 0
        if sensor_evaluated:
            activation_counter = (
                activation_counter + 1
                if hysteresis.requests_activation(sensor_after)
                else 0
            )
        if activation_blocked:
            transition = "macro->blocked"
        else:
            transition = "macro->macro"

    margin = float(realizability_margin_35(updated))
    if not np.all(np.isfinite(updated)) or margin < -5.0e-13:
        raise FloatingPointError("adaptive update produced a non-realizable state")
    next_state = AdaptiveTailMemoryState(
        moments=np.asarray(updated, dtype=float),
        microstate=microstate,
        mode=mode_after,
        global_step=adaptive.global_step + 1,
        steps_in_mode=steps_in_mode,
        activation_counter=activation_counter,
        release_counter=release_counter,
        transition_count=transition_count,
        tail_ambiguous=tail_ambiguous,
        noise_seed=adaptive.noise_seed,
        sensor_reading=sensor_after,
    )
    return next_state, AdaptiveStepDiagnostics(
        mode_before=mode_before,
        mode_after=mode_after,
        transition=transition,
        used_micro_step=used_micro_step,
        activation_requested=activation_requested,
        activation_blocked=activation_blocked,
        tail_ambiguous=tail_ambiguous,
        sensor_before=sensor_before,
        sensor_after=sensor_after,
        sensor_evaluated=sensor_evaluated,
        realizability_margin=margin,
        projection_relative_residual=projection_residual,
    )
