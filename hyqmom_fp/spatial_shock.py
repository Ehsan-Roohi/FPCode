"""One-dimensional normal-shock transport for the Stage-25 spatial gate.

The physical velocity remains three-dimensional.  Only the configuration
space is reduced to one dimension.  The module deliberately separates three
paths:

* a positive full-DVM reference with conservative upwind face fluxes;
* an inexpensive 35-moment macro path using the existing positive Gaussian
  mixture as a kinetic flux closure; and
* a causal adaptive path whose active cells transport positive DVM masses.

An alarm is never allowed to create a kinetic state from the same 35 moments.
Adaptive birth requires a known initial/inflow state or an active neighbour.
The algebraic DVM representation used at a safe macro/micro interface is
temporary and supplies only the incoming face flux; it is not retained as
kinetic memory and cannot satisfy a birth request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .adaptive import ActivationHysteresis, kinetic_activation_sensor
from .dvm_reference import (
    DVMGrid,
    DVMState,
    dvm_cubic_fp_step,
    initialize_diagonal_gaussian_mixture,
    project_cell_masses_minimum_kl,
)
from .mixture_closure import (
    finite_gaussian_mixture_fp_step,
    realizability_margin_35,
    reconstruct_gaussian_mixture_quadrature,
)
from .moments import (
    HYQMOM_35_INDICES,
    macroscopic_state,
    maxwellian_moments_35,
)


POSITION = {index: offset for offset, index in enumerate(HYQMOM_35_INDICES)}
MASS_POSITION = POSITION[(0, 0, 0)]
MOMENTUM_POSITIONS = np.asarray(
    [POSITION[(1, 0, 0)], POSITION[(0, 1, 0)], POSITION[(0, 0, 1)]],
    dtype=int,
)
ENERGY_POSITIONS = np.asarray(
    [POSITION[(2, 0, 0)], POSITION[(0, 2, 0)], POSITION[(0, 0, 2)]],
    dtype=int,
)


@dataclass(frozen=True)
class SpatialGrid1D:
    """Uniform finite-volume grid in physical space."""

    lower: float
    upper: float
    cells: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("spatial bounds must be finite")
        if self.upper <= self.lower or self.cells < 3:
            raise ValueError("spatial grid requires upper>lower and at least 3 cells")

    @property
    def width(self) -> float:
        return (self.upper - self.lower) / self.cells

    @property
    def centers(self) -> np.ndarray:
        return self.lower + (np.arange(self.cells) + 0.5) * self.width


@dataclass(frozen=True)
class NormalShockStates:
    """Rankine-Hugoniot end states for a monatomic normal shock."""

    mach: float
    gamma: float
    upstream_density: float
    upstream_velocity: float
    upstream_theta: float
    downstream_density: float
    downstream_velocity: float
    downstream_theta: float

    @property
    def upstream_moments(self) -> np.ndarray:
        return maxwellian_moments_35(
            self.upstream_density,
            (self.upstream_velocity, 0.0, 0.0),
            self.upstream_theta,
        )

    @property
    def downstream_moments(self) -> np.ndarray:
        return maxwellian_moments_35(
            self.downstream_density,
            (self.downstream_velocity, 0.0, 0.0),
            self.downstream_theta,
        )


def normal_shock_rankine_hugoniot(
    mach: float = 3.0,
    *,
    gamma: float = 5.0 / 3.0,
    upstream_density: float = 1.0,
    upstream_theta: float = 1.0,
) -> NormalShockStates:
    """Return ideal-gas end states in the shock-fixed frame."""

    if mach <= 1.0 or gamma <= 1.0:
        raise ValueError("normal shock requires Mach>1 and gamma>1")
    if upstream_density <= 0.0 or upstream_theta <= 0.0:
        raise ValueError("upstream density and theta must be positive")
    density_ratio = (
        (gamma + 1.0) * mach**2
        / ((gamma - 1.0) * mach**2 + 2.0)
    )
    pressure_ratio = 1.0 + 2.0 * gamma * (mach**2 - 1.0) / (gamma + 1.0)
    temperature_ratio = pressure_ratio / density_ratio
    upstream_velocity = mach * np.sqrt(gamma * upstream_theta)
    downstream_density = upstream_density * density_ratio
    downstream_velocity = upstream_velocity / density_ratio
    downstream_theta = upstream_theta * temperature_ratio
    return NormalShockStates(
        mach=float(mach),
        gamma=float(gamma),
        upstream_density=float(upstream_density),
        upstream_velocity=float(upstream_velocity),
        upstream_theta=float(upstream_theta),
        downstream_density=float(downstream_density),
        downstream_velocity=float(downstream_velocity),
        downstream_theta=float(downstream_theta),
    )


@dataclass(frozen=True)
class SpatialDVMState:
    """Positive DVM cell masses in every physical-space cell."""

    spatial_grid: SpatialGrid1D
    velocity_grid: DVMGrid
    masses: np.ndarray

    def __post_init__(self) -> None:
        masses = np.asarray(self.masses, dtype=float)
        expected = (self.spatial_grid.cells, self.velocity_grid.size)
        if masses.shape != expected:
            raise ValueError(f"spatial DVM masses must have shape {expected}")
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("spatial DVM masses must be finite and nonnegative")
        if np.any(np.sum(masses, axis=1) <= 0.0):
            raise ValueError("each spatial DVM cell must have positive mass")
        object.__setattr__(self, "masses", masses)

    def moments(self) -> np.ndarray:
        features = self.velocity_grid.feature_matrix(HYQMOM_35_INDICES)
        return self.masses @ features


@dataclass(frozen=True)
class TransportDiagnostics:
    cfl: float
    minimum_mass: float
    mass_balance_residual: float
    momentum_balance_residual: float
    energy_balance_residual: float
    left_boundary_flux: np.ndarray
    right_boundary_flux: np.ndarray


@dataclass(frozen=True)
class FullDVMStepDiagnostics:
    transport: TransportDiagnostics
    minimum_mass: float
    maximum_projection_residual: float
    maximum_collision_invariant_drift: float


def _maxwellian_dvm(
    grid: DVMGrid, rho: float, velocity_x: float, theta: float
) -> DVMState:
    state, _ = initialize_diagonal_gaussian_mixture(
        grid,
        [(rho, (velocity_x, 0.0, 0.0), theta * np.eye(3))],
        match_exact_moments=True,
    )
    return state


def initialize_normal_shock_dvm(
    spatial_grid: SpatialGrid1D,
    velocity_grid: DVMGrid,
    shock: NormalShockStates,
    *,
    discontinuity_location: float = 0.0,
) -> tuple[SpatialDVMState, DVMState, DVMState]:
    """Initialize a piecewise-Maxwellian shock and exact inflow states."""

    left = _maxwellian_dvm(
        velocity_grid,
        shock.upstream_density,
        shock.upstream_velocity,
        shock.upstream_theta,
    )
    right = _maxwellian_dvm(
        velocity_grid,
        shock.downstream_density,
        shock.downstream_velocity,
        shock.downstream_theta,
    )
    mask = spatial_grid.centers < discontinuity_location
    masses = np.where(mask[:, None], left.masses[None, :], right.masses[None, :])
    return SpatialDVMState(spatial_grid, velocity_grid, masses), left, right


def _dvm_face_mass_fluxes(
    state: SpatialDVMState,
    left_inflow: DVMState,
    right_inflow: DVMState,
) -> np.ndarray:
    if left_inflow.grid != state.velocity_grid or right_inflow.grid != state.velocity_grid:
        raise ValueError("DVM boundary grids must match the spatial state")
    vx = state.velocity_grid.centers()[:, 0]
    positive = np.maximum(vx, 0.0)
    negative = np.minimum(vx, 0.0)
    flux = np.empty((state.spatial_grid.cells + 1, state.velocity_grid.size))
    flux[0] = positive * left_inflow.masses + negative * state.masses[0]
    flux[1:-1] = (
        positive[None, :] * state.masses[:-1]
        + negative[None, :] * state.masses[1:]
    )
    flux[-1] = positive * state.masses[-1] + negative * right_inflow.masses
    return flux


def _balance_diagnostics(
    before: np.ndarray,
    after: np.ndarray,
    left_flux: np.ndarray,
    right_flux: np.ndarray,
    *,
    dx: float,
    dt: float,
    minimum_mass: float,
    cfl: float,
) -> TransportDiagnostics:
    residual = dx * np.sum(after - before, axis=0) + dt * (right_flux - left_flux)
    mass_scale = max(
        dx * float(np.sum(np.abs(before[:, MASS_POSITION])))
        + dt * (abs(left_flux[MASS_POSITION]) + abs(right_flux[MASS_POSITION])),
        1.0e-30,
    )
    momentum_scale = max(
        dx * float(np.sum(np.linalg.norm(before[:, MOMENTUM_POSITIONS], axis=1)))
        + dt * (
            float(np.linalg.norm(left_flux[MOMENTUM_POSITIONS]))
            + float(np.linalg.norm(right_flux[MOMENTUM_POSITIONS]))
        ),
        1.0e-30,
    )
    before_energy = np.sum(before[:, ENERGY_POSITIONS], axis=1)
    left_energy_flux = float(np.sum(left_flux[ENERGY_POSITIONS]))
    right_energy_flux = float(np.sum(right_flux[ENERGY_POSITIONS]))
    energy_scale = max(
        dx * float(np.sum(np.abs(before_energy)))
        + dt * (abs(left_energy_flux) + abs(right_energy_flux)),
        1.0e-30,
    )
    return TransportDiagnostics(
        cfl=float(cfl),
        minimum_mass=float(minimum_mass),
        mass_balance_residual=float(abs(residual[MASS_POSITION]) / mass_scale),
        momentum_balance_residual=float(
            np.linalg.norm(residual[MOMENTUM_POSITIONS]) / momentum_scale
        ),
        energy_balance_residual=float(
            abs(np.sum(residual[ENERGY_POSITIONS])) / energy_scale
        ),
        left_boundary_flux=np.asarray(left_flux, dtype=float),
        right_boundary_flux=np.asarray(right_flux, dtype=float),
    )


def dvm_upwind_transport_step(
    state: SpatialDVMState,
    dt: float,
    left_inflow: DVMState,
    right_inflow: DVMState,
    *,
    cfl_limit: float = 1.0,
) -> tuple[SpatialDVMState, TransportDiagnostics]:
    """Advance conservative positive free transport by one upwind step."""

    if dt <= 0.0:
        raise ValueError("transport step size must be positive")
    maximum_speed = float(np.max(np.abs(state.velocity_grid.centers()[:, 0])))
    cfl = dt * maximum_speed / state.spatial_grid.width
    if cfl > cfl_limit * (1.0 + 1.0e-14):
        raise ValueError(f"DVM transport CFL {cfl:.6g} exceeds {cfl_limit:.6g}")
    before_moments = state.moments()
    flux = _dvm_face_mass_fluxes(state, left_inflow, right_inflow)
    masses = state.masses - dt / state.spatial_grid.width * (flux[1:] - flux[:-1])
    tolerance = -2.0e-14 * max(float(np.max(np.sum(state.masses, axis=1))), 1.0)
    if float(np.min(masses)) < tolerance:
        raise FloatingPointError("upwind DVM transport produced negative mass")
    masses = np.maximum(masses, np.finfo(float).tiny)
    updated = SpatialDVMState(state.spatial_grid, state.velocity_grid, masses)
    features = state.velocity_grid.feature_matrix(HYQMOM_35_INDICES)
    left_flux = features.T @ flux[0]
    right_flux = features.T @ flux[-1]
    diagnostics = _balance_diagnostics(
        before_moments,
        updated.moments(),
        left_flux,
        right_flux,
        dx=state.spatial_grid.width,
        dt=dt,
        minimum_mass=float(np.min(masses)),
        cfl=cfl,
    )
    return updated, diagnostics


def full_dvm_shock_step(
    state: SpatialDVMState,
    dt: float,
    tau: float,
    left_inflow: DVMState,
    right_inflow: DVMState,
    *,
    prandtl: float = 2.0 / 3.0,
    guided: bool = True,
) -> tuple[SpatialDVMState, FullDVMStepDiagnostics]:
    """Lie-split conservative transport and independent cubic-FP collision."""

    transported, transport_diagnostics = dvm_upwind_transport_step(
        state, dt, left_inflow, right_inflow
    )
    final_masses = np.empty_like(transported.masses)
    maximum_projection_residual = 0.0
    maximum_invariant_drift = 0.0
    for cell in range(state.spatial_grid.cells):
        final, diagnostics = dvm_cubic_fp_step(
            DVMState(state.velocity_grid, transported.masses[cell]),
            dt,
            tau,
            prandtl=prandtl,
            guided=guided,
        )
        final_masses[cell] = final.masses
        if diagnostics.projection is not None:
            maximum_projection_residual = max(
                maximum_projection_residual,
                diagnostics.projection.relative_moment_residual,
            )
        maximum_invariant_drift = max(
            maximum_invariant_drift,
            diagnostics.mass_drift,
            diagnostics.momentum_drift,
            diagnostics.energy_drift,
        )
    final = SpatialDVMState(state.spatial_grid, state.velocity_grid, final_masses)
    return final, FullDVMStepDiagnostics(
        transport=transport_diagnostics,
        minimum_mass=float(np.min(final_masses)),
        maximum_projection_residual=float(maximum_projection_residual),
        maximum_collision_invariant_drift=float(maximum_invariant_drift),
    )


def initialize_normal_shock_moments(
    spatial_grid: SpatialGrid1D,
    shock: NormalShockStates,
    *,
    discontinuity_location: float = 0.0,
) -> np.ndarray:
    mask = spatial_grid.centers < discontinuity_location
    return np.where(
        mask[:, None],
        shock.upstream_moments[None, :],
        shock.downstream_moments[None, :],
    )


def _point_features(nodes: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [nodes[:, 0] ** i * nodes[:, 1] ** j * nodes[:, 2] ** k
         for i, j, k in HYQMOM_35_INDICES]
    )


def _macro_split_flux(moments: Sequence[float], positive: bool) -> tuple[np.ndarray, float]:
    quadrature = reconstruct_gaussian_mixture_quadrature(moments)
    vx = quadrature.nodes[:, 0]
    selector = vx >= 0.0 if positive else vx < 0.0
    flux = _point_features(quadrature.nodes[selector]).T @ (
        quadrature.weights[selector] * vx[selector]
    )
    return flux, float(np.max(np.abs(vx)))


def macro_upwind_transport_step(
    moments: np.ndarray,
    spatial_grid: SpatialGrid1D,
    dt: float,
    left_boundary: Sequence[float],
    right_boundary: Sequence[float],
    *,
    cfl_limit: float = 1.0,
) -> tuple[np.ndarray, TransportDiagnostics]:
    """Positive-kinetic-flux transport of the 35-moment macro state."""

    values = np.asarray(moments, dtype=float)
    if values.shape != (spatial_grid.cells, 35):
        raise ValueError("macro field must have shape (spatial cells, 35)")
    left_positive, speed_left = _macro_split_flux(left_boundary, True)
    right_negative, speed_right = _macro_split_flux(right_boundary, False)
    positive_fluxes = []
    negative_fluxes = []
    maximum_speed = max(speed_left, speed_right)
    for cell in range(spatial_grid.cells):
        positive_flux, speed_positive = _macro_split_flux(values[cell], True)
        negative_flux, speed_negative = _macro_split_flux(values[cell], False)
        positive_fluxes.append(positive_flux)
        negative_fluxes.append(negative_flux)
        maximum_speed = max(maximum_speed, speed_positive, speed_negative)
    cfl = dt * maximum_speed / spatial_grid.width
    if cfl > cfl_limit * (1.0 + 1.0e-14):
        raise ValueError(f"macro transport CFL {cfl:.6g} exceeds {cfl_limit:.6g}")
    flux = np.empty((spatial_grid.cells + 1, 35))
    flux[0] = left_positive + negative_fluxes[0]
    for face in range(1, spatial_grid.cells):
        flux[face] = positive_fluxes[face - 1] + negative_fluxes[face]
    flux[-1] = positive_fluxes[-1] + right_negative
    updated = values - dt / spatial_grid.width * (flux[1:] - flux[:-1])
    margins = np.asarray([realizability_margin_35(item) for item in updated])
    if not np.all(np.isfinite(updated)) or float(np.min(margins)) < -2.0e-12:
        raise FloatingPointError("macro kinetic-flux transport left the realizability cone")
    diagnostics = _balance_diagnostics(
        values,
        updated,
        flux[0],
        flux[-1],
        dx=spatial_grid.width,
        dt=dt,
        minimum_mass=float(np.min(updated[:, MASS_POSITION])),
        cfl=cfl,
    )
    return updated, diagnostics


def macro_shock_step(
    moments: np.ndarray,
    spatial_grid: SpatialGrid1D,
    dt: float,
    tau: float,
    left_boundary: Sequence[float],
    right_boundary: Sequence[float],
    *,
    prandtl: float = 2.0 / 3.0,
) -> tuple[np.ndarray, TransportDiagnostics]:
    transported, diagnostics = macro_upwind_transport_step(
        moments, spatial_grid, dt, left_boundary, right_boundary
    )
    updated = np.empty_like(transported)
    for cell in range(spatial_grid.cells):
        updated[cell], _ = finite_gaussian_mixture_fp_step(
            transported[cell], dt, tau, prandtl=prandtl, speed_cap=np.inf
        )
    return updated, diagnostics


def stage25_hysteresis() -> ActivationHysteresis:
    """Frozen Stage-24B thresholds used without spatial retuning."""

    return ActivationHysteresis(
        source_on=0.10124,
        source_off=0.05062,
        tail_on=0.40,
        tail_off=0.205025,
        skew_on=1.0e-3,
        skew_off=5.0e-4,
        activation_hold_steps=1,
        release_hold_steps=8,
        minimum_active_steps=20,
    )


@dataclass(frozen=True)
class AdaptiveSpatialState:
    """Macro field plus positive causal DVM memory in selected cells."""

    spatial_grid: SpatialGrid1D
    velocity_grid: DVMGrid
    moments: np.ndarray
    micro_masses: np.ndarray
    active: np.ndarray
    active_steps: np.ndarray
    release_counter: np.ndarray
    global_step: int
    transition_count: int
    blocked_births: int

    def __post_init__(self) -> None:
        nx = self.spatial_grid.cells
        moments = np.asarray(self.moments, dtype=float)
        masses = np.asarray(self.micro_masses, dtype=float)
        active = np.asarray(self.active, dtype=bool)
        if moments.shape != (nx, 35):
            raise ValueError("adaptive moments have the wrong shape")
        if masses.shape != (nx, self.velocity_grid.size):
            raise ValueError("adaptive micro masses have the wrong shape")
        if active.shape != (nx,):
            raise ValueError("adaptive active mask has the wrong shape")
        if np.any(masses[active] < 0.0) or not np.all(np.isfinite(masses[active])):
            raise ValueError("active micro masses must be finite and nonnegative")
        object.__setattr__(self, "moments", moments)
        object.__setattr__(self, "micro_masses", masses)
        object.__setattr__(self, "active", active)


@dataclass(frozen=True)
class AdaptiveSpatialStepDiagnostics:
    active_fraction: float
    activations: int
    releases: int
    blocked_births: int
    interface_macro_reconstructions: int
    maximum_micro_macro_residual: float
    minimum_micro_mass: float
    transport: TransportDiagnostics


def initialize_adaptive_normal_shock(
    spatial_grid: SpatialGrid1D,
    velocity_grid: DVMGrid,
    shock: NormalShockStates,
    *,
    discontinuity_location: float = 0.0,
    initial_active_half_width: int = 1,
) -> tuple[AdaptiveSpatialState, DVMState, DVMState]:
    """Pre-activate the known initial shock interface before tails are mixed."""

    full, left, right = initialize_normal_shock_dvm(
        spatial_grid, velocity_grid, shock,
        discontinuity_location=discontinuity_location,
    )
    moments = initialize_normal_shock_moments(
        spatial_grid, shock, discontinuity_location=discontinuity_location
    )
    closest = int(np.argmin(np.abs(spatial_grid.centers - discontinuity_location)))
    active = np.zeros(spatial_grid.cells, dtype=bool)
    lower = max(0, closest - initial_active_half_width)
    upper = min(spatial_grid.cells, closest + initial_active_half_width + 1)
    active[lower:upper] = True
    micro_masses = np.zeros_like(full.masses)
    micro_masses[active] = full.masses[active]
    return AdaptiveSpatialState(
        spatial_grid=spatial_grid,
        velocity_grid=velocity_grid,
        moments=moments,
        micro_masses=micro_masses,
        active=active,
        active_steps=np.zeros(spatial_grid.cells, dtype=int),
        release_counter=np.zeros(spatial_grid.cells, dtype=int),
        global_step=0,
        transition_count=int(np.sum(active)),
        blocked_births=0,
    ), left, right


def _safe_macro_interface_state(grid: DVMGrid, target: np.ndarray) -> DVMState:
    macro = macroscopic_state(target)
    covariance = np.diag(np.maximum(np.diag(macro.covariance), 1.0e-12 * macro.theta))
    proposal, _ = initialize_diagonal_gaussian_mixture(
        grid,
        [(macro.rho, macro.velocity, covariance)],
        match_exact_moments=False,
    )
    projected, _ = project_cell_masses_minimum_kl(proposal, target)
    return projected


def _activate_from_donor(
    target: np.ndarray,
    donor: DVMState,
) -> DVMState:
    projected, _ = project_cell_masses_minimum_kl(donor, target)
    return projected


def adaptive_shock_step(
    state: AdaptiveSpatialState,
    dt: float,
    tau: float,
    left_inflow: DVMState,
    right_inflow: DVMState,
    *,
    hysteresis: ActivationHysteresis | None = None,
    prandtl: float = 2.0 / 3.0,
) -> tuple[AdaptiveSpatialState, AdaptiveSpatialStepDiagnostics]:
    """Advance one causal hybrid shock interval with shared conservative fluxes."""

    policy = stage25_hysteresis() if hysteresis is None else hysteresis
    nx = state.spatial_grid.cells
    features = state.velocity_grid.feature_matrix(HYQMOM_35_INDICES)
    vx = state.velocity_grid.centers()[:, 0]
    positive = np.maximum(vx, 0.0)
    negative = np.minimum(vx, 0.0)
    maximum_speed = float(np.max(np.abs(vx)))
    cfl = dt * maximum_speed / state.spatial_grid.width
    if cfl > 1.0 + 1.0e-14:
        raise ValueError("adaptive DVM transport CFL exceeds unity")

    active = state.active.copy()
    masses = state.micro_masses.copy()
    active_steps = state.active_steps.copy()
    release_counter = state.release_counter.copy()
    activations = 0
    releases = 0
    blocked = 0

    readings = [
        kinetic_activation_sensor(item, tau=tau, prandtl=prandtl)
        for item in state.moments
    ]
    for cell, reading in enumerate(readings):
        if active[cell] or not policy.requests_activation(reading):
            continue
        donor = None
        if cell > 0 and active[cell - 1]:
            donor = DVMState(state.velocity_grid, masses[cell - 1])
        elif cell + 1 < nx and active[cell + 1]:
            donor = DVMState(state.velocity_grid, masses[cell + 1])
        elif cell == 0:
            donor = left_inflow
        elif cell == nx - 1:
            donor = right_inflow
        if donor is None:
            blocked += 1
            continue
        try:
            born = _activate_from_donor(state.moments[cell], donor)
        except (FloatingPointError, ValueError):
            blocked += 1
            continue
        masses[cell] = born.masses
        active[cell] = True
        active_steps[cell] = 0
        release_counter[cell] = 0
        activations += 1

    macro_positive = [None] * nx
    macro_negative = [None] * nx
    macro_speed = maximum_speed
    for cell in range(nx):
        if active[cell]:
            continue
        macro_positive[cell], speed_positive = _macro_split_flux(
            state.moments[cell], True
        )
        macro_negative[cell], speed_negative = _macro_split_flux(
            state.moments[cell], False
        )
        macro_speed = max(macro_speed, speed_positive, speed_negative)
    macro_cfl = dt * macro_speed / state.spatial_grid.width
    if macro_cfl > 1.0 + 1.0e-14:
        raise ValueError("adaptive macro transport CFL exceeds unity")

    interface_cache: dict[int, DVMState] = {}
    mass_flux: dict[int, np.ndarray] = {}
    moment_flux = np.empty((nx + 1, 35))
    interface_reconstructions = 0

    def side_dvm(cell: int, boundary: str | None = None) -> DVMState:
        nonlocal interface_reconstructions
        if boundary == "left":
            return left_inflow
        if boundary == "right":
            return right_inflow
        if active[cell]:
            return DVMState(state.velocity_grid, masses[cell])
        if cell not in interface_cache:
            interface_cache[cell] = _safe_macro_interface_state(
                state.velocity_grid, state.moments[cell]
            )
            interface_reconstructions += 1
        return interface_cache[cell]

    for face in range(nx + 1):
        left_cell = face - 1
        right_cell = face
        kinetic_face = (
            (left_cell >= 0 and active[left_cell])
            or (right_cell < nx and active[right_cell])
        )
        if kinetic_face:
            left_state = side_dvm(left_cell, "left" if face == 0 else None)
            right_state = side_dvm(right_cell, "right" if face == nx else None)
            current_mass_flux = positive * left_state.masses + negative * right_state.masses
            mass_flux[face] = current_mass_flux
            moment_flux[face] = features.T @ current_mass_flux
        elif face == 0:
            left_positive, _ = _macro_split_flux(left_inflow.moments(), True)
            moment_flux[face] = left_positive + macro_negative[0]
        elif face == nx:
            right_negative, _ = _macro_split_flux(right_inflow.moments(), False)
            moment_flux[face] = macro_positive[-1] + right_negative
        else:
            moment_flux[face] = macro_positive[left_cell] + macro_negative[right_cell]

    transported_moments = state.moments - dt / state.spatial_grid.width * (
        moment_flux[1:] - moment_flux[:-1]
    )
    transported_masses = masses.copy()
    for cell in np.flatnonzero(active):
        if cell not in mass_flux or cell + 1 not in mass_flux:
            raise RuntimeError("active cell is missing a kinetic face flux")
        transported_masses[cell] = masses[cell] - dt / state.spatial_grid.width * (
            mass_flux[cell + 1] - mass_flux[cell]
        )
    if np.any(transported_masses[active] < -2.0e-14):
        raise FloatingPointError("adaptive kinetic transport produced negative mass")
    transported_masses[active] = np.maximum(
        transported_masses[active], np.finfo(float).tiny
    )

    sync_residual = 0.0
    for cell in np.flatnonzero(active):
        micro_moments = features.T @ transported_masses[cell]
        sync_residual = max(
            sync_residual,
            float(np.linalg.norm(micro_moments - transported_moments[cell])
                  / max(np.linalg.norm(transported_moments[cell]), 1.0e-30)),
        )
        transported_moments[cell] = micro_moments

    updated_moments = np.empty_like(transported_moments)
    updated_masses = transported_masses.copy()
    for cell in range(nx):
        if active[cell]:
            final, _ = dvm_cubic_fp_step(
                DVMState(state.velocity_grid, transported_masses[cell]),
                dt,
                tau,
                prandtl=prandtl,
                guided=True,
            )
            updated_masses[cell] = final.masses
            updated_moments[cell] = final.moments()
            active_steps[cell] += 1
        else:
            updated_moments[cell], _ = finite_gaussian_mixture_fp_step(
                transported_moments[cell],
                dt,
                tau,
                prandtl=prandtl,
                speed_cap=np.inf,
            )

    final_readings = [
        kinetic_activation_sensor(item, tau=tau, prandtl=prandtl)
        for item in updated_moments
    ]
    for cell in np.flatnonzero(active):
        if policy.permits_release(final_readings[cell]):
            release_counter[cell] += 1
        else:
            release_counter[cell] = 0
        if (
            active_steps[cell] >= policy.minimum_active_steps
            and release_counter[cell] >= policy.release_hold_steps
        ):
            active[cell] = False
            updated_masses[cell] = 0.0
            active_steps[cell] = 0
            release_counter[cell] = 0
            releases += 1

    margins = np.asarray([realizability_margin_35(item) for item in updated_moments])
    if float(np.min(margins)) < -2.0e-12:
        raise FloatingPointError("adaptive spatial update left the realizability cone")
    transport_diagnostics = _balance_diagnostics(
        state.moments,
        transported_moments,
        moment_flux[0],
        moment_flux[-1],
        dx=state.spatial_grid.width,
        dt=dt,
        minimum_mass=float(np.min(transported_moments[:, MASS_POSITION])),
        cfl=max(cfl, macro_cfl),
    )
    next_state = AdaptiveSpatialState(
        spatial_grid=state.spatial_grid,
        velocity_grid=state.velocity_grid,
        moments=updated_moments,
        micro_masses=updated_masses,
        active=active,
        active_steps=active_steps,
        release_counter=release_counter,
        global_step=state.global_step + 1,
        transition_count=state.transition_count + activations + releases,
        blocked_births=state.blocked_births + blocked,
    )
    minimum_micro_mass = (
        float(np.min(updated_masses[active])) if np.any(active) else 0.0
    )
    return next_state, AdaptiveSpatialStepDiagnostics(
        active_fraction=float(np.mean(active)),
        activations=activations,
        releases=releases,
        blocked_births=blocked,
        interface_macro_reconstructions=interface_reconstructions,
        maximum_micro_macro_residual=float(sync_residual),
        minimum_micro_mass=minimum_micro_mass,
        transport=transport_diagnostics,
    )


def shock_profiles(moments: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the frozen Stage-25 profile observables."""

    values = np.asarray(moments, dtype=float)
    states = [macroscopic_state(item) for item in values]
    return {
        "rho": np.asarray([item.rho for item in states]),
        "velocity_x": np.asarray([item.velocity[0] for item in states]),
        "theta": np.asarray([item.theta for item in states]),
        "stress_xx": np.asarray([item.stress[0, 0] for item in states]),
        "heat_flux_x": np.asarray([item.heat_flux[0] for item in states]),
        "M300": values[:, POSITION[(3, 0, 0)]].copy(),
        "M400": values[:, POSITION[(4, 0, 0)]].copy(),
    }


def normalized_profile_error(model: np.ndarray, reference: np.ndarray) -> float:
    """Directed relative L2 profile error used by the spatial smoke report."""

    candidate = np.asarray(model, dtype=float)
    target = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(candidate - target) / max(np.linalg.norm(target), 1.0e-30))
