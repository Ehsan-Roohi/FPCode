"""Frozen, reference-independent initial states for Stage 58.

The case registry is deliberately deterministic and contains no QMC-derived
quantity.  It is hashed before any reference trajectory is evaluated so the
generalization test is prospective and auditable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from hyqmom_fp import HYQMOM_35_INDICES, macroscopic_state, mixture_of_gaussians_moments_35
from hyqmom_fp.moments import central_moment


ANCHOR_CASE = "stage57_anchor"
BLIND_CASES = (
    "hot_dense_shifted",
    "broad_shifted",
    "alternate_weights",
    "anisotropic_3d",
)
CASE_NAMES = (ANCHOR_CASE, *BLIND_CASES)
THIRD_INDICES = tuple(index for index in HYQMOM_35_INDICES if sum(index) == 3)


@dataclass(frozen=True)
class BlindCase:
    """One frozen positive four-Gaussian initial state."""

    name: str
    role: str
    components: tuple[tuple[float, np.ndarray, np.ndarray], ...]
    moments: np.ndarray
    configuration: dict[str, object]
    fingerprint: str
    audit: dict[str, object]


BASE_WEIGHTS = np.asarray([0.45, 0.25, 0.20, 0.10], dtype=float)
BASE_MEANS = np.asarray(
    [
        [-1.15, -0.25, 0.0],
        [0.10, 1.35, 0.0],
        [1.55, -0.45, 0.0],
        [-0.35, -1.70, 0.0],
    ],
    dtype=float,
)


def _rotation_x(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )


def _rotation_y(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def _rotation_z(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _fingerprint(configuration: dict[str, object]) -> str:
    payload = json.dumps(
        _jsonable(configuration), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_case(
    name: str,
    *,
    role: str,
    density: float,
    bulk_velocity: np.ndarray,
    energy_trace: float,
    internal_energy_fraction: float,
    weights: np.ndarray,
    raw_means: np.ndarray,
    raw_covariances: np.ndarray,
    euler_degrees: tuple[float, float, float],
) -> BlindCase:
    probabilities = np.array(weights, dtype=float, copy=True)
    probabilities /= np.sum(probabilities)
    means = np.array(raw_means, dtype=float, copy=True)
    covariances = np.array(raw_covariances, dtype=float, copy=True)
    raw_covariance_record = covariances.copy()
    velocity = np.array(bulk_velocity, dtype=float, copy=True)
    if probabilities.shape != (4,) or means.shape != (4, 3):
        raise ValueError("Stage-58 cases must have four weights and four 3-D means")
    if covariances.shape != (4, 3, 3):
        raise ValueError("raw covariances must have shape (4, 3, 3)")
    if density <= 0.0 or energy_trace <= 0.0:
        raise ValueError("density and energy trace must be positive")
    if not 0.0 < internal_energy_fraction < 1.0:
        raise ValueError("internal energy fraction must lie between zero and one")
    if np.any(probabilities <= 0.0):
        raise ValueError("all population probabilities must be positive")
    for covariance in covariances:
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("all raw covariances must be SPD")

    centered = means - probabilities @ means
    center_trace = float(np.sum(probabilities * np.sum(centered**2, axis=1)))
    covariance_trace = float(
        sum(
            probability * np.trace(covariance)
            for probability, covariance in zip(probabilities, covariances)
        )
    )
    centered *= np.sqrt(
        (1.0 - internal_energy_fraction) * energy_trace / center_trace
    )
    covariances *= internal_energy_fraction * energy_trace / covariance_trace
    rz, ry, rx = euler_degrees
    rotation = _rotation_x(rx) @ _rotation_y(ry) @ _rotation_z(rz)
    transformed_means = velocity + centered @ rotation.T
    transformed_covariances = np.asarray(
        [rotation @ covariance @ rotation.T for covariance in covariances]
    )
    components = tuple(
        (
            density * float(probability),
            mean.copy(),
            covariance.copy(),
        )
        for probability, mean, covariance in zip(
            probabilities, transformed_means, transformed_covariances
        )
    )
    moments = mixture_of_gaussians_moments_35(components)
    macro = macroscopic_state(moments)
    thirds = np.asarray([central_moment(moments, index) for index in THIRD_INDICES])
    configuration = {
        "schema": "riemann35-stage58-frozen-case-v1",
        "name": name,
        "role": role,
        "density": density,
        "bulk_velocity": velocity,
        "energy_trace": energy_trace,
        "internal_energy_fraction": internal_energy_fraction,
        "weights": probabilities,
        "raw_means": means,
        "raw_covariances": raw_covariance_record,
        "euler_degrees_z_y_x": euler_degrees,
    }
    audit = {
        "mass_error": abs(float(macro.rho) - density),
        "bulk_velocity_error": float(np.linalg.norm(macro.velocity - velocity)),
        "energy_trace_error": abs(float(np.trace(macro.covariance)) - energy_trace),
        "minimum_covariance_eigenvalue": float(
            min(np.min(np.linalg.eigvalsh(item)) for item in transformed_covariances)
        ),
        "third_component_norm": float(np.linalg.norm(thirds)),
        "minimum_absolute_third_component": float(np.min(np.abs(thirds))),
        "third_components": thirds,
    }
    return BlindCase(
        name=name,
        role=role,
        components=components,
        moments=moments,
        configuration=_jsonable(configuration),
        fingerprint=_fingerprint(configuration),
        audit=_jsonable(audit),
    )


def _isotropic_covariances() -> np.ndarray:
    return np.repeat(np.eye(3)[None, :, :], 4, axis=0)


def _case_specifications() -> dict[str, dict[str, object]]:
    alternate_means = np.asarray(
        [
            [-1.40, -0.10, 0.15],
            [0.25, 1.10, -0.20],
            [1.35, -0.75, 0.30],
            [-0.20, -1.55, -0.35],
        ]
    )
    three_dimensional_means = np.asarray(
        [
            [-1.20, -0.35, 0.55],
            [0.20, 1.45, -0.40],
            [1.60, -0.50, 0.20],
            [-0.45, -1.35, -0.65],
        ]
    )
    anisotropic_covariances = np.asarray(
        [
            [[1.00, 0.18, 0.06], [0.18, 0.42, -0.04], [0.06, -0.04, 0.20]],
            [[0.28, -0.06, 0.04], [-0.06, 1.20, 0.15], [0.04, 0.15, 0.34]],
            [[0.52, 0.10, -0.09], [0.10, 0.30, 0.03], [-0.09, 0.03, 1.35]],
            [[0.80, -0.12, 0.08], [-0.12, 0.55, -0.06], [0.08, -0.06, 0.24]],
        ]
    )
    return {
        ANCHOR_CASE: {
            "role": "stage57_anchor",
            "density": 1.0,
            "bulk_velocity": np.zeros(3),
            "energy_trace": 1.0,
            "internal_energy_fraction": 0.03,
            "weights": BASE_WEIGHTS,
            "raw_means": BASE_MEANS,
            "raw_covariances": _isotropic_covariances(),
            "euler_degrees": (17.0, 29.0, 41.0),
        },
        "hot_dense_shifted": {
            "role": "blind",
            "density": 1.60,
            "bulk_velocity": np.asarray([0.32, -0.21, 0.14]),
            "energy_trace": 1.40,
            "internal_energy_fraction": 0.03,
            "weights": BASE_WEIGHTS,
            "raw_means": BASE_MEANS,
            "raw_covariances": _isotropic_covariances(),
            "euler_degrees": (43.0, -31.0, 22.0),
        },
        "broad_shifted": {
            "role": "blind",
            "density": 0.70,
            "bulk_velocity": np.asarray([-0.24, 0.29, -0.12]),
            "energy_trace": 0.80,
            "internal_energy_fraction": 0.15,
            "weights": BASE_WEIGHTS,
            "raw_means": BASE_MEANS,
            "raw_covariances": _isotropic_covariances(),
            "euler_degrees": (-37.0, 18.0, 63.0),
        },
        "alternate_weights": {
            "role": "blind",
            "density": 1.25,
            "bulk_velocity": np.asarray([0.18, 0.11, -0.27]),
            "energy_trace": 1.10,
            "internal_energy_fraction": 0.055,
            "weights": np.asarray([0.38, 0.31, 0.19, 0.12]),
            "raw_means": alternate_means,
            "raw_covariances": _isotropic_covariances(),
            "euler_degrees": (28.0, 52.0, -19.0),
        },
        "anisotropic_3d": {
            "role": "blind",
            "density": 0.90,
            "bulk_velocity": np.asarray([-0.16, -0.23, 0.31]),
            "energy_trace": 1.25,
            "internal_energy_fraction": 0.09,
            "weights": np.asarray([0.34, 0.27, 0.23, 0.16]),
            "raw_means": three_dimensional_means,
            "raw_covariances": anisotropic_covariances,
            "euler_degrees": (-22.0, 37.0, 48.0),
        },
    }


def blind_case(name: str) -> BlindCase:
    """Materialize one named frozen case without reading reference data."""

    specifications = _case_specifications()
    if name not in specifications:
        raise KeyError(f"unknown Stage-58 case: {name}")
    return _build_case(name, **specifications[name])


def registry_manifest() -> dict[str, object]:
    """Return the complete frozen case registry and its aggregate hash."""

    cases = [blind_case(name) for name in CASE_NAMES]
    fingerprints = {case.name: case.fingerprint for case in cases}
    digest = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "riemann35-stage58-frozen-registry-v1",
        "case_order": list(CASE_NAMES),
        "blind_cases": list(BLIND_CASES),
        "case_fingerprints": fingerprints,
        "registry_fingerprint": digest,
        "qmc_used_to_define_cases": False,
        "model_only_preflight_disclosure": (
            "candidate regularization fractions 0.020 and 0.025 were excluded before "
            "any QMC evaluation because their closure-only fine/coarse full-third history "
            "changes were 0.083 and 0.072; Stage 58 makes no claim below 0.030"
        ),
    }
