"""Reference-independent harder unseen cases for Stage 71.

The complete registry is fixed and fingerprinted before any QMC trajectory is
computed.  No QMC quantity is used in choosing the cases or model parameters.
"""
from __future__ import annotations
import hashlib, json
import numpy as np
from riemann35_patch.stage58_blind_generalization.blind_cases import BlindCase, _build_case

CASE_NAMES=(
    "rare_beam_3d",
    "dense_hot_extreme",
    "dilute_broad",
    "strong_anisotropy",
    "balanced_cross_3d",
)

def _iso(): return np.repeat(np.eye(3)[None,:,:],4,axis=0)

def _fp(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":"),default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x).encode()).hexdigest()

def _wrap(name, **spec):
    base=_build_case(name,role="stage71_blind",**spec)
    config=dict(base.configuration)
    config["schema"]="riemann35-stage71-frozen-case-v1"
    config["stage71_design_note"]="Chosen before any Stage71 QMC evaluation; closure parameters and gates inherited unchanged from Stage58/70."
    return BlindCase(name=name,role="stage71_blind",components=base.components,moments=base.moments,configuration=config,fingerprint=_fp(config),audit=base.audit)

def _specs():
    rare_means=np.asarray([
        [-1.10,-0.20,0.10],[0.20,1.25,-0.25],[1.45,-0.55,0.45],[3.60,-2.40,1.80]
    ])
    dense_means=np.asarray([
        [-1.45,-0.35,0.25],[0.15,1.55,-0.45],[1.75,-0.65,0.55],[-0.55,-1.85,-0.35]
    ])
    broad_means=np.asarray([
        [-1.00,-0.15,0.20],[0.30,1.10,-0.35],[1.25,-0.80,0.30],[-0.40,-1.35,-0.25]
    ])
    aniso_means=np.asarray([
        [-1.35,-0.45,0.70],[0.30,1.55,-0.55],[1.70,-0.60,0.35],[-0.55,-1.50,-0.80]
    ])
    aniso_cov=np.asarray([
        [[1.60,0.16,0.05],[0.16,0.24,-0.02],[0.05,-0.02,0.10]],
        [[0.18,-0.04,0.02],[-0.04,1.75,0.18],[0.02,0.18,0.22]],
        [[0.26,0.05,-0.07],[0.05,0.16,0.02],[-0.07,0.02,1.85]],
        [[1.10,-0.14,0.08],[-0.14,0.32,-0.05],[0.08,-0.05,0.14]],
    ])
    cross_means=np.asarray([
        [-1.55,0.25,0.85],[0.65,1.45,-0.70],[1.50,-0.80,0.55],[-0.75,-1.35,-0.95]
    ])
    cross_cov=np.asarray([
        [[0.90,0.14,-0.06],[0.14,0.45,0.05],[-0.06,0.05,0.30]],
        [[0.35,-0.08,0.04],[-0.08,1.05,0.11],[0.04,0.11,0.42]],
        [[0.48,0.06,-0.10],[0.06,0.38,0.02],[-0.10,0.02,1.12]],
        [[0.72,-0.09,0.07],[-0.09,0.60,-0.04],[0.07,-0.04,0.33]],
    ])
    return {
      "rare_beam_3d":dict(density=1.05,bulk_velocity=np.asarray([0.28,-0.34,0.22]),energy_trace=1.55,internal_energy_fraction=0.018,weights=np.asarray([0.58,0.25,0.12,0.05]),raw_means=rare_means,raw_covariances=_iso(),euler_degrees=(61.0,-28.0,37.0)),
      "dense_hot_extreme":dict(density=2.20,bulk_velocity=np.asarray([0.55,-0.42,0.35]),energy_trace=1.90,internal_energy_fraction=0.025,weights=np.asarray([0.48,0.27,0.17,0.08]),raw_means=dense_means,raw_covariances=_iso(),euler_degrees=(-49.0,33.0,71.0)),
      "dilute_broad":dict(density=0.42,bulk_velocity=np.asarray([-0.42,0.36,-0.31]),energy_trace=0.62,internal_energy_fraction=0.24,weights=np.asarray([0.44,0.24,0.20,0.12]),raw_means=broad_means,raw_covariances=_iso(),euler_degrees=(36.0,64.0,-27.0)),
      "strong_anisotropy":dict(density=0.78,bulk_velocity=np.asarray([-0.12,0.27,0.38]),energy_trace=1.45,internal_energy_fraction=0.13,weights=np.asarray([0.32,0.28,0.22,0.18]),raw_means=aniso_means,raw_covariances=aniso_cov,euler_degrees=(-57.0,24.0,53.0)),
      "balanced_cross_3d":dict(density=1.35,bulk_velocity=np.asarray([0.22,0.19,-0.33]),energy_trace=1.25,internal_energy_fraction=0.07,weights=np.asarray([0.27,0.26,0.24,0.23]),raw_means=cross_means,raw_covariances=cross_cov,euler_degrees=(73.0,-41.0,16.0)),
    }

def hard_case(name):
    specs=_specs()
    if name not in specs: raise KeyError(name)
    return _wrap(name,**specs[name])

def registry_manifest():
    cases=[hard_case(n) for n in CASE_NAMES]
    fps={c.name:c.fingerprint for c in cases}
    digest=hashlib.sha256(json.dumps(fps,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {
      "schema":"riemann35-stage71-frozen-registry-v1",
      "case_order":list(CASE_NAMES),
      "case_fingerprints":fps,
      "registry_fingerprint":digest,
      "qmc_used_to_define_cases":False,
      "closure_parameters_refit":False,
      "inherited_gates":{"reference_spread":0.02,"time_change":0.01,"heat_flux":0.01,"third_tensor":0.03,"trace_free":0.05,"component":0.03},
    }
