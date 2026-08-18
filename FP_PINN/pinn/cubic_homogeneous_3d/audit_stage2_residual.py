#!/usr/bin/env python3
"""Evaluate a Stage-2 checkpoint on a fresh particle-free residual panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from cubic_operator import (
    ALL_CASE_NAMES,
    case_default_nu,
    case_has_heat_flux,
    case_is_axisymmetric_heat_flux,
    initial_heat_flux_qx,
)
from train_stage2 import Config, DensityModel, make_train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=ALL_CASE_NAMES, default="heat_flux")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--n-time-batch", type=int, default=12)
    parser.add_argument("--n-velocity-per-time", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--nu", type=float)
    args = parser.parse_args()
    if args.nu is None:
        args.nu = case_default_nu(args.case)
    return args


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    config = Config(
        case=args.case,
        output_dir="unused",
        reference="unused",
        width=args.width,
        depth=args.depth,
        n_time_batch=args.n_time_batch,
        n_velocity_per_time=args.n_velocity_per_time,
        fixed_velocity_quadrature=True,
        quadrature_panels=1,
        axisymmetric_heat_flux=case_is_axisymmetric_heat_flux(args.case),
        antithetic_heat_flux_quadrature=True,
        pde_weight=1.0,
        mass_weight=50.0,
        momentum_weight=30.0,
        energy_weight=50.0,
        heat_flux_weight=12.0 if case_has_heat_flux(args.case) else 0.0,
        heat_flux_scale=max(abs(initial_heat_flux_qx(args.case)), 0.25),
        nu=args.nu,
        seed=args.seed,
    )
    model = DensityModel(config)
    model.log_density(tf.zeros((1, 1)), tf.zeros((1, 3)))
    model.load_weights(Path(args.weights).resolve())
    # A zero learning rate evaluates exactly the same differentiable objective
    # and gradient path while leaving the checkpoint unchanged.
    result = make_train_step(model, tf.keras.optimizers.Adam(0.0), config)()
    metrics = {name: float(value.numpy()) for name, value in result.items()}
    metrics.update(
        {
            "weights": str(Path(args.weights).resolve()),
            "seed": args.seed,
            "particle_data_used": False,
            "panel_role": "fresh_residual_audit_not_used_by_projection",
        }
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print("RESIDUAL_AUDIT " + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
