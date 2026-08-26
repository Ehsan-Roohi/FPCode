#!/usr/bin/env python3
"""Run one frozen Stage-58 anchor or blind-generalization case."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from riemann35_patch.stage55_closure_source_audit.run_closure_method import (  # noqa: E402
    _invariants,
    _run_qmc_replicate,
)
from riemann35_patch.stage57_persistent_four_population.run_persistent_method import (  # noqa: E402
    _run_persistent_candidate,
)
from riemann35_patch.stage58_blind_generalization.blind_cases import (  # noqa: E402
    CASE_NAMES,
    blind_case,
    registry_manifest,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASE_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qmc-dt", type=float, default=3.125e-4)
    parser.add_argument("--coarse-dt", type=float, default=6.25e-4)
    parser.add_argument("--fine-dt", type=float, default=3.125e-4)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=float, default=2.5e-2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--prandtl", type=float, default=2.0 / 3.0)
    parser.add_argument("--points-per-component", type=int, default=65536)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20_260_826)
    parser.add_argument("--quadrature-nodes", type=int, default=5)
    return parser.parse_args()


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = int(round(numerator / denominator))
    if value < 1 or not np.isclose(
        value * denominator, numerator, rtol=0.0, atol=2.0e-13
    ):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _sample_steps(final_time: float, sample_interval: float, dt: float) -> tuple[int, ...]:
    steps = _integer_ratio(final_time, dt, "final-time")
    every = _integer_ratio(sample_interval, dt, "sample-interval")
    selected = tuple([0, *range(every, steps + 1, every)])
    return selected if selected[-1] == steps else (*selected, steps)


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


def _candidate_summary(result: dict[str, object]) -> dict[str, object]:
    histories = np.asarray(result["histories"])
    diagnostics = {key: value for key, value in result.items() if key != "histories"}
    return {
        "invariants": _invariants(histories[None, ...]),
        "diagnostics": diagnostics,
    }


def main() -> None:
    args = arguments()
    if min(
        args.qmc_dt,
        args.coarse_dt,
        args.fine_dt,
        args.final_time,
        args.sample_interval,
        args.tau,
    ) <= 0.0:
        raise ValueError("all time scales must be positive")
    if args.replicates < 4 or args.workers < 1:
        raise ValueError("Stage 58 requires at least four replicates and one worker")
    case = blind_case(args.case)
    case_offset = CASE_NAMES.index(args.case) * 104_729_001
    qmc_steps = _integer_ratio(args.final_time, args.qmc_dt, "qmc final-time")
    qmc_sample_steps = _sample_steps(
        args.final_time, args.sample_interval, args.qmc_dt
    )
    coarse_steps = _integer_ratio(args.final_time, args.coarse_dt, "coarse final-time")
    coarse_sample_steps = _sample_steps(
        args.final_time, args.sample_interval, args.coarse_dt
    )
    fine_steps = _integer_ratio(args.final_time, args.fine_dt, "fine final-time")
    fine_sample_steps = _sample_steps(
        args.final_time, args.sample_interval, args.fine_dt
    )
    args.output.mkdir(parents=True, exist_ok=True)
    failure_path = args.output / f"stage58_{args.case}_failure.json"
    print(
        f"[stage58] case={args.case} role={case.role} fingerprint={case.fingerprint} "
        f"started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        flush=True,
    )
    try:
        tasks = [
            (
                case.components,
                replicate,
                args.points_per_component,
                args.qmc_dt,
                qmc_steps,
                qmc_sample_steps,
                args.tau,
                args.prandtl,
                args.seed + case_offset + 15_485_863 * replicate,
            )
            for replicate in range(args.replicates)
        ]
        with ProcessPoolExecutor(
            max_workers=min(args.workers, args.replicates)
        ) as executor:
            qmc_results = list(executor.map(_run_qmc_replicate, tasks))
        coarse = _run_persistent_candidate(
            case.components,
            dt=args.coarse_dt,
            steps=coarse_steps,
            sample_steps=coarse_sample_steps,
            tau=args.tau,
            prandtl=args.prandtl,
            quadrature_nodes=args.quadrature_nodes,
        )
        fine = _run_persistent_candidate(
            case.components,
            dt=args.fine_dt,
            steps=fine_steps,
            sample_steps=fine_sample_steps,
            tau=args.tau,
            prandtl=args.prandtl,
            quadrature_nodes=args.quadrature_nodes,
        )
    except Exception as error:
        failure = {
            "schema": "riemann35-stage58-case-failure-v1",
            "case": args.case,
            "case_fingerprint": case.fingerprint,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise

    qmc_histories = np.asarray([item["histories"] for item in qmc_results])
    coarse_histories = np.asarray(coarse["histories"])[None, ...]
    fine_histories = np.asarray(fine["histories"])[None, ...]
    qmc_times = np.asarray(qmc_sample_steps, dtype=float) * args.qmc_dt / args.tau
    coarse_times = (
        np.asarray(coarse_sample_steps, dtype=float) * args.coarse_dt / args.tau
    )
    fine_times = np.asarray(fine_sample_steps, dtype=float) * args.fine_dt / args.tau
    if not (
        np.allclose(qmc_times, coarse_times, rtol=0.0, atol=2.0e-13)
        and np.allclose(qmc_times, fine_times, rtol=0.0, atol=2.0e-13)
    ):
        raise RuntimeError("Stage-58 sample times differ")
    np.savez_compressed(
        args.output / f"stage58_{args.case}.npz",
        times=qmc_times,
        qmc_histories=qmc_histories,
        persistent_coarse_histories=coarse_histories,
        persistent_fine_histories=fine_histories,
    )
    summary = {
        "schema": "riemann35-stage58-generalization-case-v1",
        "case": args.case,
        "role": case.role,
        "case_fingerprint": case.fingerprint,
        "registry": registry_manifest(),
        "configuration": case.configuration,
        "initial_audit": case.audit,
        "controls": {
            "qmc_dt_over_tau": args.qmc_dt / args.tau,
            "coarse_dt_over_tau": args.coarse_dt / args.tau,
            "fine_dt_over_tau": args.fine_dt / args.tau,
            "final_time_over_tau": args.final_time / args.tau,
            "sample_interval_over_tau": args.sample_interval / args.tau,
            "points_per_component": args.points_per_component,
            "replicates": args.replicates,
            "prandtl": args.prandtl,
            "quadrature_nodes_per_population": args.quadrature_nodes,
            "qmc_used_to_define_case": False,
            "closure_parameters_refit": False,
        },
        "qmc": {
            "invariants": _invariants(qmc_histories),
            "replicate_diagnostics": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in ("histories", "sources", "tails")
                }
                for item in qmc_results
            ],
        },
        "persistent_coarse": _candidate_summary(coarse),
        "persistent_fine": _candidate_summary(fine),
    }
    (args.output / f"stage58_{args.case}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if failure_path.exists():
        failure_path.unlink()
    print(
        json.dumps(
            {
                "case": args.case,
                "fingerprint": case.fingerprint,
                "qmc_replicates": len(qmc_results),
                "coarse_elapsed_seconds": coarse["elapsed_seconds"],
                "fine_elapsed_seconds": fine["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
