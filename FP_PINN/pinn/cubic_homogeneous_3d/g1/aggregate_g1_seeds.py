#!/usr/bin/env python3
"""Aggregate the preregistered G1 seed tasks into one deterministic verdict.

The overall status is PASS only if all three expected from-scratch tasks are
present, their configured seeds are distinct, every task passes its
deterministic gate, and their analytic-Qx L2 spread is no greater than
``--seed-spread-pp``. A failed task that never writes ``config.json`` is
therefore an explicit NO_GO, never an omitted record. The optional warm-start
task is reported but does not enter the seed-agreement gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SCRATCH_TASKS = (
    ("scratch_s1", 20260901),
    ("scratch_s2", 20260902),
    ("scratch_s3", 20260903),
)


def missing_task(name: str, seed: int | None, reason: str) -> dict:
    return {
        "task": name,
        "seed": seed,
        "warm_start": False,
        "source": "missing",
        "status": "NO_GO",
        "reason": reason,
    }


def load_task(task_dir: Path, *, default_seed: int | None = None) -> dict:
    sweep = task_dir / "checkpoint_sweep" / "checkpoint_sweep.json"
    final = task_dir / "metrics.json"
    config = task_dir / "config.json"
    if not config.exists():
        return missing_task(task_dir.name, default_seed, "missing config.json")
    cfg = json.loads(config.read_text())
    record = {
        "task": task_dir.name,
        "seed": cfg.get("seed"),
        "warm_start": bool(cfg.get("resume_base_weights") or cfg.get("resume_weights")),
    }
    if sweep.exists():
        summary = json.loads(sweep.read_text())
        metrics = summary["selected_metrics"]
        record.update({
            "source": "checkpoint_sweep",
            "selected_checkpoint": summary["selected_checkpoint"],
        })
    elif final.exists():
        metrics = json.loads(final.read_text())
        record.update({"source": "final_weights", "selected_checkpoint": "g1_final.weights.h5"})
    else:
        record.update({"source": "missing", "status": "NO_GO", "reason": "missing evaluation metrics"})
        return record
    record.update({
        "status": metrics["status"],
        "primary_pass": metrics["primary_pass"],
        "publication_pass": metrics["publication_pass"],
        "qx_analytic_l2_fine": metrics["qx_analytic_l2_fine"],
        "qx_quadrature_uncertainty_pp": metrics["qx_quadrature_uncertainty_pp"],
        "decay_rate": metrics["decay_rate"],
        "marginal_relative_l2": metrics["marginal_relative_l2"],
        "raw_max_mass_error": metrics["raw_max_mass_error"],
        "raw_max_energy_error": metrics["raw_max_energy_error"],
        "stress_anisotropy_max_abs_error": metrics["stress_anisotropy_max_abs_error"],
        "residual_rms_max": metrics.get("residual_rms_max"),
        "field_relative_l2_max": metrics.get("field_relative_l2_max"),
        "failed_checks": [key for key, value in metrics["gate_checks"].items() if not value],
    })
    return record


def aggregate_tasks(root: Path, seed_spread_pp: float) -> dict:
    """Return a verdict in which missing or partial arrays are always NO_GO."""
    expected_names = {name for name, _ in EXPECTED_SCRATCH_TASKS}
    scratch = [
        load_task(root / name, default_seed=seed) if (root / name).is_dir()
        else missing_task(name, seed, "missing task directory")
        for name, seed in EXPECTED_SCRATCH_TASKS
    ]
    extras = [
        load_task(task_dir)
        for task_dir in sorted(root.iterdir())
        if task_dir.is_dir() and task_dir.name not in expected_names
    ]
    tasks = scratch + extras

    observed = [task for task in scratch if task.get("source") != "missing"]
    observed_seeds = [task.get("seed") for task in observed]
    complete = len(observed) == len(EXPECTED_SCRATCH_TASKS) and all(
        not task.get("warm_start", False) for task in observed
    )
    unique_seeds = (
        complete
        and None not in observed_seeds
        and len(set(observed_seeds)) == len(observed_seeds)
    )
    errors = [task["qx_analytic_l2_fine"] for task in observed if "qx_analytic_l2_fine" in task]
    spread_pp = (
        (max(errors) - min(errors)) * 100.0
        if len(errors) == len(EXPECTED_SCRATCH_TASKS)
        else None
    )
    all_seeds_pass = complete and unique_seeds and all(
        task.get("status") == "PASS" for task in observed
    )
    spread_pass = spread_pp is not None and spread_pp <= seed_spread_pp
    overall = "PASS" if all_seeds_pass and spread_pass else "NO_GO"
    return {
        "overall_status": overall,
        "seed_agreement": {
            "expected_scratch_tasks": [name for name, _ in EXPECTED_SCRATCH_TASKS],
            "expected_n_scratch_seeds": len(EXPECTED_SCRATCH_TASKS),
            "n_scratch_seeds": len(observed),
            "all_expected_tasks_present": complete,
            "scratch_seeds_unique": unique_seeds,
            "qx_l2_spread_pp": spread_pp,
            "threshold_pp": seed_spread_pp,
            "all_seeds_pass": all_seeds_pass,
            "spread_pass": spread_pass,
        },
        "tasks": tasks,
    }


def render_markdown(summary: dict) -> list[str]:
    agreement = summary["seed_agreement"]
    spread = agreement["qx_l2_spread_pp"]
    spread_text = f"{spread:.2f} pp" if spread is not None else "n/a"
    lines = [
        f"# G1 heat-flux qualification: **{summary['overall_status']}**",
        "",
        f"From-scratch seeds: {agreement['n_scratch_seeds']}/{agreement['expected_n_scratch_seeds']}; "
        f"Qx L2 spread = {spread_text} (gate {agreement['threshold_pp']} pp)",
        "",
        "| task | seed | warm | status | Qx L2 (fine) | quad. unc. pp | rate | marginal | raw mass | raw energy | aniso | residual RMS | field L2 | failed checks |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task in summary["tasks"]:
        if task.get("source") == "missing":
            lines.append(
                f"| {task['task']} | {task['seed']} | {task['warm_start']} | "
                f"NO_GO ({task['reason']}) | | | | | | | | | | |"
            )
            continue
        lines.append(
            f"| {task['task']} | {task['seed']} | {task['warm_start']} | {task['status']} | "
            f"{100 * task['qx_analytic_l2_fine']:.2f} % | {task['qx_quadrature_uncertainty_pp']:.3f} | "
            f"{task['decay_rate']:.4f} | {100 * task['marginal_relative_l2']:.2f} % | "
            f"{100 * task['raw_max_mass_error']:.2f} % | {100 * task['raw_max_energy_error']:.2f} % | "
            f"{task['stress_anisotropy_max_abs_error']:.4f} | "
            f"{task['residual_rms_max'] if task['residual_rms_max'] is not None else 'n/a'} | "
            f"{task['field_relative_l2_max'] if task['field_relative_l2_max'] is not None else 'n/a'} | "
            f"{', '.join(task['failed_checks']) or '-'} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed-spread-pp", type=float, default=1.0)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = aggregate_tasks(root, args.seed_spread_pp)
    (root / "G1_SEED_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = render_markdown(summary)
    (root / "G1_SEED_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"G1_OVERALL_STATUS {summary['overall_status']}")


if __name__ == "__main__":
    main()
