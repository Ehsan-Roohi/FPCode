#!/usr/bin/env python3
"""Create one atomic ZIP archive plus SHA-256 sidecar for a G1 Slurm task.

The archive contains the case output directory (config, metrics, final
evaluation, checkpoint sweep, selected weights, loss history, plots), the
deterministic FV reference used for the gate, the Slurm logs and a metadata
file with the git commit.  A STATUS file (PASS / NO_GO) is written at the root
of the archive and the SHA-256 of the finished ZIP is written next to it as
``<archive>.sha256`` (sha256sum-compatible format) and printed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-output", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--log", action="append", default=[])
    return parser.parse_args()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_status(case_output: Path) -> tuple[str, dict]:
    """Prefer the sweep-selected checkpoint's status; fall back to the final weights."""
    sweep = case_output / "checkpoint_sweep" / "checkpoint_sweep.json"
    if sweep.exists():
        summary = json.loads(sweep.read_text())
        return summary["selected_status"], {
            "source": "checkpoint_sweep", "selected_checkpoint": summary["selected_checkpoint"],
            "qx_analytic_l2_fine": summary["selected_metrics"]["qx_analytic_l2_fine"],
            "marginal_relative_l2": summary["selected_metrics"]["marginal_relative_l2"],
        }
    final = case_output / "metrics.json"
    if final.exists():
        metrics = json.loads(final.read_text())
        return metrics["status"], {"source": "final_weights", "qx_analytic_l2_fine": metrics["qx_analytic_l2_fine"]}
    return "NO_GO", {"source": "missing", "reason": "no metrics.json or checkpoint_sweep.json"}


def main() -> None:
    args = parse_args()
    case_output = Path(args.case_output).resolve()
    archive = Path(args.archive).resolve()
    repo_root = Path(args.repo_root).resolve()
    if not case_output.is_dir():
        raise SystemExit(f"G1 case output does not exist: {case_output}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.partial")
    if temporary.exists():
        temporary.unlink()
    status, status_detail = read_status(case_output)
    metadata = {
        "stage": "heat_flux_g1",
        "array_job_id": args.array_job_id,
        "job_id": args.job_id,
        "task_id": args.task_id,
        "seed": args.seed,
        "variant": args.variant,
        "status": status,
        "status_detail": status_detail,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
    }
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        zipped.writestr("STATUS", status + "\n")
        zipped.writestr("run_metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        for path in sorted(case_output.rglob("*")):
            if path.is_file():
                # Keep every checkpoint smaller than ~2 MB and everything else.
                zipped.write(path, arcname=str(path.relative_to(case_output.parent)))
        for log in args.log:
            log_path = Path(log)
            if log_path.is_file():
                zipped.write(log_path, arcname=f"slurm_logs/{log_path.name}")
    temporary.replace(archive)
    digest = sha256_of(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"G1_ARCHIVE {archive}")
    print(f"G1_ARCHIVE_SHA256 {digest}")
    print(f"G1_ARCHIVE_STATUS {status}")


if __name__ == "__main__":
    main()
