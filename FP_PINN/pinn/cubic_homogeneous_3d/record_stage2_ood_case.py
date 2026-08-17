#!/usr/bin/env python3
"""Atomically record the terminal status of one Stage-2 V4 array case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--started-monotonic", type=float, required=True)
    parser.add_argument("--result-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve() if args.result_dir else None
    metrics_path = result_dir / "metrics.json" if result_dir else None
    metrics = None
    if metrics_path and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
    payload = {
        "array_job_id": args.array_job_id,
        "case": args.case,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": max(0.0, time.monotonic() - args.started_monotonic),
        "exit_code": args.exit_code,
        "gate_passed": bool(metrics and metrics.get("gate_passed", False)),
        "job_id": args.job_id,
        "metrics": str(metrics_path) if metrics_path and metrics_path.is_file() else None,
        "nu": args.nu,
        "result_dir": str(result_dir) if result_dir else None,
        "state": "COMPLETED" if args.exit_code == 0 else "FAILED",
        "task_id": args.task_id,
    }
    path = Path(args.status).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print("CASE_STATUS " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
