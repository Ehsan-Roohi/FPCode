#!/usr/bin/env python3
"""Create one atomic, root-level ZIP archive for a Stage-2 Slurm task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    parser.add_argument("--case", required=True)
    parser.add_argument("--log", action="append", default=[])
    return parser.parse_args()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    case_output = Path(args.case_output).resolve()
    archive = Path(args.archive).resolve()
    repo_root = Path(args.repo_root).resolve()
    if not case_output.is_dir():
        raise SystemExit(f"Stage-2 case output does not exist: {case_output}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.partial")
    if temporary.exists():
        temporary.unlink()

    metadata = {
        "array_job_id": args.array_job_id,
        "case": args.case,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "job_id": args.job_id,
        "task_id": args.task_id,
    }
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in sorted(case_output.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(case_output).as_posix())
        for raw_log in args.log:
            log = Path(raw_log)
            if log.is_file():
                bundle.write(log, f"slurm_logs/{log.name}")
        bundle.writestr("run_metadata.json", json.dumps(metadata, indent=2) + "\n")
    temporary.replace(archive)
    print(f"ROOT_ARCHIVE {archive}", flush=True)


if __name__ == "__main__":
    main()
