#!/usr/bin/env python3
"""Create one atomic root-level ZIP for the complete Stage-2 V4 suite."""

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
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--collector-job-id", required=True)
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


def add_file(
    bundle: zipfile.ZipFile,
    source: Path,
    archive_name: str,
    manifest: list[dict[str, str | int]],
) -> None:
    bundle.write(source, archive_name)
    manifest.append(
        {
            "path": archive_name,
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    archive = Path(args.archive).resolve()
    repo_root = Path(args.repo_root).resolve()
    module = Path(__file__).resolve().parent
    if not run_root.is_dir():
        raise SystemExit(f"V4 run root does not exist: {run_root}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.partial")
    if temporary.exists():
        temporary.unlink()

    manifest: list[dict[str, str | int]] = []
    aggregate = run_root / "aggregate"
    source_files = [
        path for path in module.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    ]
    log_patterns = (
        f"FP_PINN_STAGE2_V4_{args.array_job_id}_*.out",
        f"FP_PINN_STAGE2_V4_{args.array_job_id}_*.err",
        f"FP_PINN_STAGE2_V4_COLLECT_*{args.array_job_id}*.out",
        f"FP_PINN_STAGE2_V4_COLLECT_*{args.array_job_id}*.err",
    )
    logs = sorted({path for pattern in log_patterns for path in repo_root.glob(pattern)})
    metadata = {
        "array_job_id": args.array_job_id,
        "collector_job_id": args.collector_job_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "particle_data_used_in_training": False,
        "archive_location": "project_root",
    }
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        if aggregate.is_dir():
            for path in sorted(aggregate.rglob("*")):
                if path.is_file():
                    add_file(bundle, path, path.relative_to(aggregate).as_posix(), manifest)
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and aggregate not in path.parents:
                add_file(
                    bundle,
                    path,
                    "cases/" + path.relative_to(run_root).as_posix(),
                    manifest,
                )
        for path in source_files:
            add_file(
                bundle,
                path,
                "source_snapshot/" + path.relative_to(module).as_posix(),
                manifest,
            )
        for path in logs:
            add_file(bundle, path, f"slurm_logs/{path.name}", manifest)
        bundle.writestr("run_metadata.json", json.dumps(metadata, indent=2) + "\n")
        bundle.writestr(
            "MANIFEST.json",
            json.dumps({"files": manifest, "metadata": metadata}, indent=2) + "\n",
        )
    temporary.replace(archive)
    print(f"ROOT_ARCHIVE {archive}", flush=True)


if __name__ == "__main__":
    main()
