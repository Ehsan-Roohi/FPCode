#!/usr/bin/env python3
"""Select the best checkpoint in a root-level Stage-2 ZIP for continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--case-output", required=True)
    return parser.parse_args()


def allowed_relative_path(name: str) -> Path | None:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        return None
    parts = member.parts
    for index, part in enumerate(parts):
        suffix = parts[index:]
        if suffix == ("config.json",):
            return Path("config.json")
        if suffix == ("reference_particle", "reference.npz"):
            return Path(*suffix)
        if (
            len(suffix) == 2
            and suffix[0] == "checkpoints_h5"
            and suffix[1].startswith("epoch-")
            and suffix[1].endswith(".weights.h5")
        ):
            return Path(*suffix)
        if suffix == ("stage2_final.weights.h5",):
            return Path("stage2_final.weights.h5")
    return None


def main() -> None:
    args = parse_args()
    archive = Path(args.archive).resolve()
    case_output = Path(args.case_output).resolve()
    if not archive.is_file():
        raise SystemExit(f"Resume archive does not exist: {archive}")
    source = case_output / "resume_source"
    source.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = allowed_relative_path(info.filename)
            if relative is None or info.is_dir():
                continue
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as incoming, destination.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            extracted.append(relative.as_posix())

    required = [source / "config.json", source / "reference_particle" / "reference.npz"]
    if any(not path.is_file() for path in required):
        raise SystemExit("Resume ZIP is missing config.json or reference_particle/reference.npz")
    if not any((source / "checkpoints_h5").glob("epoch-*.weights.h5")):
        final = source / "stage2_final.weights.h5"
        if not final.is_file():
            raise SystemExit("Resume ZIP contains no portable Stage-2 weights")

    evaluator = Path(__file__).with_name("evaluate_stage2_checkpoints.py")
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--case-output",
            str(source),
            "--reference",
            str(source / "reference_particle" / "reference.npz"),
        ],
        check=True,
    )
    selected = source / "stage2_best.weights.h5"
    destination = case_output / "resume_input.weights.h5"
    shutil.copy2(selected, destination)
    sweep = json.loads((source / "checkpoint_sweep.json").read_text())
    metadata = {
        "source_archive": str(archive),
        "selected_checkpoint": sweep["best_checkpoint"],
        "resume_weights": destination.name,
        "extracted_members": sorted(extracted),
    }
    (case_output / "resume_selection.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print("RESUME_SELECTION " + json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
