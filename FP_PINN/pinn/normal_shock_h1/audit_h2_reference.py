#!/usr/bin/env python3
"""Gate H2-0: certify the independent reference before neural training."""
import argparse
import json
from pathlib import Path
import numpy as np
from h2_reference import load_reference, split_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--mach", type=float, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    ref = load_reference(a.reference, a.mach)
    split = split_indices(len(ref.x))
    plateau = max(
        abs(ref.qx[0]), abs(ref.qx[-1]), abs(ref.sigma_xx[0]), abs(ref.sigma_xx[-1])
    )
    scales = max(np.max(abs(ref.qx)), np.max(abs(ref.sigma_xx)), 1e-12)
    metrics = {
        "stage": "H2_GATE0_INDEPENDENT_REFERENCE",
        "status": "PASS" if plateau / scales < 5e-3 else "NO_GO",
        "mach": a.mach,
        "solver": ref.metadata["solver"],
        "points": len(ref.x),
        "anchor_points": len(split["moments"]),
        "macro_lock_points": len(split["macro"]),
        "held_out_points": len(split["held_out"]),
        "plateau_nonequilibrium_relative": float(plateau / scales),
        "reference": str(Path(a.reference).resolve()),
        "claim": "Reference audit only; no neural-solver accuracy claim."
    }
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "h2_reference_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(out / "h2_split.npz", **split)
    print("H2_REFERENCE_METRICS", json.dumps(metrics, sort_keys=True))
    return 0 if metrics["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
