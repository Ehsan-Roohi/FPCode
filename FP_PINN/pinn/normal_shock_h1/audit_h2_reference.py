#!/usr/bin/env python3
"""Gate H2-0: certify the independent reference before neural training."""
import argparse
import json
from pathlib import Path

import numpy as np

from h2_reference import (fullstate_moment_audit, load_reference,
                          split_indices, validation_regions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--mach", type=float, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    ref = load_reference(a.reference, a.mach)
    split = split_indices(len(ref.x))
    regions = validation_regions(ref.x, split["held_out"])

    global_scale = max(np.max(np.abs(ref.qx)), np.max(np.abs(ref.sigma_xx)), 1e-12)
    endpoint_noneq = max(abs(ref.qx[0]), abs(ref.qx[-1]),
                            abs(ref.sigma_xx[0]), abs(ref.sigma_xx[-1])) / global_scale
    tail = np.abs(ref.x) > 30.0
    tail_noneq = max(np.max(np.abs(ref.qx[tail])),
                     np.max(np.abs(ref.sigma_xx[tail]))) / global_scale

    direct = None
    direct_ok = True
    if ref.metadata.get("format") == "fullstate_npz":
        direct = fullstate_moment_audit(a.reference)
        direct_ok = all(v["scaled_relative_rms"] < 1e-4 for v in direct.values())

    gates = {
        "registered_provenance": ref.metadata.get("sha256") is not None,
        "endpoint_equilibrium": endpoint_noneq < 5e-3,
        "tail_artifact_bounded": tail_noneq < 5e-3,
        "direct_moment_consistency": direct_ok,
        "heldout_core_nonempty": len(regions["held_out_core"]) >= 16,
    }
    metrics = {
        "stage": "H2_GATE0_INDEPENDENT_REFERENCE",
        "status": "PASS" if all(gates.values()) else "NO_GO",
        "mach": a.mach,
        "solver": ref.metadata["solver"],
        "collision": ref.metadata.get("collision"),
        "sha256": ref.metadata.get("sha256"),
        "points": len(ref.x),
        "velocity_points": ref.metadata.get("nv"),
        "anchor_points": len(split["moments"]),
        "macro_lock_points": len(split["macro"]),
        "held_out_full_points": len(regions["held_out_full"]),
        "held_out_core_points": len(regions["held_out_core"]),
        "core_definition_mfp": [-30.0, 30.0],
        "endpoint_nonequilibrium_relative": float(endpoint_noneq),
        "outer_tail_nonequilibrium_relative": float(tail_noneq),
        "direct_moment_audit": direct,
        "gates": gates,
        "reference": str(Path(a.reference).resolve()),
        "claim": "Reference audit only; no neural-solver accuracy claim.",
    }
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "h2_reference_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(out / "h2_split.npz", **split, **regions)
    print("H2_REFERENCE_METRICS", json.dumps(metrics, sort_keys=True))
    return 0 if metrics["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
