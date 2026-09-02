"""Reference contract and held-out validation for the H2 kinetic-shock solver."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

REQUIRED = ("x", "rho", "u", "temperature", "qx", "sigma_xx")


@dataclass(frozen=True)
class ShockReference:
    x: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    temperature: np.ndarray
    qx: np.ndarray
    sigma_xx: np.ndarray
    metadata: dict


def _strictly_increasing(x):
    return len(x) >= 33 and np.all(np.diff(x) > 0)


def load_reference(path, expected_mach=None):
    """Load an independent DVM/DSMC reference; neural outputs are rejected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"REFERENCE_REQUIRED: {path}")
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        missing = sorted(set(REQUIRED) - set(z.files))
        if missing:
            raise ValueError(f"reference missing arrays: {missing}")
        meta = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else {}
        data = {k: np.asarray(z[k], dtype=float) for k in REQUIRED}
    else:
        table = np.genfromtxt(path, delimiter=",", names=True)
        names = table.dtype.names or ()
        missing = sorted(set(REQUIRED) - set(names))
        if missing:
            raise ValueError(f"reference missing CSV columns: {missing}")
        data = {k: np.asarray(table[k], dtype=float) for k in REQUIRED}
        sidecar = path.with_suffix(path.suffix + ".json")
        meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    n = len(data["x"])
    if any(v.shape != (n,) for v in data.values()):
        raise ValueError("all reference fields must be one-dimensional and co-located")
    if not _strictly_increasing(data["x"]):
        raise ValueError("reference x must be strictly increasing with at least 33 points")
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("reference contains NaN or infinity")
    if np.min(data["rho"]) <= 0 or np.min(data["temperature"]) <= 0:
        raise ValueError("reference density and temperature must be positive")
    solver = str(meta.get("solver", "")).lower()
    if not any(name in solver for name in ("dvm", "dgfs", "dsmc")):
        raise ValueError("metadata solver must identify an independent DVM, DGFS, or DSMC reference")
    if bool(meta.get("neural", False)):
        raise ValueError("a neural prediction cannot be used as the H2 reference")
    if expected_mach is not None and abs(float(meta.get("mach", np.nan)) - expected_mach) > 1e-10:
        raise ValueError(f"reference Mach does not match requested Mach {expected_mach:g}")
    return ShockReference(**data, metadata=meta)


def split_indices(n, anchor_count=16, macro_count=32):
    """Deterministic sparse training indices and disjoint dense validation indices."""
    if not (4 <= anchor_count < macro_count < n - 2):
        raise ValueError("require 4 <= anchor_count < macro_count < n-2")
    interior = np.arange(1, n - 1)
    macro = np.unique(np.rint(np.linspace(1, n - 2, macro_count)).astype(int))
    anchors = np.unique(np.rint(np.linspace(1, n - 2, anchor_count)).astype(int))
    train = np.union1d(macro, anchors)
    held_out = np.setdiff1d(interior, train)
    if len(held_out) < 16:
        raise ValueError("held-out set is too small")
    return {"macro": macro, "moments": anchors, "held_out": held_out}


def relative_l2(pred, ref, indices, floor=1e-12):
    p = np.asarray(pred)[indices]
    r = np.asarray(ref)[indices]
    return float(np.linalg.norm(p - r) / max(np.linalg.norm(r), floor))


def heldout_metrics(predictions, reference, indices):
    fields = ("rho", "u", "temperature", "qx", "sigma_xx")
    return {k: relative_l2(predictions[k], getattr(reference, k), indices) for k in fields}
