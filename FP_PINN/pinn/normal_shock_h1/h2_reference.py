"""Independent-reference contract for the H2 normal-shock solver."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np


REQUIRED = ("x", "rho", "u", "temperature", "qx", "sigma_xx")
FULLSTATE_REQUIRED = ("x_mfp", "f", "v", "w", "rho", "ux", "T", "qx", "sig")
KNOWN_FULLSTATES = {
    "8959d23bfe7643d0010bedd65516c6985103b50e3f15c1cc862893180c770a02": {
        "solver": "independent conservative BGK DVM",
        "collision": "BGK",
        "mach": 2.0,
        "neural": False,
        "nx": 1600,
        "nv": 35017,
        "x_mfp": [-40.0, 40.0],
        "reference_id": "standing_M2_hmom_x40_nx1600_v97_19_19_vmax12_fullstate",
    }
}
CORE_LIMIT_MFP = 30.0
TAIL_START_MFP = 30.0


@dataclass(frozen=True)
class ShockReference:
    x: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    temperature: np.ndarray
    qx: np.ndarray
    sigma_xx: np.ndarray
    metadata: dict


def sha256_file(path, block_size=8 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _strictly_increasing(x):
    return len(x) >= 33 and np.all(np.diff(x) > 0)


def _load_fullstate(path, files, digest):
    if digest not in KNOWN_FULLSTATES:
        raise ValueError(
            "unregistered full-state reference SHA-256; provenance must be preregistered"
        )
    missing = sorted(set(FULLSTATE_REQUIRED) - set(files))
    if missing:
        raise ValueError(f"full-state reference missing arrays: {missing}")
    # Numerical members can be read with pickle disabled. The legacy `states`
    # object is deliberately ignored; trusted metadata comes from the hash lock.
    with np.load(path, allow_pickle=False) as z:
        data = {
            "x": np.asarray(z["x_mfp"], dtype=float),
            "rho": np.asarray(z["rho"], dtype=float),
            "u": np.asarray(z["ux"], dtype=float),
            "temperature": np.asarray(z["T"], dtype=float),
            "qx": np.asarray(z["qx"], dtype=float),
            "sigma_xx": np.asarray(z["sig"], dtype=float),
        }
    meta = dict(KNOWN_FULLSTATES[digest])
    meta.update({"sha256": digest, "format": "fullstate_npz"})
    return data, meta


def load_reference(path, expected_mach=None):
    """Load an independent DVM/DSMC reference; neural outputs are rejected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"REFERENCE_REQUIRED: {path}")
    digest = sha256_file(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            files = tuple(z.files)
        if set(FULLSTATE_REQUIRED).issubset(files):
            data, meta = _load_fullstate(path, files, digest)
        else:
            with np.load(path, allow_pickle=False) as z:
                missing = sorted(set(REQUIRED) - set(z.files))
                if missing:
                    raise ValueError(f"reference missing arrays: {missing}")
                meta = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else {}
                data = {k: np.asarray(z[k], dtype=float) for k in REQUIRED}
            meta["sha256"] = digest
    else:
        table = np.genfromtxt(path, delimiter=",", names=True)
        names = table.dtype.names or ()
        missing = sorted(set(REQUIRED) - set(names))
        if missing:
            raise ValueError(f"reference missing CSV columns: {missing}")
        data = {k: np.asarray(table[k], dtype=float) for k in REQUIRED}
        sidecar = path.with_suffix(path.suffix + ".json")
        meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        meta["sha256"] = digest
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


def validation_regions(x, held_out):
    """Pre-registered full/core/tail partitions; endpoints are audited separately."""
    x = np.asarray(x)
    held_out = np.asarray(held_out, dtype=int)
    core = held_out[np.abs(x[held_out]) <= CORE_LIMIT_MFP]
    left_tail = held_out[x[held_out] < -TAIL_START_MFP]
    right_tail = held_out[x[held_out] > TAIL_START_MFP]
    return {"held_out_full": held_out, "held_out_core": core,
            "left_tail": left_tail, "right_tail": right_tail}


def fullstate_moment_audit(path, chunk_size=64):
    """Reintegrate stored f(v) and compare with all five stored moments."""
    with np.load(path, allow_pickle=False) as z:
        f, v, w = z["f"], np.asarray(z["v"], float), np.asarray(z["w"], float)
        stored = {"rho": np.asarray(z["rho"], float), "u": np.asarray(z["ux"], float),
                  "temperature": np.asarray(z["T"], float), "qx": np.asarray(z["qx"], float),
                  "sigma_xx": np.asarray(z["sig"], float)}
        vx, vy, vz = v.T
        result = {k: np.empty(f.shape[0]) for k in stored}
        for lo in range(0, f.shape[0], chunk_size):
            hi = min(lo + chunk_size, f.shape[0])
            fw = np.asarray(f[lo:hi], float) * w[None, :]
            rho = fw.sum(1)
            u = (fw * vx[None, :]).sum(1) / rho
            cx = vx[None, :] - u[:, None]
            c2 = cx*cx + vy[None, :]**2 + vz[None, :]**2
            result["rho"][lo:hi] = rho
            result["u"][lo:hi] = u
            result["temperature"][lo:hi] = (fw*c2).sum(1)/(3*rho)
            result["qx"][lo:hi] = 0.5*(fw*cx*c2).sum(1)
            result["sigma_xx"][lo:hi] = (fw*(cx*cx-c2/3)).sum(1)
    metrics = {}
    for key, direct in result.items():
        ref = stored[key]
        scale = float(np.dot(direct, ref) / max(np.dot(direct, direct), np.finfo(float).tiny))
        metrics[key] = {
            "best_scale": scale,
            "scaled_relative_rms": float(np.linalg.norm(scale*direct-ref) /
                                         max(np.linalg.norm(ref), np.finfo(float).tiny)),
        }
    return metrics


def relative_l2(pred, ref, indices, floor=1e-12):
    p = np.asarray(pred)[indices]
    r = np.asarray(ref)[indices]
    return float(np.linalg.norm(p-r) / max(np.linalg.norm(r), floor))


def heldout_metrics(predictions, reference, indices):
    fields = ("rho", "u", "temperature", "qx", "sigma_xx")
    return {k: relative_l2(predictions[k], getattr(reference, k), indices) for k in fields}
