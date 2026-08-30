#!/usr/bin/env python3
"""Deterministic axisymmetric FV reference for the G2 stress benchmark.

The anisotropic Gaussian f0 with variances (1.6, 0.7, 0.7) and the homogeneous
cubic FP operator are invariant under rotations about the x axis, so the exact solution is f(t, cx, rho),
rho = sqrt(cy^2 + cz^2).  With the repository closure (cubic_operator.solve_closure) the
axisymmetric drift is

    a_x   = -nu x + C_xx x + Gamma_x (r^2 - DM2) + lambda (x r^2 - Q_x)
    a_rho = (-nu + C_yy + lambda r^2) rho

and the FP equation in cylindrical velocity coordinates reads

    df/dt = -d/dx(a_x f) - (1/rho) d/drho(rho a_rho f) + D [d2f/dx2 + (1/rho) d/drho(rho df/drho)]

with D = nu * DM2 / 3 (self-consistent temperature).  A conservative second-order
finite-volume discretisation with RK4 time stepping is used.  Moments are grid integrals,
the 9x9 closure is re-solved from the grid moments every RK stage, so the reference is an
independent check of the heat-flux law dQ_x/dt = -(4/3) nu Q_x and an essentially
noise-free reference for marginals and pointwise density.  The closed-form
stress law is used only after integration as an independent qualification.

Usage (from FP_PINN/pinn/cubic_homogeneous_3d):
    python g2/axisym_stress_reference.py --output-dir REF --nx 400 --nr 200 --L 8 --dt 2e-4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (str(HERE.parent), str(HERE)):   # cubic_operator.py lives in the stage directory
    if _path not in sys.path:
        sys.path.insert(0, _path)
from cubic_operator import Moments, solve_closure  # noqa: E402
from stress_history import (  # noqa: E402
    STRESS_DELTA0,
    STRESS_PPERP0,
    STRESS_PXX0,
    analytic_stress_components,
    analytic_stress_delta,
    fit_decay_rate,
    relative_l2,
)


def trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Trapezoidal integral compatible with NumPy 1.x and 2.x."""
    method = getattr(np, "trapezoid", None)
    if method is None:
        method = np.trapz
    return float(method(values, coordinates))


def initial_axisym(cx: np.ndarray, rho: np.ndarray) -> np.ndarray:
    norm = (2.0 * np.pi) ** (-1.5) / np.sqrt(STRESS_PXX0 * STRESS_PPERP0**2)
    return norm * np.exp(-0.5 * (cx * cx / STRESS_PXX0 + rho * rho / STRESS_PPERP0))


class AxisymGrid:
    def __init__(self, L: float, nx: int, nr: int):
        self.L, self.nx, self.nr = L, nx, nr
        self.dx = 2.0 * L / nx
        self.dr = L / nr
        self.x = -L + (np.arange(nx) + 0.5) * self.dx          # cell centres
        self.r = (np.arange(nr) + 0.5) * self.dr                # cell centres (rho>0)
        self.xf = -L + np.arange(nx + 1) * self.dx               # x faces
        self.rf = np.arange(nr + 1) * self.dr                    # rho faces (rf[0]=0)
        X, R = np.meshgrid(self.x, self.r, indexing="ij")
        self.X, self.R = X, R
        self.vol = (2.0 * np.pi * R * self.dr * self.dx)         # cell volumes (axisymmetric)
        self.R2 = X * X + R * R

    def moments(self, f: np.ndarray) -> Moments:
        w = self.vol * f
        mass = w.sum()
        mx = (w * self.X).sum() / mass
        vx = self.X - mx
        r2 = vx * vx + self.R * self.R
        avg = lambda g: float((w * g).sum() / mass)
        pxx = avg(vx * vx)
        pyy = 0.5 * avg(self.R ** 2)
        qx = avg(vx * r2)
        m3xxx = avg(vx ** 3)
        m3xyy = 0.5 * avg(vx * self.R ** 2)
        m4xx = avg(vx * vx * r2)
        m4yy = 0.5 * avg(self.R ** 2 * r2)
        m5x = avg(vx * r2 * r2)
        dm2 = avg(r2)
        dm4 = avg(r2 * r2)
        return Moments(
            mass=float(mass), mean=np.array([mx, 0.0, 0.0]),
            pij=np.array([pxx, 0, 0, pyy, 0, pyy]), q=np.array([qx, 0, 0]),
            m3=np.array([m3xxx, 0, 0, m3xyy, 0, m3xyy, 0, 0, 0, 0]),
            m4=np.array([m4xx, 0, 0, m4yy, 0, m4yy]), m5=np.array([m5x, 0, 0]),
            dm2=dm2, dm4=dm4,
        )


def rhs(grid: AxisymGrid, f: np.ndarray, nu: float) -> tuple[np.ndarray, dict]:
    m = grid.moments(f)
    cl = solve_closure(m, nu=nu)
    Cxx, Cyy, Gx, lam = cl.matrix[0, 0], cl.matrix[1, 1], cl.gamma[0], cl.cubic_lambda
    D = nu * m.dm2 / 3.0
    ux = m.mean[0]
    # ---- x fluxes on x faces (nx+1, nr) -------------------------------------------
    xf = grid.xf[:, None]
    r_c = grid.r[None, :]
    vxf = xf - ux
    r2f = vxf * vxf + r_c * r_c
    ax = -nu * vxf + Cxx * vxf + Gx * (r2f - m.dm2) + lam * (vxf * r2f - m.q[0])
    fface = np.zeros((grid.nx + 1, grid.nr))
    fface[1:-1] = 0.5 * (f[1:] + f[:-1])
    dfdx = np.zeros_like(fface)
    dfdx[1:-1] = (f[1:] - f[:-1]) / grid.dx
    Fx = ax * fface - D * dfdx
    Fx[0] = 0.0; Fx[-1] = 0.0                       # zero flux at far boundaries
    # ---- rho fluxes on rho faces (nx, nr+1) --------------------------------------
    rf = grid.rf[None, :]
    x_c = grid.x[:, None] - ux
    r2r = x_c * x_c + rf * rf
    ar = (-nu + Cyy + lam * r2r) * rf
    fface_r = np.zeros((grid.nx, grid.nr + 1))
    fface_r[:, 1:-1] = 0.5 * (f[:, 1:] + f[:, :-1])
    dfdr = np.zeros_like(fface_r)
    dfdr[:, 1:-1] = (f[:, 1:] - f[:, :-1]) / grid.dr
    Fr = rf * (ar * fface_r - D * dfdr)             # includes the rho factor
    Fr[:, 0] = 0.0; Fr[:, -1] = 0.0                 # axis and far boundary
    dfdt = -(Fx[1:] - Fx[:-1]) / grid.dx - (Fr[:, 1:] - Fr[:, :-1]) / (grid.R * grid.dr)
    info = {"Cxx": Cxx, "Cyy": Cyy, "Gx": Gx, "lam": lam, "cond": cl.condition_number}
    return dfdt, info


def marginals(grid: AxisymGrid, f: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return marginal_x(centers) and marginal_y(centers) (= marginal_z) by interpolation."""
    # marginal in x: integrate over rho
    mx_grid = (f * 2.0 * np.pi * grid.R * grid.dr).sum(axis=1)
    marg_x = np.interp(centers, grid.x, mx_grid, left=0.0, right=0.0)
    # marginal in y: m_y(cy) = int dx int dz f(x, sqrt(cy^2+cz^2)); integrate over x first -> g(rho)
    g_rho = (f * grid.dx).sum(axis=0)                               # g(rho) = int f dx
    cz = np.linspace(-grid.L, grid.L, 2 * grid.nr + 1)
    marg_y = np.empty_like(centers)
    for i, cy in enumerate(centers):
        rr = np.sqrt(cy * cy + cz * cz)
        marg_y[i] = trapezoidal_integral(
            np.interp(rr, grid.r, g_rho, left=g_rho[0], right=0.0), cz,
        )
    return marg_x, marg_y


def run(args: argparse.Namespace) -> dict:
    grid = AxisymGrid(args.L, args.nx, args.nr)
    f = initial_axisym(grid.X, grid.R)
    f /= grid.moments(f).mass          # remove O(h^2) quadrature mass error of the sampled IC
    steps = int(round(args.tmax / args.dt))
    save_every = max(1, int(round(args.save_interval / args.dt)))
    progress_every = max(1, steps // 10)
    edges = np.linspace(-args.hist_vmax, args.hist_vmax, args.hist_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    rec = {k: [] for k in ("time", "mass", "mean", "dm2", "pij", "q", "m3", "m4", "dm4",
                           "Cxx", "Cyy", "Gx", "lam", "cond", "marginal_x", "marginal_y", "marginal_z")}
    fields = []

    def record(t, f):
        m = grid.moments(f)
        _, info = rhs(grid, f, args.nu)
        rec["time"].append(t); rec["mass"].append(m.mass); rec["mean"].append(m.mean.copy())
        rec["dm2"].append(m.dm2); rec["pij"].append(m.pij.copy()); rec["q"].append(m.q.copy())
        rec["m3"].append(m.m3.copy()); rec["m4"].append(m.m4.copy()); rec["dm4"].append(m.dm4)
        for k in ("Cxx", "Cyy", "Gx", "lam", "cond"):
            rec[k].append(info[k])
        mxg, myg = marginals(grid, f, centers)
        rec["marginal_x"].append(mxg); rec["marginal_y"].append(myg); rec["marginal_z"].append(myg)
        fields.append(f.copy())

    t0 = time.perf_counter()
    record(0.0, f)
    dt = args.dt
    for n in range(1, steps + 1):
        k1, _ = rhs(grid, f, args.nu)
        k2, _ = rhs(grid, f + 0.5 * dt * k1, args.nu)
        k3, _ = rhs(grid, f + 0.5 * dt * k2, args.nu)
        k4, _ = rhs(grid, f + dt * k3, args.nu)
        f = f + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if n % save_every == 0 or n == steps:
            record(n * dt, f)
        if n % progress_every == 0:
            m = grid.moments(f)
            print(f"  t={n*dt:.3f} mass={m.mass:.10f} dm2={m.dm2:.10f} qx={m.q[0]:+.2e} "
                  f"delta={m.pij[0]-m.pij[3]:.8f} exact={STRESS_DELTA0*np.exp(-2*args.nu*n*dt):.8f} "
                  f"min f={f.min():.2e} [{time.perf_counter()-t0:.0f}s]", flush=True)
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    times = arrays["time"]
    exact_delta = analytic_stress_delta(times, nu=args.nu)
    exact_pxx, exact_pperp = analytic_stress_components(times, nu=args.nu)
    delta = arrays["pij"][:, 0] - arrays["pij"][:, 3]
    rel_l2 = relative_l2(delta, exact_delta)
    component_l2 = relative_l2(
        np.column_stack([arrays["pij"][:, 0], arrays["pij"][:, 3], arrays["pij"][:, 5]]),
        np.column_stack([exact_pxx, exact_pperp, exact_pperp]),
    )
    rate = fit_decay_rate(times, delta)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reference.npz", histogram_centers=centers, histogram_edges=edges,
                        grid_x=grid.x, grid_rho=grid.r, fields=np.asarray(fields), **arrays)
    metrics = {
        "solver": "axisymmetric_fv_rk4", "L": args.L, "nx": args.nx, "nr": args.nr, "dt": args.dt,
        "case": "axisymmetric_stress", "stress_delta_vs_exact_relative_l2": rel_l2,
        "stress_components_vs_exact_relative_l2": component_l2,
        "fitted_decay_rate": rate, "exact_decay_rate": 2.0 * args.nu,
        "max_mass_error": float(np.max(np.abs(arrays["mass"] - 1.0))),
        "max_energy_error": float(np.max(np.abs(arrays["dm2"] - 3.0))),
        "max_momentum": float(np.max(np.abs(arrays["mean"][:, 0]))),
        "max_heat_flux_norm": float(np.max(np.linalg.norm(arrays["q"], axis=1))),
        "max_transverse_stress_split": float(np.max(np.abs(arrays["pij"][:, 3] - arrays["pij"][:, 5]))),
        "max_abs_lambda": float(np.max(np.abs(arrays["lam"]))),
        "min_density": float(min(fl.min() for fl in fields)),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (out / "reference_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("REFERENCE_METRICS " + json.dumps(metrics))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--L", type=float, default=8.0)
    p.add_argument("--nx", type=int, default=400)
    p.add_argument("--nr", type=int, default=200)
    p.add_argument("--dt", type=float, default=2.0e-4)
    p.add_argument("--tmax", type=float, default=1.0)
    p.add_argument("--save-interval", type=float, default=0.05)
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--hist-bins", type=int, default=200)
    p.add_argument("--hist-vmax", type=float, default=5.0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
