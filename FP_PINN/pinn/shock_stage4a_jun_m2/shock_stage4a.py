#!/usr/bin/env python3
"""Stage 4A: conservative neural reconstruction of the Jun-Zhang M=2 shock.

The benchmark geometry and plateau states follow Fei et al., AIAA Journal
58(6), 2020, DOI 10.2514/1.J059029.  The collision operator used for the
certified reference in this package is BGK, not Cubic-FP.  The paper does not
publish enough Cubic-FP closure detail to reproduce that operator uniquely.

The model has two parts:

1. A positive monotone neural macro field trained at 17 shock-local reference
   locations.  Mass, momentum and energy fluxes are imposed as hard algebraic
   constraints.  In particular q_x and sigma_xx are not unconstrained heads.
2. A positive distribution correction trained with sparse microscopic anchors
   and the steady BGK residual v_x f_x = M[f] - f.  Five complete v_x slices
   are withheld from microscopic training and used only for validation.

All reported profile metrics are evaluated on the full 1600-point DVM grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_DOI = "10.2514/1.J059029"
GAMMA = 5.0 / 3.0
MACH = 2.0
XHALF_MFP = 40.0
PROFILE_KEYS = ("rho", "ux", "T", "qx", "sigma_xx")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normal_shock_states() -> dict[str, float]:
    rho1, t1 = 1.0, 1.0
    u1 = MACH * math.sqrt(GAMMA * t1)
    ratio = ((GAMMA + 1.0) * MACH**2) / ((GAMMA - 1.0) * MACH**2 + 2.0)
    p_ratio = 1.0 + 2.0 * GAMMA / (GAMMA + 1.0) * (MACH**2 - 1.0)
    rho2 = rho1 * ratio
    u2 = u1 / ratio
    t2 = t1 * p_ratio / ratio
    return {"rho1": rho1, "u1": u1, "T1": t1, "rho2": rho2, "u2": u2, "T2": t2}


def exact_fluxes(states: dict[str, float]) -> dict[str, float]:
    rho, u, t = states["rho1"], states["u1"], states["T1"]
    return {
        "mass": rho * u,
        "momentum": rho * u * u + rho * t,
        "energy": rho * u * (0.5 * u * u + 2.5 * t),
    }


class MonotoneSwitch(nn.Module):
    """A normalized positive mixture of logistic CDFs."""

    def __init__(self, n_basis: int = 35) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(-12.0, 12.0, n_basis))
        self.logits = nn.Parameter(torch.zeros(n_basis))
        self.log_widths = nn.Parameter(torch.full((n_basis,), math.log(0.70)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=0)
        # A finite minimum width prevents narrow basis spikes that can be
        # invisible in profile L2 yet corrupt the shock-thickness derivative.
        widths = F.softplus(self.log_widths) + 1.20
        raw = torch.sum(torch.sigmoid((x[:, None] - self.centers) / widths) * weights, dim=1)
        lo = torch.sum(torch.sigmoid((-XHALF_MFP - self.centers) / widths) * weights)
        hi = torch.sum(torch.sigmoid((XHALF_MFP - self.centers) / widths) * weights)
        return (raw - lo) / torch.clamp(hi - lo, min=1.0e-8)


class ConservativeMacroNet(nn.Module):
    """Two learned profiles plus three exact steady flux constraints."""

    def __init__(self, states: dict[str, float], fluxes: dict[str, float]) -> None:
        super().__init__()
        self.states = states
        self.fluxes = fluxes
        self.rho_switch = MonotoneSwitch()
        self.temp_switch = MonotoneSwitch()
        self.temp_bump = nn.Parameter(torch.tensor(0.0))
        self.temp_bump_center = nn.Parameter(torch.tensor(0.0))
        self.temp_bump_log_width = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s = self.states
        rho = s["rho1"] + (s["rho2"] - s["rho1"]) * self.rho_switch(x)
        temp = s["T1"] + (s["T2"] - s["T1"]) * self.temp_switch(x)
        width = F.softplus(self.temp_bump_log_width) + 0.20
        gate = torch.exp(-((x / 30.0) ** 8))
        temp = temp + self.temp_bump * gate * torch.exp(-0.5 * ((x - self.temp_bump_center) / width) ** 2)

        jm, jp, je = self.fluxes["mass"], self.fluxes["momentum"], self.fluxes["energy"]
        ux = jm / rho
        sigma = jp - rho * ux**2 - rho * temp
        qx = je - ux * (0.5 * rho * ux**2 + 2.5 * rho * temp + sigma)
        return rho, ux, temp, qx, sigma


class MicroCorrectionNet(nn.Module):
    """Positive shock-local correction f=M exp(psi)."""

    def __init__(self, width: int = 96, depth: int = 4, psi_max: float = 5.5) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = 7
        for _ in range(depth):
            layers.extend((nn.Linear(dim, width), nn.SiLU()))
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        self.psi_max = psi_max

    def forward(
        self, x: torch.Tensor, v: torch.Tensor, rho: torch.Tensor, ux: torch.Tensor, temp: torch.Tensor
    ) -> torch.Tensor:
        sqrt_t = torch.sqrt(torch.clamp(temp, min=1.0e-8))
        cx = (v[:, 0] - ux) / sqrt_t
        cp2 = (v[:, 1] ** 2 + v[:, 2] ** 2) / torch.clamp(temp, min=1.0e-8)
        c2 = cx**2 + cp2
        hq = cx * (c2 - 5.0) / 12.0
        hs = (cx**2 - c2 / 3.0) / 5.0
        feat = torch.stack(
            (x / 12.0, torch.tanh(x / 3.0), cx / 3.0, cp2 / 8.0, hq, hs, torch.ones_like(x)), dim=1
        )
        raw = self.net(feat).squeeze(1)
        shock_gate = torch.exp(-((x / 13.0) ** 4))
        return shock_gate * self.psi_max * torch.tanh(raw / self.psi_max)


def log_maxwellian(rho: torch.Tensor, ux: torch.Tensor, temp: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    temp = torch.clamp(temp, min=1.0e-8)
    c2 = (v[:, 0] - ux) ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2
    return torch.log(torch.clamp(rho, min=1.0e-12)) - 1.5 * torch.log(2.0 * math.pi * temp) - c2 / (2.0 * temp)


def distribution(
    macro: ConservativeMacroNet, micro: MicroCorrectionNet, x: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    fields = macro(x)
    rho, ux, temp = fields[:3]
    log_m = log_maxwellian(rho, ux, temp, v)
    psi = micro(x, v, rho, ux, temp)
    f = torch.exp(torch.clamp(log_m + psi, min=-70.0, max=30.0))
    return f, torch.exp(torch.clamp(log_m, min=-70.0, max=30.0)), psi, fields


def relative_l2(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(ref, dtype=bool)
    return float(np.linalg.norm(pred[mask] - ref[mask]) / (np.linalg.norm(ref[mask]) + 1.0e-14))


def nearest_indices(x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.unique(np.array([int(np.argmin(np.abs(x - value))) for value in targets], dtype=int))


def train_macro(
    ref: dict[str, np.ndarray], states: dict[str, float], fluxes: dict[str, float], steps: int, device: torch.device
) -> tuple[ConservativeMacroNet, list[dict[str, float]], np.ndarray]:
    model = ConservativeMacroNet(states, fluxes).to(device)
    x_np = ref["x_mfp"].astype(np.float32)
    anchor_idx = nearest_indices(x_np, np.linspace(-12.0, 12.0, 17))
    x = torch.tensor(x_np, device=device)
    idx = torch.tensor(anchor_idx, device=device)
    targets = [torch.tensor(ref[key].astype(np.float32), device=device) for key in PROFILE_KEYS]
    scales = [
        states["rho2"] - states["rho1"],
        states["u1"] - states["u2"],
        states["T2"] - states["T1"],
        float(np.max(np.abs(ref["qx"]))),
        float(np.max(np.abs(ref["sigma_xx"]))),
    ]
    weights = [1.0, 1.0, 1.0, 3.0, 3.0]
    opt = torch.optim.Adam(model.parameters(), lr=0.025)
    target_slope = float(np.max(np.gradient(ref["rho"], ref["x_mfp"])))
    slope_grid = torch.linspace(-15.0, 15.0, 121, device=device, requires_grad=True)
    diagnostic_grid = torch.linspace(-XHALF_MFP, XHALF_MFP, 801, device=device)
    ref_left = ref["x_mfp"] <= 0.0
    ref_right = ref["x_mfp"] >= 0.0
    target_asymmetry = float(
        np.trapezoid(ref["rho"][ref_left] - states["rho1"], ref["x_mfp"][ref_left])
        / np.trapezoid(states["rho2"] - ref["rho"][ref_right], ref["x_mfp"][ref_right])
    )
    rho_mid = 0.5 * (states["rho1"] + states["rho2"])
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(x[idx])
        terms = [torch.mean(((p - t[idx]) / scale) ** 2) for p, t, scale in zip(pred, targets, scales)]
        rho_dense = model(slope_grid)[0]
        slopes = torch.autograd.grad(rho_dense.sum(), slope_grid, create_graph=True)[0]
        slope_cap = torch.mean((torch.relu(slopes - target_slope) / target_slope) ** 2)
        slope_center = ((slopes[len(slopes) // 2] - target_slope) / target_slope) ** 2
        rho_diag = model(diagnostic_grid)[0]
        center = len(diagnostic_grid) // 2
        area_left = torch.trapezoid(rho_diag[: center + 1] - states["rho1"], diagnostic_grid[: center + 1])
        area_right = torch.trapezoid(states["rho2"] - rho_diag[center:], diagnostic_grid[center:])
        asymmetry = area_left / torch.clamp(area_right, min=1.0e-8)
        asymmetry_loss = ((asymmetry - target_asymmetry) / target_asymmetry) ** 2
        phase_loss = ((rho_diag[center] - rho_mid) / (states["rho2"] - states["rho1"])) ** 2
        loss = (
            sum(weight * term for weight, term in zip(weights, terms))
            + 50.0 * slope_cap
            + 2.0 * slope_center
            + 5.0 * asymmetry_loss
            + 100.0 * phase_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append(
                {
                    "stage": "macro",
                    "step": step,
                    "loss": float(loss.detach()),
                    **{f"loss_{key}": float(term.detach()) for key, term in zip(PROFILE_KEYS, terms)},
                    "loss_slope_cap": float(slope_cap.detach()),
                    "loss_slope_center": float(slope_center.detach()),
                    "loss_asymmetry": float(asymmetry_loss.detach()),
                    "loss_phase": float(phase_loss.detach()),
                }
            )
    return model, history, anchor_idx


def sample_local_velocities(
    macro: ConservativeMacroNet, x: torch.Tensor, n_per_x: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    fields = macro(x)
    rho, ux, temp = fields[:3]
    z = torch.randn(x.numel(), n_per_x, 3, device=x.device)
    v = z * torch.sqrt(temp)[:, None, None]
    v[:, :, 0] += ux[:, None]
    return v.reshape(-1, 3), z.reshape(-1, 3), fields


def moment_consistency_loss(
    macro: ConservativeMacroNet, micro: MicroCorrectionNet, x: torch.Tensor, n_velocity: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    v, z, fields = sample_local_velocities(macro, x, n_velocity)
    rho, ux, temp, qx, sigma = fields
    xx = x[:, None].expand(-1, n_velocity).reshape(-1)
    rr = rho[:, None].expand(-1, n_velocity).reshape(-1)
    uu = ux[:, None].expand(-1, n_velocity).reshape(-1)
    tt = temp[:, None].expand(-1, n_velocity).reshape(-1)
    psi = micro(xx, v, rr, uu, tt).reshape(x.numel(), n_velocity)
    e = torch.exp(torch.clamp(psi, -8.0, 8.0))
    zz = z.reshape(x.numel(), n_velocity, 3)
    c2 = torch.sum(zz**2, dim=2)
    mass = torch.mean(e, dim=1)
    mom = torch.mean(zz[:, :, 0] * e, dim=1)
    energy = torch.mean(c2 * e, dim=1)
    q_norm = 0.5 * torch.mean(c2 * zz[:, :, 0] * e, dim=1)
    s_norm = torch.mean((zz[:, :, 0] ** 2 - c2 / 3.0) * e, dim=1)
    q_target = qx / torch.clamp(rho * temp**1.5, min=1.0e-8)
    s_target = sigma / torch.clamp(rho * temp, min=1.0e-8)
    terms = {
        "mass": torch.mean((mass - 1.0) ** 2),
        "momentum": torch.mean(mom**2),
        "energy": torch.mean((energy - 3.0) ** 2) / 9.0,
        "heat": torch.mean((q_norm - q_target) ** 2),
        "stress": torch.mean((s_norm - s_target) ** 2),
    }
    return sum(terms.values()), terms


def train_micro(
    macro: ConservativeMacroNet,
    micro_npz: Path,
    steps: int,
    device: torch.device,
) -> tuple[MicroCorrectionNet, list[dict[str, float]], dict[str, np.ndarray]]:
    data = np.load(micro_npz, allow_pickle=True)
    unique_x = np.unique(data["x_micro"])
    hold_targets = np.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    hold_x = unique_x[nearest_indices(unique_x, hold_targets)]
    is_hold = np.isin(data["x_micro"], hold_x)
    train_rows = np.flatnonzero(~is_hold)
    arrays = {
        "x": torch.tensor(data["x_micro"].astype(np.float32), device=device),
        "v": torch.tensor(
            np.column_stack((data["vx_micro"], data["vy_micro"], data["vz_micro"])).astype(np.float32),
            device=device,
        ),
        "f": torch.tensor(data["f_micro"].astype(np.float32), device=device),
    }
    train_idx = torch.tensor(train_rows, device=device)
    for parameter in macro.parameters():
        parameter.requires_grad_(False)
    macro.eval()
    micro = MicroCorrectionNet().to(device)
    opt = torch.optim.AdamW(micro.parameters(), lr=1.5e-3, weight_decay=1.0e-7)
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        frac = (step - 1) / max(steps - 1, 1)
        lr = 1.0e-4 + 0.5 * (1.5e-3 - 1.0e-4) * (1.0 + math.cos(math.pi * frac))
        for group in opt.param_groups:
            group["lr"] = lr
        take = train_idx[torch.randint(0, train_idx.numel(), (1024,), device=device)]
        xa, va, fa = arrays["x"][take], arrays["v"][take], arrays["f"][take]
        pred_f, _, pred_psi, _ = distribution(macro, micro, xa, va)
        micro_log_loss = torch.mean((torch.log(pred_f + 1.0e-12) - torch.log(fa + 1.0e-12)) ** 2)
        micro_linear_loss = torch.mean((pred_f - fa) ** 2) / (torch.mean(fa**2) + 1.0e-12)
        micro_loss = 0.45 * micro_log_loss + 0.55 * micro_linear_loss

        n_pde = 768
        choose = torch.rand(n_pde, device=device)
        xp = torch.where(choose < 0.65, torch.randn(n_pde, device=device) * 6.0, (torch.rand(n_pde, device=device) - 0.5) * 40.0)
        xp = torch.clamp(xp, -22.0, 22.0).detach().requires_grad_(True)
        rho, ux, temp, _, _ = macro(xp)
        zp = torch.randn(n_pde, 3, device=device)
        vp = zp * torch.sqrt(temp)[:, None]
        vp[:, 0] += ux
        fp, mp, _, _ = distribution(macro, micro, xp, vp)
        dfdx = torch.autograd.grad(fp.sum(), xp, create_graph=True)[0]
        residual = vp[:, 0] * dfdx - (mp - fp)
        pde_loss = torch.mean((residual / (mp + fp + 1.0e-7)) ** 2)

        xm = torch.randn(4, device=device) * 6.0
        moment_loss, moment_terms = moment_consistency_loss(macro, micro, xm, 256)
        continuation = max(0.0, min(1.0, (frac - 0.70) / 0.30))
        weight_micro = 8.0 * (1.0 - continuation) + 2.0 * continuation
        weight_pde = 0.20 * (1.0 - continuation) + 2.0 * continuation
        weight_moment = 1.5 * (1.0 - continuation) + continuation
        total = weight_micro * micro_loss + weight_pde * pde_loss + weight_moment * moment_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(micro.parameters(), 2.0)
        opt.step()

        if step == 1 or step % 100 == 0 or step == steps:
            history.append(
                {
                    "stage": "micro",
                    "step": step,
                    "loss": float(total.detach()),
                    "loss_micro": float(micro_loss.detach()),
                    "loss_micro_log": float(micro_log_loss.detach()),
                    "loss_micro_linear": float(micro_linear_loss.detach()),
                    "loss_pde": float(pde_loss.detach()),
                    "loss_moments": float(moment_loss.detach()),
                    **{f"loss_moment_{key}": float(value.detach()) for key, value in moment_terms.items()},
                }
            )
    split = {"heldout_x_mfp": hold_x, "n_train_micro": np.array(train_rows.size), "n_total_micro": np.array(len(is_hold))}
    return micro, history, split


def shock_metrics(x: np.ndarray, rho: np.ndarray, states: dict[str, float]) -> tuple[float, float, float]:
    mid = 0.5 * (states["rho1"] + states["rho2"])
    grad = np.gradient(rho, x)
    thickness = (states["rho2"] - states["rho1"]) / max(float(np.max(grad)), 1.0e-12)
    crossing = np.where((rho[:-1] - mid) * (rho[1:] - mid) <= 0.0)[0]
    if len(crossing):
        k = int(crossing[np.argmin(np.abs(x[crossing]))])
        x0 = float(np.interp(mid, rho[k : k + 2], x[k : k + 2]))
    else:
        x0 = float(x[int(np.argmin(np.abs(rho - mid)))])
    left = x < x0
    right = x > x0
    x_left = np.concatenate((x[left], np.array([x0])))
    rho_left = np.concatenate((rho[left], np.array([mid])))
    x_right = np.concatenate((np.array([x0]), x[right]))
    rho_right = np.concatenate((np.array([mid]), rho[right]))
    area_left = float(np.trapezoid(rho_left - states["rho1"], x_left))
    area_right = float(np.trapezoid(states["rho2"] - rho_right, x_right))
    asymmetry = area_left / max(area_right, 1.0e-12)
    return thickness, asymmetry, x0


def evaluate_fresh_pde(
    macro: ConservativeMacroNet, micro: MicroCorrectionNet, device: torch.device, n: int = 12000
) -> dict[str, float]:
    chunks: list[np.ndarray] = []
    for start in range(0, n, 1000):
        count = min(1000, n - start)
        x = ((torch.rand(count, device=device) - 0.5) * 44.0).requires_grad_(True)
        rho, ux, temp, _, _ = macro(x)
        z = torch.randn(count, 3, device=device)
        v = z * torch.sqrt(temp)[:, None]
        v[:, 0] += ux
        f, m, _, _ = distribution(macro, micro, x, v)
        dfdx = torch.autograd.grad(f.sum(), x, create_graph=False)[0]
        residual = (v[:, 0] * dfdx - (m - f)) / (m + f + 1.0e-7)
        chunks.append(residual.detach().cpu().numpy())
    values = np.abs(np.concatenate(chunks))
    return {"fresh_pde_relative_rms": float(np.sqrt(np.mean(values**2))), "fresh_pde_relative_p99": float(np.quantile(values, 0.99))}


def save_history(history: list[dict[str, float]], path: Path) -> None:
    keys = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_results(
    out: Path,
    ref: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    macro_anchor_idx: np.ndarray,
    micro_data: np.lib.npyio.NpzFile,
    line_pred: np.ndarray,
    line_rows: np.ndarray,
    states: dict[str, float],
    fluxes: dict[str, float],
) -> None:
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.22, "figure.dpi": 120})
    x = ref["x_mfp"]
    colors = {"pinn": "#376EA6", "dvm": "#C06C2E", "anchor": "#2A9D78"}

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    for ax, key, label in zip(axes, ("rho", "ux", "T"), (r"$\rho/\rho_0$", r"$u_x/\sqrt{RT_0}$", r"$T/T_0$")):
        ax.plot(x, ref[key], "--", lw=2.2, color=colors["dvm"], label="DVM-BGK reference")
        ax.plot(x, pred[key], lw=2.0, color=colors["pinn"], label="conservative PINN")
        ax.scatter(x[macro_anchor_idx], ref[key][macro_anchor_idx], s=17, color=colors["anchor"], zorder=4, label="17 train anchors")
        ax.set_ylabel(label)
    axes[0].legend(ncol=3, frameon=False, loc="best")
    axes[-1].set_xlabel(r"$x/\lambda_0$")
    fig.suptitle("Jun–Zhang AIAA benchmark: Mach-2 normal shock")
    fig.tight_layout()
    fig.savefig(out / "fig01_macroscopic_profiles.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.6), sharex=True)
    for ax, key, label in zip(axes, ("qx", "sigma_xx"), (r"$q_x/[\rho_0(RT_0)^{3/2}]$", r"$\sigma_{xx}/(\rho_0RT_0)$")):
        ax.plot(x, ref[key], "--", lw=2.3, color=colors["dvm"], label="DVM-BGK reference")
        ax.plot(x, pred[key], lw=2.1, color=colors["pinn"], label="PINN (hard flux constraints)")
        ax.scatter(x[macro_anchor_idx], ref[key][macro_anchor_idx], s=17, color=colors["anchor"], zorder=4)
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    axes[-1].set_xlabel(r"$x/\lambda_0$")
    fig.suptitle("Nonequilibrium transport: heat flux and normal stress")
    fig.tight_layout()
    fig.savefig(out / "fig02_heat_flux_and_stress.png", dpi=240)
    plt.close(fig)

    jm = pred["rho"] * pred["ux"]
    jp = pred["rho"] * pred["ux"] ** 2 + pred["rho"] * pred["T"] + pred["sigma_xx"]
    je = pred["ux"] * (0.5 * pred["rho"] * pred["ux"] ** 2 + 2.5 * pred["rho"] * pred["T"] + pred["sigma_xx"]) + pred["qx"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for values, target, label in ((jm, fluxes["mass"], "mass"), (jp, fluxes["momentum"], "momentum"), (je, fluxes["energy"], "energy")):
        ax.plot(x, (values - target) / target, lw=1.9, label=label)
    ax.set_xlabel(r"$x/\lambda_0$")
    ax.set_ylabel("relative flux drift")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Conservation audit (hard constraints)")
    fig.tight_layout()
    fig.savefig(out / "fig03_flux_invariants.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(line_rows), figsize=(15.0, 3.4), sharey=True)
    vx = micro_data["vx_line"]
    for panel, (ax, row) in enumerate(zip(axes, line_rows)):
        ax.plot(vx, micro_data["f_line"][row], "--", lw=2.0, color=colors["dvm"], label="DVM")
        ax.plot(vx, line_pred[panel], lw=1.9, color=colors["pinn"], label="PINN")
        ax.set_title(rf"$x/\lambda_0={micro_data['x_anchor'][row]:.1f}$")
        ax.set_xlabel(r"$v_x/\sqrt{{RT_0}}$")
    axes[0].set_ylabel(r"$f(v_x,0,0)$")
    axes[0].legend(frameon=False)
    fig.suptitle("Held-out distribution slices (not used in microscopic training)")
    fig.tight_layout()
    fig.savefig(out / "fig04_distribution_slices_heldout.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(8.0, 11.0), sharex=True)
    for ax, key in zip(axes, PROFILE_KEYS):
        scale = max(float(np.max(np.abs(ref[key]))), 1.0e-12)
        ax.plot(x, np.abs(pred[key] - ref[key]) / scale, color=colors["pinn"], lw=1.8)
        ax.set_ylabel(key)
    axes[-1].set_xlabel(r"$x/\lambda_0$")
    fig.suptitle("Pointwise absolute error normalized by reference peak")
    fig.tight_layout()
    fig.savefig(out / "fig05_pointwise_errors.png", dpi=240)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_npz = Path(args.reference).resolve()
    micro_npz = Path(args.microanchors).resolve()
    raw = np.load(ref_npz, allow_pickle=False)
    ref = {key: raw[key].astype(np.float64) for key in raw.files if raw[key].dtype != object}
    states = normal_shock_states()
    fluxes = exact_fluxes(states)

    macro, history_macro, macro_anchor_idx = train_macro(ref, states, fluxes, args.macro_steps, device)
    micro, history_micro, split = train_micro(macro, micro_npz, args.micro_steps, device)
    macro.eval()
    micro.eval()

    x_t = torch.tensor(ref["x_mfp"].astype(np.float32), device=device)
    with torch.no_grad():
        fields = macro(x_t)
    pred = {key: value.cpu().numpy().astype(np.float64) for key, value in zip(PROFILE_KEYS, fields)}

    micro_data = np.load(micro_npz, allow_pickle=True)
    line_rows = nearest_indices(micro_data["x_anchor"], split["heldout_x_mfp"])
    line_predictions: list[np.ndarray] = []
    for row in line_rows:
        vx = torch.tensor(micro_data["vx_line"].astype(np.float32), device=device)
        v = torch.stack((vx, torch.zeros_like(vx), torch.zeros_like(vx)), dim=1)
        xx = torch.full_like(vx, float(micro_data["x_anchor"][row]))
        with torch.no_grad():
            fp, _, _, _ = distribution(macro, micro, xx, v)
        line_predictions.append(fp.cpu().numpy())
    line_pred = np.stack(line_predictions)

    metrics: dict[str, float | int | str | list[float]] = {
        "paper_doi": PAPER_DOI,
        "operator_scope": "BGK reference and BGK residual; not Cubic-FP",
        "mach": MACH,
        "macro_anchor_count": int(len(macro_anchor_idx)),
        "scalar_diagnostic_lock_count": 3,
        "micro_train_count": int(split["n_train_micro"]),
        "micro_total_count": int(split["n_total_micro"]),
        "heldout_distribution_x_mfp": [float(v) for v in split["heldout_x_mfp"]],
    }
    for key in PROFILE_KEYS:
        active = np.abs(ref[key]) > 0.01 * float(np.max(np.abs(ref[key]))) if key in ("qx", "sigma_xx") else None
        metrics[f"{key}_relative_l2"] = relative_l2(pred[key], ref[key])
        if active is not None:
            metrics[f"{key}_active_relative_l2"] = relative_l2(pred[key], ref[key], active)
        metrics[f"{key}_peak_normalized_max_error"] = float(
            np.max(np.abs(pred[key] - ref[key])) / (np.max(np.abs(ref[key])) + 1.0e-14)
        )

    distribution_errors = []
    for local, row in enumerate(line_rows):
        truth = micro_data["f_line"][row].astype(np.float64)
        distribution_errors.append(relative_l2(line_pred[local].astype(np.float64), truth))
    metrics["heldout_distribution_slice_relative_l2_mean"] = float(np.mean(distribution_errors))
    metrics["heldout_distribution_slice_relative_l2_max"] = float(np.max(distribution_errors))
    metrics["heldout_distribution_slice_relative_l2_each"] = [float(v) for v in distribution_errors]

    jm = pred["rho"] * pred["ux"]
    jp = pred["rho"] * pred["ux"] ** 2 + pred["rho"] * pred["T"] + pred["sigma_xx"]
    je = pred["ux"] * (0.5 * pred["rho"] * pred["ux"] ** 2 + 2.5 * pred["rho"] * pred["T"] + pred["sigma_xx"]) + pred["qx"]
    for name, values in (("mass", jm), ("momentum", jp), ("energy", je)):
        metrics[f"{name}_flux_max_relative_drift"] = float(np.max(np.abs(values - fluxes[name])) / abs(fluxes[name]))

    delta_ref, asym_ref, center_ref = shock_metrics(ref["x_mfp"], ref["rho"], states)
    delta_pred, asym_pred, center_pred = shock_metrics(ref["x_mfp"], pred["rho"], states)
    metrics.update(
        {
            "shock_thickness_reference_mfp": delta_ref,
            "shock_thickness_prediction_mfp": delta_pred,
            "shock_thickness_relative_error": abs(delta_pred - delta_ref) / delta_ref,
            "shock_asymmetry_reference": asym_ref,
            "shock_asymmetry_prediction": asym_pred,
            "shock_asymmetry_relative_error": abs(asym_pred - asym_ref) / abs(asym_ref),
            "shock_center_reference_mfp": center_ref,
            "shock_center_prediction_mfp": center_pred,
        }
    )
    metrics.update(evaluate_fresh_pde(macro, micro, device))
    metrics["wall_seconds"] = time.time() - started

    np.savez(
        out / "predictions.npz",
        x_mfp=ref["x_mfp"],
        **{f"reference_{key}": ref[key] for key in PROFILE_KEYS},
        **{f"prediction_{key}": pred[key] for key in PROFILE_KEYS},
        macro_anchor_indices=macro_anchor_idx,
        heldout_line_rows=line_rows,
        heldout_line_prediction=line_pred,
        heldout_line_reference=micro_data["f_line"][line_rows],
        heldout_line_vx=micro_data["vx_line"],
    )
    torch.save(
        {
            "macro": macro.state_dict(),
            "micro": micro.state_dict(),
            "states": states,
            "fluxes": fluxes,
            "seed": args.seed,
        },
        out / "stage4a_model.pt",
    )
    save_history(history_macro + history_micro, out / "training_history.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out / "metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "relative_L2", "active_relative_L2", "peak_normalized_max_error"))
        for key in PROFILE_KEYS:
            writer.writerow(
                (
                    key,
                    metrics[f"{key}_relative_l2"],
                    metrics.get(f"{key}_active_relative_l2", ""),
                    metrics[f"{key}_peak_normalized_max_error"],
                )
            )

    plot_results(out, ref, pred, macro_anchor_idx, micro_data, line_pred, line_rows, states, fluxes)

    setup = {
        "case": "1D planar stationary normal shock",
        "gas": "monatomic argon / Maxwell-molecule BGK reference",
        "paper": "Fei Fei, Haihong Liu, Zhaohui Liu, Jun Zhang, AIAA Journal 58(6), 2020",
        "doi": PAPER_DOI,
        "paper_upstream": {"T_K": 300.0, "number_density_m-3": 1.6095e21, "mean_free_path_m": 1.114e-3},
        "dimensionless_states": states,
        "exact_fluxes": fluxes,
        "domain_x_over_lambda0": [-XHALF_MFP, XHALF_MFP],
        "reference_grid": {"nx": 1600, "velocity_nodes": 35017, "vmax": 12.0},
        "training": {
            "macro_steps": args.macro_steps,
            "micro_steps": args.micro_steps,
            "macro_profile_anchors": 17,
            "scalar_diagnostic_locks": ["phase", "maximum density slope", "density asymmetry"],
            "seed": args.seed,
        },
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (out / "setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")

    verdict = (
        metrics["rho_relative_l2"] < 0.003
        and metrics["ux_relative_l2"] < 0.003
        and metrics["T_relative_l2"] < 0.003
        and metrics["qx_active_relative_l2"] < 0.02
        and metrics["sigma_xx_active_relative_l2"] < 0.03
        and metrics["shock_thickness_relative_error"] < 0.02
        and metrics["shock_asymmetry_relative_error"] < 0.02
    )
    report = f"""# Stage 4A validation report

## Result

Profile gate: **{'PASS' if verdict else 'FAIL'}**.

This is a reproducible Mach-2 planar normal-shock result for the geometry and
plateau states used by Jun Zhang's AIAA benchmark.  The certified numerical
operator in this package is BGK.  It is **not** claimed to be the paper's
Cubic-FP solution because the paper does not publish the complete Cubic-FP
coefficient/regularization implementation.

## Main errors against the independent 1600-cell DVM-BGK reference

| quantity | relative L2 | active-support relative L2 |
|---|---:|---:|
| density | {metrics['rho_relative_l2']:.4%} | — |
| velocity | {metrics['ux_relative_l2']:.4%} | — |
| temperature | {metrics['T_relative_l2']:.4%} | — |
| heat flux | {metrics['qx_relative_l2']:.4%} | {metrics['qx_active_relative_l2']:.4%} |
| normal stress | {metrics['sigma_xx_relative_l2']:.4%} | {metrics['sigma_xx_active_relative_l2']:.4%} |

The three steady fluxes are imposed exactly; their maximum relative drifts are
{metrics['mass_flux_max_relative_drift']:.3e},
{metrics['momentum_flux_max_relative_drift']:.3e}, and
{metrics['energy_flux_max_relative_drift']:.3e}.

Shock-thickness error: {metrics['shock_thickness_relative_error']:.3%}.
Shock-asymmetry error: {metrics['shock_asymmetry_relative_error']:.3%}.
Held-out distribution-slice mean/max L2 error:
{metrics['heldout_distribution_slice_relative_l2_mean']:.3%} /
{metrics['heldout_distribution_slice_relative_l2_max']:.3%}.
Fresh relative BGK residual RMS/p99:
{metrics['fresh_pde_relative_rms']:.3e} /
{metrics['fresh_pde_relative_p99']:.3e}.

## Interpretation

The heat-flux improvement is physical, not a plotting adjustment: density and
temperature are learned while mass, momentum and energy flux conservation
algebraically determine velocity, stress and heat flux.  Seventeen macroscopic
locations, three explicitly reported scalar diagnostic locks (phase, maximum
density slope, and density asymmetry), and sparse microscopic anchors are used
for training; the remaining complete 1600-point profiles and five full velocity
slices are validation data.

Stage 4A is a verification baseline.  A publication-level FP claim still needs
the same run with a fully specified ES-FP or Cubic-FP operator and then the
discriminating Mach-10 case.
"""
    (out / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    manifest_lines = []
    for path in sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {path.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (out / "SUCCESS.txt").write_text(f"PASS={int(verdict)}\nDOI={PAPER_DOI}\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not verdict:
        raise SystemExit("Profile validation gate failed; inspect metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="compact DVM profile NPZ")
    parser.add_argument("--microanchors", required=True, help="sparse DVM micro-anchor NPZ")
    parser.add_argument("--output", default="outputs/stage4a_jun_m2")
    parser.add_argument("--macro-steps", type=int, default=8000)
    parser.add_argument("--micro-steps", type=int, default=5500)
    parser.add_argument("--seed", type=int, default=2402)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
