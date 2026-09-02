#!/usr/bin/env python3
"""H2: positive macro--micro PINN for the stationary Mach-2 BGK shock."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from h2_bgk import (H2_GATES, KNUDSEN_EFFECTIVE, PSI_MAX,
                    compact_quadrature_arrays)
from h2_reference import (heldout_metrics, load_reference, split_indices,
                          validation_regions)
from shock_physics import analytic_fluxes, normal_shock_states

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--nx-pde", type=int, default=161)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=600)
    parser.add_argument("--ramp-epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def network(width, depth):
    return tf.keras.Sequential(
        [tf.keras.layers.Input((5,))]
        + [tf.keras.layers.Dense(width, activation=tf.nn.silu) for _ in range(depth)]
        + [tf.keras.layers.Dense(5, kernel_initializer="zeros")]
    )


def features(y):
    z = 2.0 * y
    return tf.stack(
        (z, tf.sin(2.0 * np.pi * y), tf.cos(2.0 * np.pi * y),
         tf.sin(4.0 * np.pi * y), tf.cos(4.0 * np.pi * y)), axis=-1
    )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference, expected_mach=args.mach)
    split = split_indices(len(reference.x))
    regions = validation_regions(reference.x, split["held_out"])
    upstream, downstream = normal_shock_states(args.mach)
    vx_np, r2_np, weights_np = compact_quadrature_arrays(args.reference)
    vx = tf.constant(vx_np, DTYPE)
    r2 = tf.constant(r2_np, DTYPE)
    weights = tf.constant(weights_np, DTYPE)
    v2 = vx * vx + r2
    basis = tf.stack((tf.ones_like(vx), vx, v2), axis=-1)
    invariant_flux_target = tf.constant(analytic_fluxes(upstream), DTYPE)

    ypde = np.linspace(-0.5, 0.5, args.nx_pde, dtype=np.float32)
    train_indices = np.union1d(split["macro"], split["moments"])
    ytrain = (reference.x[train_indices] / 80.0).astype(np.float32)
    yall = tf.constant(np.concatenate((ypde, ytrain)), DTYPE)
    n_pde = len(ypde)
    index_of = {int(j): n_pde + i for i, j in enumerate(train_indices)}
    macro_local = tf.constant([index_of[int(j)] for j in split["macro"]], tf.int32)
    moment_local = tf.constant([index_of[int(j)] for j in split["moments"]], tf.int32)

    macro_targets = {
        "rho": tf.constant(reference.rho[split["macro"]], DTYPE),
        "u": tf.constant(reference.u[split["macro"]], DTYPE),
        "temperature": tf.constant(reference.temperature[split["macro"]], DTYPE),
    }
    q_target = tf.constant(reference.qx[split["moments"]], DTYPE)
    sigma_target = tf.constant(reference.sigma_xx[split["moments"]], DTYPE)
    scales = {
        "rho": float(np.max(np.abs(reference.rho))),
        "u": float(np.max(np.abs(reference.u))),
        "temperature": float(np.max(np.abs(reference.temperature))),
        "qx": float(np.max(np.abs(reference.qx))) + 1e-7,
        "sigma_xx": float(np.max(np.abs(reference.sigma_xx))) + 1e-7,
    }

    model = network(args.width, args.depth)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate, max(args.epochs, 1), alpha=0.08
    )
    optimizer = tf.keras.optimizers.Adam(schedule)

    def distribution_at(y, training):
        out = model(features(y), training=training)
        x_mfp = 80.0 * y
        transition = 0.5 * (1.0 + tf.tanh(x_mfp / 4.0))
        boundary_gate = tf.maximum(0.0, 1.0 - tf.square(2.0 * y))
        rho0 = upstream.rho + transition * (downstream.rho - upstream.rho)
        u0 = upstream.u + transition * (downstream.u - upstream.u)
        t0 = upstream.temperature + transition * (downstream.temperature - upstream.temperature)
        rho_macro = rho0 * tf.exp(0.45 * boundary_gate * tf.tanh(out[:, 0]))
        u_macro = u0 + 0.60 * boundary_gate * tf.tanh(out[:, 1])
        t_macro = t0 * tf.exp(0.40 * boundary_gate * tf.tanh(out[:, 2]))
        aq = 0.55 * boundary_gate * tf.tanh(out[:, 3])
        astress = 0.55 * boundary_gate * tf.tanh(out[:, 4])
        cxhat = (vx[None, :] - u_macro[:, None]) / tf.sqrt(t_macro[:, None])
        c2hat = cxhat * cxhat + r2[None, :] / t_macro[:, None]
        phi_q = ((0.5 * c2hat - 2.5) * cxhat) / 8.0
        phi_sigma = (cxhat * cxhat - c2hat / 3.0) / 4.0
        psi = PSI_MAX * tf.tanh(
            (aq[:, None] * phi_q + astress[:, None] * phi_sigma) / PSI_MAX
        )
        log_m = (
            tf.math.log(rho_macro[:, None])
            - 1.5 * tf.math.log(2.0 * np.pi * t_macro[:, None])
            - 0.5 * ((vx[None, :] - u_macro[:, None]) ** 2 + r2[None, :])
            / t_macro[:, None]
        )
        return tf.exp(tf.clip_by_value(log_m + psi, -80.0, 25.0))

    def moments(f):
        wf = f * weights[None, :]
        rho = tf.reduce_sum(wf, axis=1)
        momentum = tf.reduce_sum(wf * vx[None, :], axis=1)
        u = momentum / rho
        cx = vx[None, :] - u[:, None]
        c2 = cx * cx + r2[None, :]
        temperature = tf.reduce_sum(wf * c2, axis=1) / (3.0 * rho)
        return {
            "rho": rho,
            "u": u,
            "temperature": temperature,
            "qx": 0.5 * tf.reduce_sum(wf * cx * c2, axis=1),
            "sigma_xx": tf.reduce_sum(wf * (cx * cx - c2 / 3.0), axis=1),
            "flux": tf.stack(
                (momentum,
                 tf.reduce_sum(wf * vx[None, :] ** 2, axis=1),
                 0.5 * tf.reduce_sum(wf * vx[None, :] * v2[None, :], axis=1)),
                axis=1,
            ),
        }

    def discrete_maxwellian(m):
        rho, u, temperature = m["rho"], m["u"], m["temperature"]
        theta = tf.stack(
            (tf.math.log(rho) - 1.5 * tf.math.log(2.0 * np.pi * temperature)
             - u * u / (2.0 * temperature),
             u / temperature, -0.5 / temperature), axis=1
        )
        target = tf.stack((rho, rho * u, rho * (u * u + 3.0 * temperature)), axis=1)
        for damping in (0.8, 1.0):
            md = tf.exp(tf.clip_by_value(tf.einsum("xi,vi->xv", theta, basis), -80.0, 25.0))
            wm = md * weights[None, :]
            current = tf.einsum("xv,vi->xi", wm, basis)
            jacobian = tf.einsum("xv,vi,vj->xij", wm, basis, basis)
            delta = tf.linalg.solve(
                jacobian + 1e-8 * tf.eye(3, dtype=DTYPE)[None, :, :],
                (target - current)[..., None],
            )[..., 0]
            theta = theta + damping * tf.clip_by_value(delta, -0.5, 0.5)
        return tf.exp(tf.clip_by_value(tf.einsum("xi,vi->xv", theta, basis), -80.0, 25.0))

    velocity_weight = weights * (1.0 + 0.1 * v2 * v2)
    velocity_weight /= tf.reduce_sum(velocity_weight)
    dy = float(ypde[1] - ypde[0])

    def objective(pde_weight, training=True):
        f_all = distribution_at(yall, training)
        m_all = moments(f_all)
        f = f_all[:n_pde]
        m = {k: v[:n_pde] for k, v in m_all.items()}
        md = discrete_maxwellian(m)
        dfdx = (f[2:] - f[:-2]) / (2.0 * dy)
        residual = vx[None, :] * dfdx - (md[1:-1] - f[1:-1]) / KNUDSEN_EFFECTIVE
        collision_scale = tf.sqrt(
            tf.reduce_mean(tf.square((md[1:-1] - f[1:-1]) / KNUDSEN_EFFECTIVE))
        ) + 1e-8
        absolute_pde = tf.reduce_mean(
            tf.reduce_sum(tf.square(residual / collision_scale) * velocity_weight[None, :], axis=1)
        )
        denominator = tf.abs(f[1:-1]) + tf.abs(md[1:-1]) + 1e-7
        relative_pde = tf.reduce_mean(
            tf.reduce_sum(tf.square(residual / denominator) * velocity_weight[None, :], axis=1)
        )
        macro = tf.constant(0.0, DTYPE)
        for key in ("rho", "u", "temperature"):
            value = tf.gather(m_all[key], macro_local)
            factor = 12.0 if key == "u" else 1.0
            macro += factor * tf.reduce_mean(tf.square((value - macro_targets[key]) / scales[key]))
        qloss = tf.reduce_mean(tf.square((tf.gather(m_all["qx"], moment_local) - q_target) / scales["qx"]))
        sloss = tf.reduce_mean(tf.square((tf.gather(m_all["sigma_xx"], moment_local) - sigma_target) / scales["sigma_xx"]))
        flux_error = (m["flux"] - invariant_flux_target[None, :]) / tf.abs(invariant_flux_target[None, :])
        flux_loss = tf.reduce_mean(tf.square(flux_error))
        regularizer = tf.add_n([tf.reduce_mean(tf.square(v)) for v in model.trainable_variables])
        total = (25.0 * macro + 12.0 * qloss + 12.0 * sloss + 20.0 * flux_loss
                 + pde_weight * (absolute_pde + 0.02 * relative_pde) + 1e-7 * regularizer)
        return total, absolute_pde, relative_pde, macro, qloss, sloss, flux_loss, residual, m

    @tf.function
    def train_step(pde_weight):
        with tf.GradientTape() as tape:
            values = objective(pde_weight, training=True)
        gradients = tape.gradient(values[0], model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return values

    history = []
    best = np.inf
    best_weights = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            pde_weight = 0.0
        else:
            pde_weight = min(1.0, (epoch - args.warmup_epochs) / max(args.ramp_epochs, 1))
        values = train_step(tf.constant(pde_weight, DTYPE))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            row = [epoch, pde_weight] + [float(v) for v in values[:7]]
            history.append(row)
            if pde_weight >= 0.999 and row[2] < best:
                best = row[2]
                best_weights = model.get_weights()
            print(
                "h2 epoch=%d loss=%.6e pde=%.3e rel=%.3e macro=%.3e q=%.3e sigma=%.3e flux=%.3e ramp=%.2f"
                % tuple([epoch] + row[2:9] + [pde_weight]), flush=True
            )
    if best_weights is not None:
        model.set_weights(best_weights)
    final = objective(tf.constant(1.0, DTYPE), training=False)

    def predict_all(chunk=32):
        collected = {k: [] for k in ("rho", "u", "temperature", "qx", "sigma_xx", "flux")}
        minimum = np.inf
        y = (reference.x / 80.0).astype(np.float32)
        for start in range(0, len(y), chunk):
            f = distribution_at(tf.constant(y[start:start + chunk], DTYPE), False)
            minimum = min(minimum, float(tf.reduce_min(f)))
            mb = moments(f)
            for key in collected:
                collected[key].append(mb[key].numpy())
        return ({key: np.concatenate(value, axis=0) for key, value in collected.items()},
                minimum)

    prediction, minimum_distribution = predict_all()
    core = heldout_metrics(prediction, reference, regions["held_out_core"])
    full = heldout_metrics(prediction, reference, regions["held_out_full"])
    left = heldout_metrics(prediction, reference, regions["left_tail"])
    right = heldout_metrics(prediction, reference, regions["right_tail"])
    flux = prediction["flux"]
    spreads = {
        key: float(np.ptp(flux[:, i]) / max(abs(np.mean(flux[:, i])), 1e-12))
        for i, key in enumerate(("mass", "momentum", "energy"))
    }
    residual = final[7].numpy()
    fscale = float(np.sqrt(np.mean(((final[8]["flux"].numpy() - invariant_flux_target.numpy())
                                   / invariant_flux_target.numpy()) ** 2)))
    residual_rms = float(final[1]) ** 0.5
    boundary_error = float(max(
        abs(prediction["rho"][0] / upstream.rho - 1.0),
        abs(prediction["rho"][-1] / downstream.rho - 1.0),
        abs(prediction["u"][0] / upstream.u - 1.0),
        abs(prediction["u"][-1] / downstream.u - 1.0),
        abs(prediction["temperature"][0] / upstream.temperature - 1.0),
        abs(prediction["temperature"][-1] / downstream.temperature - 1.0),
    ))
    gates = {
        "finite": bool(all(np.isfinite(v).all() for v in prediction.values())),
        "positive": bool(np.isfinite(minimum_distribution) and minimum_distribution > 0.0),
        "boundary": bool(boundary_error < H2_GATES["boundary_relative_error"]),
        "rho_core": bool(core["rho"] < H2_GATES["rho_core_relative_l2"]),
        "u_core": bool(core["u"] < H2_GATES["u_core_relative_l2"]),
        "temperature_core": bool(core["temperature"] < H2_GATES["temperature_core_relative_l2"]),
        "qx_core": bool(core["qx"] < H2_GATES["qx_core_relative_l2"]),
        "sigma_xx_core": bool(core["sigma_xx"] < H2_GATES["sigma_xx_core_relative_l2"]),
        "fluxes": bool(max(spreads.values()) < H2_GATES["maximum_flux_relative_spread"]),
        "residual": bool(residual_rms < H2_GATES["relative_residual_rms"]),
    }
    status = "PILOT_PASS" if all(gates.values()) else "NO_GO"
    metrics = {
        "stage": "H2_GATE1_STATIONARY_BGK_PINN",
        "status": status,
        "mach": args.mach,
        "effective_knudsen": KNUDSEN_EFFECTIVE,
        "reference_sha256": reference.metadata["sha256"],
        "quadrature_points_full": int(reference.metadata["nv"]),
        "quadrature_points_axisymmetric": int(len(vx_np)),
        "macro_lock_points": int(len(split["macro"])),
        "moment_anchor_points": int(len(split["moments"])),
        "held_out_points": int(len(regions["held_out_full"])),
        "held_out_core_points": int(len(regions["held_out_core"])),
        "errors_core_relative_l2": core,
        "errors_full_relative_l2": full,
        "errors_left_tail_relative_l2": left,
        "errors_right_tail_relative_l2": right,
        "flux_relative_spreads": spreads,
        "flux_training_rms": fscale,
        "relative_residual_rms": residual_rms,
        "boundary_relative_error": boundary_error,
        "minimum_distribution": minimum_distribution,
        "epochs": args.epochs,
        "elapsed_seconds": time.time() - started,
        "gates_preregistered": H2_GATES,
        "gates": gates,
        "claim": "Sparse-moment BGK reproduction; dense DVM profiles are held out.",
    }
    (output / "h2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        output / "h2_profiles.csv",
        np.column_stack((reference.x, reference.rho, prediction["rho"], reference.u,
                         prediction["u"], reference.temperature, prediction["temperature"],
                         reference.qx, prediction["qx"], reference.sigma_xx,
                         prediction["sigma_xx"], prediction["flux"])),
        delimiter=",",
        header="x_mfp,rho_dvm,rho_pinn,u_dvm,u_pinn,T_dvm,T_pinn,qx_dvm,qx_pinn,sigma_dvm,sigma_pinn,mass_flux,momentum_flux,energy_flux",
        comments="",
    )
    np.savetxt(
        output / "h2_history.csv", np.asarray(history), delimiter=",",
        header="epoch,pde_ramp,total,pde_absolute,pde_relative,macro,qx,sigma,flux", comments=""
    )
    np.savez(output / "h2_split.npz", **split, **regions)
    model.save_weights(output / "h2.weights.h5")

    figure, axes = plt.subplots(2, 3, figsize=(15.2, 9.0))
    figure.subplots_adjust(top=0.77, hspace=0.68, wspace=0.30)
    figure.suptitle("H2 stationary BGK normal shock — independent DVM validation", y=0.985, fontsize=18)
    panels = [
        ("rho", reference.rho, r"Density $\rho$"),
        ("u", reference.u, r"Velocity $u_x$"),
        ("temperature", reference.temperature, r"Temperature $T$"),
        ("qx", reference.qx, r"Heat flux $q_x$"),
        ("sigma_xx", reference.sigma_xx, r"Normal stress $\sigma_{xx}$"),
    ]
    for axis, (key, truth, title) in zip(axes.flat[:5], panels):
        axis.plot(reference.x, truth, color="black", linewidth=2.1, label="DVM reference")
        axis.plot(reference.x, prediction[key], color="#d95f02", linewidth=1.8,
                  linestyle="--", label="H2 BGK–PINN")
        anchor = split["moments"] if key in ("qx", "sigma_xx") else split["macro"]
        axis.scatter(reference.x[anchor], truth[anchor], s=16, color="#1b9e77",
                     zorder=3, label="training anchors")
        axis.set_title(title, pad=46)
        axis.set_xlabel(r"$x/\lambda_1$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    hist = np.asarray(history)
    axis = axes.flat[5]
    axis.semilogy(hist[:, 0], np.maximum(hist[:, 3], 1e-12), label="weighted BGK residual")
    axis.semilogy(hist[:, 0], np.maximum(hist[:, 8], 1e-12), label="invariant-flux loss")
    axis.set_title("Physics optimization", pad=46)
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    figure.savefig(output / "h2_bgk_physics.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    print("H2_METRICS", json.dumps(metrics, sort_keys=True), flush=True)
    print(f"H2_OUTPUT={output}", flush=True)
    return 0 if status == "PILOT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
