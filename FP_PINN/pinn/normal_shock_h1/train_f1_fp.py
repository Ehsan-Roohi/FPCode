#!/usr/bin/env python3
"""F1 operator-switch continuation for a stationary Dougherty--FP shock.

The Mach-2 BGK DVM is never used in the loss.  It is loaded only after the
physics optimization to provide a labelled cross-model comparison.
"""
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

from f1_fp import (F1_GATES, analytic_dougherty_collision_tf,
                   structured_axisymmetric_quadrature)
from h2_bgk import (KNUDSEN_EFFECTIVE, LOG_CEILING, LOG_FLOOR,
                    LOG_TILT_LIMIT, PSI_MAX)
from h2_reference import heldout_metrics, load_reference, validation_regions
from shock_physics import analytic_fluxes, normal_shock_states
from train_h2r2_bgk import H2R2Model

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--restart", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=4500)
    parser.add_argument("--nx-pde", type=int, default=129)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--projection-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    restart = Path(args.restart)
    if not restart.is_file():
        raise FileNotFoundError(f"F1 restart not found: {restart}")

    # The reference is audited up front for provenance, but its values enter
    # only the post-training cross-model report below.
    reference = load_reference(args.reference, expected_mach=args.mach)
    upstream, downstream = normal_shock_states(args.mach)
    invariant_flux = np.asarray(analytic_fluxes(upstream), np.float32)
    velocity_scale = float(upstream.u)

    vx_1d, s_1d, weight_grid = structured_axisymmetric_quadrature(args.reference)
    nvx, ns = len(vx_1d), len(s_1d)
    vx_np = np.repeat(vx_1d, ns).astype(np.float32)
    s_np = np.tile(s_1d, nvx).astype(np.float32)
    weight_np = weight_grid.reshape(-1).astype(np.float32)
    vx = tf.constant(vx_np, DTYPE)
    s = tf.constant(s_np, DTYPE)
    weights = tf.constant(weight_np, DTYPE)
    speed2 = vx*vx+s
    basis5 = tf.stack((tf.ones_like(vx), vx/velocity_scale,
                       speed2/velocity_scale**2, vx*vx/velocity_scale**2,
                       0.5*vx*speed2/velocity_scale**3), axis=1)
    flux_target = tf.constant(invariant_flux, DTYPE)

    model = H2R2Model(upstream, downstream, invariant_flux, args.width, args.depth)
    model(tf.constant([-40.0, 0.0, 40.0], DTYPE), training=False)
    model.load_weights(restart)
    initial_weights = [tf.constant(variable.numpy()) for variable in model.trainable_variables]

    def projection_targets(fields):
        rho, u = fields["rho"], fields["u"]
        temperature, sigma = fields["temperature"], fields["sigma_xx"]
        return tf.stack((rho, rho*u/velocity_scale,
                         rho*(u*u+3*temperature)/velocity_scale**2,
                         (rho*u*u+rho*temperature+sigma)/velocity_scale**2,
                         tf.fill(tf.shape(rho), flux_target[2]/velocity_scale**3)), axis=1)

    def project_five(log_raw, target):
        beta = tf.zeros((tf.shape(log_raw)[0], 5), DTYPE)
        eye = tf.eye(5, dtype=DTYPE)[None, :, :]
        base = tf.clip_by_value(log_raw, LOG_FLOOR+LOG_TILT_LIMIT, LOG_CEILING)
        for _ in range(args.projection_steps):
            tilt = tf.clip_by_value(tf.einsum("xi,vi->xv", beta, basis5),
                                    -LOG_TILT_LIMIT, LOG_TILT_LIMIT)
            distribution = tf.exp(tf.clip_by_value(base+tilt, LOG_FLOOR, LOG_CEILING))
            weighted = distribution*weights[None, :]
            current = tf.einsum("xv,vi->xi", weighted, basis5)
            jacobian = tf.einsum("xv,vi,vj->xij", weighted, basis5, basis5)
            delta = tf.linalg.solve(jacobian+2.0e-7*eye,
                                    (target-current)[..., None])[..., 0]
            beta += 0.72*tf.clip_by_value(delta, -0.7, 0.7)
        tilt = tf.clip_by_value(tf.einsum("xi,vi->xv", beta, basis5),
                                -LOG_TILT_LIMIT, LOG_TILT_LIMIT)
        return tf.exp(tf.clip_by_value(base+tilt, LOG_FLOOR, LOG_CEILING)), beta

    def represented_log(v, radial2, fields, heat, stress, beta):
        rho, u, temperature = fields["rho"], fields["u"], fields["temperature"]
        cx = (v-u[:, None])/tf.sqrt(temperature[:, None])
        c2 = cx*cx+radial2/temperature[:, None]
        heat_mode = ((0.5*c2-2.5)*cx)/8.0
        stress_mode = (cx*cx-c2/3.0)/4.0
        correction = PSI_MAX*tf.tanh(
            (heat[:, None]*heat_mode+stress[:, None]*stress_mode)/PSI_MAX)
        log_maxwellian = (tf.math.log(rho[:, None])
                          -1.5*tf.math.log(2*np.pi*temperature[:, None])
                          -0.5*((v-u[:, None])**2+radial2)/temperature[:, None])
        local_basis = tf.stack((tf.ones_like(v), v/velocity_scale,
                                (v*v+radial2)/velocity_scale**2,
                                v*v/velocity_scale**2,
                                0.5*v*(v*v+radial2)/velocity_scale**3), axis=2)
        base = tf.clip_by_value(log_maxwellian+correction,
                                LOG_FLOOR+LOG_TILT_LIMIT, LOG_CEILING)
        tilt = tf.clip_by_value(tf.einsum("xi,xvi->xv", beta, local_basis),
                                -LOG_TILT_LIMIT, LOG_TILT_LIMIT)
        return tf.clip_by_value(base+tilt, LOG_FLOOR, LOG_CEILING)

    def distribution(y, training, return_state=False):
        x = 80.0*y
        fields = model.macro_fields(x)
        heat, stress = model.micro_coefficients(x, training)
        rho, u, temperature = fields["rho"], fields["u"], fields["temperature"]
        cx = (vx[None, :]-u[:, None])/tf.sqrt(temperature[:, None])
        c2 = cx*cx+s[None, :]/temperature[:, None]
        heat_mode = ((0.5*c2-2.5)*cx)/8.0
        stress_mode = (cx*cx-c2/3.0)/4.0
        correction = PSI_MAX*tf.tanh(
            (heat[:, None]*heat_mode+stress[:, None]*stress_mode)/PSI_MAX)
        log_maxwellian = (tf.math.log(rho[:, None])
                          -1.5*tf.math.log(2*np.pi*temperature[:, None])
                          -0.5*((vx[None, :]-u[:, None])**2+s[None, :])
                          /temperature[:, None])
        f, beta = project_five(log_maxwellian+correction, projection_targets(fields))
        if return_state:
            return f, (fields, heat, stress, beta)
        return f

    def moments(f):
        weighted = f*weights[None, :]
        rho = tf.reduce_sum(weighted, axis=1)
        momentum = tf.reduce_sum(weighted*vx[None, :], axis=1)
        u = momentum/rho
        cx = vx[None, :]-u[:, None]
        c2 = cx*cx+s[None, :]
        temperature = tf.reduce_sum(weighted*c2, axis=1)/(3.0*rho)
        flux = tf.stack((momentum,
                         tf.reduce_sum(weighted*vx[None, :]**2, axis=1),
                         0.5*tf.reduce_sum(weighted*vx[None, :]*speed2[None, :], axis=1)), axis=1)
        return {"rho": rho, "u": u, "temperature": temperature,
                "qx": 0.5*tf.reduce_sum(weighted*cx*c2, axis=1),
                "sigma_xx": tf.reduce_sum(weighted*(cx*cx-c2/3.0), axis=1),
                "flux": flux}

    velocity_weight = weights*(1.0+0.1*speed2*speed2)
    velocity_weight /= tf.reduce_sum(velocity_weight)

    def objective(y, training):
        with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as accumulator:
            f, state = distribution(y, training, return_state=True)
        dfd_y = accumulator.jvp(f)
        values = moments(f)
        fields, heat, stress, beta = state
        collision, invariant_error, correction_error = analytic_dougherty_collision_tf(
            lambda local_v, local_s: represented_log(
                local_v, local_s, fields, heat, stress, beta),
            f, values, vx, s, weights)
        rhs = collision/KNUDSEN_EFFECTIVE
        residual = vx[None, :]*dfd_y-rhs
        scale = tf.sqrt(tf.reduce_mean(tf.square(rhs)))+1.0e-8
        absolute = tf.reduce_mean(tf.reduce_sum(
            tf.square(residual/scale)*velocity_weight[None, :], axis=1))
        relative = tf.reduce_mean(tf.reduce_sum(
            tf.square(residual/(tf.abs(vx[None, :]*dfd_y)+tf.abs(rhs)+1.0e-7))
            *velocity_weight[None, :], axis=1))
        projection = tf.reduce_mean(tf.square(
            (values["flux"]-flux_target[None, :])/tf.abs(flux_target[None, :])))
        center = model.macro_fields(tf.constant([0.0], DTYPE))["rho"][0]
        midpoint = 0.5*(upstream.rho+downstream.rho)
        gauge = tf.square((center-midpoint)/(downstream.rho-upstream.rho))
        trust = tf.add_n([
            tf.reduce_mean(tf.square((variable-start)/(1.0+tf.abs(start))))
            for variable, start in zip(model.trainable_variables, initial_weights)])
        loss = (absolute+0.01*relative+100.0*projection+10.0*gauge
                +0.1*correction_error+1.0e-6*trust)
        return (loss, absolute, relative, projection, invariant_error,
                correction_error, gauge, residual, values)

    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate, args.epochs, alpha=0.08))

    @tf.function
    def training_step(epoch):
        interior = args.nx_pde-2
        local_count = int(round(0.65*interior))
        seed = tf.stack((tf.cast(args.seed, tf.int32), epoch))
        local = tf.random.stateless_normal((local_count,), seed, stddev=6.0/80.0)
        broad = tf.random.stateless_uniform(
            (interior-local_count,), seed+tf.constant([31, 73]), minval=-0.5, maxval=0.5)
        y = tf.concat((tf.constant([-0.5]), tf.sort(tf.clip_by_value(
            tf.concat((local, broad), axis=0), -0.499, 0.499)), tf.constant([0.5])), axis=0)
        with tf.GradientTape() as tape:
            result = objective(y, True)
        gradients = tape.gradient(result[0], model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return result

    history = []
    best_loss = np.inf
    best_weights = None
    started = time.time()
    for epoch in range(1, args.epochs+1):
        result = training_step(tf.constant(epoch, tf.int32))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            row = [epoch]+[float(value) for value in result[:7]]
            history.append(row)
            if row[1] < best_loss:
                best_loss, best_weights = row[1], model.get_weights()
            print("f1 epoch=%d loss=%.6e pde=%.3e rel=%.3e flux=%.3e inv=%.3e corr=%.3e gauge=%.3e"
                  % tuple(row), flush=True)
    if best_weights is not None:
        model.set_weights(best_weights)

    audit = objective(tf.linspace(-0.5, 0.5, 321), False)
    held = np.arange(1, len(reference.x)-1)
    regions = validation_regions(reference.x, held)

    def predict_all(chunk=32):
        result = {key: [] for key in ("rho", "u", "temperature", "qx", "sigma_xx", "flux")}
        minimum = np.inf
        y = (reference.x/80.0).astype(np.float32)
        for lower in range(0, len(y), chunk):
            f = distribution(tf.constant(y[lower:lower+chunk]), False)
            minimum = min(minimum, float(tf.reduce_min(f)))
            values = moments(f)
            for key in result:
                result[key].append(values[key].numpy())
        return {key: np.concatenate(value, axis=0) for key, value in result.items()}, minimum

    predicted, minimum_distribution = predict_all()
    cross_core = heldout_metrics(predicted, reference, regions["held_out_core"])
    cross_full = heldout_metrics(predicted, reference, regions["held_out_full"])
    flux_spreads = {name: float(np.ptp(predicted["flux"][:, i])
                                      /max(abs(np.mean(predicted["flux"][:, i])), 1.0e-12))
                    for i, name in enumerate(("mass", "momentum", "energy"))}
    residual_rms = float(audit[1])**0.5
    invariant_rms = float(audit[4])
    correction_rms = float(audit[5])
    boundary = float(max(
        abs(predicted["rho"][0]/upstream.rho-1.0),
        abs(predicted["rho"][-1]/downstream.rho-1.0),
        abs(predicted["u"][0]/upstream.u-1.0),
        abs(predicted["u"][-1]/downstream.u-1.0),
        abs(predicted["temperature"][0]/upstream.temperature-1.0),
        abs(predicted["temperature"][-1]/downstream.temperature-1.0)))
    gates = {
        "finite": bool(all(np.isfinite(value).all() for value in predicted.values())),
        "positive": bool(np.isfinite(minimum_distribution) and minimum_distribution > 0.0),
        "boundary": bool(boundary < F1_GATES["boundary_relative_error"]),
        "fluxes": bool(max(flux_spreads.values()) < F1_GATES["maximum_flux_relative_spread"]),
        "residual": bool(residual_rms < F1_GATES["relative_residual_rms"]),
        "collision_invariants": bool(invariant_rms < F1_GATES["collision_invariant_relative_rms"]),
        "collision_projection": bool(correction_rms < F1_GATES["collision_projection_relative_rms"]),
    }
    status = "F1_PILOT_PASS" if all(gates.values()) else "NO_GO"
    metrics = {
        "stage": "F1_CONSERVATIVE_DOUGHERTY_FP_OPERATOR_SWITCH", "status": status,
        "mach": args.mach, "effective_knudsen": KNUDSEN_EFFECTIVE,
        "restart": str(restart), "reference_sha256": reference.metadata["sha256"],
        "reference_role": "post-training BGK cross-model comparison only; absent from F1 loss",
        "collision": "nonlinear local-moment Dougherty--Fokker--Planck",
        "collision_discretization": "axisymmetric divergence in (vx,r^2) plus invariant-nullspace projection",
        "velocity_grid": {"nvx": nvx, "nr2": ns, "points": nvx*ns},
        "cross_model_bgk_errors_core_relative_l2": cross_core,
        "cross_model_bgk_errors_full_relative_l2": cross_full,
        "flux_relative_spreads": flux_spreads,
        "relative_residual_rms": residual_rms,
        "collision_invariant_relative_rms": invariant_rms,
        "collision_projection_relative_rms": correction_rms,
        "boundary_relative_error": boundary,
        "minimum_distribution": minimum_distribution,
        "epochs": args.epochs, "elapsed_seconds": time.time()-started,
        "gates_preregistered": F1_GATES, "gates": gates,
        "claim": "Operator-switch continuation pilot, not same-operator FP validation."
    }
    (output/"f1_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    np.savetxt(output/"f1_profiles.csv", np.column_stack((
        reference.x, reference.rho, predicted["rho"], reference.u, predicted["u"],
        reference.temperature, predicted["temperature"], reference.qx, predicted["qx"],
        reference.sigma_xx, predicted["sigma_xx"], predicted["flux"])),
        delimiter=",", comments="", header=(
            "x_mfp,rho_bgk_dvm,rho_fp,u_bgk_dvm,u_fp,T_bgk_dvm,T_fp,"
            "qx_bgk_dvm,qx_fp,sigma_bgk_dvm,sigma_fp,mass_flux,momentum_flux,energy_flux"))
    np.savetxt(output/"f1_history.csv", np.asarray(history), delimiter=",", comments="",
               header="epoch,total,pde_absolute,pde_relative,projection_flux,collision_invariant,collision_projection,gauge")
    model.save_weights(output/"f1.weights.h5")

    figure, axes = plt.subplots(2, 3, figsize=(15.2, 9.2))
    figure.subplots_adjust(top=0.72, hspace=0.82, wspace=0.30)
    figure.suptitle("F1 conservative Dougherty–FP shock — BGK cross-model audit",
                    y=0.985, fontsize=18)
    panels = [("rho", reference.rho, r"Density $\rho$"),
              ("u", reference.u, r"Velocity $u_x$"),
              ("temperature", reference.temperature, r"Temperature $T$"),
              ("qx", reference.qx, r"Heat flux $q_x$"),
              ("sigma_xx", reference.sigma_xx, r"Normal stress $\sigma_{xx}$")]
    for axis, (key, bgk, title) in zip(axes.flat[:5], panels):
        axis.plot(reference.x, bgk, color="black", lw=2.1, label="BGK DVM control")
        axis.plot(reference.x, predicted[key], color="#377eb8", lw=1.9, ls="--",
                  label="F1 Dougherty–FP PINN")
        axis.set_title(title, pad=58)
        axis.set_xlabel(r"$x/\lambda_1$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
    history_array = np.asarray(history)
    axis = axes.flat[5]
    axis.semilogy(history_array[:, 0], np.maximum(history_array[:, 2], 1.0e-14),
                  label="FP residual")
    axis.semilogy(history_array[:, 0], np.maximum(history_array[:, 6], 1.0e-14),
                  label="conservative correction")
    axis.set_title("Operator-switch optimization", pad=58)
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
    figure.savefig(output/"f1_fp_physics.png", dpi=240, bbox_inches="tight")
    plt.close(figure)

    print("F1_METRICS", json.dumps(metrics, sort_keys=True), flush=True)
    print(f"F1_OUTPUT={output}", flush=True)
    return 0 if status == "F1_PILOT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
