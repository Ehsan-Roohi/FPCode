#!/usr/bin/env python3
"""H2R2 structure-preserving stationary Mach-2 BGK control problem."""
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

from h2_bgk import H2_GATES, KNUDSEN_EFFECTIVE, PSI_MAX, compact_quadrature_arrays
from h2_reference import heldout_metrics, load_reference, validation_regions
from shock_physics import analytic_fluxes, normal_shock_states

tf.keras.backend.set_floatx("float32")
DTYPE = tf.float32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mach", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--macro-epochs", type=int, default=1400)
    p.add_argument("--nx-pde", type=int, default=193)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--projection-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--print-every", type=int, default=100)
    return p.parse_args()


def nearest_indices(x, stations):
    x = np.asarray(x)
    return np.unique([int(np.argmin(np.abs(x-s))) for s in stations])


class MonotoneSwitch(tf.keras.layers.Layer):
    """Positive logistic mixture normalized at the finite boundaries."""
    def __init__(self, centers):
        super().__init__()
        self.centers = tf.constant(np.asarray(centers, np.float32), DTYPE)
        n = len(centers)
        self.logits = self.add_weight(name="logits", shape=(n,), initializer="zeros")
        initial = np.log(np.expm1(1.5))
        self.log_widths = self.add_weight(
            name="log_widths", shape=(n,),
            initializer=tf.keras.initializers.Constant(initial))

    def call(self, x):
        widths = 0.3 + tf.nn.softplus(self.log_widths)
        weights = tf.nn.softmax(self.logits)
        def mixture(z):
            return tf.reduce_sum(weights[None, :]*tf.sigmoid(
                (z[:, None]-self.centers[None, :])/widths[None, :]), axis=1)
        raw = mixture(x)
        left = mixture(tf.constant([-40.0], DTYPE))[0]
        right = mixture(tf.constant([40.0], DTYPE))[0]
        return tf.clip_by_value((raw-left)/(right-left), 0.0, 1.0)


class H2R2Model(tf.keras.Model):
    def __init__(self, upstream, downstream, fluxes, width, depth):
        super().__init__()
        centers = np.linspace(-12.0, 12.0, 25)
        self.rho_switch = MonotoneSwitch(centers)
        self.temperature_switch = MonotoneSwitch(centers)
        self.micro = tf.keras.Sequential(
            [tf.keras.layers.Input((3,))]
            + [tf.keras.layers.Dense(width, activation=tf.nn.silu) for _ in range(depth)]
            + [tf.keras.layers.Dense(2, kernel_initializer="zeros")])
        self.upstream, self.downstream = upstream, downstream
        self.fluxes = tf.constant(fluxes, DTYPE)

    def macro_fields(self, x):
        sr, st = self.rho_switch(x), self.temperature_switch(x)
        rho = self.upstream.rho + (self.downstream.rho-self.upstream.rho)*sr
        temperature = (self.upstream.temperature
                       +(self.downstream.temperature-self.upstream.temperature)*st)
        mass, momentum, energy = tf.unstack(self.fluxes)
        u = mass/rho
        sigma = momentum-rho*u*u-rho*temperature
        qx = energy-u*(0.5*rho*u*u+2.5*rho*temperature+sigma)
        return {"rho": rho, "u": u, "temperature": temperature,
                "qx": qx, "sigma_xx": sigma}

    def micro_coefficients(self, x, training):
        features = tf.stack((x/12.0, tf.tanh(x/3.0), tf.exp(-tf.square(x/8.0))), 1)
        out = self.micro(features, training=training)
        gate = tf.exp(-tf.pow(x/13.0, 4.0))
        return 0.55*gate*tf.tanh(out[:, 0]), 0.55*gate*tf.tanh(out[:, 1])

    def call(self, x, training=False):
        fields = self.macro_fields(x)
        aq, astress = self.micro_coefficients(x, training)
        return tf.stack((fields["rho"], fields["u"], fields["temperature"],
                         fields["qx"], fields["sigma_xx"], aq, astress), axis=1)


def main():
    a = parse_args()
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = load_reference(a.reference, expected_mach=a.mach)
    up, down = normal_shock_states(a.mach)
    flux_np = np.asarray(analytic_fluxes(up), np.float32)
    us = float(up.u)
    vx_np, r2_np, w_np = compact_quadrature_arrays(a.reference)
    vx, r2, weights = map(lambda z: tf.constant(z, DTYPE), (vx_np, r2_np, w_np))
    v2 = vx*vx+r2
    b5 = tf.stack((tf.ones_like(vx), vx/us, v2/us**2, vx*vx/us**2,
                   0.5*vx*v2/us**3), 1)
    b3 = tf.stack((tf.ones_like(vx), vx, v2), 1)
    flux_target = tf.constant(flux_np, DTYPE)

    anchors = nearest_indices(ref.x, np.linspace(-12.0, 12.0, 17))
    held = np.setdiff1d(np.arange(1, len(ref.x)-1), anchors)
    split = {"macro": anchors, "moments": anchors, "held_out": held}
    regions = validation_regions(ref.x, held)
    xa = tf.constant(ref.x[anchors], DTYPE)
    targets = {k: tf.constant(getattr(ref, k)[anchors], DTYPE)
               for k in ("rho", "u", "temperature", "qx", "sigma_xx")}
    scales = {k: tf.constant(float(np.max(np.abs(getattr(ref, k))))+1e-7, DTYPE)
              for k in targets}

    model = H2R2Model(up, down, flux_np, a.width, a.depth)
    model(tf.constant([-40.0, 0.0, 40.0], DTYPE), training=False)

    def macro_loss():
        pred = model.macro_fields(xa)
        term = {k: tf.reduce_mean(tf.square((pred[k]-targets[k])/scales[k]))
                for k in targets}
        total = (term["rho"]+3*term["u"]+term["temperature"]
                 +2*term["qx"]+1.5*term["sigma_xx"])
        return total, term

    macro_opt = tf.keras.optimizers.Adam(8e-3)

    @tf.function
    def macro_step():
        variables = (model.rho_switch.trainable_variables
                     +model.temperature_switch.trainable_variables)
        with tf.GradientTape() as tape:
            total, term = macro_loss()
        grad = tape.gradient(total, variables)
        grad, _ = tf.clip_by_global_norm(grad, 5.0)
        macro_opt.apply_gradients(zip(grad, variables))
        return total, term

    for epoch in range(1, a.macro_epochs+1):
        total, term = macro_step()
        if epoch == 1 or epoch % a.print_every == 0 or epoch == a.macro_epochs:
            print("h2r2 macro epoch=%d loss=%.6e rho=%.3e u=%.3e T=%.3e q=%.3e sigma=%.3e"
                  % (epoch, float(total), float(term["rho"]), float(term["u"]),
                     float(term["temperature"]), float(term["qx"]),
                     float(term["sigma_xx"])), flush=True)
    model.rho_switch.trainable = False
    model.temperature_switch.trainable = False

    def projection_targets(fields):
        rho, u = fields["rho"], fields["u"]
        t, sigma = fields["temperature"], fields["sigma_xx"]
        return tf.stack((rho, rho*u/us, rho*(u*u+3*t)/us**2,
                         (rho*u*u+rho*t+sigma)/us**2,
                         tf.fill(tf.shape(rho), flux_target[2]/us**3)), 1)

    def project_five(log_raw, target):
        beta = tf.zeros((tf.shape(log_raw)[0], 5), DTYPE)
        eye = tf.eye(5, dtype=DTYPE)[None, :, :]
        for _ in range(a.projection_steps):
            expo = tf.einsum("xi,vi->xv", beta, b5)
            # Combine both exponentials before clipping.  Multiplying an
            # already tiny Maxwellian by the tilt can flush float32 subnormal
            # tail values to zero on a GPU even though the ansatz is positive.
            f = tf.exp(tf.clip_by_value(log_raw+expo, -80.0, 25.0))
            wf = f*weights[None, :]
            current = tf.einsum("xv,vi->xi", wf, b5)
            jac = tf.einsum("xv,vi,vj->xij", wf, b5, b5)
            delta = tf.linalg.solve(jac+2e-7*eye, (target-current)[..., None])[..., 0]
            beta += 0.72*tf.clip_by_value(delta, -0.7, 0.7)
        expo = tf.einsum("xi,vi->xv", beta, b5)
        return tf.exp(tf.clip_by_value(log_raw+expo, -80.0, 25.0))

    def distribution(y, training):
        x = 80*y
        fields = model.macro_fields(x)
        aq, ast = model.micro_coefficients(x, training)
        rho, u, t = fields["rho"], fields["u"], fields["temperature"]
        cx = (vx[None, :]-u[:, None])/tf.sqrt(t[:, None])
        c2 = cx*cx+r2[None, :]/t[:, None]
        pq = ((0.5*c2-2.5)*cx)/8.0
        ps = (cx*cx-c2/3.0)/4.0
        psi = PSI_MAX*tf.tanh((aq[:, None]*pq+ast[:, None]*ps)/PSI_MAX)
        logm = (tf.math.log(rho[:, None])-1.5*tf.math.log(2*np.pi*t[:, None])
                -0.5*((vx[None, :]-u[:, None])**2+r2[None, :])/t[:, None])
        return project_five(logm+psi, projection_targets(fields))

    def moments(f):
        wf = f*weights[None, :]
        rho = tf.reduce_sum(wf, 1)
        momentum = tf.reduce_sum(wf*vx[None, :], 1)
        u = momentum/rho
        cx = vx[None, :]-u[:, None]
        c2 = cx*cx+r2[None, :]
        t = tf.reduce_sum(wf*c2, 1)/(3*rho)
        return {"rho": rho, "u": u, "temperature": t,
                "qx": 0.5*tf.reduce_sum(wf*cx*c2, 1),
                "sigma_xx": tf.reduce_sum(wf*(cx*cx-c2/3), 1),
                "flux": tf.stack((momentum, tf.reduce_sum(wf*vx[None, :]**2, 1),
                                  0.5*tf.reduce_sum(wf*vx[None, :]*v2[None, :], 1)), 1)}

    def maxwellian(m):
        rho, u, t = m["rho"], m["u"], m["temperature"]
        theta = tf.stack((tf.math.log(rho)-1.5*tf.math.log(2*np.pi*t)-u*u/(2*t),
                          u/t, -0.5/t), 1)
        target = tf.stack((rho, rho*u, rho*(u*u+3*t)), 1)
        for _ in range(3):
            md = tf.exp(tf.clip_by_value(tf.einsum("xi,vi->xv", theta, b3), -80.0, 25.0))
            wm = md*weights[None, :]
            cur = tf.einsum("xv,vi->xi", wm, b3)
            jac = tf.einsum("xv,vi,vj->xij", wm, b3, b3)
            delta = tf.linalg.solve(jac+1e-8*tf.eye(3, dtype=DTYPE)[None, :, :],
                                    (target-cur)[..., None])[..., 0]
            theta += 0.8*tf.clip_by_value(delta, -0.5, 0.5)
        return tf.exp(tf.clip_by_value(tf.einsum("xi,vi->xv", theta, b3), -80.0, 25.0))

    vw = weights*(1+0.1*v2*v2)
    vw /= tf.reduce_sum(vw)

    def objective(y, training):
        with tf.autodiff.ForwardAccumulator(y, tf.ones_like(y)) as acc:
            f = distribution(y, training)
        dfdx = acc.jvp(f)
        m = moments(f)
        md = maxwellian(m)
        residual = vx[None, :]*dfdx-(md-f)/KNUDSEN_EFFECTIVE
        scale = tf.sqrt(tf.reduce_mean(tf.square((md-f)/KNUDSEN_EFFECTIVE)))+1e-8
        absolute = tf.reduce_mean(tf.reduce_sum(tf.square(residual/scale)*vw[None, :], 1))
        relative = tf.reduce_mean(tf.reduce_sum(
            tf.square(residual/(tf.abs(f)+tf.abs(md)+1e-7))*vw[None, :], 1))
        projection = tf.reduce_mean(tf.square(
            (m["flux"]-flux_target[None, :])/tf.abs(flux_target[None, :])))
        reg = tf.add_n([tf.reduce_mean(tf.square(v)) for v in model.micro.trainable_variables])
        return absolute+0.01*relative+200*projection+1e-7*reg, absolute, relative, projection, residual, m

    micro_epochs = max(a.epochs-a.macro_epochs, 1)
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(
        a.learning_rate, micro_epochs, alpha=0.08))

    @tf.function
    def micro_step(epoch):
        n = a.nx_pde-2
        nl = int(round(0.65*n))
        seed = tf.stack((tf.cast(a.seed, tf.int32), epoch))
        local = tf.random.stateless_normal((nl,), seed, stddev=6/80, dtype=DTYPE)
        broad = tf.random.stateless_uniform((n-nl,), seed+tf.constant([31, 73]),
                                            minval=-0.5, maxval=0.5, dtype=DTYPE)
        y = tf.concat((tf.constant([-0.5], DTYPE), tf.sort(tf.clip_by_value(
            tf.concat((local, broad), 0), -0.499, 0.499)), tf.constant([0.5], DTYPE)), 0)
        with tf.GradientTape() as tape:
            values = objective(y, True)
        grad = tape.gradient(values[0], model.micro.trainable_variables)
        grad, _ = tf.clip_by_global_norm(grad, 5.0)
        opt.apply_gradients(zip(grad, model.micro.trainable_variables))
        return values

    history, best, best_weights = [], np.inf, None
    started = time.time()
    for epoch in range(1, micro_epochs+1):
        values = micro_step(tf.constant(epoch, tf.int32))
        if epoch == 1 or epoch % a.print_every == 0 or epoch == micro_epochs:
            row = [epoch]+[float(v) for v in values[:4]]
            history.append(row)
            if row[1] < best:
                best, best_weights = row[1], model.micro.get_weights()
            print("h2r2 micro epoch=%d loss=%.6e pde=%.3e rel=%.3e projection=%.3e"
                  % tuple(row), flush=True)
    if best_weights is not None:
        model.micro.set_weights(best_weights)

    final = objective(tf.linspace(tf.constant(-0.5, DTYPE), tf.constant(0.5, DTYPE), 641), False)

    def predict_all(chunk=32):
        result = {k: [] for k in ("rho", "u", "temperature", "qx", "sigma_xx", "flux")}
        minimum = np.inf
        y = (ref.x/80).astype(np.float32)
        for lo in range(0, len(y), chunk):
            f = distribution(tf.constant(y[lo:lo+chunk], DTYPE), False)
            minimum = min(minimum, float(tf.reduce_min(f)))
            m = moments(f)
            for k in result:
                result[k].append(m[k].numpy())
        return {k: np.concatenate(v, 0) for k, v in result.items()}, minimum

    pred, fmin = predict_all()
    core = heldout_metrics(pred, ref, regions["held_out_core"])
    full = heldout_metrics(pred, ref, regions["held_out_full"])
    left = heldout_metrics(pred, ref, regions["left_tail"])
    right = heldout_metrics(pred, ref, regions["right_tail"])
    spreads = {k: float(np.ptp(pred["flux"][:, i])/max(abs(np.mean(pred["flux"][:, i])), 1e-12))
               for i, k in enumerate(("mass", "momentum", "energy"))}
    residual_rms = float(final[1])**0.5
    projection_rms = float(final[3])**0.5
    boundary = float(max(abs(pred["rho"][0]/up.rho-1), abs(pred["rho"][-1]/down.rho-1),
                         abs(pred["u"][0]/up.u-1), abs(pred["u"][-1]/down.u-1),
                         abs(pred["temperature"][0]/up.temperature-1),
                         abs(pred["temperature"][-1]/down.temperature-1)))
    gates = {"finite": bool(all(np.isfinite(v).all() for v in pred.values())),
             "positive": bool(np.isfinite(fmin) and fmin > 0),
             "boundary": bool(boundary < H2_GATES["boundary_relative_error"]),
             "rho_core": bool(core["rho"] < H2_GATES["rho_core_relative_l2"]),
             "u_core": bool(core["u"] < H2_GATES["u_core_relative_l2"]),
             "temperature_core": bool(core["temperature"] < H2_GATES["temperature_core_relative_l2"]),
             "qx_core": bool(core["qx"] < H2_GATES["qx_core_relative_l2"]),
             "sigma_xx_core": bool(core["sigma_xx"] < H2_GATES["sigma_xx_core_relative_l2"]),
             "fluxes": bool(max(spreads.values()) < H2_GATES["maximum_flux_relative_spread"]),
             "residual": bool(residual_rms < H2_GATES["relative_residual_rms"])}
    status = "PILOT_PASS" if all(gates.values()) else "NO_GO"
    metrics = {"stage": "H2R2_STRUCTURE_PRESERVING_BGK_CONTROL", "status": status,
               "mach": a.mach, "effective_knudsen": KNUDSEN_EFFECTIVE,
               "reference_sha256": ref.metadata["sha256"],
               "quadrature_points_full": int(ref.metadata["nv"]),
               "quadrature_points_axisymmetric": int(len(vx_np)),
               "pde_collocation_points_per_epoch": int(a.nx_pde),
               "pde_collocation": "65% shock-local normal plus 35% global uniform; AD residual",
               "projection": "positive five-raw-moment exponential Newton tilt",
               "projection_steps": a.projection_steps,
               "projection_flux_relative_rms": projection_rms,
               "residual_audit_points": 641, "macro_lock_points": int(len(anchors)),
               "moment_anchor_points": int(len(anchors)),
               "held_out_points": int(len(regions["held_out_full"])),
               "held_out_core_points": int(len(regions["held_out_core"])),
               "errors_core_relative_l2": core, "errors_full_relative_l2": full,
               "errors_left_tail_relative_l2": left, "errors_right_tail_relative_l2": right,
               "flux_relative_spreads": spreads, "relative_residual_rms": residual_rms,
               "boundary_relative_error": boundary, "minimum_distribution": fmin,
               "epochs": a.epochs, "macro_epochs": a.macro_epochs,
               "micro_epochs": micro_epochs, "elapsed_seconds": time.time()-started,
               "gates_preregistered": H2_GATES, "gates": gates,
               "claim": "Structure-preserving BGK control with dense DVM profiles held out; "
                        "the Fokker--Planck claim is a separate next-stage comparison."}
    (outdir/"h2_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    np.savetxt(outdir/"h2_profiles.csv", np.column_stack((
        ref.x, ref.rho, pred["rho"], ref.u, pred["u"], ref.temperature,
        pred["temperature"], ref.qx, pred["qx"], ref.sigma_xx, pred["sigma_xx"], pred["flux"])),
        delimiter=",", comments="", header="x_mfp,rho_dvm,rho_pinn,u_dvm,u_pinn,T_dvm,T_pinn,qx_dvm,qx_pinn,sigma_dvm,sigma_pinn,mass_flux,momentum_flux,energy_flux")
    np.savetxt(outdir/"h2_history.csv", np.asarray(history), delimiter=",", comments="",
               header="micro_epoch,total,pde_absolute,pde_relative,projection_flux")
    np.savez(outdir/"h2_split.npz", **split, **regions)
    model.save_weights(outdir/"h2r2.weights.h5")

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2))
    fig.subplots_adjust(top=0.72, hspace=0.82, wspace=0.30)
    fig.suptitle("H2R2 structure-preserving BGK shock — independent DVM audit", y=0.985, fontsize=18)
    panels = [("rho", ref.rho, r"Density $\rho$"), ("u", ref.u, r"Velocity $u_x$"),
              ("temperature", ref.temperature, r"Temperature $T$"),
              ("qx", ref.qx, r"Heat flux $q_x$"),
              ("sigma_xx", ref.sigma_xx, r"Normal stress $\sigma_{xx}$")]
    for ax, (key, truth, title) in zip(axes.flat[:5], panels):
        ax.plot(ref.x, truth, color="black", lw=2.1, label="DVM reference")
        ax.plot(ref.x, pred[key], color="#d95f02", lw=1.8, ls="--", label="H2R2 BGK–PINN")
        ax.scatter(ref.x[anchors], truth[anchors], s=16, color="#1b9e77", zorder=3,
                   label="17 sparse anchors")
        ax.set_title(title, pad=58)
        ax.set_xlabel(r"$x/\lambda_1$")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
    hist = np.asarray(history)
    ax = axes.flat[5]
    ax.semilogy(hist[:, 0], np.maximum(hist[:, 2], 1e-14), label="BGK residual")
    ax.semilogy(hist[:, 0], np.maximum(hist[:, 4], 1e-14), label="projection error")
    ax.set_title("Microscopic physics optimization", pad=58)
    ax.set_xlabel("micro epoch")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2)
    fig.savefig(outdir/"h2_bgk_physics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print("H2_METRICS", json.dumps(metrics, sort_keys=True), flush=True)
    print(f"H2_OUTPUT={outdir}", flush=True)
    return 0 if status == "PILOT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
