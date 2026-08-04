#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rebuild a COMPLETE 58generalcavity.py from the working 77CavityUQL2.py.

The current 58generalcavity.py is truncated and starts mid-line, so it cannot
be patched. This script ignores the broken 58 and rebuilds it from the working
77CavityUQL2.py solver functions.
"""

from pathlib import Path
import shutil

src = Path("77CavityUQL2.py")
dst = Path("58generalcavity.py")

if not src.exists():
    raise SystemExit("ERROR: 77CavityUQL2.py not found. Put this script in the same folder as 77CavityUQL2.py.")

if dst.exists():
    backup = Path("58generalcavity_broken_or_old_backup.py")
    if not backup.exists():
        shutil.copy2(dst, backup)
        print(f"Backup of old 58 created: {backup}")

text = src.read_text(encoding="utf-8")

marker = "# ============================================================================\n# 6. Main Execution"
if marker in text:
    pre = text.split(marker)[0]
else:
    idx = text.find("\ndef main():")
    if idx < 0:
        raise SystemExit("ERROR: Could not find main() in 77CavityUQL2.py.")
    pre = text[:idx]

if "import os" not in pre:
    pre = pre.replace("import time\n", "import time\nimport os\n", 1)

new_main = """
# ============================================================================
# 6. Main Execution: physics data generation for high-U cavity training
# ============================================================================

def _collect_training_snapshot(grid_gpu, coeffs_gpu):
    \"\"\"Return X(16) and y(9) for all cells at the current time step.\"\"\"
    X_gpu = cp.column_stack([
        grid_gpu['rho'],
        grid_gpu['T'],
        grid_gpu['U'][:, 0],
        grid_gpu['U'][:, 1],
        grid_gpu['U'][:, 2],
        grid_gpu['PIJ'][:, 0],
        grid_gpu['PIJ'][:, 1],
        grid_gpu['PIJ'][:, 2],
        grid_gpu['PIJ'][:, 3],
        grid_gpu['PIJ'][:, 4],
        grid_gpu['PIJ'][:, 5],
        grid_gpu['Q'][:, 0],
        grid_gpu['Q'][:, 1],
        grid_gpu['Q'][:, 2],
        grid_gpu['DM2'],
        grid_gpu['nu'],
    ])

    y_gpu = cp.concatenate([coeffs_gpu['A'], coeffs_gpu['B']], axis=1)

    X = cp.asnumpy(X_gpu).astype(np.float32, copy=False)
    y = cp.asnumpy(y_gpu).astype(np.float32, copy=False)

    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
    return X[mask], y[mask]


def main():
    global UW_LID, DT, N_STEPS_PER_RUN, NTSS, PARTICLES_PER_CELL_TARGET, NP, W_PARTICLE

    print("="*70)
    print("  COMPLETE 2D Cavity Physics Data Generator rebuilt from 77CavityUQL2.py")
    print("="*70)
    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

    lid_velocities = [
        float(v.strip()) for v in os.environ.get("CAVITY_LID_VELOCITIES", "700,800,900").split(",")
        if v.strip()
    ]

    N_STEPS_PER_RUN = int(os.environ.get("CAVITY_STEPS", str(N_STEPS_PER_RUN)))
    NTSS = int(os.environ.get("CAVITY_NTSS", str(NTSS)))
    save_interval = int(os.environ.get("CAVITY_SAVE_INTERVAL", "20"))
    output_file = os.environ.get("CAVITY_OUTPUT_FILE", "ml_training_data_highU_700_800_900.npz")

    PARTICLES_PER_CELL_TARGET = int(os.environ.get("CAVITY_PPC", str(PARTICLES_PER_CELL_TARGET)))
    NP = PARTICLES_PER_CELL_TARGET * NC
    W_PARTICLE = (LX * LY * RHO_IN_BASE) / float(NP)

    print(f"  Grid: {NX}x{NY} ({NC} cells)")
    print(f"  PPC: {PARTICLES_PER_CELL_TARGET}, Particles: {NP}")
    print(f"  Steps/run: {N_STEPS_PER_RUN}, NTSS: {NTSS}, save interval: {save_interval}")
    print(f"  Lid velocities: {lid_velocities}")
    print(f"  Combined output: {output_file}")

    all_X = []
    all_y = []
    total_start = time.time()

    for uw in lid_velocities:
        UW_LID = float(uw)
        DT = 0.2 * min(DX, DY) / max(UW_LID, THETA_IN)

        print("\\n" + "="*70)
        print(f"Starting PHYSICS data generation for UW_LID={UW_LID:.1f} m/s")
        print(f"DT={DT:.3e} s")
        print("="*70)

        grid_gpu, coeffs_gpu, linsys_gpu = initialize_grid_cupy(NX, NY, LX, LY)
        p_data = initialize_particles_cupy(NP, LX, LY, THETA_IN, W_PARTICLE)

        run_X = []
        run_y = []
        run_start = time.time()

        for nt in range(1, N_STEPS_PER_RUN + 1):
            p_data[9][:] = p_data[0]
            p_data[10][:] = p_data[1]
            p_data[0][:] = p_data[9] + p_data[3] * DT
            p_data[1][:] = p_data[10] + p_data[4] * DT

            apply_boundary_cavity_cupy(p_data, LX, LY, DT)

            sort_and_calc_moments_cupy_FULL(p_data, grid_gpu, NC, NX, NY, LX, LY)
            build_linear_systems_cupy(grid_gpu, linsys_gpu)
            solve_linear_systems_cupy(linsys_gpu, coeffs_gpu)

            if nt > NTSS and (nt % save_interval == 0):
                Xs, ys = _collect_training_snapshot(grid_gpu, coeffs_gpu)
                run_X.append(Xs)
                run_y.append(ys)

            evolve_velocities_cupy(p_data, grid_gpu, coeffs_gpu, DT, NC)

            if nt % 1000 == 0 or nt == N_STEPS_PER_RUN:
                cp.cuda.Stream.null.synchronize()
                elapsed = time.time() - run_start
                eta = (N_STEPS_PER_RUN - nt) * elapsed / max(nt, 1)
                avg_T = float(cp.asnumpy(cp.mean(grid_gpu['T'])))
                nsamp = sum(x.shape[0] for x in run_X) if run_X else 0
                print(
                    f"\\r  U={UW_LID:.0f} Step {nt}/{N_STEPS_PER_RUN} | "
                    f"Time {elapsed:.1f}s | ETA {eta:.1f}s | AvgT {avg_T:.2f} K | "
                    f"samples {nsamp}",
                    end="",
                    flush=True
                )

        print("")
        if run_X:
            X_run = np.concatenate(run_X, axis=0)
            y_run = np.concatenate(run_y, axis=0)
        else:
            X_run = np.empty((0, 16), dtype=np.float32)
            y_run = np.empty((0, 9), dtype=np.float32)

        per_file = f"training_data_{int(round(UW_LID))}ms.npz"
        np.savez_compressed(
            per_file,
            inputs=X_run,
            targets=y_run,
            lid_velocity=np.array([UW_LID], dtype=np.float64),
            feature_names=np.array([
                "rho","T","Ux","Uy","Uz","Pxx","Pxy","Pxz","Pyy","Pyz","Pzz",
                "Qx","Qy","Qz","DM2","nu"
            ]),
            target_names=np.array(["Axx","Axy","Axz","Ayy","Ayz","Azz","Bx","By","Bz"]),
        )
        print(f"Saved {per_file}: X={X_run.shape}, y={y_run.shape}")

        all_X.append(X_run)
        all_y.append(y_run)

    if all_X:
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
    else:
        X_all = np.empty((0, 16), dtype=np.float32)
        y_all = np.empty((0, 9), dtype=np.float32)

    np.savez_compressed(
        output_file,
        inputs=X_all,
        targets=y_all,
        lid_velocities=np.array(lid_velocities, dtype=np.float64),
        feature_names=np.array([
            "rho","T","Ux","Uy","Uz","Pxx","Pxy","Pxz","Pyy","Pyz","Pzz",
            "Qx","Qy","Qz","DM2","nu"
        ]),
        target_names=np.array(["Axx","Axy","Axz","Ayy","Ayz","Azz","Bx","By","Bz"]),
    )

    print("\\n" + "="*70)
    print(f"All runs completed in {(time.time() - total_start)/60.0:.2f} minutes")
    print(f"Combined data saved to {output_file}: X={X_all.shape}, y={y_all.shape}")
    print("="*70)


if __name__ == "__main__":
    main()
"""

dst.write_text(pre + new_main, encoding="utf-8")
print("Wrote complete new 58generalcavity.py using 77CavityUQL2.py as source.")
print("Now check:")
print("  python -m py_compile 58generalcavity.py")
