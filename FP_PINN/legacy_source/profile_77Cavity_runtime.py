#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runtime component profiler for 77CavityUQL2.py.

This profiler does NOT edit 77CavityUQL2.py. It imports it as a module,
monkey-patches key CuPy functions with timers, runs main(), and writes:

  component_profile_raw.csv
  component_profile_summary.csv
  component_profile_table.tex
  profile_notes.md

What is measured
----------------
Physics run:
  boundary treatment
  FULL moment calculation
  averaging
  9x9 matrix construction
  9x9 batched solve
  velocity evolution

ML run:
  boundary treatment
  LITE moment calculation
  averaging
  DNN forward pass
  velocity evolution

The residual is reported as:
  particle move + loop/report/uncaptured overhead

Timing mode
-----------
GPU work is asynchronous, so accurate timing requires synchronization.
To reduce overhead, this script synchronizes and times only every N-th call
(default --sample-every 20), then estimates total time from sampled means.
"""

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import time
from collections import defaultdict

import numpy as np


def load_module_from_path(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("cavity77_profiled", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ComponentProfiler:
    def __init__(self, cp, expected_steps, sample_every=20, sync=True):
        self.cp = cp
        self.expected_steps = int(expected_steps)
        self.sample_every = max(1, int(sample_every))
        self.sync = bool(sync)

        self.phase = "physics"
        self.phase_start = {}
        self.phase_end = {}

        self.call_counts = defaultdict(int)
        self.sample_counts = defaultdict(int)
        self.sample_time = defaultdict(float)

    def maybe_switch_phase(self):
        # Physics loop runs first. After full_moments has been called expected_steps
        # times, the next loop is the ML loop; this catches the ML boundary call
        # before the first LITE moment call.
        if self.phase == "physics" and self.call_counts[("physics", "full_moments")] >= self.expected_steps:
            self.phase = "ml"

    def wrap(self, name, func, fixed_phase=None):
        def wrapped(*args, **kwargs):
            if fixed_phase is None:
                self.maybe_switch_phase()
                phase = self.phase
            else:
                phase = fixed_phase
                if fixed_phase == "ml":
                    self.phase = "ml"

            key = (phase, name)
            self.call_counts[key] += 1
            call_id = self.call_counts[key]

            if phase not in self.phase_start:
                self.phase_start[phase] = time.perf_counter()

            do_sample = (call_id % self.sample_every == 0)

            if do_sample:
                if self.sync:
                    self.cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                out = func(*args, **kwargs)
                if self.sync:
                    self.cp.cuda.Stream.null.synchronize()
                dt = time.perf_counter() - t0
                self.sample_counts[key] += 1
                self.sample_time[key] += dt
            else:
                out = func(*args, **kwargs)

            self.phase_end[phase] = time.perf_counter()
            return out
        return wrapped

    def rows(self):
        all_keys = sorted(set(self.call_counts.keys()))
        rows = []
        for phase, name in all_keys:
            calls = self.call_counts[(phase, name)]
            samples = self.sample_counts[(phase, name)]
            sampled_time = self.sample_time[(phase, name)]
            mean_sample_s = sampled_time / samples if samples > 0 else np.nan
            est_total_s = mean_sample_s * calls if samples > 0 else np.nan
            rows.append({
                "phase": phase,
                "component": name,
                "calls": calls,
                "timed_samples": samples,
                "sample_every": self.sample_every,
                "sampled_time_s": sampled_time,
                "mean_sample_s": mean_sample_s,
                "estimated_total_s": est_total_s,
                "estimated_ms_per_call": 1000.0 * mean_sample_s if samples > 0 else np.nan,
            })
        return rows

    def phase_totals(self):
        out = {}
        for phase in sorted(self.phase_start.keys()):
            out[phase] = self.phase_end.get(phase, self.phase_start[phase]) - self.phase_start[phase]
        return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows, phase_totals):
    # Group by phase and add residual.
    out = []
    for phase in ["physics", "ml"]:
        phase_rows = [r for r in rows if r["phase"] == phase]
        total_wall = phase_totals.get(phase, np.nan)
        estimated_sum = np.nansum([r["estimated_total_s"] for r in phase_rows])
        for r in phase_rows:
            rr = dict(r)
            rr["phase_wall_s"] = total_wall
            rr["fraction_of_phase_wall"] = rr["estimated_total_s"] / total_wall if total_wall and np.isfinite(rr["estimated_total_s"]) else np.nan
            out.append(rr)

        if np.isfinite(total_wall):
            residual = total_wall - estimated_sum
            out.append({
                "phase": phase,
                "component": "particle_move_plus_loop_report_uncaptured",
                "calls": "",
                "timed_samples": "",
                "sample_every": "",
                "sampled_time_s": "",
                "mean_sample_s": "",
                "estimated_total_s": residual,
                "estimated_ms_per_call": "",
                "phase_wall_s": total_wall,
                "fraction_of_phase_wall": residual / total_wall if total_wall else np.nan,
            })
    return out


def latex_component_table(summary_rows, out_tex):
    # Compact table for manuscript/response.
    label_map = {
        "boundary": "Boundary treatment",
        "full_moments": "FULL high-order moments",
        "lite_moments": "LITE low-order moments",
        "averaging": "Averaging",
        "build_linear_system": "Build $9\\times9$ system",
        "solve_linear_system": "Solve $9\\times9$ system",
        "dnn_forward": "DNN forward pass",
        "velocity_evolution": "Velocity evolution",
        "particle_move_plus_loop_report_uncaptured": "Particle move + overhead",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Component-level runtime profile of the q-weighted cavity simulation. Timings are estimated from synchronized sampled calls.}")
    lines.append(r"\label{tab:cavity_component_profile}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\hline")
    lines.append(r"Run & Component & Estimated time (s) & Phase fraction (\%) \\")
    lines.append(r"\hline")
    for r in summary_rows:
        if r["component"] == "averaging":
            continue
        comp = label_map.get(r["component"], r["component"])
        phase = "Physics" if r["phase"] == "physics" else "ML"
        t = r["estimated_total_s"]
        frac = r["fraction_of_phase_wall"]
        if isinstance(t, str) or not np.isfinite(float(t)):
            continue
        lines.append(f"{phase} & {comp} & {float(t):.2f} & {100.0*float(frac):.1f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(out_tex).write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="77CavityUQL2.py")
    ap.add_argument("--steps", type=int, default=None, help="Override N_STEPS_PER_RUN for profiling.")
    ap.add_argument("--ntss", type=int, default=None, help="Override NTSS for profiling.")
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--outdir", default="cavity77_component_profile")
    ap.add_argument("--no-sync", action="store_true", help="Do not synchronize GPU before/after sampled calls.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mod = load_module_from_path(args.module)
    cp = mod.cp

    if args.steps is not None:
        mod.N_STEPS_PER_RUN = int(args.steps)
    if args.ntss is not None:
        mod.NTSS = int(args.ntss)

    # Disable optional diagnostic add-ons if present, so the runtime profile is the solver profile.
    if hasattr(mod, "HIGH_MOMENTS_EVERY"):
        mod.HIGH_MOMENTS_EVERY = 0
    if hasattr(mod, "ENTROPY_EVERY"):
        mod.ENTROPY_EVERY = 0

    print("="*70)
    print("Component profiling for 77CavityUQL2.py")
    print(f"Steps: {mod.N_STEPS_PER_RUN}, NTSS: {mod.NTSS}, sample_every: {args.sample_every}")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print("="*70, flush=True)

    prof = ComponentProfiler(cp, expected_steps=mod.N_STEPS_PER_RUN, sample_every=args.sample_every, sync=(not args.no_sync))

    # Monkey-patch components.
    mod.apply_boundary_cavity_cupy = prof.wrap("boundary", mod.apply_boundary_cavity_cupy)
    mod.sort_and_calc_moments_cupy_FULL = prof.wrap("full_moments", mod.sort_and_calc_moments_cupy_FULL, fixed_phase="physics")
    mod.sort_and_calc_moments_cupy_LITE = prof.wrap("lite_moments", mod.sort_and_calc_moments_cupy_LITE, fixed_phase="ml")
    mod.average_results_cupy = prof.wrap("averaging", mod.average_results_cupy)
    mod.build_linear_systems_cupy = prof.wrap("build_linear_system", mod.build_linear_systems_cupy, fixed_phase="physics")
    mod.solve_linear_systems_cupy = prof.wrap("solve_linear_system", mod.solve_linear_systems_cupy, fixed_phase="physics")
    mod.predict_coeffs_cupy_native = prof.wrap("dnn_forward", mod.predict_coeffs_cupy_native, fixed_phase="ml")
    mod.evolve_velocities_cupy = prof.wrap("velocity_evolution", mod.evolve_velocities_cupy)

    t0 = time.perf_counter()
    mod.main()
    total_wall = time.perf_counter() - t0

    raw_rows = prof.rows()
    phase_totals = prof.phase_totals()
    summary_rows = summarize(raw_rows, phase_totals)

    raw_fields = [
        "phase", "component", "calls", "timed_samples", "sample_every",
        "sampled_time_s", "mean_sample_s", "estimated_total_s", "estimated_ms_per_call"
    ]
    write_csv(outdir / "component_profile_raw.csv", raw_rows, raw_fields)

    summary_fields = raw_fields + ["phase_wall_s", "fraction_of_phase_wall"]
    write_csv(outdir / "component_profile_summary.csv", summary_rows, summary_fields)

    latex_component_table(summary_rows, outdir / "component_profile_table.tex")

    notes = []
    notes.append("# Component profiling notes")
    notes.append("")
    notes.append(f"- Total wrapper wall time: {total_wall:.3f} s")
    notes.append(f"- Steps: {mod.N_STEPS_PER_RUN}")
    notes.append(f"- NTSS: {mod.NTSS}")
    notes.append(f"- Sample every: {args.sample_every}")
    notes.append(f"- GPU synchronization during sampled timing: {not args.no_sync}")
    notes.append("")
    notes.append("## Phase wall times from profiler")
    for ph, val in phase_totals.items():
        notes.append(f"- {ph}: {val:.3f} s")
    notes.append("")
    notes.append("## Interpretation")
    notes.append("The residual row is the approximate cost of particle position update plus Python loop/reporting and any uninstrumented work. It should be described conservatively.")
    notes.append("")
    notes.append("## Files")
    notes.append("- component_profile_raw.csv")
    notes.append("- component_profile_summary.csv")
    notes.append("- component_profile_table.tex")
    (outdir / "profile_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("\nSaved profiling outputs:")
    for p in sorted(outdir.iterdir()):
        print(" ", p)
    print("\nSummary preview:")
    for r in summary_rows:
        if r["component"] == "averaging":
            continue
        t = r["estimated_total_s"]
        frac = r["fraction_of_phase_wall"]
        if isinstance(t, str):
            continue
        print(f"{r['phase']:7s} | {r['component']:40s} | {float(t):10.3f} s | {100*float(frac):6.2f}%")

if __name__ == "__main__":
    main()
