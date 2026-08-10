#!/usr/bin/env julia

using DelimitedFiles
using LinearAlgebra
using Printf
using Riemann35

const M000 = IJK_INDEX[(0,0,0)]
const M100 = IJK_INDEX[(1,0,0)]
const M010 = IJK_INDEX[(0,1,0)]
const M001 = IJK_INDEX[(0,0,1)]
const M200 = IJK_INDEX[(2,0,0)]
const M020 = IJK_INDEX[(0,2,0)]
const M002 = IJK_INDEX[(0,0,2)]

mutable struct AdaptiveStats
    accepted_steps::Int
    rejected_steps::Int
    minimum_h::Float64
    maximum_source_norm::Float64
    minimum_trial_margin::Float64
    minimum_accepted_margin::Float64
end

AdaptiveStats(initial_margin) = AdaptiveStats(
    0, 0, Inf, 0.0, Float64(initial_margin), Float64(initial_margin),
)

function parse_cli(arguments)
    length(arguments) in (8, 9) || error(
        "usage: run_homogeneous_stage2.jl INITIAL HISTORY METRICS STEPS DT TAU PR GAMMA_SCALE [raw|bounded]",
    )
    source_mode = length(arguments) == 9 ? Symbol(arguments[9]) : :raw
    source_mode in (:raw, :bounded) || error("source mode must be raw or bounded")
    return (
        initial=arguments[1],
        history=arguments[2],
        metrics=arguments[3],
        steps=parse(Int, arguments[4]),
        dt=parse(Float64, arguments[5]),
        tau=parse(Float64, arguments[6]),
        Pr=parse(Float64, arguments[7]),
        gamma_scale=parse(Float64, arguments[8]),
        source_mode=source_mode,
        speed_cap=25.0,
    )
end

function read_initial_moments(path)
    table = readdlm(path, ',', Any, '\n')
    size(table) == (2, 35) || error("expected a two-row, 35-column initial-state CSV")
    M = Float64[parse(Float64, string(table[2, column])) for column in 1:35]
    all(isfinite, M) || error("initial state contains NaN or Inf")
    is_realizable(M) || error("particle-derived initial M4 state is not realizable")
    return M
end

function diagnostics(step, dt, M, stats)
    state = fp_macroscopic35(M)
    energy = M[M200] + M[M020] + M[M002]
    recorded_minimum_h = isfinite(stats.minimum_h) ? stats.minimum_h : 0.0
    return Float64[
        step,
        step*dt,
        state.rho,
        energy,
        M[M200],
        M[IJK_INDEX[(3,0,0)]],
        M[IJK_INDEX[(4,0,0)]],
        norm(state.stress),
        norm(state.heat_flux),
        realizability_margin(M),
        stats.accepted_steps,
        stats.rejected_steps,
        recorded_minimum_h,
    ]
end

function write_history(path, rows)
    open(path, "w") do stream
        println(
            stream,
            "step,time,rho,energy_trace,M200,M300,M400,stress_norm,heat_flux_norm,realizability_margin,accepted_microsteps,rejected_microsteps,minimum_h",
        )
        for row in rows
            @printf(
                stream,
                "%.0f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.0f,%.0f,%.17g\n",
                row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                row[8], row[9], row[10], row[11], row[12], row[13],
            )
        end
    end
end

function write_metrics(path, metrics)
    open(path, "w") do stream
        println(stream, "key,value")
        for (key, value) in metrics
            println(stream, key, ",", value)
        end
    end
end

function legacy_substep_probe(M, controls)
    raw = copy(M)
    for step in 1:controls.steps
        try
            raw = fp_collision35(
                raw,
                controls.dt,
                controls.tau;
                Pr=controls.Pr,
                gamma_scale=controls.gamma_scale,
            )
        catch exception
            source = fp_collision_source35(
                raw,
                controls.tau;
                Pr=controls.Pr,
                gamma_scale=controls.gamma_scale,
            )
            probes = Pair{String,Float64}[]
            for power in 0:16
                h = controls.dt / (2.0^power)
                margin = realizability_margin(raw .+ h.*source)
                push!(probes, "legacy_probe_margin_dt_div_2pow$(power)" => margin)
            end
            return (
                failure_step=step,
                state_margin=realizability_margin(raw),
                exception=replace(sprint(showerror, exception), ',' => ';'),
                probes=probes,
            )
        end
    end
    return (
        failure_step=0,
        state_margin=realizability_margin(raw),
        exception="none",
        probes=Pair{String,Float64}[],
    )
end

function evaluate_source(M, controls)
    if controls.source_mode == :raw
        return fp_collision_source35(
            M,
            controls.tau;
            Pr=controls.Pr,
            gamma_scale=controls.gamma_scale,
        )
    end
    return fp_collision_source35_bounded(
        M,
        controls.tau;
        Pr=controls.Pr,
        gamma_scale=controls.gamma_scale,
        speed_cap=controls.speed_cap,
    )
end

function advance_to_target(
    M,
    current_time,
    target_time,
    h,
    controls,
    stats;
    minimum_h=controls.dt/(2.0^24),
    maximum_trials=2_000_000,
)
    success_streak = 0
    tolerance = 64.0*eps(Float64)*max(1.0, abs(target_time))

    while target_time-current_time > tolerance
        stats.accepted_steps + stats.rejected_steps < maximum_trials || error(
            "adaptive FP source exceeded $maximum_trials total trial steps",
        )
        trial_h = min(h, target_time-current_time)
        trial_h >= minimum_h || error(
            "adaptive FP source requires h=$trial_h below minimum_h=$minimum_h",
        )

        source = evaluate_source(M, controls)
        stats.maximum_source_norm = max(stats.maximum_source_norm, norm(source))
        candidate = M .+ trial_h.*source
        margin = realizability_margin(candidate)
        stats.minimum_trial_margin = min(stats.minimum_trial_margin, margin)

        if margin >= 0.0
            M = candidate
            current_time += trial_h
            stats.accepted_steps += 1
            stats.minimum_h = min(stats.minimum_h, trial_h)
            stats.minimum_accepted_margin = min(stats.minimum_accepted_margin, margin)
            success_streak += 1
            if success_streak >= 16
                h = min(2.0*h, controls.dt)
                success_streak = 0
            end
        else
            stats.rejected_steps += 1
            h = trial_h/2.0
            success_streak = 0
        end
    end

    return M, target_time, h
end

function main(arguments)
    controls = parse_cli(arguments)
    controls.steps > 0 || error("steps must be positive")
    controls.dt > 0.0 || error("dt must be positive")
    controls.tau > 0.0 || error("tau must be positive")
    0.0 < controls.Pr <= 1.0 || error("Pr must lie in (0,1]")

    M = read_initial_moments(controls.initial)
    initial = copy(M)
    initial_margin = realizability_margin(M)

    legacy_probe = legacy_substep_probe(M, controls)
    @printf("Selected source mode: %s\n", string(controls.source_mode))
    @printf("CHyQMOM initial margin: %.8e\n", initial_margin)
    if legacy_probe.failure_step == 0
        println("Legacy max_substeps=256 trajectory reached final time.")
    else
        @printf(
            "Legacy max_substeps=256 first failure: step %d, time %.8e\n",
            legacy_probe.failure_step,
            legacy_probe.failure_step*controls.dt,
        )
        @printf("State margin before capped step: %.8e\n", legacy_probe.state_margin)
        println("Legacy cap failure: ", legacy_probe.exception)
        for (key, value) in legacy_probe.probes
            @printf("%s: %.8e\n", key, value)
        end
    end

    stats = AdaptiveStats(initial_margin)
    rows = Vector{Vector{Float64}}()
    push!(rows, diagnostics(0, controls.dt, M, stats))
    current_time = 0.0
    h = controls.dt
    adaptive_reached_final_time = true
    adaptive_failure_step = 0
    adaptive_exception = "none"

    try
        for step in 1:controls.steps
            M, current_time, h = advance_to_target(
                M, current_time, step*controls.dt, h, controls, stats,
            )
            if step % 10 == 0 || step == controls.steps
                push!(rows, diagnostics(step, controls.dt, M, stats))
                @printf(
                    "adaptive progress: step=%d/%d time=%.8e h/dt=%.8e accepted=%d rejected=%d margin=%.8e\n",
                    step,
                    controls.steps,
                    current_time,
                    h/controls.dt,
                    stats.accepted_steps,
                    stats.rejected_steps,
                    realizability_margin(M),
                )
                flush(stdout)
            end
        end
    catch exception
        adaptive_reached_final_time = false
        adaptive_failure_step = floor(Int, current_time/controls.dt) + 1
        adaptive_exception = replace(sprint(showerror, exception), ',' => ';')
        println("Adaptive integration failure: ", adaptive_exception)
    end

    write_history(controls.history, rows)
    momentum_positions = (M100, M010, M001)
    initial_energy = initial[M200] + initial[M020] + initial[M002]
    final_energy = M[M200] + M[M020] + M[M002]
    mass_drift = abs(M[M000] - initial[M000])
    momentum_drift = maximum(abs(M[n] - initial[n]) for n in momentum_positions)
    energy_drift = abs(final_energy - initial_energy)
    final_margin = realizability_margin(M)

    metrics = Pair{String,Any}[
        "source_mode" => string(controls.source_mode),
        "bounded_speed_cap" => controls.speed_cap,
        "legacy_cap_failure_step" => legacy_probe.failure_step,
        "legacy_failure_state_margin" => legacy_probe.state_margin,
        "adaptive_reached_final_time" => adaptive_reached_final_time,
        "adaptive_failure_step" => adaptive_failure_step,
        "adaptive_exception" => adaptive_exception,
        "adaptive_accepted_microsteps" => stats.accepted_steps,
        "adaptive_rejected_microsteps" => stats.rejected_steps,
        "adaptive_minimum_h" => stats.minimum_h,
        "adaptive_minimum_h_over_dt" => stats.minimum_h/controls.dt,
        "adaptive_maximum_source_norm" => stats.maximum_source_norm,
        "minimum_trial_margin" => stats.minimum_trial_margin,
        "minimum_accepted_margin" => stats.minimum_accepted_margin,
        "initial_margin" => initial_margin,
        "final_margin" => final_margin,
        "julia_mass_drift" => mass_drift,
        "julia_momentum_drift" => momentum_drift,
        "julia_energy_drift" => energy_drift,
    ]
    for (key, value) in legacy_probe.probes
        push!(metrics, key => value)
    end
    write_metrics(controls.metrics, metrics)

    @printf("Adaptive Julia samples written:       %d\n", length(rows))
    @printf("Adaptive reached final time:          %s\n", string(adaptive_reached_final_time))
    @printf("Accepted / rejected microsteps:       %d / %d\n", stats.accepted_steps, stats.rejected_steps)
    @printf("Minimum h / dt:                       %.8e\n", stats.minimum_h/controls.dt)
    @printf("Maximum source norm:                  %.8e\n", stats.maximum_source_norm)
    @printf("Minimum accepted margin:              %.8e\n", stats.minimum_accepted_margin)
    @printf("Julia mass drift:                     %.3e\n", mass_drift)
    @printf("Julia momentum drift:                 %.3e\n", momentum_drift)
    @printf("Julia energy drift:                   %.3e\n", energy_drift)

    adaptive_reached_final_time || error(
        "$(controls.source_mode) adaptive closure failed at macro step $adaptive_failure_step: $adaptive_exception",
    )
    mass_drift <= 1.0e-12 || error("mass conservation gate failed")
    momentum_drift <= 1.0e-12 || error("momentum conservation gate failed")
    energy_drift <= 1.0e-10 || error("energy conservation gate failed")
    final_margin >= 0.0 || error("final state failed realizability gate")
end

main(ARGS)
