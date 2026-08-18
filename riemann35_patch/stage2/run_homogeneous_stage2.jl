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
        "usage: run_homogeneous_stage2.jl INITIAL HISTORY METRICS STEPS DT TAU PR GAMMA_SCALE [raw|bounded|finite|finite_subcycled]",
    )
    source_mode = length(arguments) == 9 ? Symbol(arguments[9]) : :raw
    source_mode in (:raw, :bounded, :finite, :finite_subcycled) || error(
        "source mode must be raw, bounded, finite, or finite_subcycled",
    )
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
        finite_max_substeps=parse(Int, get(ENV, "FINITE_MAX_SUBSTEPS", "256")),
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

function try_finite_substeps(M, substeps, controls)
    trial = copy(M)
    h = controls.dt/substeps
    attempted_substeps = 0
    minimum_margin = realizability_margin(trial)
    minimum_alpha = Inf
    maximum_alpha = 0.0
    maximum_node_c2_over_theta = 0.0
    maximum_quadrature_residual_norm = 0.0
    maximum_collision_increment_norm = 0.0
    maximum_source_norm = 0.0

    for substep in 1:substeps
        previous = trial
        candidate = nothing
        map_diagnostics = nothing
        attempted_substeps += 1
        try
            candidate, map_diagnostics = fp_collision_step35_bounded(
                previous,
                h,
                controls.tau;
                Pr=controls.Pr,
                gamma_scale=controls.gamma_scale,
                speed_cap=controls.speed_cap,
                with_diagnostics=true,
            )
        catch exception
            return (
                success=false,
                state=M,
                attempted_substeps=attempted_substeps,
                failure_substep=substep,
                minimum_margin=-Inf,
                exception=replace(sprint(showerror, exception), ',' => ';'),
            )
        end

        margin = realizability_margin(candidate)
        safe_margin = isfinite(margin) ? margin : -Inf
        minimum_margin = min(minimum_margin, safe_margin)
        if safe_margin < 0.0 || !is_realizable(candidate)
            return (
                success=false,
                state=M,
                attempted_substeps=attempted_substeps,
                failure_substep=substep,
                minimum_margin=minimum_margin,
                exception="finite FP map left the realizability cone",
            )
        end

        trial = candidate
        minimum_alpha = min(minimum_alpha, map_diagnostics.alpha)
        maximum_alpha = max(maximum_alpha, map_diagnostics.alpha)
        maximum_node_c2_over_theta = max(
            maximum_node_c2_over_theta,
            map_diagnostics.maximum_c2_over_theta,
        )
        if hasproperty(map_diagnostics, :quadrature_residual_norm)
            maximum_quadrature_residual_norm = max(
                maximum_quadrature_residual_norm,
                map_diagnostics.quadrature_residual_norm,
            )
            maximum_collision_increment_norm = max(
                maximum_collision_increment_norm,
                map_diagnostics.collision_increment_norm,
            )
        end
        maximum_source_norm = max(
            maximum_source_norm,
            norm(candidate-previous)/h,
        )
    end

    return (
        success=true,
        state=trial,
        attempted_substeps=attempted_substeps,
        failure_substep=0,
        minimum_margin=minimum_margin,
        exception="none",
        h=h,
        minimum_alpha=minimum_alpha,
        maximum_alpha=maximum_alpha,
        maximum_node_c2_over_theta=maximum_node_c2_over_theta,
        maximum_quadrature_residual_norm=maximum_quadrature_residual_norm,
        maximum_collision_increment_norm=maximum_collision_increment_norm,
        maximum_source_norm=maximum_source_norm,
    )
end

function advance_finite_subcycled(M, controls)
    attempts = NamedTuple[]
    rejected_microsteps = 0
    restarts = 0
    substeps = 1

    while substeps <= controls.finite_max_substeps
        result = try_finite_substeps(M, substeps, controls)
        push!(attempts, (
            substeps=substeps,
            success=result.success,
            attempted_substeps=result.attempted_substeps,
            failure_substep=result.failure_substep,
            minimum_margin=result.minimum_margin,
            exception=result.exception,
        ))
        if result.success
            return (
                success=true,
                state=result.state,
                substeps=substeps,
                rejected_microsteps=rejected_microsteps,
                restarts=restarts,
                attempts=attempts,
                h=result.h,
                minimum_margin=result.minimum_margin,
                minimum_alpha=result.minimum_alpha,
                maximum_alpha=result.maximum_alpha,
                maximum_node_c2_over_theta=result.maximum_node_c2_over_theta,
                maximum_quadrature_residual_norm=result.maximum_quadrature_residual_norm,
                maximum_collision_increment_norm=result.maximum_collision_increment_norm,
                maximum_source_norm=result.maximum_source_norm,
                exception="none",
            )
        end
        rejected_microsteps += result.attempted_substeps
        restarts += 1
        substeps *= 2
    end

    last_attempt = attempts[end]
    return (
        success=false,
        state=M,
        substeps=0,
        rejected_microsteps=rejected_microsteps,
        restarts=restarts,
        attempts=attempts,
        exception=(
            "finite FP subcycling exceeded max_substeps=$(controls.finite_max_substeps); " *
            "last failure at nsub=$(last_attempt.substeps) substep=$(last_attempt.failure_substep): " *
            last_attempt.exception
        ),
    )
end

function main(arguments)
    controls = parse_cli(arguments)
    controls.steps > 0 || error("steps must be positive")
    controls.dt > 0.0 || error("dt must be positive")
    controls.tau > 0.0 || error("tau must be positive")
    0.0 < controls.Pr <= 1.0 || error("Pr must lie in (0,1]")
    controls.finite_max_substeps > 0 || error("FINITE_MAX_SUBSTEPS must be positive")

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
    minimum_alpha = Inf
    maximum_alpha = 0.0
    maximum_node_c2_over_theta = 0.0
    maximum_quadrature_residual_norm = 0.0
    maximum_collision_increment_norm = 0.0
    finite_maximum_substeps_per_macro = 1
    finite_retried_macrosteps = 0
    finite_total_restarts = 0
    finite_first_step_attempts = NamedTuple[]

    try
        for step in 1:controls.steps
            if controls.source_mode == :finite
                previous = M
                M, map_diagnostics = fp_collision_step35_bounded(
                    M,
                    controls.dt,
                    controls.tau;
                    Pr=controls.Pr,
                    gamma_scale=controls.gamma_scale,
                    speed_cap=controls.speed_cap,
                    with_diagnostics=true,
                )
                margin = realizability_margin(M)
                margin >= 0.0 || error(
                    "finite FP map left the realizability cone at step $step",
                )
                current_time = step*controls.dt
                h = controls.dt
                stats.accepted_steps += 1
                stats.minimum_h = min(stats.minimum_h, controls.dt)
                stats.maximum_source_norm = max(
                    stats.maximum_source_norm,
                    norm(M-previous)/controls.dt,
                )
                stats.minimum_trial_margin = min(stats.minimum_trial_margin, margin)
                stats.minimum_accepted_margin = min(stats.minimum_accepted_margin, margin)
                minimum_alpha = min(minimum_alpha, map_diagnostics.alpha)
                maximum_alpha = max(maximum_alpha, map_diagnostics.alpha)
                maximum_node_c2_over_theta = max(
                    maximum_node_c2_over_theta,
                    map_diagnostics.maximum_c2_over_theta,
                )
                if hasproperty(map_diagnostics, :quadrature_residual_norm)
                    maximum_quadrature_residual_norm = max(
                        maximum_quadrature_residual_norm,
                        map_diagnostics.quadrature_residual_norm,
                    )
                    maximum_collision_increment_norm = max(
                        maximum_collision_increment_norm,
                        map_diagnostics.collision_increment_norm,
                    )
                end
            elseif controls.source_mode == :finite_subcycled
                advance = advance_finite_subcycled(M, controls)
                if step == 1
                    finite_first_step_attempts = advance.attempts
                end
                for attempt in advance.attempts
                    if step == 1 || !attempt.success
                        @printf(
                            "finite subcycle attempt: macro_step=%d nsub=%d success=%s attempted=%d failure_substep=%d minimum_margin=%.8e message=%s\n",
                            step,
                            attempt.substeps,
                            string(attempt.success),
                            attempt.attempted_substeps,
                            attempt.failure_substep,
                            attempt.minimum_margin,
                            attempt.exception,
                        )
                    end
                    stats.minimum_trial_margin = min(
                        stats.minimum_trial_margin,
                        attempt.minimum_margin,
                    )
                end
                stats.rejected_steps += advance.rejected_microsteps
                finite_total_restarts += advance.restarts
                finite_retried_macrosteps += (advance.restarts > 0 ? 1 : 0)
                advance.success || error(advance.exception)
                M = advance.state
                margin = realizability_margin(M)
                current_time = step*controls.dt
                h = advance.h
                stats.accepted_steps += advance.substeps
                stats.minimum_h = min(stats.minimum_h, advance.h)
                stats.maximum_source_norm = max(
                    stats.maximum_source_norm,
                    advance.maximum_source_norm,
                )
                stats.minimum_accepted_margin = min(
                    stats.minimum_accepted_margin,
                    advance.minimum_margin,
                )
                minimum_alpha = min(minimum_alpha, advance.minimum_alpha)
                maximum_alpha = max(maximum_alpha, advance.maximum_alpha)
                maximum_node_c2_over_theta = max(
                    maximum_node_c2_over_theta,
                    advance.maximum_node_c2_over_theta,
                )
                maximum_quadrature_residual_norm = max(
                    maximum_quadrature_residual_norm,
                    advance.maximum_quadrature_residual_norm,
                )
                maximum_collision_increment_norm = max(
                    maximum_collision_increment_norm,
                    advance.maximum_collision_increment_norm,
                )
                finite_maximum_substeps_per_macro = max(
                    finite_maximum_substeps_per_macro,
                    advance.substeps,
                )
            else
                M, current_time, h = advance_to_target(
                    M, current_time, step*controls.dt, h, controls, stats,
                )
            end
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
        "finite_minimum_alpha" => minimum_alpha,
        "finite_maximum_alpha" => maximum_alpha,
        "finite_maximum_node_c2_over_theta" => maximum_node_c2_over_theta,
        "finite_maximum_quadrature_residual_norm" => maximum_quadrature_residual_norm,
        "finite_maximum_collision_increment_norm" => maximum_collision_increment_norm,
        "finite_max_substeps_limit" => controls.finite_max_substeps,
        "finite_maximum_substeps_per_macro" => finite_maximum_substeps_per_macro,
        "finite_retried_macrosteps" => finite_retried_macrosteps,
        "finite_total_restarts" => finite_total_restarts,
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
    for attempt in finite_first_step_attempts
        prefix = "finite_first_step_nsub_$(attempt.substeps)"
        push!(metrics, "$(prefix)_success" => attempt.success)
        push!(metrics, "$(prefix)_attempted" => attempt.attempted_substeps)
        push!(metrics, "$(prefix)_failure_substep" => attempt.failure_substep)
        push!(metrics, "$(prefix)_minimum_margin" => attempt.minimum_margin)
        push!(metrics, "$(prefix)_exception" => attempt.exception)
    end
    write_metrics(controls.metrics, metrics)

    @printf("Adaptive Julia samples written:       %d\n", length(rows))
    @printf("Adaptive reached final time:          %s\n", string(adaptive_reached_final_time))
    @printf("Accepted / rejected microsteps:       %d / %d\n", stats.accepted_steps, stats.rejected_steps)
    @printf("Minimum h / dt:                       %.8e\n", stats.minimum_h/controls.dt)
    @printf("Maximum source norm:                  %.8e\n", stats.maximum_source_norm)
    @printf("Minimum accepted margin:              %.8e\n", stats.minimum_accepted_margin)
    if controls.source_mode in (:finite, :finite_subcycled)
        @printf("Minimum / maximum alpha:              %.8e / %.8e\n", minimum_alpha, maximum_alpha)
        @printf("Maximum node c2 / theta:              %.8e\n", maximum_node_c2_over_theta)
        @printf("Maximum quadrature residual norm:     %.8e\n", maximum_quadrature_residual_norm)
        @printf("Maximum collision increment norm:     %.8e\n", maximum_collision_increment_norm)
        if controls.source_mode == :finite_subcycled
            @printf("Maximum substeps per macro interval:  %d\n", finite_maximum_substeps_per_macro)
            @printf("Retried macro intervals / restarts:  %d / %d\n", finite_retried_macrosteps, finite_total_restarts)
        end
    end
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
