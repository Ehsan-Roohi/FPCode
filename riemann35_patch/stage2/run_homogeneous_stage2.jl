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
const M110 = IJK_INDEX[(1,1,0)]
const M101 = IJK_INDEX[(1,0,1)]
const M011 = IJK_INDEX[(0,1,1)]

function parse_cli(arguments)
    length(arguments) == 8 || error(
        "usage: run_homogeneous_stage2.jl INITIAL HISTORY METRICS STEPS DT TAU PR GAMMA_SCALE",
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

function diagnostics(step, dt, M, projection_count)
    state = fp_macroscopic35(M)
    energy = M[M200] + M[M020] + M[M002]
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
        projection_count,
    ]
end

function write_history(path, rows)
    open(path, "w") do stream
        println(
            stream,
            "step,time,rho,energy_trace,M200,M300,M400,stress_norm,heat_flux_norm,realizability_margin,projection_count",
        )
        for row in rows
            @printf(
                stream,
                "%.0f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.0f\n",
                row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                row[8], row[9], row[10], row[11],
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

function gaussian_anchor(M)
    rho = M[M000]
    u = (M[M100]/rho, M[M010]/rho, M[M001]/rho)
    C200 = M[M200]/rho - u[1]^2
    C020 = M[M020]/rho - u[2]^2
    C002 = M[M002]/rho - u[3]^2
    C110 = M[M110]/rho - u[1]*u[2]
    C101 = M[M101]/rho - u[1]*u[3]
    C011 = M[M011]/rho - u[2]*u[3]
    return InitializeM4_35(
        rho, u[1], u[2], u[3], C200, C110, C101, C020, C011, C002,
    )
end

function move_to_interior(M; target_margin=1.0e-10)
    margin = realizability_margin(M)
    margin >= target_margin && return Float64.(M), 0.0, margin
    anchor = gaussian_anchor(M)
    for weight in (1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8,
                   1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3,
                   1.0e-2, 1.0e-1, 1.0)
        candidate = (1.0-weight).*M .+ weight.*anchor
        candidate_margin = realizability_margin(candidate)
        if candidate_margin >= target_margin
            return candidate, weight, candidate_margin
        end
    end
    error("could not move the projected state into the realizability interior")
end

function projected_source_step(M, dt, tau, Ma; Pr, gamma_scale)
    source = fp_collision_source35(M, tau; Pr=Pr, gamma_scale=gamma_scale)
    candidate = M .+ dt.*source
    trial_margin = realizability_margin(candidate)
    if trial_margin >= 1.0e-10
        return candidate, false, 0.0, 0.0, trial_margin, trial_margin
    end

    projected = realizable_3D_M4(candidate, Ma)
    projected, interior_weight, final_margin = move_to_interior(projected)
    relative_correction = norm(projected-candidate) / max(norm(candidate), eps(Float64))
    return projected, true, relative_correction, interior_weight, trial_margin, final_margin
end

function raw_realizability_probe(M, controls)
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
                push!(probes, "raw_probe_margin_dt_div_2pow$(power)" => margin)
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

function main(arguments)
    controls = parse_cli(arguments)
    controls.steps > 0 || error("steps must be positive")
    controls.dt > 0.0 || error("dt must be positive")
    controls.tau > 0.0 || error("tau must be positive")
    0.0 < controls.Pr <= 1.0 || error("Pr must lie in (0,1]")

    M = read_initial_moments(controls.initial)
    initial = copy(M)
    initial_state = fp_macroscopic35(M)
    Ma = norm(collect(initial_state.velocity)) / sqrt((5.0/3.0)*initial_state.theta)
    initial_margin = realizability_margin(M)

    raw_probe = raw_realizability_probe(M, controls)
    @printf("Raw CHyQMOM-M6 initial margin: %.8e\n", initial_margin)
    if raw_probe.failure_step == 0
        println("Raw CHyQMOM-M6 trajectory reached final time without a realizability failure.")
    else
        @printf(
            "Raw CHyQMOM-M6 first realizability failure: step %d, time %.8e\n",
            raw_probe.failure_step,
            raw_probe.failure_step*controls.dt,
        )
        @printf("Raw state margin before failed step: %.8e\n", raw_probe.state_margin)
        println("Raw failure: ", raw_probe.exception)
        for (key, value) in raw_probe.probes
            @printf("%s: %.8e\n", key, value)
        end
    end

    sample_every = 10
    rows = Vector{Vector{Float64}}()
    projection_count = 0
    maximum_relative_projection = 0.0
    maximum_interior_weight = 0.0
    minimum_trial_margin = initial_margin
    minimum_accepted_margin = initial_margin
    push!(rows, diagnostics(0, controls.dt, M, projection_count))

    for step in 1:controls.steps
        M, projected, correction, interior_weight, trial_margin, accepted_margin =
            projected_source_step(
                M,
                controls.dt,
                controls.tau,
                Ma;
                Pr=controls.Pr,
                gamma_scale=controls.gamma_scale,
            )
        if projected
            projection_count += 1
            maximum_relative_projection = max(maximum_relative_projection, correction)
            maximum_interior_weight = max(maximum_interior_weight, interior_weight)
        end
        minimum_trial_margin = min(minimum_trial_margin, trial_margin)
        minimum_accepted_margin = min(minimum_accepted_margin, accepted_margin)
        if step % sample_every == 0 || step == controls.steps
            push!(rows, diagnostics(step, controls.dt, M, projection_count))
        end
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
        "raw_failure_step" => raw_probe.failure_step,
        "raw_reached_final_time" => (raw_probe.failure_step == 0),
        "raw_failure_state_margin" => raw_probe.state_margin,
        "initial_margin" => initial_margin,
        "projection_count" => projection_count,
        "projection_fraction" => projection_count/controls.steps,
        "maximum_relative_projection" => maximum_relative_projection,
        "maximum_interior_weight" => maximum_interior_weight,
        "minimum_trial_margin" => minimum_trial_margin,
        "minimum_accepted_margin" => minimum_accepted_margin,
        "final_margin" => final_margin,
        "julia_mass_drift" => mass_drift,
        "julia_momentum_drift" => momentum_drift,
        "julia_energy_drift" => energy_drift,
    ]
    for (key, value) in raw_probe.probes
        push!(metrics, key => value)
    end
    write_metrics(controls.metrics, metrics)

    @printf("Projected Julia samples written:          %d\n", length(rows))
    @printf("Projection count / fraction:             %d / %.3f\n", projection_count, projection_count/controls.steps)
    @printf("Maximum relative projection correction:  %.3e\n", maximum_relative_projection)
    @printf("Maximum Gaussian interiorization weight: %.3e\n", maximum_interior_weight)
    @printf("Minimum accepted margin:                 %.3e\n", minimum_accepted_margin)
    @printf("Julia mass drift:                        %.3e\n", mass_drift)
    @printf("Julia momentum drift:                    %.3e\n", momentum_drift)
    @printf("Julia energy drift:                      %.3e\n", energy_drift)

    mass_drift <= 1.0e-12 || error("mass conservation gate failed")
    momentum_drift <= 1.0e-12 || error("momentum conservation gate failed")
    energy_drift <= 1.0e-10 || error("energy conservation gate failed")
    final_margin >= 1.0e-10 || error("final state failed interior realizability gate")
end

main(ARGS)
