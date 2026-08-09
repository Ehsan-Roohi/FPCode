#!/usr/bin/env julia

using DelimitedFiles
using LinearAlgebra
using Printf
using Riemann35

function parse_cli(arguments)
    length(arguments) == 7 || error(
        "usage: run_homogeneous_stage2.jl INITIAL OUTPUT STEPS DT TAU PR GAMMA_SCALE",
    )
    return (
        initial=arguments[1],
        output=arguments[2],
        steps=parse(Int, arguments[3]),
        dt=parse(Float64, arguments[4]),
        tau=parse(Float64, arguments[5]),
        Pr=parse(Float64, arguments[6]),
        gamma_scale=parse(Float64, arguments[7]),
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

function diagnostics(step, dt, M)
    state = fp_macroscopic35(M)
    energy = M[IJK_INDEX[(2,0,0)]] + M[IJK_INDEX[(0,2,0)]] + M[IJK_INDEX[(0,0,2)]]
    return Float64[
        step,
        step*dt,
        state.rho,
        energy,
        M[IJK_INDEX[(2,0,0)]],
        M[IJK_INDEX[(3,0,0)]],
        M[IJK_INDEX[(4,0,0)]],
        norm(state.stress),
        norm(state.heat_flux),
    ]
end

function write_history(path, rows)
    open(path, "w") do stream
        println(stream, "step,time,rho,energy_trace,M200,M300,M400,stress_norm,heat_flux_norm")
        for row in rows
            @printf(
                stream,
                "%.0f,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                row[1], row[2], row[3], row[4], row[5], row[6], row[7],
                row[8], row[9],
            )
        end
    end
end

function main(arguments)
    controls = parse_cli(arguments)
    controls.steps > 0 || error("steps must be positive")
    controls.dt > 0.0 || error("dt must be positive")
    controls.tau > 0.0 || error("tau must be positive")
    0.0 < controls.Pr <= 1.0 || error("Pr must lie in (0,1]")

    M = read_initial_moments(controls.initial)
    initial = copy(M)
    sample_every = 10
    rows = Vector{Vector{Float64}}()
    push!(rows, diagnostics(0, controls.dt, M))

    for step in 1:controls.steps
        M = fp_collision35(
            M,
            controls.dt,
            controls.tau;
            Pr=controls.Pr,
            gamma_scale=controls.gamma_scale,
        )
        if step % sample_every == 0 || step == controls.steps
            push!(rows, diagnostics(step, controls.dt, M))
        end
    end

    write_history(controls.output, rows)
    momentum_positions = (
        IJK_INDEX[(1,0,0)], IJK_INDEX[(0,1,0)], IJK_INDEX[(0,0,1)],
    )
    initial_energy = diagnostics(0, controls.dt, initial)[4]
    final_energy = diagnostics(controls.steps, controls.dt, M)[4]
    mass_drift = abs(M[IJK_INDEX[(0,0,0)]] - initial[IJK_INDEX[(0,0,0)]])
    momentum_drift = maximum(abs(M[n] - initial[n]) for n in momentum_positions)
    energy_drift = abs(final_energy - initial_energy)

    @printf("Julia CHyQMOM-M6 samples written: %d\n", length(rows))
    @printf("Julia mass drift:       %.3e\n", mass_drift)
    @printf("Julia momentum drift:   %.3e\n", momentum_drift)
    @printf("Julia energy drift:     %.3e\n", energy_drift)
    @printf("Julia final realizable: %s\n", string(is_realizable(M)))

    mass_drift <= 1.0e-12 || error("mass conservation gate failed")
    momentum_drift <= 1.0e-12 || error("momentum conservation gate failed")
    energy_drift <= 1.0e-10 || error("energy conservation gate failed")
    is_realizable(M) || error("final state failed realizability gate")
end

main(ARGS)
