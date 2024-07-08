include("bench_variance_utils.jl")


if Threads.nthreads() != 1
    @warn "Using several threads: $(Threads.nthreads())"
end

function run_experiments(; n_repeats, rounds, N)
    result = DataFrame(
        time = Float64[],
        lognorm = Float64[],
        type = Symbol[],
        scheme = Symbol[],
        T = Int[],
        backend=Symbol[]
    )

    for backend in backends()
        @show backend
        target = build_target(backend)
        for scheme_type in [SAIS, FixedSchedule, ZJA]
            @show scheme_type
            for n_rounds in rounds
                for seed in 1:(n_repeats+1)
                    s = scheme(scheme_type, n_rounds)
                    a = ais(target, s; seed, N, backend, elt_type = Float64, show_report = false)
                    time = a.full_timing.time
                    lognorm = a.particles.log_normalization 
                    scheme_symbol = Symbol(string(s))
                    if seed > 1 # skip first: compile time..
                        push!(result, (; 
                            time, 
                            lognorm,
                            type = Symbol(scheme_type), 
                            scheme = scheme_symbol,
                            T = length(a.schedule),
                            backend = backend_label(backend),
                            )
                        )
                    end
                end
            end
        end
    end

    return result
end

result = run_experiments(; n_repeats = 10, rounds = 4:8, N = 2^14)
CSV.write("bench_variance.csv", result; delim = ";")