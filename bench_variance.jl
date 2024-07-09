include("bench_variance_utils.jl")


if Threads.nthreads() != 1
    error("Using several threads: $(Threads.nthreads())")
end

function run_experiments(; seeds, rounds, n_particles)
    result = DataFrame(
        time = Float64[],
        lognorm = Float64[],
        type = Symbol[],
        scheme = Symbol[],
        T = Int[],
        backend=Symbol[]
    )

    collected_seeds = collect(seeds)
    if 0 ∈ collected_seeds 
        push!(collected_seeds, 0)
    end

    for backend in backends()
        @show backend
        target = build_target(backend)
        for scheme_type in [SAIS, FixedSchedule, ZJA]
            @show scheme_type
            for n_rounds in rounds
                for seed in collected_seeds
                    s = scheme(scheme_type, n_rounds)
                    a = ais(target, s; 
                        seed, 
                        N = seed == 0 ? 1 : n_particles, 
                        backend, 
                        elt_type = Float64, 
                        show_report = false)
                    time = a.full_timing.time
                    lognorm = a.particles.log_normalization 
                    scheme_symbol = Symbol(string(s))
                    if seed > 0 # skip first: compile time..
                        push!(result, 
                            (; 
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

#result = run_experiments(; seeds = 1:10, rounds = 4:8, N = 2^14)
#CSV.write("bench_variance.csv", result; delim = ";")