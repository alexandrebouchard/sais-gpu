include("bench_variance_utils.jl")


if Threads.nthreads() != 1
    error("Using several threads: $(Threads.nthreads())")
end

function run_experiments(; seeds, rounds, n_particles, models = [Unid, SimpleMixture], scheme_types = [SAIS, ZJA])
    
    result = DataFrame(
        backend=Symbol[],
        model = String[],
        type = Symbol[],
        n_rounds = Int[],
        time = Float64[],
        lognorm = Float64[],
        relnorm = Float64[], # hatZ / referenceZ
        scheme = String[], # we need string otherwise does not get quoted properly
        T = Int[],
    )
    collected_seeds = collect(seeds)
    if !(0 ∈ collected_seeds) 
        push!(collected_seeds, 0)
    end
    sort!(collected_seeds)

    for model_type in models 
        @show model_type
        for backend in backends()
            @show backend
            target = build_target(backend, model_type)
            model_approx_gcb = approx_gcb(target)
            approx_log_Z = ref_log_Z(target)

            for scheme_type in scheme_types
                @show scheme_type
                for n_rounds in rounds
                    for seed in collected_seeds
                        @show seed
                        s = scheme(scheme_type, n_rounds, model_approx_gcb)
                        a = ais(target, s; 
                            seed, 
                            N = seed == 0 ? 1 : n_particles, 
                            backend, 
                            elt_type = Float64, 
                            show_report = false)
                        time = a.full_timing.time
                        lognorm = a.particles.log_normalization 
                        relnorm = exp(lognorm - approx_log_Z)
                        scheme_symbol = string(s)
                        if seed > 0 # skip first: compile time..
                            @show time 
                            push!(result, 
                                (; 
                                    n_rounds,
                                    model = string(model_type),
                                    time, 
                                    lognorm,
                                    relnorm,
                                    type = Symbol(scheme_type), 
                                    scheme = scheme_symbol,
                                    T = length(a.schedule),
                                    backend = backend_label(backend),
                                )
                            )
                        else
                            println("Skipped: $time")
                        end
                    end
                end
            end
        end
    end
    sort!(result)
    return result
end