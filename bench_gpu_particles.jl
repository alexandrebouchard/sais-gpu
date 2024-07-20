include("bench_variance_utils.jl")
using CairoMakie
using AlgebraOfGraphics

function run_bench(; n_rounds, seed, model_type, scheme_type, elt_type, max_particle_exponent = 21)
    result = DataFrame(
                N=Int[], 
                time=Float64[], 
                model = String[],
                type = Symbol[],
                backend=Symbol[],
                elt_type = String[],
                )

    for backend in backends()
        @show backend
        target = build_target(backend, model_type)
        model_approx_gcb = approx_gcb(target)
        s = scheme(scheme_type, n_rounds, model_approx_gcb)
                        
        # warm-up 
        a = ais(target, s; seed, N = 1, backend, elt_type, show_report = false)
        println("Warm up: $(a.full_timing.time)") 

        # actual
        for N in map(i -> 2^i, (0:max_particle_exponent))
            @show N
            a = ais(target, s; seed, N, backend, elt_type, show_report = false)
            push!(result, (; 
                N, 
                time = a.full_timing.time,
                model = string(model_type),
                type = Symbol(scheme_type),
                backend = backend_label(backend),
                elt_type = string(elt_type)
            ))
        end
    end
    return result
end

plot_gpu_particles(result) =
    data(result) * 
        visual(Lines) *
        mapping(
            :N => "Number of particles", 
            :time => "Wallclock time (s)", 
            color = :backend) 

function plot_gpu_particles(result, to_file) 
    p = plot_gpu_particles(result) 
    axis = (width = 225, height = 225, xscale = log2, yscale = log2)
    fg = draw(p; axis)
    save(to_file, fg)
end

