using CUDA
using Roots
include("ais.jl")

struct ZJA 
    divergence::Float64
end

@kernel function log_ratios_(
        @Const(path),
        states,         # D x N
        output         # N
        ) 

    n = @index(Global)  # ∈ 1 .. N 
    state = @view states[:, n]
    output[n] = log_density_ratio(path, state)
end 

@kernel function propagate_(
        rngs, @Const(path), @Const(explorer), 
        states,         # D x N
        buffers,        # D x N
        beta
        ) 

    n = @index(Global)  # ∈ 1 .. N 
    rng = rngs[n]
    state = @view states[:, n]
    buffer = @view buffers[:, n]
    explore!(rng, explorer, path, state, buffer, beta)
end 


function ais(
        path, schedule_optimizer::ZJA; 
        backend::Backend = CPU(), 
        N::Int = backend isa CPU ? 2^10 : 2^14, 
        seed = 1,
        multi_threaded = true,
        explorer = RWMH(), 
        elt_type::Type{E} = Float32, 
        show_report::Bool = true
        ) where {E}

    full_timing = @timed begin

        rngs = SplitRandomArray(N; backend, seed) 
        D = dimensionality(path)
        states = KernelAbstractions.zeros(backend, E, D, N)

        # initialization: iid sampling from reference
        iid_(backend, cpu_args(multi_threaded, N, backend)...)(rngs, path, states, ndrange = N) 
        KernelAbstractions.synchronize(backend)

        schedule = Float64[0]
        buffers = KernelAbstractions.zeros(backend, E, D, N) 
        log_weights = KernelAbstractions.zeros(backend, E, N)
        log_ratios =  KernelAbstractions.zeros(backend, E, N)

        log_ratios! = log_ratios_(backend, cpu_args(multi_threaded, N, backend)...)
        propagate! = propagate_(backend, cpu_args(multi_threaded, N, backend)...)

        timing = @timed begin
            while schedule[end] < 1
                log_ratios!(path, states, log_ratios, ndrange = N)
                KernelAbstractions.synchronize(backend)

                # find the delta 
                max = 1 - schedule[end]
                delta = find_delta(log_weights, log_ratios, schedule_optimizer.divergence, max)
                next_beta = schedule[end] + delta 
                push!(schedule, next_beta)

                # update weights and perform MCMC simultaneously
                propagate!(rngs, path, explorer, states, buffers, E(next_beta), ndrange = N)
                log_weights .= log_weights .+ delta .* log_ratios
                KernelAbstractions.synchronize(backend)
            end
            nothing
        end
        particles = Particles(states, log_weights)
        nothing
    end
    return AIS(particles, backend, timing, full_timing, schedule, nothing, nothing)
end

function find_delta(log_weights, log_ratios, divergence, max) 
    f(x) = objective(log_weights, log_ratios, x) - divergence
    if f(max) ≤ 0
        return max
    else
        find_zero(f, (0, max))
    end
end

objective(log_weights, log_ratios, delta) =
    compute_log_g(log_weights, log_ratios, delta, 2) - 2 * 
    compute_log_g(log_weights, log_ratios, delta, 1)

function compute_log_g(log_weights, log_ratios, delta, exponent::Int)
    log_sum = log_weights .+ (exponent * delta) .* log_ratios 
    return logsumexp(log_sum) .- logsumexp(log_weights)
end

