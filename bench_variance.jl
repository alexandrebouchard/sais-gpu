include("sais.jl")
include("simple_mixture.jl")
using DataFrames 
using CairoMakie 

function reference_run() 
    backend = gpu_available() ? CUDABackend() : CPU() 
    target = SimpleMixture(backend) 
    return ais(target, SAIS(15); elt_type = Float64, seed = 2)
end
#ref = reference_run() 

# seed 1 => -782.1570317771477
# seed 2 => -781.9446689097091

# Note: control n threads

# turn off GC

scheme(::Type{SAIS}, n_rounds) = SAIS(n_rounds)
scheme(::Type{Int}, n_rounds) = 2^n_rounds 
scheme(::Type{ZJA}, n_rounds) = ZJA((7.0 / 2^n_rounds)^2)

function run_experiments(; n_repeats = 100)
    result = DataFrame(
        time = Float64[],
        lognorm = Float64[],
    )

    for backend in backends()
        for scheme_type in [SAIS, Int, ZJA]
            for n_rounds in 1:5
                for seed in 1:n_repeats
                    a = ais(target, scheme(scheme_type, n_rounds); seed, backend)
                    time = a.full_timing.time
                    lognorm = a.particles.log_normalization 
                    push!(result, (; ))
                end
            end
        end
    end


    return result
end








# function test_speed(data_size) 
#     backend = gpu_available() ? CUDABackend() : CPU() 
#     array = randn(data_size)
#     target = SimpleMixture(array)
#     explorer = RWMH(3) 
#     state = zeros(5)
#     buffer = zeros(5)
#     rng = SplittableRandom(1)
#     iid_sample!(rng, target, state) 
#     log_weights = zeros(1)

#     for j in 1:2
#         @time for i in 1:16000 
#             explore!(rng, explorer, target, state, buffer, 1.0)
#             log_increment = weigh(target, explorer, 0.9, 1.0, state)
#             log_weights[1] += log_increment
#         end
#     end
# end
# test_speed(100)
# #test_speed(1000)
# #test_speed(10000)

