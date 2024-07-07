include("sais.jl")
include("zja.jl")
include("simple_mixture.jl")
using DataFrames 
using CairoMakie 
using AlgebraOfGraphics

function reference_run() 
    backend = gpu_available() ? CUDABackend() : CPU() 
    target = SimpleMixture(backend) 
    return ais(target, SAIS(15); elt_type = Float64, seed = 2)
end
#ref = reference_run() 

# seed 1 => -782.1570317771477
# seed 2 => -781.9446689097091



scheme(::Type{SAIS}, n_rounds) = SAIS(n_rounds)
scheme(::Type{FixedSchedule}, n_rounds) = FixedSchedule(2^n_rounds) 
scheme(::Type{ZJA}, n_rounds) = ZJA((7.0 / 2^n_rounds)^2)

@assert Threads.nthreads() == 1

function run_experiments(; n_repeats = 100, max_rounds = 8, N = 2^14)
    result = DataFrame(
        time = Float64[],
        lognorm = Float64[],
        type = Symbol[],
        scheme = Symbol[],
        T = Int[],
        backend=Symbol[]
    )

    for backend in [CUDABackend()] #backends()
        target = SimpleMixture(backend)
        for scheme_type in [SAIS, FixedSchedule, ZJA]
            for n_rounds in 3:max_rounds
                for seed in 1:(n_repeats+1)
                    s = scheme(scheme_type, n_rounds)
                    #GC.enable(false)
                    a = ais(target, s; seed, backend, show_report = false)
                    #GC.enable(true)
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

result = run_experiments(; n_repeats = 2)

plot(result) = 
    data(result) * 
    visual() * 
    mapping(
        :T, #:time, 
        :lognorm, 
        #marker = :scheme,
        color = :type,
        row = :backend,
    )

draw(plot(result))


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

