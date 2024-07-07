include("sais.jl")
include("zja.jl")
include("simple_mixture.jl")
include("toy_unid.jl")
using DataFrames 
using CairoMakie 
using AlgebraOfGraphics

build_target(backend) = Unid(10^11, 10^10)
    #SimpleMixture(backend)

function reference_run() 
    backend = gpu_available() ? CUDABackend() : CPU() 
    target = build_target(backend) 
    return ais(target, SAIS(15); elt_type = Float64, seed = 1)
end
#ref = reference_run() 

# For mixture:
# seed 1 => -782.1570317771477
# seed 2 => -781.9446689097091

# For unid: ( 5.24e+05   1.02e+03)
# seed 1 => -24.6 
# seed 2 => -24.4
ref_log_Z = -24.5
relative_z_hat(z_hat) = exp(z_hat - ref_log_Z)

scheme(::Type{SAIS}, n_rounds) = SAIS(n_rounds)
scheme(::Type{FixedSchedule}, n_rounds) = FixedSchedule(2^n_rounds) 
scheme(::Type{ZJA}, n_rounds) = ZJA((7.0 / 2^n_rounds)^2)

if Threads.nthreads() != 1
    @warn "Using several threads: $(Threads.nthreads())"
end

function run_experiments(; n_repeats, rounds, N)
    result = DataFrame(
        time = Float64[],
        lognorm = Float64[],
        relnorm = Float64[],
        type = Symbol[],
        scheme = Symbol[],
        T = Int[],
        backend=Symbol[]
    )

    for backend in backends()
        target = build_target(backend)
        for scheme_type in [SAIS, FixedSchedule, ZJA]
            for n_rounds in rounds
                for seed in 1:(n_repeats+1)
                    s = scheme(scheme_type, n_rounds)
                    GC.enable(false)
                    a = ais(target, s; seed, N, backend, elt_type = Float64, show_report = false)
                    GC.enable(true)
                    GC.gc()
                    time = a.full_timing.time
                    lognorm = a.particles.log_normalization 
                    scheme_symbol = Symbol(string(s))
                    if seed > 1 # skip first: compile time..
                        push!(result, (; 
                            time, 
                            lognorm,
                            relnorm = relative_z_hat(lognorm),
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

#result = run_experiments(; n_repeats = 10, rounds = 7:10, N = 2^10)

rmse(x) = sqrt(sum(x .- ref_log_Z).^2)

# function plot(result) 
#     result = subset(result, :type => t -> t .!= :FixedSchedule)
#     groups = groupby(result, [:T, :type, :backend])
#     transformed = combine(groups, :lognorm => rmse)
#     data(transformed) *
#         visual() * 
#         mapping(
#             :T, #:time, 
#             :lognorm_rmse, 
#             #marker = :scheme,
#             color = :type,
#             row = :backend,
#         )

# end

function plot(result) 
    result = subset(result, :type => t -> t .!= :FixedSchedule)
    data(result) * 
        visual() * 
        mapping(
            :T, #:time, 
            :lognorm, 
            #marker = :scheme,
            color = :type,
            row = :backend,
        )
end

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

