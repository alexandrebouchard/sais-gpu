include("ais.jl")
include("toy_normal.jl")

function test_repro()

    N = 500
    T = 500 # with 5000/5000 no longer pass (slight differences, probably guard digits? see: https://discourse.julialang.org/t/different-results-when-running-on-cpu-or-gpu/42200/8)
    
    results = map(backends) do backend
        p = ais(NormalPath(2), T; N, backend).particles
        ensure_to_cpu(p)
    end
    for result in results 
        @assert result.probabilities ≈ results[1].probabilities
        @assert result.states ≈ results[1].states
        @assert result.log_normalization ≈ results[1].log_normalization
    end
end
test_repro()

function test_moments(; kwargs...)
    ndims = 2 
    μ = ais(NormalPath(ndims), 500; kwargs...).particles 

    # first moment check 
    m = ∫(x -> x, μ)
    @assert all(x -> isapprox(x, 1; atol = 0.1), m)

    # second moment check 
    m = ∫(x -> (x .- 1).^2, μ)
    @assert all(x -> isapprox(x, 4; atol = 0.1), m)
end
test_moments(N = 100000, elt_type = Float64) 
test_moments(N = 100000, elt_type = Float32)


function test_log_norm(; kwargs...)
    ndims = 1
    μ = ais(NormalPath(ndims), 500; kwargs...).particles
    @assert isapprox(μ.log_normalization, log(2); atol = 0.02)
end
test_log_norm(N = 1000, elt_type = Float64) 
test_log_norm(N = 1000, elt_type = Float32)