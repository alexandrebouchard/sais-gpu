include("barriers.jl")
include("ais.jl")
using CUDA 
using KernelAbstractions 


T = 5
N = 10

function check_cumsum()
    a = ais(NormalPath(2); T, N = 10, compute_increments = true) 
    probabilities = exp.(cumsum(a.log_increments, dims = 1)[end,:]) 
    probabilities .= probabilities ./ sum(probabilities)
    @assert vec(probabilities) ≈ vec(a.particles.probabilities)
end
check_cumsum()

function naive_log_g(log_increments, t, exponent::Int) # t ∈ {2, ..., T}
    T, N = size(log_increments)
    weights = exp.(cumsum(log_increments, dims = 1)) 
    increments = exp.(log_increments)[t, :]

    prev_weights = weights[t-1, :] 
    prev_probabilities = prev_weights  / sum(prev_weights)
    
    s = 0.0
    for i in 1:N 
        s += prev_probabilities[i] * increments[i]^exponent
    end
    return log(s)
end

function test_barriers(backend, exponent) 
    rng = SplittableRandom(1)
    array = rand(rng, Float32, T, N)

    @show log_increments = copy_to_device(array, backend)
    @show tested = ensure_to_cpu(compute_log_g(log_increments, exponent))

    naive = [naive_log_g(array, t, exponent) for t in 1:T]
    
    @assert vec(naive) ≈ vec(tested)
    @assert eltype(tested) == Float32
    return tested
end

for exponent in [1, 2]
    cpu = test_barriers(CPU(), exponent)
    gpu = test_barriers(CUDABackend(), exponent) 
    @assert cpu ≈ gpu
end


nothing