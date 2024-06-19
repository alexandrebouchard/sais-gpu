include("ais.jl")


struct NormalPath
    dim::Int 
end

# reference is standard normal
function iid_sample!(rng, path::NormalPath, state::AbstractVector{E}) where {E}
    for d in eachindex(state)  
        state[d] = randn(rng, E)
    end
end

function log_density(path::NormalPath, beta::E, state::AbstractVector{E}) where {E}
    sum = zero(E)
    for d in eachindex(state) 
        sum += -(beta - state[d])^2 / 2
    end
    return sum
end

dimensionality(path::NormalPath) = path.dim

N = 50000
T = 10000
for backend in [CPU(), CUDABackend()]
    @show backend 
    @time ais(NormalPath(2), T, N; backend)
end

#=
Proof of concept timing:

N = 50000
T = 10000

backend = CPU(false)
 78.033562 seconds (197.33 k allocations: 14.095 MiB, 0.35% compilation time: 64% of which was recompilation)
backend = CUDABackend(false, false)
  1.734952 seconds (344.06 k allocations: 25.049 MiB, 43.55% gc time, 11.53% compilation time: 25% of which was recompilation)
=#