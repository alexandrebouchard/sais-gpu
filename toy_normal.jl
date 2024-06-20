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

N = 100
T = 500000

for backend in [CPU()]
    a = AIS(; backend)
    @show backend 
    for i in 1:3
        @show i 
        ais(a, NormalPath(2); N, T, backend)
    end
end

#=
Proof of concept timing:

N = 1000
T = 500000

backend = CPU(false)
 74.247462 seconds (149.30 k allocations: 11.519 MiB, 0.03% gc time, 0.39% compilation time: 61% of which was recompilation)
backend = CUDABackend(false, false)
  3.014277 seconds (295.61 k allocations: 22.513 MiB, 6.90% compilation time: 22% of which was recompilation)=#