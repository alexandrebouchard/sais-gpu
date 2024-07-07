include("sais.jl")
include("toy_unid.jl")
using StatsFuns

T = 50
N = 10000

function test_unid() 
    outputs = map(backends()) do backend 
        @show backend
        target = Unid(100, 50)
        # warm-up
        ais(target, 2; N=1, backend)
        # actual sampling
        @time a = ais(target, T; N, backend)
        @show a.timing
        return ensure_to_cpu(a.particles)
   end
   if gpu_available()
        @assert outputs[1].probabilities ≈ outputs[2].probabilities
        @assert outputs[1].states ≈ outputs[2].states
   end
   return nothing
end 
test_unid() 