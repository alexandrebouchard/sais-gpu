include("sais.jl") 
include("simple_mixture.jl")
include("barriers.jl")

T = 5
N = 10000

function test_mix_repro()
     outputs = map([CPU(), CUDABackend()]) do backend 
          @show backend
          target = SimpleMixture(backend)
          # warm-up
          ais(target, 2; N=1, backend)
          # actual sampling
          @time a = ais(target, T; N, backend)
          @show a.timing
          return ensure_to_cpu(a.particles)
     end
     @assert outputs[1].probabilities ≈ outputs[2].probabilities
     @assert outputs[1].states ≈ outputs[2].states
     return nothing
end 
if gpu_available
     test_mix_repro()
end
