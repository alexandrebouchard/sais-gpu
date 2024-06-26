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
test_mix_repro()

function test_mix_moments() 
     backend = CUDABackend()
     target = SimpleMixture(backend) 
     a = ais(target; backend)
     return a 
end
test_mix_moments() 

# function test_mix_barrier() 
#      T = 200
#      N = 10000
#      backend = CUDABackend() 
#      target = SimpleMixture(backend)
#      a = ais(target, T; N, backend, compute_barriers = true)
#      betas = range(0, 1, length=T)  
#      barriers = a.barriers 
#      return lines(0..1, x -> barriers.localbarrier(x))
# end

# using CairoMakie 
# p = test_mix_barrier() 
# save("test_mix_barrier.png", p)