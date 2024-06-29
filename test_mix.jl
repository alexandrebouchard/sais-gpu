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
     a = ais(target; backend, elt_type = Float64)
     μ = ensure_to_cpu(a.particles)
     # compare against Blang's results
     @show m = mean(μ)
     @show s = std(μ)
     for i in [1, 2]
          @assert isapprox(m[i], 145, atol=10) 
          @assert isapprox(s[i], 30, atol=10)
     end
     for i in [3, 4]
          @assert isapprox(m[i], 30, atol=3) 
          @assert isapprox(s[i], 8, atol=2)
     end
     @assert isapprox(m[5], 0.5, atol=0.05) 
     @assert isapprox(s[5], 0.2, atol=0.05) 
     return μ 
end
if gpu_available()
     test_mix_repro()
end
