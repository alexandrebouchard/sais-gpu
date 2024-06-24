include("ais.jl") 
include("simple_mixture.jl")

T = 5
N = 10000

for backend in [CPU(), CUDABackend()]
     @show backend
     target = SimpleMixture(backend)
     # warm-up
     ais(target; T=2, N=1, backend)
     # actual sampling
     @time a = ais(target; T, N, backend)
     @show a.timing
end