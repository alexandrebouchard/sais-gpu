include("ais.jl")
include("toy_normal.jl")

backend = CUDABackend()
a = AIS(; backend)

N = 500
T = 500

μ = ais(a, NormalPath(1); N, T, backend)


#∫(x -> x, μ)