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


#=

Nice example of importance of adaptive schedule 
julia> ais(target; backend)
───────────────────────────────────────────────────────────────────────────────────────
    T          N        time(s)    %t in k    allc(B)     ess         Λ      log(Z₁/Z₀)
────────── ────────── ────────── ────────── ────────── ────────── ────────── ──────────
        2   1.64e+04    0.00212       0.06   2.78e+03       53.1       2.39      -12.2 
        4   1.64e+04    0.00177      0.113   2.78e+03       53.4       2.89      -12.2 
        8   1.64e+04    0.00206      0.202   2.78e+03       57.8       3.88      -12.2 
       16   1.64e+04    0.00261      0.368    2.8e+03       66.9        5.1      -12.3 
       32   1.64e+04    0.00352      0.553    2.8e+03        144       6.26      -12.1 
       64   1.64e+04    0.00554      0.719    2.8e+03        270       7.58      -12.3 
      128   1.64e+04    0.00988      0.795    2.8e+03        713       8.05      -12.2 
      256   1.64e+04     0.0185      0.834    2.8e+03    1.6e+03       8.21      -12.2 
      512   1.64e+04     0.0365      0.869    2.8e+03   3.29e+03       8.31      -12.2 
 1.02e+03   1.64e+04      0.069      0.879    2.8e+03   5.76e+03       8.36      -12.2 
───────────────────────────────────────────────────────────────────────────────────────
AIS(backend=CUDABackend, T=1024, N=16384, time=0.06069079s, ess=5756.2417, lognorm=-12.173146)

julia> ais(target, 2^10; backend)
AIS(backend=CUDABackend, T=1024, N=16384, time=0.048907059s, ess=331.65457, lognorm=-12.209261)


=#