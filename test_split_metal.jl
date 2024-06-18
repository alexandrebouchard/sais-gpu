using Metal 

include("SplitRandom.jl")

using KernelAbstractions
using BenchmarkTools

@kernel function test_rng(rngs, seeds, out)
    i = @index(Global)
    rng = SplitRandom(seeds)
    out[i] = randn(rng, Float32)
end

f(x) = 2x

N = 2
gpu = true
rngs = SplitRandomArray(N; gpu)


out = gpu ? Metal.ones(N) : ones(N)
seeds = (gpu ? MtlArray : Array){UInt64}([42, 43]) 

@btime begin 
    backend = get_backend(out)
    test_rng(backend)(rngs, seeds, out, ndrange=size(out))
    KernelAbstractions.synchronize(backend)
end