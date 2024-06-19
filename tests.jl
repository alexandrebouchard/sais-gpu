using CUDA

include("SplitRandom.jl") 

@kernel function test!(rngs, out) 
    i = @index(Global)
    rng = rngs[i]
    out[i] = randn(rng, Float32)
end

N = 5


for backend in [CPU(), CUDABackend()]
    @show backend
    for i in 1:3

        rngs = SplitRandomArray(N; backend)
        out = KernelAbstractions.zeros(backend, Float32, N)

        test!(backend)(rngs, out, ndrange=size(out))
        KernelAbstractions.synchronize(backend)
        @show out
    end
end
