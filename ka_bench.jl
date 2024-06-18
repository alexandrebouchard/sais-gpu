N = 1024

### performance for KA

using KernelAbstractions 
using BenchmarkTools

@kernel function test(A)
    I = @index(Global)
    for j in 1:10
        A[I] = exp(A[I])
    end
end

A = ones(N, N)
@btime begin 
    backend = get_backend(A)
    test(backend)(A, ndrange=size(A))
    synchronize(dev)
end


# using CUDA: CuArray
using CUDA
A = CuArray(ones(N, N))
@btime begin 
    backend = get_backend(A)
    test(backend)(A, ndrange=size(A))
    synchronize(dev)
end