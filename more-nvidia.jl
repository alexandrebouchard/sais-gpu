N = 2^25

using CUDA 
using BenchmarkTools 

println("On GPU (CUDA)")
gpu_vector = CUDA.ones(N)

function my_fct(x::T)::T where {T}
    z = randn(T)
    #z = rand(T)
    return exp(x) + x + z
end

function add_cuda!(x)
    CUDA.@sync sum(my_fct, x)
    return
end
@btime add_cuda!(gpu_vector)

println("On CPU")
cpu_vector = ones(Float32, N) 
function add_cpu!(x)
    sum(my_fct, x)
    return
end
@btime add_cpu!(cpu_vector)
