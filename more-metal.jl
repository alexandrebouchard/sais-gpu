using Metal 
using BenchmarkTools
using LinearAlgebra

N = 2^25

gpu_vector = Metal.ones(N)

# randn(Float32) does not work on Metal!
# Allocating does not work on Metal!

# next: try this with CUDA

function my_fct(x::T)::T where {T}
    #z = randn(T)
    #z = rand(T)
    return exp(x) + x 
end


println("On GPU (Metal)")
gpu_vector = Metal.ones(Float32, N)
function add_metal!(x)
    Metal.@sync sum(my_fct, x)
    return
end
@btime add_metal!(gpu_vector)

println("On CPU")
cpu_vector = ones(Float32, N) 
function add_cpu!(x)
    sum(my_fct, x)
    return
end
@btime add_cpu!(cpu_vector)
