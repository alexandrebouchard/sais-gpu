N = 2^20

using CUDA 
using BenchmarkTools 

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

@btime add_broadcast!($y_d, $x_d)