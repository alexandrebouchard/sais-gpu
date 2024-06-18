using CUDA

include("responsibilities.jl")

function ais_steps!(states, weights)
    for i in responsibilities(weights)
        states[i] = weights[i]
    end
end




function ais(; target, reference, use_gpu::Bool)

    # init states with reference and map construct

    # call 


end


## temp


N = 2^20

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0



function bench!(y, x)
    kernel = @cuda launch=false ais_steps!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end

using BenchmarkTools
@btime bench!($y_d, $x_d)