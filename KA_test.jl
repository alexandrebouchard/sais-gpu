using CUDA
using StatsFuns
using KernelAbstractions

const backend = CUDABackend()

@kernel function ais!(states, to)
    i = @index(Global)
    x = (a = 12, b = 15)
    y = [1, 2, 3]
    to[i] = states[i] + f(x)
    #state = @view states[:, i]
    #compute!(state)

end

f(tuple) = tuple.a + 2

function mycopy!(A, B)
    backend = get_backend(A)
    println(backend)
    @assert size(A) == size(B)
    @assert get_backend(B) == backend

    kernel = ais!(backend)
    kernel(A, B, ndrange=length(A))
end

A = KernelAbstractions.zeros(backend, Float64, 128, 128)
B = KernelAbstractions.ones(backend, Float64, 128, 128)
mycopy!(A, B)
KernelAbstractions.synchronize(backend)
println("ok")
#@assert A == B

# function propagate(initial_proposal!, particles)
#     i = threadIdx().x

    
#     initial_proposal!(@view particles[:, i])

#     return
# end

# function test!(particle)
#     randn!(particle)
#     return 
# end

# function main()
#     n_particles = 1000
#     n_dims = 10
#     particles = CuArray{Float32}(undef, n_dims, n_particles)
#     @cuda threads=n_particles propagate(test!, particles)
# end

# main();