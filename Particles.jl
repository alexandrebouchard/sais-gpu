include("utils.jl")
import Pigeons: @auto

@auto struct Particles
    states
    probabilities
    log_normalization
end

ensure_to_cpu(p::Particles) = 
    Particles(ensure_to_cpu(p.states), ensure_to_cpu(p.probabilities), p.log_normalization)

n_particles(p) = length(p.probabilities)

function Particles(particles::AbstractArray, log_weights::AbstractVector)
    log_normalization = exp_normalize!(log_weights) - log(length(log_weights))
    return Particles(particles, log_weights, log_normalization)
end

integrate(f::Function, p::Particles) =
    sum(1:n_particles(p)) do i 
        state = @view p.states[:, i] 
        f(state) * p.probabilities[i]
    end

∫ = integrate 

# function integrate(f::Function, p::Particles{S,P,L,D}) where {D <: KernelAbstractions.GPU} 
#     # mapslice currently gives 'Invocation of getindex' so need to write our own
#     out = out_buffer(f, p)
#     TODO: save compiled kernel somewhere? 
#           once it's a package, do it when package loads + Extension for CUDA/etc
#     sum(out)
# end 

# @kernel integrate_(f, states, probabilities, out)
#     i = @index(Global) 
#     state = @view states[:, i]
#     out[i] = f(state) * probabilities[i]
# end

# function out_buffer(f::Function, p::Particle) 
#     D, N = size(p.states) 
#     state = @view p.states[:, 1]
#     proto = f(copy_to_cpu(state)) 
#     return KernelAbstractions.zeros(backend, eltype(proto), length(proto), N)
# end

    

# intuitive shortcut
