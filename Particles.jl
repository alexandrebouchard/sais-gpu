include("utils.jl")

struct Particles{P,W,L}
    particles::P 
    probabilities::W 
    log_normalization::L 
end

n_particles(p) = length(p.particles)

function Particles(particles::AbstractArray, log_weights::AbstractVector)
    log_normalization = exp_normalize!(log_weights) 
    return Particles(particles, log_weights, log_normalization)
end

integrate(f::Function, p::Particles) =
    sum(1:n_particles) do i 
        state = @view p.particles[:, i] 
        f(state) * p.probabilities[i]
    end

# intuitive shortcut
∫ = integrate 