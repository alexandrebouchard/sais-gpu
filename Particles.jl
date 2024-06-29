include("utils.jl")
import Pigeons: @auto
using Statistics 

@auto struct Particles
    states
    probabilities
    log_normalization
end

ess(p::Particles) = 1 / sum(x -> x^2, p.probabilities)

ensure_to_cpu(p::Particles) = 
    Particles(ensure_to_cpu(p.states), ensure_to_cpu(p.probabilities), p.log_normalization)

n_particles(p) = length(p.probabilities)

function Particles(particles::AbstractArray, log_weights::AbstractVector{E}) where {E}
    log_normalization = exp_normalize!(log_weights) - log(E(length(log_weights)))
    return Particles(particles, log_weights, log_normalization)
end

integrate(f::Function, p::Particles) =
    sum(1:n_particles(p)) do i 
        state = @view p.states[:, i] 
        f(state) * p.probabilities[i]
    end

∫ = integrate 

Statistics.mean(μ::Particles) = ∫(x -> x, μ) 
function Statistics.var(μ::Particles) 
    m = Statistics.mean(μ)
    return ∫(x -> (x - m).^2, μ)
end
Statistics.std(μ::Particles) = sqrt.(var(μ))