@kwdef struct RWMH 
    n_passes::Int = 3
end

explore!(rng, explorer::RWMH, path, state, buffer, beta) = 
    for i in 1:explorer.n_passes
        mh!(rng, path, state, buffer, beta) 
    end


function mh!(rng, path, 
        state::AbstractVector{E}, 
        buffer::AbstractVector{E}, 
        beta::E, 
        ) where {E <: AbstractFloat}
    log_path_before = log_density(path, beta, state)

    # propose
    exponent = rand(rng, -1:1)
    sd = E(10)^exponent 
    for d in eachindex(state) 
        buffer[d] = state[d]
        state[d] += randn(rng, E) * sd 
    end

    log_path_proposed = log_density(path, beta, state) 

    if rand(rng) < exp(log_path_proposed - log_path_before) 
        # accept: nothing to do (proposed in place)
    else
        # reject
        for d in eachindex(state) 
            state[d] = buffer[d]
        end
    end
    return nothing
end