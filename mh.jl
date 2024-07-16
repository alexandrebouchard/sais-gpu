@kwdef struct RWMH
    n_passes::Int = 1
    sd_exponents::UnitRange{Int} = (-1:1)
end

buffer_size(::RWMH, path) = dimensionality(path)

explore!(rng, 
            explorer::RWMH, 
            path, 
            state::AbstractVector{E}, 
            buffer::AbstractVector{E}, 
            beta::E) where {E} = 
    for i in 1:explorer.n_passes
        for exponent in explorer.sd_exponents
            sd = E(10)^exponent
            mh!(rng, path, state, buffer, beta, sd) 
        end
    end


function mh!(rng, path, 
        state::AbstractVector{E}, 
        buffer::AbstractVector{E}, 
        beta::E, 
        sd::E
        ) where {E <: AbstractFloat}

    log_path_before = log_density(path, beta, state)

    # propose
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