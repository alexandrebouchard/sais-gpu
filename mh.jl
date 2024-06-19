function mh!(rng, path, 
        state::AbstractVector{E}, 
        buffer::AbstractVector{E}, 
        beta::T) where {E}
    log_path_before = log_density(path, beta, state)

    # propose
    for d in eachindex(state) 
        buffer[d] = state[d]
        state[d] = randn(rng, T)
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