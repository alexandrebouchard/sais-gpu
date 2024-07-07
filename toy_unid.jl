
struct Unid 
    n_trials::Int
    n_successes::Int
end

function iid_sample!(rng, ::Unid, state::AbstractVector{E}) where {E}
    for d in 1:2  
        state[d] = rand(rng, E)
    end
end

function log_reference(::Unid, state::AbstractVector{E}) where {E}
    p1, p2 = state
    if !(0 < p1 < 1) || !(0 < p2 < 1)
        return -Pigeons.inf(E)
    else 
        return zero(E)
    end
end

function log_density_ratio(target::Unid, state::AbstractVector{E}) where {E}
    p1, p2 = state
    if !isfinite(log_reference(target, state)) 
        return -Pigeons.inf(E)
    end
    p = p1 * p2
    return StatsFuns.binomlogpdf(target.n_trials, p, target.n_successes)
end

dimensionality(::Unid) = 2