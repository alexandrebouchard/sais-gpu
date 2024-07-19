using DataFrames 
using CSV
using KernelAbstractions
using StatsFuns
using Adapt
include("utils.jl")
include("logistic_regression_data.jl")

struct LogisticRegression{S, T}
    n::Int
    p::Int 
    covariates::S # stored as n x p 
    observations::T
end 
Adapt.@adapt_structure LogisticRegression

function LogisticRegression(covariates::Matrix, observations::Vector; backend::Backend)
    n, p = size(covariates)
    @assert length(observations) == n
    labels = sort(unique(observations))
    @assert issubset(labels, [0, 1]) "Expected labels: [0, 1], got: $labels"
    if labels != [0, 1]
        @warn "Less than two labels encountered: $labels"
    end
    return LogisticRegression(n, p, copy_to_device(covariates, backend), copy_to_device(observations, backend))
end 

function LogisticRegression(; backend::Backend, preprocessing_options...) 
    covariates, observations = load_and_preprocess_data(; preprocessing_options...)
    return LogisticRegression(covariates, observations; backend)
end 

# order: params; caches; sum_sq; log_ref; log_ratio
dimensionality(lr::LogisticRegression) = lr.p + lr.n + 3

function log_reference(lr::LogisticRegression, state)
    p = lr.p
    n = lr.n 
    return state[n + p + 2]
end

function log_density_ratio(lr::LogisticRegression, state)
    p = lr.p
    n = lr.n 
    return state[n + p + 3]
end

function iid_sample!(rng, lr::LogisticRegression, state::AbstractVector{E}) where {E}
    for i in 1:lr.p 
        update!(lr, state, i, randn(rng, E))
    end
    return nothing
end

function update!(lr, state::AbstractVector{E}, entry::Int, new_value) where {E}
    @assert 1 ≤ entry ≤ lr.p
    p = lr.p
    n = lr.n 

    old_value = state[entry] 
    state[entry] = new_value
    
    # likelihood update
    log_likelihood = zero(E) 
    for i in 1:n
        idx = p + i
        old_cache = state[idx] 
        dot_product_delta = (new_value - old_value) * lr.covariates[i, entry]
        new_cache = old_cache + dot_product_delta 
        log_likelihood += (lr.observations[i] == 1 ? new_cache : zero(E)) - log1pexp(new_cache)
        state[idx] = new_cache
    end
    state[n + p + 3] = log_likelihood

    # prior update
    old_sqs = state[n + p + 1] 
    new_sqs = old_sqs - old_value^2 + new_value^2 
    log_prior = -new_sqs/E(2)
    state[n + p + 1] = new_sqs
    state[n + p + 2] = log_prior
    
    return nothing
end

@kwdef struct WithinGibbs 
    n_passes::Int = 1
    sd_exponents::UnitRange{Int} = (-1:0)
end 
default_explorer(::LogisticRegression) = WithinGibbs()

buffer_size(::WithinGibbs, _) = 0

function explore!(rng, 
            explorer::WithinGibbs, 
            path, 
            state::AbstractVector{E}, 
            _, 
            beta::E) where {E} 

    for _ in 1:explorer.n_passes
        for i in 1:path.p
            for exponent in explorer.sd_exponents
                sd = E(10)^exponent
                mh_within_gibbs!(rng, path, state, i, beta, sd) 
            end
        end
    end
end

function mh_within_gibbs!(rng, path, 
    state, 
    entry::Int, 
    beta::E, 
    sd::E
    ) where {E <: AbstractFloat}

    log_path_before = log_density(path, beta, state)

    # propose
    state_before = state[entry] 
    proposal = state_before + randn(rng, E) * sd 
    update!(path, state, entry, proposal)
    log_path_proposed = log_density(path, beta, state) 

    if rand(rng) < exp(log_path_proposed - log_path_before) 
        # accept: nothing to do (proposed in place)
    else
        # reject
        update!(path, state, entry, state_before)
    end
    return nothing
end
