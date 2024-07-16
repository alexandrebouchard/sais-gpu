using DataFrames 
using CSV
using KernelAbstractions
using Statistics
using StatsFuns
using Adapt
include("utils.jl")

struct LogisticRegression{S, T}
    n::Int
    p::Int 
    covariates::S # stored as n x p 
    observations::T
end 
Adapt.@adapt_structure LogisticRegression

function LogisticRegression(covariates::Matrix, observations::Vector; 
        backend::Backend, add_ones = true, standardize = true)

    if standardize 
        foreach(c -> c .= (c .- mean(c)) ./ std(c), eachcol(covariates))
    end
    if add_ones 
        covariates = hcat(covariates, ones(eltype(covariates), size(covariates)[1]))
    end
    n, p = size(covariates)
    @assert length(observations) == n
    return LogisticRegression(n, p, copy_to_device(covariates, backend), copy_to_device(observations, backend))
end 

function logistic_regression(; file = "data/data_banknote_authentication.txt", lr_args...)
    df = DataFrame(CSV.File(file)) 
    observations = df[:, end] 
    covariates = Matrix(df[:, 1:end-1])
    return LogisticRegression(covariates, observations; lr_args...)
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
    result = zero(E) 

    # likelihood update
    for i in 1:n
        idx = p + i
        old_cache = state[idx] 
        dot_product_delta = (new_value - old_value) * lr.covariates[i, entry]
        new_cache = old_cache + dot_product_delta 
        current_pr = log1pexp(new_cache)
        result += lr.observations[i] == 1 ? current_pr : 1 - current_pr 
        state[idx] = new_cache
    end
    state[n + p + 3] = result

    # prior update
    old_sqs = state[n + p + 1] 
    new_sqs = old_sqs - old_value^2 + new_value^2 
    log_prior = -new_sqs/2
    result += log_prior
    state[n + p + 1] = new_sqs
    state[n + p + 2] = log_prior
    
    return result
end

@kwdef struct WithinGibbs 
    n_passes::Int = 1
    sd_exponents::UnitRange{Int} = (-1:1)
end 

buffer_size(::WithinGibbs, _) = 0

function explore!(rng, 
            explorer::WithinGibbs, 
            path, 
            state::AbstractVector{E}, 
            _, 
            beta::E) where {E} 

    for i in 1:explorer.n_passes
        for i in 1:lr.p
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
    update!(lr, state, entry, proposal)
    log_path_proposed = log_density(path, beta, state) 

    if rand(rng) < exp(log_path_proposed - log_path_before) 
        # accept: nothing to do (proposed in place)
    else
        # reject
        update!(lr, state, entry, state_before)
    end
    return nothing
end


## quick test - remove m

include("sais.jl")

backend = CUDABackend()

target = logistic_regression(; backend)

ais(target; backend)