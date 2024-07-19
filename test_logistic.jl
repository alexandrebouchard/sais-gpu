include("logistic_regression.jl")

function _naive_recompute(lr, state)

    p = lr.p
    n = lr.n 

    log_prior = 0.0
    for i in 1:p 
        log_prior += -state[i]^2/2
    end

    log_like = 0.0
    for j in 1:n 
        dot_prod = 0.0 
        for i in 1:p 
            dot_prod += state[i] * lr.covariates[j, i]
        end
        log_like += (lr.observations[j] == 1 ? dot_prod : 0) - log1pexp(dot_prod) 
    end

    return log_prior, log_like
end

function check(target, state)
    naive_log_prior, naive_log_like = _naive_recompute(target, state) 
    @assert naive_log_prior ≈ log_reference(target, state) 
    @assert naive_log_like ≈ log_density_ratio(target, state)
end

backend = CPU() 
target = LogisticRegression(; backend, generated_dim = 200, generated_size = 50) 

rng = SplittableRandom(1) 
state = zeros(dimensionality(target))
iid_sample!(rng, target, state) 
check(target, state)

explorer = WithinGibbs() 
explore!(rng, explorer, target, state, nothing, 0.0) 
check(target, state)
explore!(rng, explorer, target, state, nothing, 0.2) 
check(target, state)