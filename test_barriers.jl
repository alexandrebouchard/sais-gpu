using SplittableRandoms 
include("barriers.jl")

rng = SplittableRandom(1) 

T = 2 
N = 1

log_increments = rand(rng, T, N) 

# TODO: check last cumsum is the same as the weights?


function naive_log_g(log_increments, t, exponent::Int) # t ∈ {2, ..., T}
    T, N = size(log_increments)
    weights = exp.(cumsum(log_increments, dims = 1)) 
    increments = exp.(log_increments)[t, :]

    prev_weights = weights[t-1, :] 
    prev_probabilities = prev_weights  / sum(prev_weights)
    
    s = 0.0
    for i in 1:N 
        s += prev_probabilities[i] * increments[i]^exponent
    end
    return log(s)
end

exponent = 2

@show naive_log_g(log_increments, 2, exponent)


@show tested = compute_log_g(log_increments, exponent)




nothing