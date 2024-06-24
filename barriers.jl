include("utils.jl")

# idea: first write naive version 

# then matrix-based version 

# finally test on GPUs afterwards

# input for all these is the matrix of incremental weights 


function compute_log_g(log_increments, exponent::Int) 
    T, _ = size(log_increments) 
    log_weights = cumsum(log_increments, dims = 1)

    log_sum = 
        @view(log_weights[1:(T-1), :]) + 
        @view(log_increments[2:T, :]) .* exponent 
    result = log_sum_exp(log_sum, dims = 2) 

    weight_log_norms = log_sum_exp(log_weights, dims = 2)
    result .= result .- @view(weight_log_norms[1:(T-1), :])

    return result
end

