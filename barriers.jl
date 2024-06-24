include("utils.jl")

# idea: first write naive version 

# then matrix-based version 

# finally test on GPUs afterwards

# input for all these is the matrix of incremental weights 


function compute_log_g(log_increments, exponent::Int) 
    T, _ = size(log_increments) 
    log_weights = cumsum(log_increments, dims = 1)
    weight_log_norms = log_sum_exp(log_weights, dims = 2)

    @view(log_weights[2:T, :]) .= 
        @view(log_weights[1:(T-1), :]) .+ 
        @view(log_increments[2:T, :]) .* exponent 
    result = log_sum_exp(log_weights, dims = 2) 

    result .= result .- weight_log_norms
    return @show result
end

