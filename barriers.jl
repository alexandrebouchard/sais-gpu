include("utils.jl")

function intensity(log_increments) 
    log_weights = cumsum(log_increments, dims = 1)
    g1 = compute_log_g(log_weights, log_increments, 1)
    g2 = compute_log_g(log_weights, log_increments, 2)
    return sqrt.(ensure_non_negative.(g2 .- 2 .* g1))
end


intensity(log_increments::AbstractArray{Float64}, _) = intensity(log_increments)
intensity(log_increments::AbstractArray{Float32}, backend) =
    # force to move to Float64, otherwise global barrier estimate 
    # too imprecise (detected in test_large_t.jl)
    intensity(copy_to_device(log_increments, backend, Float64))


function compute_log_g(log_weights, log_increments, exponent::Int) 
    T, _ = size(log_increments) 
    
    log_sum = 
        @view(log_weights[1:(T-1), :]) + 
        @view(log_increments[2:T, :]) .* exponent 
    result = log_sum_exp(log_sum, dims = 2) 

    weight_log_norms = log_sum_exp(log_weights, dims = 2)
    result .= result .- @view(weight_log_norms[1:(T-1), :])

    return vec(result)
end

# we have to do this because some slightly negative 
# (i.e., larger than -1e4) intensities do pop up 
# because of numerical error
ensure_non_negative(x) = max(x, 0)