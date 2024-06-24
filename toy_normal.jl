# A path between N(0, 1) and N(0, sd = 2)
struct NormalPath
    dim::Int 
end

# reference is standard normal
function iid_sample!(rng, ::NormalPath, state::AbstractVector{E}) where {E}
    for d in eachindex(state)  
        state[d] = randn(rng, E)
    end
end

function log_density(::NormalPath, beta::E, state::AbstractVector{E}) where {E}
    sum = zero(E)
    scaling = inv(1 + beta)
    for d in eachindex(state) 
        sum += -(scaling * (beta - state[d]))^2 / 2
    end
    return sum
end

dimensionality(path::NormalPath) = path.dim
