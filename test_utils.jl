include("utils.jl")

using Random 

function test_vector(seed = 1) 
    rng = MersenneTwister(seed)
    return rand(rng, Float32, 100)
end

function test_alloc_type() 
    ws = test_vector()
    exp_normalize!(ws)
    t = @timed exp_normalize!(ws)

    @assert t.bytes ≤ 32
    @assert exp_normalize!(ws) isa Float32
end
test_alloc_type() 


function naive_exp_norm!(log_weights) 
    prs = exp.(log_weights)
    norm = sum(prs)
    log_weights .= prs/sum(prs)
    return log(norm)
end

function test_naive() 
    w1 = test_vector() 
    w2 = test_vector() 

    @assert exp_normalize!(w1) ≈ naive_exp_norm!(w2) 
    @assert w1 ≈ w2

end
test_naive()

function test_vectorized() 
    T = 2
    N = 100
    matrix = zeros(T, N)
    matrix[1, :] .= test_vector(1)
    matrix[2, :] .= test_vector(2) 

    log_norms = log_sum_exp(matrix, dims = 2) 

    w1 = test_vector(1) 
    w2 = test_vector(2)

    @assert log_norms[1] ≈ exp_normalize!(w1)
    @assert log_norms[2] ≈ exp_normalize!(w2)
end
test_vectorized()
