# D = dim 
# N = num particles 
# T = num annealing params

@kernel function iid_(
        rngs, @Const(path), 
        states          # D x N
        )

    n = @index(Global)  # ∈ 1 .. N 
    rng = rngs[n]
    state = @view states[:, n]
    iid_sample!(rng, path, state)
end

@kernel function propagate_and_weigh_(
        rngs, @Const(path), @Const(explorer), 
        states,         # D x N
        buffers,        # D x N
        log_weights,    # N 
        log_increments, # T x N of nothing
        @Const(betas)   # T
        ) 

    n = @index(Global)  # ∈ 1 .. N 
    rng = rngs[n]
    T = length(betas) 
    for t in 2:T 
        state = @view states[:, n]
        buffer = @view buffers[:, n]
        log_increment = weigh(path, explorer, betas[t-1], betas[t], state)
        update_log_increment!(log_increments, t, n, log_increment)
        log_weights[n] += log_increment
        explore!(rng, explorer, path, state, buffer, betas[t])
    end
end 

# Default weighing scheme: linear path 
log_density(path, beta, state) = log_reference(path, state) + beta * log_density_ratio(path, state) 
weigh(path, _, prev_beta, cur_beta, state) = (cur_beta - prev_beta) *  log_density_ratio(path, state) 

# Only record those when adaptation needs it
function update_log_increment!(log_increments, t, n, log_increment) 
    log_increments[t, n] = log_increment
end
update_log_increment!(::Nothing, _, _, _) = nothing