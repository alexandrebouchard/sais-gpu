using CUDA
include("SplitRandom.jl") 
include("mh.jl")
include("Particles.jl")

# D = dim 
# N = num particles 
# T = num annealing params

@kernel function iid_(rngs, @Const(path), 
        states          # D x N
        ) 
    i = @index(Global)  # ∈ 1 .. N 
    rng = rngs[i]
    state = @view states[:, i]
    iid_sample!(rng, path, state)
end

@kernel function propagate_and_weigh_(rngs, path, 
        states,         # D x N
        buffers,        # D x N
        log_weights,    # N 
        @Const(betas)   # T
        )   
    i = @index(Global)  # ∈ 1 .. N 
    rng = rngs[i]
    T = length(betas) 
    for t in 2:T 
        state = @view states[:, i]
        buffer = @view buffers[:, i]
        log_weights[i] += log_density(path, betas[t], state) - log_density(path, betas[t-1], state)
        mh!(rng, path, state, buffer, betas[t]) 
    end
end 

function ais(path; 
        T::Int, 
        N::Int, 
        backend::Backend = CPU(), 
        seed::Int = 1,
        n_workers = nothing, 
        elt_type::Type{E} = Float32
        ) where {E}
    rngs = SplitRandomArray(N; backend, seed) 
    D = dimensionality(path)
    states = KernelAbstractions.zeros(backend, E, D, N)

    # initialization: iid sampling from reference
    iid_kernel = isnothing(n_workers) ? iid_(backend) : iid_(backend, n_workers)
    iid_kernel(rngs, path, states, ndrange = N) 
    KernelAbstractions.synchronize(backend)

    # parallel propagation 
    betas = copy_to_device(range(zero(E), stop=one(E), length=T), backend)
    buffers = KernelAbstractions.zeros(backend, E, D, N) 
    log_weights = KernelAbstractions.zeros(backend, E, N) 
    prop_kernel = isnothing(n_workers) ? propagate_and_weigh_(backend) : propagate_and_weigh_(backend, n_workers)
    timing = @timed begin
        prop_kernel(rngs, path, states, buffers, log_weights, betas, ndrange = N)
        KernelAbstractions.synchronize(backend)
    end 
    println("Ran T=$T, N=$N in $(timing.time) sec [$(timing.bytes) bytes allocated]")

    return Particles(states, log_weights), timing
end

