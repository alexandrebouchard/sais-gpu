using CUDA
using Pigeons
include("SplitRandom.jl") 
include("mh.jl")
include("Particles.jl")
include("barriers.jl")
include("kernels.jl")

@auto struct AIS 
    particles 
    backend
    timing # for the kernel only
    full_timing # for the full function call
    schedule 
    intensity
    barriers 
end

Base.show(io::IO, a::AIS) = print(io, "AIS(backend=$(typeof(a.backend)), T=$(length(a.schedule)), N=$(n_particles(a.particles)), time=$(a.timing.time)s, ess=$(ess(a.particles)), lognorm=$(a.particles.log_normalization))")

struct FixedSchedule 
    n_points::Int 
end
ais(path, s::FixedSchedule; kwargs...) = ais(path, s.n_points; kwargs...)
ais(path, T::Int; kwargs...) = ais(path, collect(range(0, 1, length = T)); kwargs...)

function ais(
        path, schedule::AbstractVector; 
        backend::Backend = CPU(), 
        N::Int = backend isa CPU ? 2^10 : 2^14, 
        seed = 1,
        multi_threaded = true,
        compute_barriers = false, 
        explorer = default_explorer(path), 
        elt_type::Type{E} = Float64,
        show_report::Bool = true
        ) where {E}

    full_timing = @timed begin
    
        @assert multi_threaded || backend isa CPU
        
        T = length(schedule) 
        rngs = SplitRandomArray(N; backend, seed) 
        D = dimensionality(path)
        states = KernelAbstractions.zeros(backend, E, D, N)

        # initialization: iid sampling from reference
        iid_(backend, cpu_args(multi_threaded, N, backend)...)(rngs, path, states, ndrange = N) 
        KernelAbstractions.synchronize(backend)

        # parallel propagation 
        converted_schedule = Array{E}(schedule)
        betas = copy_to_device(converted_schedule, backend)
        buffers = KernelAbstractions.zeros(backend, E, buffer_size(explorer, path), N) 
        log_weights = KernelAbstractions.zeros(backend, E, N)
        log_increments = compute_barriers ? KernelAbstractions.zeros(backend, E, T, N) : nothing 
        propagate_and_weigh! = propagate_and_weigh_(backend, cpu_args(multi_threaded, N, backend)...)
        timing = @timed begin
            propagate_and_weigh!(rngs, path, explorer, states, buffers, log_weights, log_increments, betas, ndrange = N)
            KernelAbstractions.synchronize(backend)
            nothing
        end 

        particles = Particles(states, log_weights)
        intensity_vector = compute_barriers ? ensure_to_cpu(intensity(log_increments, backend)) : nothing 
        barriers = compute_barriers ? Pigeons.communication_barriers(intensity_vector, converted_schedule) : nothing
        nothing
    end
    return AIS(particles, backend, timing, full_timing, converted_schedule, intensity_vector, barriers)
end

# workaround counter intuitive behaviour of KA on CPUs
cpu_args(multi_threaded::Bool, N::Int, ::CPU) = multi_threaded ? 1 : N
# the above is not needed for GPUs
cpu_args(_, _, _) = ()