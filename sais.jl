include("ais.jl")

@auto struct SAIS 
    n_rounds
    initial_schedule   
end
SAIS(n_rounds = 10) = SAIS(n_rounds, range(0, 1, length = 2))

ais(path; kwargs...) = ais(path, SAIS(); kwargs...)
function ais(path, sais::SAIS; kwargs...) 
    kwargs = (compute_barriers = true, kwargs...)
    schedule = collect(sais.initial_schedule) 
    for r in 1:sais.n_rounds 
        a = ais(path, schedule; kwargs...)
        if r == sais.n_rounds 
            return a 
        else
            T = length(schedule) 
            schedule = Pigeons.optimal_schedule(a.intensity, schedule, 2T)
        end
    end
end