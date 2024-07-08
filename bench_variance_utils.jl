include("sais.jl")
include("zja.jl")
include("simple_mixture.jl")
include("toy_unid.jl")
using DataFrames 
using CSV

build_target(backend) = Unid(10^11, 10^10)
    #SimpleMixture(backend)

function reference_run() 
    backend = gpu_available() ? CUDABackend() : CPU() 
    target = build_target(backend) 
    return ais(target, SAIS(15); elt_type = Float64, seed = 1, backend)
end
#ref = reference_run() 

# For mixture:
# seed 1 => -782.1570317771477
# seed 2 => -781.9446689097091

# For unid: ( 5.24e+05   1.02e+03)
# seed 1 => -24.6 
# seed 2 => -24.4
ref_log_Z = -24.5
Λ = 17.0


relative_z_hat(z_hat) = exp(z_hat - ref_log_Z)
rmse(x) = sqrt(sum(x .- ref_log_Z).^2)

scheme(::Type{SAIS}, n_rounds) = SAIS(n_rounds)
scheme(::Type{FixedSchedule}, n_rounds) = FixedSchedule(2^(n_rounds-1)) 
scheme(::Type{ZJA}, n_rounds) = ZJA((Λ / 2^(n_rounds-1))^2)