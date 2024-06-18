
using LinearAlgebra 

function ais!() 

    # initialize: t = 1;  log_weights_t = 0, sample iid from pi0 


    # loop: t = 2 ... T 

        # in general, the weight fct might depend on x_t 
        # however, with the standard choice of backward/forward kernels
        # it does not, so we do the weighing first to accommodate 
        # Zhou et al adaptive method
        # w_t = (gamma_t / gamma_{t-1})(x_{t-1})
        # logw_t = (loggamma_t - loggamma_{t-1})(x_{t-1})

        # x_t ~ M_{t-1, t}(x_{t-1}, .), where M_{t-1, t} is pi_t invariant

end

function rwmh_move!(state, path, beta)
    log_gamma_before = log_gamma(path, beta, state)
    delta = randn()
    state .= state .+ delta 
    log_gamma_after = log_gamma(path, beta, state) 
    if rand() < exp(log_gamma_after - log_gamma_before)
        # accept, nothing to do since working in place 
    else
        state .= state .- delta 
    end
    return nothing
end

struct StdNormal end 

log_gamma(::StdNormal, beta, x) = - norm(x) / 2 TODO: use beta..
