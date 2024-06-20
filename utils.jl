"""
Normalize a vector of non-negative weights stored in 
log-scale. Returns the log normalization. 
Works on both CPU and GPU. 

I.e., given input `log_weights` (an array), 
perform in-place, numerically stable implementation of: 
```
prs = exp.(log_weights)
log_weights .= prs/sum(prs)
```

Moreover, returns a numerically stable version of 
```
log(sum(prs))
```
"""
function exp_normalize!(log_weights)
    m = maximum(log_weights)
    log_weights .= exp.(log_weights .- m) 
    return m + log(normalize!(log_weights))
end 

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end

