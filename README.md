Julia code supporting the preprint 
"Optimized Annealed Sequential Monte Carlo Samplers", 2024, 
S. Syed, A. Bouchard-Côté, K Chern, A. Doucet. 

Code is not yet ready for general use as the manuscript is under review. 
An open source license will be added once the paper is accepted. 
Please contact us if you would like to use the software in the meantime. 


## Setup

Test with Julia 1.10.2 and CUDA runtime 12.5. 

To setup:

```
using Pkg
Pkg.instantiate() 
``` 

To run a battery of test cases:

```
include("tests.jl")
```

## Usage

### Defining the reference and target (path)

To see an example of how to setup target distributions, see 
`simple_mixture.jl`. The main idea is that a custom struct is 
defined to hold the dataset, and dispatches are defined to 
specify target and reference.

The functions to dispatch are:

- `iid_sample!(...)` to sample from the reference, 
- `log_reference(...)` to evaluate the log density of the reference, 
- `log_density_ratio(...)` to evaluate the log of the un-normalized 
    ratio between target and reference (e.g., in Bayesian models this 
    is just the likelihood), 
- `dimensionality(...)` to obtain the dimensionality of the latent space. 

After doing so, create a variable say `path` and assign to it 
an instance of your custom struct. For example:

```
include("simple_mixture.jl")
backend = CPU() # or: CUDABackend()
path = SimpleMixture(backend) # since it holds data array, it needs to know if will be in CPU or GPU
```

### Sampling 

The interface for sampling is `ais(...)`. The followoing methods are implemented:

- In `sais.jl`, we implement our SAIS (Sequential Annealed Importance Sampling) algorithm. 
- In `zha.jl`, we implement Zhou et al (2016).

For example:

```
include("sais.jl")
a = ais(path, SAIS(10);
    N = 100, # number of particles
    backend,
    seed = 1)
```

### Approximating expectation and estimating normalization constant

To approximate expectations:

```
∫(x -> x.^2, a.particles)
```

For the normalization constant:

```
a.particles.log_normalization
```

For ESS:

```
ess(a.particles)
```