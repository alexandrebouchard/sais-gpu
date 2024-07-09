include { crossProduct; collectCSVs; deliverables } from './utils.nf'

params.dryRun = false

def julia_depot_dir = file("/home/alexbou/st-alexbou-1/abc/depot")
def toml_files = file("../*.toml")
def jl_files = file("../*.jl")
def deliv = deliverables(workflow)

def variables = [
    seed: (1..100),
]

workflow  {
    args = crossProduct(variables, params.dryRun)
    results = run_experiment(julia_depot_dir, toml_files, jl_files, args, params.dryRun) // | collectCSVs

}

process run_experiment {
    debug true
    time 5.min
    scratch true 
    clusterOptions '--nodes 1', '--account st-alexbou-1-gpu', '--gpus 1'
    input:
        env JULIA_DEPOT_PATH
        path toml_files
        path jl_files
        val arg
        val dryRun
    output:
        tuple val(arg), path('csvs')
    """
    #!/usr/bin/env -S julia --project=@.

    include(pwd() * "/bench_variance.jl")
    result = run_experiments(; seeds = ${arg.seed}:${arg.seed}, rounds = ${dryRun ? "4:4" : "4:8"}, n_particles = ${dryRun ? "1" : "2^14"})
    
    mkdir("csvs")
    CSV.write("csvs/bench_variance.csv", result; delim = ";")
    """
}

/*
process plot {


}
*/