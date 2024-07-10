include { crossProduct; collectCSVs; deliverables; } from './utils.nf'

params.dryRun = false

def julia_depot_dir = file("/home/alexbou/st-alexbou-1/abc/depot")
def toml_files = file("../*.toml")

def experiment_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,bench_variance.jl".split(",").collect{file("../" + it)}
def plot_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,bench_variance_plot.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl".split(",").collect{file("../" + it)}

def deliv = deliverables(workflow)

def variables = [
    seed: (1..1000),
]

workflow  {
    args = crossProduct(variables, params.dryRun)
    results = run_experiment(julia_depot_dir, toml_files, experiment_jl, args, params.dryRun) | collectCSVs    
    plot(julia_depot_dir, toml_files, plot_jl, results)
}

process run_experiment {
    debug false
    time 20.min
    memory = 16.GB
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
    result = run_experiments(; seeds = ${arg.seed}:${arg.seed}, rounds = ${dryRun ? "4:4" : "5:10"}, n_particles = ${dryRun ? "2^10" : "2^14"})
    
    mkdir("csvs")
    CSV.write("csvs/bench_variance.csv", result; quotestrings = true)
    """
}

process plot {
    debug true
    time 5.min
    memory = 16.GB
    input:
        env JULIA_DEPOT_PATH
        path toml_files
        path jl_files
        path aggregated 
    output:
        path "bench_variance.png"
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    
    """ 
    #!/usr/bin/env -S julia --project=@.

    include(pwd() * "/bench_variance_plot.jl")
    fg = create_fig("aggregated/bench_variance.csv")
    save("bench_variance.png", fg)
    """
}