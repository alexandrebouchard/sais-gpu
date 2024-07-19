include { crossProduct; collectCSVs; deliverables; } from './utils.nf'

params.dryRun = false

def julia_depot_dir = file("/home/alexbou/st-alexbou-1/abc/depot")
def toml_files = file("../*.toml")

def experiment_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,bench_variance.jl".split(",").collect{file("../" + it)}
def plot_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,bench_variance_plot.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl".split(",").collect{file("../" + it)}

def deliv = deliverables(workflow)

def variables = [
    job_seed: (1..1000),
    job_model: ["Unid", "SimpleMixture"],
    job_scheme_types: ["SAIS", "ZJA"],
]

workflow  {
    args = crossProduct(variables, params.dryRun)
    results = run_experiment(julia_depot_dir, toml_files, experiment_jl, args, params.dryRun) | collectCSVs    
    plot(julia_depot_dir, toml_files, plot_jl, results)
}

process run_experiment {
    debug false
    time 40.min
    memory = 16.GB
    errorStrategy 'ignore'
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

    min_round = ${arg.job_model == "Unid" ? 5 : 3} 
    max_round = min_round + ${dryRun ? 0 : 5}

    include(pwd() * "/bench_variance.jl")
    result = run_experiments(; 
                scheme_types = [${arg.job_scheme_types}],
                models = [${arg.job_model}],
                seeds = ${arg.job_seed}:${arg.job_seed}, 
                rounds = min_round:max_round, 
                n_particles = ${dryRun ? "2" : "2^14"}
            )
    
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
    result = DataFrame(CSV.File("aggregated/bench_variance.csv"))

    fg = create_vars_fig(result)
    save("bench_variance.png", fg)

    
    """
}