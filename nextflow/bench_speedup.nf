include { crossProduct; collectCSVs; deliverables; } from './utils.nf'

params.dryRun = false

def julia_depot_dir = file("/home/alexbou/st-alexbou-1/abc/depot")
def toml_files = file("../*.toml")

def experiment_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,bench_gpu_particles.jl".split(",").collect{file("../" + it)}
def plot_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,bench_speedup_plot.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl".split(",").collect{file("../" + it)}

def deliv = deliverables(workflow)

def variables = [
    job_seed: (1..10),
    job_model: ["Unid", "SimpleMixture"],
    job_scheme_types: ["SAIS", "ZJA"],
    job_elt_type: ["Float64", "Float32"],
]

workflow  {
    args = crossProduct(variables, params.dryRun)
    results = run_experiment(julia_depot_dir, toml_files, experiment_jl, args, params.dryRun) | collectCSVs    
    plot(julia_depot_dir, toml_files, plot_jl, results)
}

process run_experiment {
    debug false
    scratch true
    time 400.min
    errorStrategy 'ignore'
    memory = 16.GB
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

    n_rounds = ${arg.job_model == "Unid" ? 7 : 5} 

    include(pwd() * "/bench_gpu_particles.jl")
    result = run_bench(; 
                n_rounds, 
                seed = ${arg.job_seed},
                model_type = ${arg.job_model},
                scheme_type = ${arg.job_scheme_types},
                elt_type = ${arg.job_elt_type},
            )
    
    mkdir("csvs")
    CSV.write("csvs/bench_speedup.csv", result; quotestrings = true)
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
        path "bench_speedup.png"
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    
    """ 
    #!/usr/bin/env -S julia --project=@.

    include(pwd() * "/bench_speedup_plot.jl")
    result = DataFrame(CSV.File("aggregated/bench_speedup.csv"))

    fg = create_speedup_fig(result)
    save("bench_speedup.png", fg)
    """
}