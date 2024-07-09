include { crossProduct; collectCSVs; deliverables } from './utils.nf'

def julia_depot_dir = file("/home/alexbou/st-alexbou-1/abc/depot")
def julia_env_dir = file("..")
def deliv = deliverables(workflow)

workflow  {
    test(julia_depot_dir, julia_env_dir)
}

// TODO: add explicit deps to toml/julia files to ensure resume works properly...

process test {
    debug true
    cpus 1
    time '5m'
    memory '16 GB'
    scratch false
    input:
        env JULIA_DEPOT_PATH
        path julia_env
    """
    #!/usr/bin/env -S julia 

    using Pkg
    Pkg.activate("$julia_env")

    include("$julia_env/test_unid.jl")
    """
}