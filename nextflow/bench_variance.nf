include { crossProduct; collectCSVs; deliverables } from './utils.nf'

def julia_depot_dir = file("/home/alexbou/st-alexbou-1/abc/depot")
def toml_files = file("../*.toml")
def jl_files = file("../*.jl")
def deliv = deliverables(workflow)

workflow  {
    test(julia_depot_dir, toml_files, jl_files)
}

process test {
    debug true
    cpus 1
    time '5m'
    memory '16 GB'
    scratch false
    input:
        env JULIA_DEPOT_PATH
        path toml_files
        path jl_files
    """
    #!/usr/bin/env -S julia --project=@.

    include("test_unid.jl")
    """
}