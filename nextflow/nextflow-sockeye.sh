#!/bin/bash

# explicit number of nodes needed since Nov-23
cl_ops="--nodes=1"

# need GPUs in this project
cl_ops="$cl_ops --gpus=1"


cl_ops="$cl_ops --account=st-alexbou-1-gpu"

export CLUSTER_OPTIONS="$cl_ops"

echo "Using CLUSTER_OPTIONS=$CLUSTER_OPTIONS"
./nextflow $@ -profile cluster