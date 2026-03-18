#!/bin/bash

module load NRIS/GPU  
module load OpenMPI/5.0.9-GCC-14.3.0
module load Julia/1.12.2
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.resolve()'