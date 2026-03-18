#!/bin/bash
#SBATCH --account=NN12110K           # project account to bill 
#SBATCH --partition=accel            # other options are small and large
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 32)
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1          # Use one task for one GPU
#SBATCH --output=output-%j
#SBATCH --error=error-%j
#SBATCH --time=01:00:00              # time limit

script=${1:-"simulations/script.jl"}

module load NRIS/GPU  
module load OpenMPI/5.0.9-GCC-14.3.0
module load Julia/1.12.2
 
julia --project=. "$script"