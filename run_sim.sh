#!/bin/bash
#SBATCH --account=NN12110K           # project account to bill 
#SBATCH --partition=accel            # other options are small and large
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 32)
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1          # Use one task for one GPU
#SBATCH --time=25:00:00              # time limit

# Log files
#SBATCH --output=outputs/output-%j.out
#SBATCH --error=errors/error-%j.err

set -euo pipefail

# Require a Julia script argument
if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  sbatch [slurm-options] run_sim.sh script.jl"
    echo
    echo "Examples:"
    echo "  sbatch run_sim.sh simulations/test.jl"
    echo "  sbatch --array=1-10 run_sim.sh simulations/test.jl"
    exit 1
fi

script="$1"

mkdir -p outputs errors results

module load NRIS/GPU
module load OpenMPI/5.0.9-GCC-14.3.0
module load Julia/1.12.2

echo "===================================="
echo "Job ID      : ${SLURM_JOB_ID:-none}"
echo "Array Job   : ${SLURM_ARRAY_JOB_ID:-none}"
echo "Task ID     : ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "Script      : $script"
echo "===================================="

julia --project=. "$script"

echo "Finished at $(date)"