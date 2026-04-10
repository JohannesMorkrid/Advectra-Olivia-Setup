#!/bin/bash
#SBATCH --account=NN12110K           # project account to bill 
#SBATCH --partition=small            # other options are accel and large
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=outputs/script-output-%j
#SBATCH --error=errors/error-%j
#SBATCH --time=00:02:00

module load NRIS/CPU  
module load h5py

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

python scripts/h5extract_diagnostics.py \
    --input "$1" \
    --output "$2" \
    --diagnostics "$3" \
    ${4:+--force}