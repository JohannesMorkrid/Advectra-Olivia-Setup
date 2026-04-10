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

# Usage - single file mode:
#   sbatch run.sh -p /data -i input.h5 -o output.h5 -d "All probe,ExB CFL"
# Usage - batch mode:
#   sbatch run.sh -p /data -F input1.h5 input2.h5 -s probes -d "All probe,ExB CFL"
# Append --force to overwrite existing output files.

python scripts/h5extract_diagnostics.py "$@"