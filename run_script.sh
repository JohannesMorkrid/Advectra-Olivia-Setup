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

#python scripts/h5extract_diagnostics.py "$@"
python scripts/h5extract_diagnostics.py -p /cluster/work/projects/nn12110k/joemork/GD-sheath-scan --files GDSI_sigma-0.001_hybrid.h5 GDSI_sigma-0.005_hybrid.h5 GDSI_sigma-0.02_hybrid.h5 GDSI_sigma-0.1_hybrid.h5 GDSI_sigma-0.5_hybrid.h5 GDSI_sigma-0.002_hybrid.h5 GDSI_sigma-0.01_hybrid.h5 GDSI_sigma-0.05_hybrid.h5 GDSI_sigma-0.2_hybrid.h5 GDSI_sigma-1.0_hybrid.h5 --suffix cfl -d "ExB CFL"