# Advectra-Olivia-Setup
Structure for running Advectra.jl simulations on Olivia

## Setup

Allocate a GPU compatible partition:
```bash
  salloc -N 1 -A <project_number> --mem 200G -t 600 -p accel --gpus-per-node=1
```
Run instantiate.sh from the repo directory on a ***accel partition***, check that it is executable, to ensure all packages are installed and pre-compiled:
```bash
  ./instantiate.sh
```

## Run simulations
Modify the script.jl and run.sh to your liking and then submit it to slurm using:

```bash 
  sbatch run.sh
```

Currently the normal terminal log is logged to the error-* file.
