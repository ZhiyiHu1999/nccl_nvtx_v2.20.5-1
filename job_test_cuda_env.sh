#!/bin/bash -l
#SBATCH --job-name="check-cuda"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --partition=normal
#SBATCH --account=a-g34
#SBATCH --time=00:10:00            # total run time limit (HH:MM:SS)
#SBATCH --output=check_cuda.%j.o
#SBATCH --error=check_cuda.%j.e

srun --environment=megatron bash -c "
    which nvcc
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    which nsys
    nsys --version
"