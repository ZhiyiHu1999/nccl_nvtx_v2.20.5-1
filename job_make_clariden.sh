#!/bin/bash -l
#SBATCH --job-name="nccl-build"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # Do not change
#SBATCH --partition=normal
#SBATCH --account=a-g34
#SBATCH --time=02:20:00            # total run time limit (HH:MM:SS)
#SBATCH --output=nccl_build.%j.o
#SBATCH --error=nccl_build.%j.e

srun --environment=megatron bash -c "
cd /users/zhu/nccl_nvtx_v2.20.5-1/nccl
export NVTX_FLAGS=\"-DENABLE_API_NVTX -DENABLE_INIT_NVTX -DENABLE_ENQUEUE_NVTX\"
export TRACING_FLAGS=\"\$NVTX_FLAGS\"
make -j src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE=\"-gencode=arch=compute_90,code=sm_90\" TRACING_FLAGS=\"\$TRACING_FLAGS\"
"
