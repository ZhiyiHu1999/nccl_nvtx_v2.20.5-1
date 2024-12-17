#!/bin/bash

cd nccl

# module load cuda/11.6.2
# module load cuda/12.1.1
module load cuda/11.8.0
module list

make clean

export NVTX_FLAGS="-DENABLE_API_NVTX -DENABLE_INIT_NVTX -DENABLE_ENQUEUE_NVTX -DENABLE_NET_NVTX"

# export NVTX_FLAGS="-DENABLE_NET_NVTX -DENABLE_API_NVTX\
#  -DENABLE_NVTX_EVENT_NET_SEND_ENTRY -DENABLE_NVTX_EVENT_NET_SEND_EXIT -DENABLE_NVTX_EVENT_NET_RECV_ENTRY -DENABLE_NVTX_EVENT_NET_RECV_EXIT \
#  -DENABLE_NVTX_EVENT_NET_SEND_TEST_ENTRY -DENABLE_NVTX_EVENT_NET_SEND_TEST_EXIT -DENABLE_NVTX_EVENT_NET_RECV_TEST_ENTRY -DENABLE_NVTX_EVENT_NET_RECV_TEST_EXIT \
#  -DENABLE_NVTX_EVENT_RING_CHANNEL -DENABLE_NVTX_EVENT_TREE_CHANNEL -DENABLE_NVTX_EVENT_CHANNELS -DENABLE_ENQUEUE_NVTX"

# export NVTX_FLAGS="-DENABLE_NET_NVTX \
#  -DENABLE_NVTX_EVENT_NET_SEND_ENTRY -DENABLE_NVTX_EVENT_NET_SEND_EXIT -DENABLE_NVTX_EVENT_NET_RECV_ENTRY -DENABLE_NVTX_EVENT_NET_RECV_EXIT \
#  -DENABLE_NVTX_EVENT_RING_CHANNEL -DENABLE_NVTX_EVENT_TREE_CHANNEL -DENABLE_NVTX_EVENT_CHANNELS -DENABLE_ENQUEUE_NVTX"

export NPKIT_FLAG="-DENABLE_NPKIT"

export NPKIT_NCCLKERNEL_ALLREDUCE_FLAGS="-DENABLE_NPKIT_EVENT_NCCLKERNEL_ALL_REDUCE_RING_ENTRY -DENABLE_NPKIT_EVENT_NCCLKERNEL_ALL_REDUCE_RING_EXIT \
 -DENABLE_NPKIT_EVENT_NCCLKERNEL_ALL_REDUCE_TREE_UPDOWN_ENTRY -DENABLE_NPKIT_EVENT_NCCLKERNEL_ALL_REDUCE_TREE_UPDOWN_EXIT \
 -DENABLE_NPKIT_EVENT_NCCLKERNEL_ALL_REDUCE_TREE_SPLIT_ENTRY -DENABLE_NPKIT_EVENT_NCCLKERNEL_ALL_REDUCE_TREE_SPLIT_EXIT"

export NPKIT_NCCLKERNEL_SENDRECV_FLAGS="-DENABLE_NPKIT_EVENT_NCCLKERNEL_SEND_RECV_SEND_ENTRY -DENABLE_NPKIT_EVENT_NCCLKERNEL_SEND_RECV_SEND_EXIT \
 -DENABLE_NPKIT_EVENT_NCCLKERNEL_SEND_RECV_RECV_ENTRY -DENABLE_NPKIT_EVENT_NCCLKERNEL_SEND_RECV_RECV_EXIT"

export NPKIT_PRIMS_SIMPLE_FLAGS="-DENABLE_NPKIT_EVENT_PRIM_SIMPLE_DATA_PROCESS_ENTRY -DENABLE_NPKIT_EVENT_PRIM_SIMPLE_DATA_PROCESS_EXIT"
export NPKIT_PRIMS_LL_FLAGS="-DENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_ENTRY -DENABLE_NPKIT_EVENT_PRIM_LL_DATA_PROCESS_EXIT"
export NPKIT_PRIMS_LL128_FLAGS="-DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_ENTRY -DENABLE_NPKIT_EVENT_PRIM_LL128_DATA_PROCESS_EXIT"

export NPKIT_FLAGS="$NPKIT_FLAG $NPKIT_NCCLKERNEL_ALLREDUCE_FLAGS $NPKIT_NCCLKERNEL_SENDRECV_FLAGS $NPKIT_PRIMS_SIMPLE_FLAGS $NPKIT_PRIMS_LL_FLAGS $NPKIT_PRIMS_LL128_FLAGS"

# export TRACING_FLAGS="$NVTX_FLAGS $NPKIT_FLAGS"
export TRACING_FLAGS="$NVTX_FLAGS"
# export TRACING_FLAGS="$NPKIT_FLAGS"

make -j src.build CUDA_HOME=/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80" TRACING_FLAGS="$TRACING_FLAGS"
# make -j src.build CUDA_HOME=/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
