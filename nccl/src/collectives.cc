/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "nccl.h"

#if defined(ENABLE_API_NVTX)
#include "nvtx3/nvToolsExt.h"
#endif

#if defined(ENABLE_API_NVTX)
  const uint32_t colors[] = {
    0xffe91e63, 
    0xff2196f3, 
    0xff4caf50, 
    0xffffc107, 
    0xff9c27b0, 
    0xffff5722, 
    0xff00bcd4, 
    0xff673ab7, 
    0xffff9800, 
    0xff03a9f4};
  nvtxEventAttributes_t eventAttrib_allgather = {0};  
  nvtxEventAttributes_t eventAttrib_allreduce = {0};
  nvtxEventAttributes_t eventAttrib_broadcast = {0};
  nvtxEventAttributes_t eventAttrib_reduce = {0};
  nvtxEventAttributes_t eventAttrib_reducescatter = {0};
  nvtxEventAttributes_t eventAttrib_send = {0};
  nvtxEventAttributes_t eventAttrib_recv = {0};
#endif

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_AllGather[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_AllGather, sizeof(nvtxMsg_AllGather), 
                  "ncclAllGather(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  sendcount * ncclTypeSize(datatype),
                  ncclTypeSize(datatype),
                  pid);

  eventAttrib_allgather.version = NVTX_VERSION;
  eventAttrib_allgather.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_allgather.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_allgather.colorType = NVTX_COLOR_ARGB;
  eventAttrib_allgather.message.ascii = nvtxMsg_AllGather;
  eventAttrib_allgather.color = colors[0];

  nvtxRangePushEx(&eventAttrib_allgather);
#endif  

  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };

  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_AllReduce[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_AllReduce, sizeof(nvtxMsg_AllReduce), 
                  "ncclAllReduce(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, red_op %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  count * ncclTypeSize(datatype),
                  ncclTypeSize(datatype),
                  op,
                  pid);

  eventAttrib_allreduce.version = NVTX_VERSION;
  eventAttrib_allreduce.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_allreduce.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_allreduce.colorType = NVTX_COLOR_ARGB;
  eventAttrib_allreduce.message.ascii = nvtxMsg_AllReduce;
  eventAttrib_allreduce.color = colors[1];

  nvtxRangePushEx(&eventAttrib_allreduce);
#endif

  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsAllReduce, op)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };

  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Broadcast[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Broadcast, sizeof(nvtxMsg_Broadcast), 
                  "ncclBroadcast(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, root %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  count * ncclTypeSize(datatype),
                  ncclTypeSize(datatype), 
                  root,
                  pid);

  eventAttrib_broadcast.version = NVTX_VERSION;
  eventAttrib_broadcast.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_broadcast.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_broadcast.colorType = NVTX_COLOR_ARGB;
  eventAttrib_broadcast.message.ascii = nvtxMsg_Broadcast;
  eventAttrib_broadcast.color = colors[2];

  nvtxRangePushEx(&eventAttrib_broadcast);
#endif

  struct NvtxParamsBroadcast {
    size_t bytes;
    int root;
  };
  constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)}
  };
  NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root};
  NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };

  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif
  
  return ret;
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Reduce[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Reduce, sizeof(nvtxMsg_Reduce), 
                  "ncclReduce(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, red_op %d, root %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  count * ncclTypeSize(datatype),
                  ncclTypeSize(datatype), 
                  op,
                  root,
                  pid);

  eventAttrib_reduce.version = NVTX_VERSION;
  eventAttrib_reduce.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_reduce.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_reduce.colorType = NVTX_COLOR_ARGB;
  eventAttrib_reduce.message.ascii = nvtxMsg_Reduce;
  eventAttrib_reduce.color = colors[4];

  nvtxRangePushEx(&eventAttrib_reduce);
#endif

  struct NvtxParamsReduce {
    size_t bytes;
    int root;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsReduce, root)},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduce, op)}
  };
  NvtxParamsReduce payload{count * ncclTypeSize(datatype), root, op};
  NVTX3_FUNC_WITH_PARAMS(Reduce, ReduceSchema, payload)

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };

  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_ReduceScatter[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_ReduceScatter, sizeof(nvtxMsg_ReduceScatter), 
                  "ncclReduceScatter(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, red_op %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream, 
                  recvcount * ncclTypeSize(datatype),
                  ncclTypeSize(datatype), 
                  op,
                  pid);

  eventAttrib_reducescatter.version = NVTX_VERSION;
  eventAttrib_reducescatter.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib_reducescatter.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib_reducescatter.colorType = NVTX_COLOR_ARGB;
  eventAttrib_reducescatter.message.ascii = nvtxMsg_ReduceScatter;
  eventAttrib_reducescatter.color = colors[3];

  nvtxRangePushEx(&eventAttrib_reducescatter);
#endif

  struct NvtxParamsReduceScatter {
    size_t bytes;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduceScatter, op)}
  };
  NvtxParamsReduceScatter payload{recvcount * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };

  ncclResult_t ret;
  ret = ncclEnqueueCheck(&info);

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Send[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Send, sizeof(nvtxMsg_Send), 
                  "ncclSend(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, receiver_rank %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream,
                  count * ncclTypeSize(datatype), 
                  ncclTypeSize(datatype), 
                  peer,
                  pid);

              eventAttrib_send.version = NVTX_VERSION;
              eventAttrib_send.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
              eventAttrib_send.messageType = NVTX_MESSAGE_TYPE_ASCII;
              eventAttrib_send.colorType = NVTX_COLOR_ARGB;
              eventAttrib_send.message.ascii = nvtxMsg_Send;
              eventAttrib_send.color = colors[5];

  nvtxRangePushEx(&eventAttrib_send);
#endif

  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {

#if defined(ENABLE_API_NVTX)
  char nvtxMsg_Recv[256];
  pid_t pid = getpid();
  snprintf(nvtxMsg_Recv, sizeof(nvtxMsg_Recv), 
                  "ncclRecv(): commHash 0x%llx, stream %p, data_size %zu, type_size %d, sender_rank %d, pid %d", 
                  (unsigned long long)comm->commHash, 
                  stream,
                  count * ncclTypeSize(datatype), 
                  ncclTypeSize(datatype), 
                  peer,
                  pid);

              eventAttrib_recv.version = NVTX_VERSION;
              eventAttrib_recv.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
              eventAttrib_recv.messageType = NVTX_MESSAGE_TYPE_ASCII;
              eventAttrib_recv.colorType = NVTX_COLOR_ARGB;
              eventAttrib_recv.message.ascii = nvtxMsg_Recv;
              eventAttrib_recv.color = colors[6];

  nvtxRangePushEx(&eventAttrib_recv);
#endif

  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(ncclGroupEnd());

#if defined(ENABLE_API_NVTX)
  nvtxRangePop();
#endif

  return ret;
}
