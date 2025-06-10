#include "core/nvshmem_utils.h"
#include "core/utils.h"
#include "internode.h"

#include <cuda.h>
#include <nvshmem.h>
#include <nvtx3/nvToolsExt.h>

using namespace pplx;

template <typename T, typename U, size_t NUM_WARPS, bool DO_SEND, bool DO_RECV>
__global__ __launch_bounds__(NUM_WARPS * 32, 1) void combineKernel(
    U *outTokens,
    size_t outTokensStrideElem,
    const uint32_t *indices,
    size_t indicesStrideElem,
    size_t indicesStrideRow,
    const float *weights,
    size_t weightsStrideElem,
    size_t weightsStrideRow,
    const T *expertX,
    size_t expertXStrideElem,
    size_t expertXStrideRow,
    size_t expertsPerToken,
    size_t maxNumTokens,
    size_t numExperts,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    unsigned *boundM,
    unsigned m,
    uint32_t *sourceExpert,
    const uint32_t *sourceIndex,
    const uint32_t *sourceOffset,
    const uint32_t *sourceGroup,
    uint64_t *combineSignalBuffer,
    uint64_t *combineSyncBuffer,
    uint32_t &globalTokenIndex,
    std::byte *xBufferIn,
    std::byte *xBufferOut
) {
  const unsigned numLocalExperts = numExperts / worldSize;
  const size_t stride = hiddenDim * sizeof(T);
  constexpr unsigned WARP_SIZE = 32;
  uint32_t warpId = threadIdx.x / WARP_SIZE;
  const unsigned laneId = threadIdx.x % WARP_SIZE;

  if (DO_SEND) {

    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/send Step 1\n", rank);
    }

    const size_t numSendTokens = __ldg(&globalTokenIndex);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += gridDim.x * blockDim.x) {
      nvshmemx_signal_op(&combineSyncBuffer[rank], 1, NVSHMEM_SIGNAL_SET, i);
    }

    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/send Step 2\n", rank);
    }

    // Dispatch the tokens from the expert to the DP groups.
    for (uint32_t token = blockIdx.x; token < numSendTokens; token += gridDim.x) {
      const uint32_t expert = __ldg(&sourceExpert[token]);
      const uint32_t offset = __ldg(&sourceOffset[token]);

      // Copy the token to symmetric memory for send.
      std::byte *xTokenPtr = xBufferIn + token * stride;
      {
        const T *expertXTokenPtr = expertX + expert * expertXStrideRow + offset * expertXStrideElem;

        int4 *dstPtr = (int4 *)xTokenPtr;
        const int4 *srcPtr = (const int4 *)expertXTokenPtr;
        dstPtr += threadIdx.x;
        srcPtr += threadIdx.x;

        const unsigned n = stride / sizeof(int4);
#pragma unroll(4)
        for (unsigned j = threadIdx.x; j < n; j += blockDim.x) {
          *dstPtr = __ldg(srcPtr);
          dstPtr += blockDim.x;
          srcPtr += blockDim.x;
        }
        __syncthreads();
      }

      const uint32_t dstExpert = rank * numLocalExperts + expert;

      const uint32_t source = __ldg(&sourceIndex[token]);
      const uint32_t dp = __ldg(&sourceGroup[token]);
      for (unsigned i = warpId; i < dpSize; i += NUM_WARPS) {
        const int dstRank = dp * dpSize + i;
        const unsigned index = dstExpert * maxNumTokens + source;
        std::byte *dstPtr = xBufferOut + index * stride;
        uint64_t *sig_addr = nullptr;
#if FORCE_ZCOPY


        if ( laneId == 0){
          printf ("[%d:%d:%d] COMBINE Return ltoken %d to rank=%d, index=%d (exp_offs=%d, lidx=%d)\n", 
                  rank, blockIdx.x, threadIdx.x,
                  i, dstRank, index, (unsigned)(dstExpert * maxNumTokens), source);
        }

        sig_addr = &combineSignalBuffer[index];
#else

        if ( laneId == 0){
          printf ("[%d:%d:%d] COMBINE Return ltoken %d to rank=%d, index=%d (exp_offs=%d, lidx=%d)\n", 
                  rank, blockIdx.x, threadIdx.x,
                  i, dstRank, index, (unsigned)(dstExpert * maxNumTokens), source);
        }
        sig_addr = &combineSignalBuffer[source];
#endif
        nvshmemx_putmem_signal_nbi_warp(
            dstPtr, xTokenPtr, stride, sig_addr, 1, NVSHMEM_SIGNAL_ADD, dstRank
        );
      }
    }
    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/send Step 3\n", rank);
    }
  }

  // Synchronize the grid to ensure that tokens routed within the rank are
  // correctly transported from one block to another.
  if (DO_RECV) {


    extern __shared__ std::byte sharedMemory[];
    uint32_t *tokenIndex = reinterpret_cast<uint32_t *>(sharedMemory);
    for (uint32_t i = threadIdx.x; i < numExperts; i += blockDim.x) {
      tokenIndex[i] = 0;
    }

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/recv Step 1\n", rank);
    }

    if (DO_SEND) {
      cooperative_groups::this_grid().sync();
    }

    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/recv Step 2\n", rank);
    }

    // Compute the weighed sum of the input tokens.
    const size_t localNumTokens = boundM ? __ldg(boundM) : m;
    for (unsigned i = 0; i < localNumTokens; i++) {

      // printf ("[%d:%d:%d] COMBINE Check token ltoken %d (expertsPerToken=%d)\n", 
      //         rank, blockIdx.x, threadIdx.x, i, expertsPerToken);
      
      
      if ( (i % gridDim.x) == blockIdx.x) {

        // printf ("[%d:%d:%d] COMBINE Wait for token ltoken %d\n", 
        //   rank, blockIdx.x, threadIdx.x, i);

        if (0 == threadIdx.x && rank == 0) {
            printf("\t%d [blk %d]: combine/recv Step 2. token=%d wait for combineSig\n", rank, blockIdx.x, i);
        }

#if FORCE_ZCOPY
        /* Parallel wait for the tokens to arrive */
        for (unsigned k = threadIdx.x; k < expertsPerToken; k += blockDim.x ) {
          const uint32_t expert = __ldg(&indices[i * expertsPerToken + k]);
          const unsigned syncIdx = expert * maxNumTokens + tokenIndex[expert];

          if (1 || rank == 0) {
              printf ("[%d:%d:%d] COMBINE Wait for token ltoken %d from rank=%d, index=%d (exp_offs=%d, lidx=%d)\n", 
                      rank, blockIdx.x, threadIdx.x,
                      i, expert/numLocalExperts, syncIdx, expert * maxNumTokens, tokenIndex[expert]);
          }
          nvshmem_uint64_wait_until(&combineSignalBuffer[syncIdx], NVSHMEM_CMP_EQ, 1);
          combineSignalBuffer[syncIdx] = 0;
        }
#else
        nvshmem_uint64_wait_until(&combineSignalBuffer[i], NVSHMEM_CMP_EQ, expertsPerToken);
#endif
        __syncthreads();

#if !FORCE_ZCOPY
        combineSignalBuffer[i] = 0;
#endif

        if (0 == threadIdx.x && rank == 0) {
          printf("\t%d [blk %d]: combine/recv Step 2. token=%d __syncthreads()\n", rank, blockIdx.x, i);
        }

        if (0 == threadIdx.x && rank == 0) {
            printf("\t%d [blk %d]: combine/recv Step 2. token=%d process the token\n", rank, blockIdx.x, i);
        
        }

#if 1
        U *dstPtr = outTokens + i * outTokensStrideElem;
        constexpr unsigned VEC_SIZE = 8;
        for (unsigned j = threadIdx.x * VEC_SIZE; j < hiddenDim; j += blockDim.x * VEC_SIZE) {
          float sum[VEC_SIZE];

  #pragma unroll
          for (unsigned l = 0; l < VEC_SIZE; ++l) {
            sum[l] = 0.0f;
          }

          for (unsigned k = 0; k < expertsPerToken; ++k) {
            const uint32_t expert = __ldg(&indices[i * expertsPerToken + k]);
            const float weight = __ldg(&weights[i * weightsStrideRow + k]);

  #pragma unroll
            for (unsigned l = 0; l < VEC_SIZE; ++l) {
#if FORCE_ZCOPY
              std::byte *xDstPtr = xBufferOut + (expert * maxNumTokens + tokenIndex[expert]) * stride;
#else
              std::byte *xDstPtr = xBufferOut + (expert * maxNumTokens + i) * stride;
#endif
              sum[l] += weight * (float)((T *)xDstPtr)[j + l];
            }
          }

  #pragma unroll
          for (unsigned l = 0; l < VEC_SIZE; ++l) {
            dstPtr[j + l] = sum[l];
          }
        }
#endif

      }

#if FORCE_ZCOPY
      // Replicate the token count calculation across all blocks.
      if (threadIdx.x < expertsPerToken) {
        uint32_t dstExpert = __ldg(&indices[i * expertsPerToken + threadIdx.x]);
        tokenIndex[dstExpert]++;
          printf ("[%d:%d:%d] COMBINE token %d, update expert %d local indexes to %d\n", 
                        rank, blockIdx.x, threadIdx.x,
                        i, dstExpert, tokenIndex[dstExpert]);
      }
#endif

    }


    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/recv Step 3\n", rank);
    }

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < worldSize;
         i += gridDim.x * blockDim.x) {
      nvshmem_uint64_wait_until(&combineSyncBuffer[i], NVSHMEM_CMP_EQ, 1);
      combineSyncBuffer[i] = 0;
    }

    if (0 == threadIdx.x && blockIdx.x == 0) {
      printf("\t%d: combine/recv Step 4\n", rank);
    }


    if (blockIdx.x == 0 && threadIdx.x == 0) {
      globalTokenIndex = 0;
    }
  }
}

template <typename T, typename U>
void AllToAllInterNode::combine(
    const Strided1D<U> &outTokens,
    const Strided2D<uint32_t> &indices,
    const Strided2D<float> &weights,
    const Strided2D<T> &expertX,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {
  constexpr size_t NUM_WARPS = 16;

  const size_t numLocalExperts = numExperts / worldSize;
  const size_t numDPGroups = worldSize / dpSize;
  const size_t batchNumTokens = numLocalExperts * numDPGroups * maxNumTokens;
  const size_t numBlocks = std::min(static_cast<size_t>(numSMs), batchNumTokens);

  assert(hiddenDimBytes % 16 == 0);

  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);

  const size_t sharedMemoryRecv = sizeof(uint32_t) * numExperts;

  void *args[] = {
      const_cast<U **>(&outTokens.data),
      const_cast<size_t *>(&outTokens.strideElem),
      const_cast<uint32_t **>(&indices.data),
      const_cast<size_t *>(&indices.strideElem),
      const_cast<size_t *>(&indices.strideRow),
      const_cast<float **>(&weights.data),
      const_cast<size_t *>(&weights.strideElem),
      const_cast<size_t *>(&weights.strideRow),
      const_cast<T **>(&expertX.data),
      const_cast<size_t *>(&expertX.strideElem),
      const_cast<size_t *>(&expertX.strideRow),
      const_cast<size_t *>(&expertsPerToken),
      const_cast<size_t *>(&maxNumTokens),
      const_cast<size_t *>(&numExperts),
      const_cast<unsigned *>(&rank),
      const_cast<unsigned *>(&worldSize),
      const_cast<unsigned *>(&dpSize),
      const_cast<size_t *>(&hiddenDim),
      const_cast<unsigned **>(&boundM),
      &m,
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceGroup,
      &combineSignalBuffer,
      &combineSyncBuffer,
      &tokenIndex,
      &xCombineIn,
      &xCombineOut};

  nvtxRangePush("combine");
  switch (splitMode) {
  case SplitMode::SEND:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, true, false>,
        dimGrid,
        dimBlock,
        args,
        0,
        stream
    ));
    break;
  case SplitMode::RECV:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, false, true>,
        dimGrid,
        dimBlock,
        args,
        sharedMemoryRecv,
        stream
    ));
    break;
  case SplitMode::NONE:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&combineKernel<T, U, NUM_WARPS, true, true>, 
        dimGrid,
        dimBlock, 
        args,
        sharedMemoryRecv,
        stream
    ));
    break;
  default:
    PPLX_UNREACHABLE("invalid split mode");
  }
  nvtxRangePop();
}

#define INSTANTIATE_COMBINE(T, U)                                                                  \
  template void AllToAllInterNode::combine<T, U>(                                                  \
      const Strided1D<U> &outTokens,                                                               \
      const Strided2D<uint32_t> &indices,                                                          \
      const Strided2D<float> &weights,                                                             \
      const Strided2D<T> &expertX,                                                                 \
      unsigned m,                                                                                  \
      const unsigned *boundM,                                                                      \
      SplitMode splitMode,                                                                         \
      cudaStream_t stream                                                                          \
  );

INSTANTIATE_COMBINE(float, nv_bfloat16)
INSTANTIATE_COMBINE(float, half)
INSTANTIATE_COMBINE(nv_bfloat16, nv_bfloat16)
INSTANTIATE_COMBINE(nv_bfloat16, half)
INSTANTIATE_COMBINE(half, nv_bfloat16)
INSTANTIATE_COMBINE(half, half)
