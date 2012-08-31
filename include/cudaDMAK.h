/*
 *  Copyright 2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

__device__ __forceinline__ void ptx_cudaDMA_barrier_blocking (const int name, const int num_barriers)
{
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(num_barriers) : "memory" );
}

__device__ __forceinline__ void ptx_cudaDMA_barrier_nonblocking (const int name, const int num_barriers)
{
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(num_barriers) : "memory" );
}

#define CUDADMA_BASE cudaDMA
#define CUDADMA_DMA_TID (threadIdx.x-dma_threadIdx_start)
#define WARP_SIZE 32
#define GUARD_ZERO(expr) ((expr == 0) ? 1 : expr)

// Enable the restrict keyword to allow additional compiler optimizations
// Note that this can increase register pressure (see appendix B.2.4 of
// CUDA Programming Guide)
#define ENABLE_RESTRICT

#ifdef ENABLE_RESTRICT
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// Enable the use of LDG load operations
//#define ENABLE_LDG

class cudaDMA
{
public:
  __device__ cudaDMA(const int dmaID,
  		     const int num_dma_threads,
		     const int num_compute_threads,
		     const int dma_threadIdx_start)
    : 
    is_dma_thread ((int(threadIdx.x)>=dma_threadIdx_start) && (int(threadIdx.x)<(dma_threadIdx_start+num_dma_threads))),
    dma_tid (CUDADMA_DMA_TID),
    barrierID_empty ((dmaID<<1)+1),
    barrierID_full (dmaID<<1),
    barrier_size (num_dma_threads+num_compute_threads)
      {
      }

  // Intraspective function
  __device__ __forceinline__ bool owns_this_thread(void) const { return is_dma_thread; }

  // Compute thread synchronization calls
  __device__ __forceinline__ void start_async_dma(void) const
  {
    ptx_cudaDMA_barrier_nonblocking(barrierID_empty,barrier_size);
  }
  __device__ __forceinline__ void wait_for_dma_finish(void) const
  {
    ptx_cudaDMA_barrier_blocking(barrierID_full,barrier_size);
  }
  // DMA thread synchronization calls
  __device__ __forceinline__ void wait_for_dma_start(void) const
  {
    ptx_cudaDMA_barrier_blocking(barrierID_empty, barrier_size);
  }
  __device__ __forceinline__ void finish_async_dma(void) const
  {
    ptx_cudaDMA_barrier_nonblocking(barrierID_full,barrier_size);
  }
protected:
  template<typename T>
  __device__ __forceinline__ void perform_load(const void *src_ptr, void *dst_ptr) const
  {
#ifdef ENABLE_LDG
    *((T*)dst_ptr) = __ldg(((T*)src_ptr));
#else
    *((T*)dst_ptr) = *((T*)src_ptr);
#endif
  }
  template<typename T>
  __device__ __forceinline__ void perform_store(const void *src_ptr, void *dst_ptr) const
  {
    *((T*)dst_ptr) = *((T*)src_ptr);
  }
protected:
  const bool is_dma_thread;
  const int dma_tid;
  const int barrierID_empty;
  const int barrierID_full;
  const int barrier_size;
};

#define MAX_LDS_PER_THREAD (BYTES_PER_THREAD/ALIGNMENT)

#define LDS_PER_ELMT ((BYTES_PER_ELMT+ALIGNMENT-1)/ALIGNMENT)
#define FULL_LDS_PER_ELMT (BYTES_PER_ELMT/ALIGNMENT)

// Figure out if we need to split a warp across multiple elements
// because the elements are very small (i.e. total loads per element <= 32)
#define SPLIT_WARP (LDS_PER_ELMT <= WARP_SIZE)
#define THREADS_PER_ELMT (LDS_PER_ELMT > (WARP_SIZE/2) ? WARP_SIZE : \
			 LDS_PER_ELMT > (WARP_SIZE/4) ? (WARP_SIZE/2) : \
			 LDS_PER_ELMT > (WARP_SIZE/8) ? (WARP_SIZE/4) : \
			 LDS_PER_ELMT > (WARP_SIZE/16) ? (WARP_SIZE/8) : \
			 LDS_PER_ELMT > (WARP_SIZE/32) ? (WARP_SIZE/16) : WARP_SIZE/32)
#define ELMT_PER_STEP_SPLIT ((DMA_THREADS/THREADS_PER_ELMT) * MAX_LDS_PER_THREAD)
#define ROW_ITERS_SPLIT	 (MAX_LDS_PER_THREAD)
#define HAS_PARTIAL_ELMTS_SPLIT ((NUM_ELMTS % ELMT_PER_STEP_SPLIT) != 0)
//#define HAS_PARTIAL_BYTES_SPLIT ((LDS_PER_ELMT % THREADS_PER_ELMT) != 0)
#define HAS_PARTIAL_BYTES_SPLIT ((BYTES_PER_ELMT % (THREADS_PER_ELMT*ALIGNMENT)) != 0)
#define COL_ITERS_SPLIT  ((LDS_PER_ELMT == THREADS_PER_ELMT) ? 1 : 0)
#define STEP_ITERS_SPLIT (NUM_ELMTS/ELMT_PER_STEP_SPLIT)

#define NUM_WARPS (DMA_THREADS/WARP_SIZE)

// Next we'll handle the case where all the warps performing as many loads as
// possible in a step can't handle an entire element.
#define BIG_ELMTS ((DMA_THREADS*MAX_LDS_PER_THREAD) < LDS_PER_ELMT)
// This is the number of steps where ALL threads can perform the maximum number of loads
#define MAX_ITERS_BIG (LDS_PER_ELMT/(DMA_THREADS*MAX_LDS_PER_THREAD))
// This is actually whether there are leftovers from the max loading phase
#define HAS_PARTIAL_ELMTS_BIG (((LDS_PER_ELMT % (DMA_THREADS*MAX_LDS_PER_THREAD)) / DMA_THREADS) > 0)
// This is the number of loads to be performed for remaining bytes after max loads
#define PART_ITERS_BIG ((LDS_PER_ELMT % (DMA_THREADS*MAX_LDS_PER_THREAD)) / DMA_THREADS)
#define REMAINING_BYTES_BIG (BYTES_PER_ELMT - (MAX_ITERS_BIG*DMA_THREADS*MAX_LDS_PER_THREAD*ALIGNMENT + \
                                                  PART_ITERS_BIG*DMA_THREADS*ALIGNMENT))
// This is actually whether there are leftovers from the partial loading phase
#define HAS_PARTIAL_BYTES_BIG (REMAINING_BYTES_BIG > 0)
#define STEP_ITERS_BIG (NUM_ELMTS)

// Now handle the case where we don't have to split a warp across multiple elements.
// Now see if we can have a single warp handle an entire element in a single pass.
// We only do this if every warp will be busy, otherwise we'll allocate warps to elements
// to maximize MLP.
#define SINGLE_WARP ((LDS_PER_ELMT <= (WARP_SIZE*MAX_LDS_PER_THREAD)) && (NUM_WARPS <= NUM_ELMTS))
// If we can't do single, figure out the minimum number of warps needed to cover
// the element doing as many loads as possible and allocate those warps.  The idea here is to get
// a small number of warps on each element to minimize the wasting of warps.
#define MINIMUM_COVER ((LDS_PER_ELMT+(WARP_SIZE*MAX_LDS_PER_THREAD)-1)/(WARP_SIZE*MAX_LDS_PER_THREAD))
// Otherwise we'll allocate as many warps as possible to a single element to maximize MLP.
// Try to allocate as many warps as possible to an element, each performing one load
// to maximize MLP, if we exceed the maximum, then split the group of warps across
// multiple elements.  Once we've allocated warps to elements, see how many elements we
// can handle based on the number of outstanding loads each thread can have.
// Include the BIG_ELMTS case to avoid divide by zero errors in template instantiation.
#define MAX_WARPS_PER_ELMT ((LDS_PER_ELMT+WARP_SIZE-1)/WARP_SIZE)
#define WARPS_PER_ELMT (SINGLE_WARP ? 1 : \
                        BIG_ELMTS ? NUM_WARPS : \
                        (NUM_WARPS <= NUM_ELMTS) ? MINIMUM_COVER : \
                        ((MAX_WARPS_PER_ELMT >= NUM_WARPS) ? NUM_WARPS : MAX_WARPS_PER_ELMT))
// Figure out how many loads need to be done per thread per element (round up)
#define LDS_PER_ELMT_PER_THREAD ((LDS_PER_ELMT+(WARPS_PER_ELMT*WARP_SIZE)-1)/(WARPS_PER_ELMT*WARP_SIZE))
// This assumes that the number of warps allocated to the element were enough to
// cover the size of the element. Note we mask out the result if we're in a BIG_ELMTS
// case because it can lead to divide by zero errors in the template instantiation.
#define ELMT_PER_STEP_PER_THREAD (BIG_ELMTS ? 1 : MAX_LDS_PER_THREAD/LDS_PER_ELMT_PER_THREAD)
// Now we can figure out how many elements we can handle per step by multiplying
// the total number of elements to be handled by each thread in a step by
// the total number of groups of warps (also total groups of threads)
#define ELMT_PER_STEP_FULL (ELMT_PER_STEP_PER_THREAD * (NUM_WARPS/WARPS_PER_ELMT))
#define ROW_ITERS_FULL (ELMT_PER_STEP_PER_THREAD)
#define HAS_PARTIAL_ELMTS_FULL ((NUM_ELMTS % ELMT_PER_STEP_FULL) != 0)
//#define HAS_PARTIAL_BYTES_FULL ((LDS_PER_ELMT % (WARPS_PER_ELMT*WARP_SIZE)) != 0)
#define HAS_PARTIAL_BYTES_FULL ((BYTES_PER_ELMT % (WARPS_PER_ELMT*WARP_SIZE*ALIGNMENT)) != 0)
#define COL_ITERS_FULL (LDS_PER_ELMT/(WARPS_PER_ELMT*WARP_SIZE))
//#define STEP_ITERS_FULL ((NUM_ELMTS >= ELMT_PER_STEP_FULL) ? NUM_ELMTS/ELMT_PER_STEP_FULL : \
//                                                             NUM_ELMTS/(NUM_WARPS/WARPS_PER_ELMT))
#define STEP_ITERS_FULL (NUM_ELMTS/ELMT_PER_STEP_FULL)

#define HAS_PARTIAL_BYTES (SPLIT_WARP ? HAS_PARTIAL_BYTES_SPLIT : \
                           BIG_ELMTS ? HAS_PARTIAL_BYTES_BIG : HAS_PARTIAL_BYTES_FULL)
#define HAS_PARTIAL_ELMTS (SPLIT_WARP ? HAS_PARTIAL_ELMTS_SPLIT : \
                           BIG_ELMTS ? HAS_PARTIAL_BYTES_BIG : HAS_PARTIAL_ELMTS_FULL)

// Finally, let's compute all the initial values based on the things above.
// First we'll do the split versions
#define ELMT_ID_SPLIT (CUDADMA_DMA_TID/THREADS_PER_ELMT)
#define SPLIT_GROUP_TID (threadIdx.x - (dma_threadIdx_start + (ELMT_ID_SPLIT * THREADS_PER_ELMT)))
#define INIT_SRC_OFFSET_SPLIT(_src_stride) (ELMT_ID_SPLIT * _src_stride + SPLIT_GROUP_TID * ALIGNMENT)
#define INIT_DST_OFFSET_SPLIT(_dst_stride) (ELMT_ID_SPLIT * _dst_stride + SPLIT_GROUP_TID * ALIGNMENT)
#define INIT_SRC_STEP_STRIDE_SPLIT(_src_stride) (ELMT_PER_STEP_SPLIT * _src_stride)
#define INIT_DST_STEP_STRIDE_SPLIT(_dst_stride) (ELMT_PER_STEP_SPLIT * _dst_stride)
#define INIT_SRC_ELMT_STRIDE_SPLIT(_src_stride) ((DMA_THREADS/THREADS_PER_ELMT) * _src_stride)
#define INIT_DST_ELMT_STRIDE_SPLIT(_dst_stride) ((DMA_THREADS/THREADS_PER_ELMT) * _dst_stride)
#define INIT_INTRA_ELMT_STRIDE_SPLIT (THREADS_PER_ELMT * ALIGNMENT) // Shouldn't really matter
#define REMAINING_LOADS_SPLIT (FULL_LDS_PER_ELMT % THREADS_PER_ELMT)
// Three cases:
//     1. group id < remaining loads -> partial bytes is ALIGNMENT
//     2. group id > remaining loads -> partial bytes is 0
//     3. group id == remaining loads -> partial bytes is difference between total bytes and full loads * ALIGNMENT
#define INIT_PARTIAL_BYTES_SPLIT ((SPLIT_GROUP_TID > REMAINING_LOADS_SPLIT) ? 0 : \
                                  (SPLIT_GROUP_TID == REMAINING_LOADS_SPLIT) ? (BYTES_PER_ELMT - (FULL_LDS_PER_ELMT * ALIGNMENT)) : \
                                  ALIGNMENT)
#define REMAINING_ELMTS_SPLIT (NUM_ELMTS % ELMT_PER_STEP_SPLIT)
#define FULL_REMAINING_SPLIT (REMAINING_ELMTS_SPLIT / (DMA_THREADS/THREADS_PER_ELMT))
#define LAST_REMAINING_SPLIT (REMAINING_ELMTS_SPLIT % (DMA_THREADS/THREADS_PER_ELMT))
// Two cases:
//     1. element id < last_remaining -> full_remaining+1
//     2. element id >= last_remaining -> full_remaining
#define INIT_PARTIAL_ELMTS_SPLIT (FULL_REMAINING_SPLIT + \
                                  ((LAST_REMAINING_SPLIT==0) ? 0 : \
                                   ((ELMT_ID_SPLIT >= LAST_REMAINING_SPLIT) ? 0 : 1)))

// Now for the big versions
#define INIT_SRC_OFFSET_BIG (CUDADMA_DMA_TID * ALIGNMENT)
#define INIT_DST_OFFSET_BIG (CUDADMA_DMA_TID * ALIGNMENT)
#define INIT_SRC_STEP_STRIDE_BIG(_src_stride) (_src_stride)
#define INIT_DST_STEP_STRIDE_BIG(_dst_stride) (_dst_stride)
#define INIT_SRC_ELMT_STRIDE_BIG(_src_stride) (_src_stride)
#define INIT_DST_ELMT_STRIDE_BIG(_dst_stride) (_dst_stride)
#define INIT_INTRA_ELMT_STRIDE_BIG (DMA_THREADS * ALIGNMENT)
#define INIT_PARTIAL_ELMTS_BIG (0) // No partial elements in the big elements case
#define INIT_PARTIAL_BYTES_BIG ((CUDADMA_DMA_TID > (REMAINING_BYTES_BIG/ALIGNMENT)) ? 0 : \
                                (CUDADMA_DMA_TID == (REMAINING_BYTES_BIG/ALIGNMENT)) ? \
                                                  (BYTES_PER_ELMT - (FULL_LDS_PER_ELMT * ALIGNMENT)) : ALIGNMENT)

// Now we do the full versions 
#define ELMT_ID_FULL (CUDADMA_DMA_TID/(WARPS_PER_ELMT*WARP_SIZE))
#define FULL_GROUP_TID (threadIdx.x - (dma_threadIdx_start + (ELMT_ID_FULL * WARPS_PER_ELMT * WARP_SIZE)))
#define INIT_SRC_OFFSET_FULL(_src_stride) (ELMT_ID_FULL * _src_stride + FULL_GROUP_TID * ALIGNMENT)
#define INIT_DST_OFFSET_FULL(_dst_stride) (ELMT_ID_FULL * _dst_stride + FULL_GROUP_TID * ALIGNMENT)
#define INIT_SRC_STEP_STRIDE_FULL(_src_stride) (ELMT_PER_STEP_FULL * _src_stride)
#define INIT_DST_STEP_STRIDE_FULL(_dst_stride) (ELMT_PER_STEP_FULL * _dst_stride)
#define INIT_SRC_ELMT_STRIDE_FULL(_src_stride) ((NUM_WARPS/WARPS_PER_ELMT) * _src_stride)
#define INIT_DST_ELMT_STRIDE_FULL(_dst_stride) ((NUM_WARPS/WARPS_PER_ELMT) * _dst_stride)
#define INIT_INTRA_ELMT_STRIDE_FULL (WARPS_PER_ELMT * WARP_SIZE * ALIGNMENT)
#define REMAINING_LOADS_FULL (FULL_LDS_PER_ELMT % (WARPS_PER_ELMT * WARP_SIZE))
// Same three cases as for split
#define INIT_PARTIAL_BYTES_FULL ((REMAINING_LOADS_FULL==0) ? 0 : \
                                 (FULL_GROUP_TID > REMAINING_LOADS_FULL) ? 0 : \
                                 (FULL_GROUP_TID == REMAINING_LOADS_FULL) ? (BYTES_PER_ELMT - (FULL_LDS_PER_ELMT * ALIGNMENT)) : \
                                  ALIGNMENT)
#define REMAINING_ELMTS_FULL (NUM_ELMTS % ELMT_PER_STEP_FULL)
#define FULL_REMAINING_FULL (REMAINING_ELMTS_FULL / (DMA_THREADS/(WARPS_PER_ELMT * WARP_SIZE)))
#define LAST_REMAINING_FULL (REMAINING_ELMTS_FULL % (DMA_THREADS/(WARPS_PER_ELMT * WARP_SIZE)))
// Same two cases as for split
#define INIT_PARTIAL_ELMTS_FULL (FULL_REMAINING_FULL + \
                                  ((LAST_REMAINING_FULL==0) ? 0 : \
                                   ((ELMT_ID_FULL < LAST_REMAINING_FULL) ? 1 : 0)))
// We also have one more case here for full warp allocation:
// to determine if our warp is one of the active warps
#define WARP_ID (CUDADMA_DMA_TID/WARP_SIZE)
#define NUM_ACTIVE_WARPS ((NUM_WARPS > (NUM_ELMTS*WARPS_PER_ELMT)) ? NUM_ELMTS*WARPS_PER_ELMT : \
                                                                    (NUM_WARPS - (NUM_WARPS % WARPS_PER_ELMT)))
#define INIT_ACTIVE_WARP (WARP_ID < NUM_ACTIVE_WARPS)
#define ALL_WARPS_ACTIVE (NUM_WARPS == NUM_ACTIVE_WARPS) 

#define INIT_SRC_OFFSET(_src_stride) (SPLIT_WARP ? INIT_SRC_OFFSET_SPLIT(_src_stride) : \
                                      BIG_ELMTS  ? INIT_SRC_OFFSET_BIG : INIT_SRC_OFFSET_FULL(_src_stride))
#define INIT_DST_OFFSET(_dst_stride) (SPLIT_WARP ? INIT_DST_OFFSET_SPLIT(_dst_stride) : \
                                      BIG_ELMTS  ? INIT_DST_OFFSET_BIG : INIT_DST_OFFSET_FULL(_dst_stride))
#define INIT_SRC_STEP_STRIDE(_src_stride) (SPLIT_WARP ? INIT_SRC_STEP_STRIDE_SPLIT(_src_stride) : \
                                           BIG_ELMTS  ? INIT_SRC_STEP_STRIDE_BIG(_src_stride) : INIT_SRC_STEP_STRIDE_FULL(_src_stride))
#define INIT_DST_STEP_STRIDE(_dst_stride) (SPLIT_WARP ? INIT_DST_STEP_STRIDE_SPLIT(_dst_stride) : \
                                           BIG_ELMTS  ? INIT_DST_STEP_STRIDE_BIG(_dst_stride) : INIT_DST_STEP_STRIDE_FULL(_dst_stride))
#define INIT_SRC_ELMT_STRIDE(_src_stride) (SPLIT_WARP ? INIT_SRC_ELMT_STRIDE_SPLIT(_src_stride) : \
                                           BIG_ELMTS  ? INIT_SRC_ELMT_STRIDE_BIG(_src_stride) : INIT_SRC_ELMT_STRIDE_FULL(_src_stride))
#define INIT_DST_ELMT_STRIDE(_dst_stride) (SPLIT_WARP ? INIT_DST_ELMT_STRIDE_SPLIT(_dst_stride) : \
                                           BIG_ELMTS  ? INIT_DST_ELMT_STRIDE_BIG(_dst_stride) : INIT_DST_ELMT_STRIDE_FULL(_dst_stride))
#define INIT_INTRA_ELMT_STRIDE (SPLIT_WARP ? INIT_INTRA_ELMT_STRIDE_SPLIT : \
                                BIG_ELMTS  ? INIT_INTRA_ELMT_STRIDE_BIG : INIT_INTRA_ELMT_STRIDE_FULL)
#define INIT_PARTIAL_BYTES (SPLIT_WARP ? INIT_PARTIAL_BYTES_SPLIT : \
                            BIG_ELMTS  ? INIT_PARTIAL_BYTES_BIG : INIT_PARTIAL_BYTES_FULL)
#define INIT_PARTIAL_ELMTS (SPLIT_WARP ? INIT_PARTIAL_ELMTS_SPLIT : \
                            BIG_ELMTS  ? INIT_PARTIAL_ELMTS_BIG : INIT_PARTIAL_ELMTS_FULL)
#define INIT_PARTIAL_OFFSET (SPLIT_WARP ? 0 : BIG_ELMTS ? 0 : \
                            ((FULL_LDS_PER_ELMT - (FULL_LDS_PER_ELMT % (WARPS_PER_ELMT * WARP_SIZE))) * ALIGNMENT))

#include <cstdio>
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
__host__ void print_strided_variables(void)
{
#define PRINT_VAR(var_name) printf(#var_name " %d\n", (var_name))
  PRINT_VAR(MAX_LDS_PER_THREAD);
  PRINT_VAR(LDS_PER_ELMT);
  PRINT_VAR(FULL_LDS_PER_ELMT);
  PRINT_VAR(SPLIT_WARP);
  PRINT_VAR(BIG_ELMTS);
  printf("---------- Split math ---------\n");
  PRINT_VAR(THREADS_PER_ELMT);
  PRINT_VAR(ELMT_PER_STEP_SPLIT);
  PRINT_VAR(ROW_ITERS_SPLIT);
  PRINT_VAR(HAS_PARTIAL_ELMTS_SPLIT);
  PRINT_VAR(HAS_PARTIAL_BYTES_SPLIT);
  PRINT_VAR(COL_ITERS_SPLIT);
  PRINT_VAR(STEP_ITERS_SPLIT); 
  printf("---------- Big math -----------\n");
  PRINT_VAR(HAS_PARTIAL_ELMTS_BIG);
  PRINT_VAR(HAS_PARTIAL_BYTES_BIG);
  PRINT_VAR(MAX_ITERS_BIG);
  PRINT_VAR(PART_ITERS_BIG);
  PRINT_VAR(STEP_ITERS_BIG);
  PRINT_VAR(REMAINING_BYTES_BIG);
  printf("---------- Full math ----------\n");
  PRINT_VAR(NUM_WARPS);
  PRINT_VAR(SINGLE_WARP);
  PRINT_VAR(MAX_WARPS_PER_ELMT);
  PRINT_VAR(WARPS_PER_ELMT);
  PRINT_VAR(LDS_PER_ELMT_PER_THREAD);
  PRINT_VAR(ELMT_PER_STEP_PER_THREAD);
  PRINT_VAR(ELMT_PER_STEP_FULL);
  PRINT_VAR(ROW_ITERS_FULL);
  PRINT_VAR(HAS_PARTIAL_ELMTS_FULL);
  PRINT_VAR(HAS_PARTIAL_BYTES_FULL);
  PRINT_VAR(COL_ITERS_FULL);
  PRINT_VAR(STEP_ITERS_FULL);
  PRINT_VAR(ALL_WARPS_ACTIVE);
  PRINT_VAR(NUM_ACTIVE_WARPS);
  PRINT_VAR(REMAINING_ELMTS_FULL);
  printf("----------- Offsets -----------\n");
  PRINT_VAR(INIT_SRC_STEP_STRIDE(BYTES_PER_ELMT));
  PRINT_VAR(INIT_DST_STEP_STRIDE(BYTES_PER_ELMT));
  PRINT_VAR(INIT_SRC_ELMT_STRIDE(BYTES_PER_ELMT));
  PRINT_VAR(INIT_DST_ELMT_STRIDE(BYTES_PER_ELMT));
  PRINT_VAR(INIT_INTRA_ELMT_STRIDE);
  PRINT_VAR(INIT_PARTIAL_OFFSET);
#undef PRINT_VAR
}

// Bytes-per-thread will manage the number of registers available for each thread
// Bytes-per-thread must be divisible by alignment
template<bool DO_SYNC_TOP, int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class cudaDMAStrided : public CUDADMA_BASE
{
public:
  // Constructor for when dst_stride == BYTES_PER_ELMT
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int el_stride)
    : CUDADMA_BASE(dmaID,DMA_THREADS,num_compute_threads,dma_threadIdx_start),
      dma_src_offset(INIT_SRC_OFFSET(el_stride)),
      dma_dst_offset(INIT_DST_OFFSET(el_stride)),
      dma_src_step_stride(INIT_SRC_STEP_STRIDE(el_stride)),
      dma_dst_step_stride(INIT_DST_STEP_STRIDE(el_stride)),
      dma_src_elmt_stride(INIT_SRC_ELMT_STRIDE(el_stride)),
      dma_dst_elmt_stride(INIT_DST_ELMT_STRIDE(el_stride)),
      dma_intra_elmt_stride(INIT_INTRA_ELMT_STRIDE),
      dma_partial_bytes(INIT_PARTIAL_BYTES),
      dma_partial_offset(INIT_PARTIAL_OFFSET),
      dma_partial_elmts(INIT_PARTIAL_ELMTS),
      dma_active_warp(INIT_ACTIVE_WARP)
  {
#ifdef CUDADMA_DEBUG_ON
    assert((BYTES_PER_THREAD%ALIGNMENT) == 0);
#endif
  }

  __device__ cudaDMAStrided (const int dmaID,
  			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int src_stride,
			     const int dst_stride)
    : CUDADMA_BASE(dmaID,DMA_THREADS,num_compute_threads,dma_threadIdx_start),
      dma_src_offset(INIT_SRC_OFFSET(src_stride)),
      dma_dst_offset(INIT_DST_OFFSET(dst_stride)),
      dma_src_step_stride(INIT_SRC_STEP_STRIDE(src_stride)),
      dma_dst_step_stride(INIT_DST_STEP_STRIDE(dst_stride)),
      dma_src_elmt_stride(INIT_SRC_ELMT_STRIDE(src_stride)),
      dma_dst_elmt_stride(INIT_DST_ELMT_STRIDE(dst_stride)),
      dma_intra_elmt_stride(INIT_INTRA_ELMT_STRIDE),
      dma_partial_bytes(INIT_PARTIAL_BYTES),
      dma_partial_offset(INIT_PARTIAL_OFFSET),
      dma_partial_elmts(INIT_PARTIAL_ELMTS),
      dma_active_warp(INIT_ACTIVE_WARP)
  {
#ifdef CUDADMA_DEBUG_ON
    assert((BYTES_PER_THREAD%ALIGNMENT) == 0);
#endif
  }
public:
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) const
  {
    switch (ALIGNMENT)
    {
      case 4:
      	{
          execute_internal<float,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(src_ptr,dst_ptr);
	  break;
	}
    case 8:
      	{
	  execute_internal<float2,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(src_ptr,dst_ptr);
	  break;
	}
    case 16:
      	{
	  execute_internal<float4,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(src_ptr,dst_ptr);
	  break;
	}
#ifdef CUDADMA_DEBUG_ON
	  default:
		printf("Invalid ALIGNMENT %d must be one of (4,8,16)\n",ALIGNMENT);
		assert(false);
		break;
#endif
    }
  }

  template<typename BULK_TYPE, bool DMA_IS_SPLIT, bool DMA_IS_BIG,
           int DMA_STEP_ITERS_SPLIT, int DMA_ROW_ITERS_SPLIT, int DMA_COL_ITERS_SPLIT,
           int DMA_STEP_ITERS_BIG,   int DMA_MAX_ITERS_BIG,   int DMA_PART_ITERS_BIG,
           int DMA_STEP_ITERS_FULL,  int DMA_ROW_ITERS_FULL,  int DMA_COL_ITERS_FULL,
	   bool DMA_PARTIAL_BYTES, bool DMA_PARTIAL_ROWS, bool DMA_ALL_WARPS_ACTIVE>
  __device__ __forceinline__ void execute_internal(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) const
  {
    const char * src_off_ptr = ((const char*)src_ptr) + dma_src_offset;
    char * dst_off_ptr = ((char*)dst_ptr) + dma_dst_offset;
    if (DMA_IS_SPLIT)
    {
      if (DMA_STEP_ITERS_SPLIT == 0)
      {
        all_partial_cases<BULK_TYPE,DO_SYNC_TOP,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
				DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(src_off_ptr, dst_off_ptr);
      }
      else
      {
        if (DO_SYNC_TOP)
	  CUDADMA_BASE::wait_for_dma_start();
        for (int i = 0; i < DMA_STEP_ITERS_SPLIT; i++)
	{
	  all_partial_cases<BULK_TYPE,false/*do sync*/,DMA_PARTIAL_BYTES,false/*partial rows*/,
	  			DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(src_off_ptr, dst_off_ptr);
          src_off_ptr += dma_src_step_stride;
          dst_off_ptr += dma_dst_step_stride;
	}
        if (DMA_PARTIAL_ROWS)
        {
          all_partial_cases<BULK_TYPE,false/*do sync*/,DMA_PARTIAL_BYTES,true/*partial rows*/,
                                DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(src_off_ptr, dst_off_ptr);
        }
      }
    }
    else if (DMA_IS_BIG)
    {
      // No optimized case here, we have at least one full column iteration to perform
      if (DO_SYNC_TOP)
        CUDADMA_BASE::wait_for_dma_start();
      for (int i = 0; i < DMA_STEP_ITERS_BIG; i++)
      {
        do_copy_elmt<BULK_TYPE,DMA_MAX_ITERS_BIG,DMA_PART_ITERS_BIG,DMA_PARTIAL_ROWS,DMA_PARTIAL_BYTES>
            (src_off_ptr, dst_off_ptr, dma_intra_elmt_stride, dma_partial_bytes);
        src_off_ptr += dma_src_elmt_stride;
        dst_off_ptr += dma_dst_elmt_stride;
      }
    }
    else // Not split and not big
    {
      if (DMA_ALL_WARPS_ACTIVE) // Check to see if all the warps are active
      {
        if (DMA_STEP_ITERS_FULL == 0)
        {
          all_partial_cases<BULK_TYPE,DO_SYNC_TOP,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
                                  DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(src_off_ptr,dst_off_ptr);
        }
        else
        {
          if (DO_SYNC_TOP)
            CUDADMA_BASE::wait_for_dma_start();
          for (int i = 0; i < DMA_STEP_ITERS_FULL; i++)
          {
            all_partial_cases<BULK_TYPE,false/*do sync*/,DMA_PARTIAL_BYTES,false/*partial rows*/,
                                  DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(src_off_ptr,dst_off_ptr);
            src_off_ptr += dma_src_step_stride;
            dst_off_ptr += dma_dst_step_stride;
          }
          if (DMA_PARTIAL_ROWS)
          {
            all_partial_cases<BULK_TYPE,false/*do sync*/,DMA_PARTIAL_BYTES,true/*partial rows*/,
                                  DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(src_off_ptr,dst_off_ptr);
          }
        }
      }
      else if (dma_active_warp) // Otherwise mask off unused warps
      {
        if (DMA_STEP_ITERS_FULL == 0)
        {
          all_partial_cases<BULK_TYPE,DO_SYNC_TOP,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
                                  DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(src_off_ptr,dst_off_ptr);
        }
        else
        {
          if (DO_SYNC_TOP)
            CUDADMA_BASE::wait_for_dma_start();
          for (int i = 0; i < DMA_STEP_ITERS_FULL; i++)
          {
            all_partial_cases<BULK_TYPE,false/*do sync*/,DMA_PARTIAL_BYTES,false/*partial rows*/,
                                  DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(src_off_ptr,dst_off_ptr);
            src_off_ptr += dma_src_step_stride;
            dst_off_ptr += dma_dst_step_stride;
          }
          if (DMA_PARTIAL_ROWS)
          {
            all_partial_cases<BULK_TYPE,false/*do sync*/,DMA_PARTIAL_BYTES,true/*partial rows*/,
                                  DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(src_off_ptr,dst_off_ptr);
          }
        }
      }
      else if (DO_SYNC_TOP) // Otherwise unused warps still have to synchronize
      {
        CUDADMA_BASE::wait_for_dma_start();
      }
    }
    if (DO_SYNC_TOP)
      CUDADMA_BASE::finish_async_dma();
  }

  template<typename BULK_TYPE, bool DO_SYNC, bool DMA_PARTIAL_BYTES, bool DMA_PARTIAL_ROWS,
           int DMA_ROW_ITERS, int DMA_COL_ITERS>
  __device__ __forceinline__ void all_partial_cases(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr) const
  {
    if (!DMA_PARTIAL_BYTES)
    {
      if (!DMA_PARTIAL_ROWS)
      {
        // Optimized case
	do_strided<BULK_TYPE,DO_SYNC,true,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, dst_ptr, dma_src_elmt_stride, dma_dst_elmt_stride,
			 dma_intra_elmt_stride, 0/*no partial bytes*/);
      }
      else
      {
        // Partial rows 
        do_strided_upper<BULK_TYPE,DO_SYNC,true,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, dst_ptr, dma_src_elmt_stride, dma_dst_elmt_stride,
			 dma_intra_elmt_stride, 0/*no partial bytes*/, dma_partial_elmts);
      }
    }
    else
    {
      if (!DMA_PARTIAL_ROWS)
      {
        // Partial bytes  
	do_strided<BULK_TYPE,DO_SYNC,false,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, dst_ptr, dma_src_elmt_stride, dma_dst_elmt_stride,
			 dma_intra_elmt_stride, dma_partial_bytes);
      }
      else
      {
        // Partial bytes and partial rows
	do_strided_upper<BULK_TYPE,DO_SYNC,false,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, dst_ptr, dma_src_elmt_stride, dma_dst_elmt_stride,
			dma_intra_elmt_stride, dma_partial_bytes, dma_partial_elmts);
      }
    }
  }

  template<typename BULK_TYPE, bool DO_SYNC, bool DMA_ALL_ACTIVE, int DMA_ROW_ITERS, int DMA_COL_ITERS>
  __device__ __forceinline__ void do_strided(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr,
  					     const int src_elmt_stride, const int dst_elmt_stride,
					     const int intra_elmt_stride, const int partial_bytes) const
  {
    BULK_TYPE temp[GUARD_ZERO(DMA_ROW_ITERS*DMA_COL_ITERS)];
    // Perform loads from the source
    {
      unsigned idx = 0;
      const char *src_row_ptr = src_ptr;
      for (int row = 0; row < DMA_ROW_ITERS; row++)
      {
        const char *src_col_ptr = src_row_ptr;
        for (int col = 0; col < DMA_COL_ITERS; col++)
        {
  	  perform_load<BULK_TYPE>(src_col_ptr, &(temp[idx++]));
	  src_col_ptr += intra_elmt_stride;
        }
        src_row_ptr += src_elmt_stride; 
      }
    }
    if (!DMA_ALL_ACTIVE)
    {
      do_strided_across<DO_SYNC,DMA_ROW_ITERS>(src_ptr+dma_partial_offset,
                                               dst_ptr+dma_partial_offset,
                                               src_elmt_stride,
                                               dst_elmt_stride,
                                               partial_bytes);
    }
    else if (DO_SYNC) // Otherwise check to see if we should do the sync here
    {
      CUDADMA_BASE::wait_for_dma_start();
    }
    // Perform the destination stores
    {
      unsigned idx = 0;
      char *dst_row_ptr = dst_ptr;
      for (int row = 0; row < DMA_ROW_ITERS; row++)
      {
	char *dst_col_ptr = dst_row_ptr;
	for (int col = 0; col < DMA_COL_ITERS; col++)
	{
	  perform_store<BULK_TYPE>(&(temp[idx++]), dst_col_ptr);
	  dst_col_ptr += intra_elmt_stride;
	}
	dst_row_ptr += dst_elmt_stride;
      }
    }
  }

  template<typename BULK_TYPE, int DMA_MAX_ITERS, int DMA_PART_ITERS, bool DMA_PARTIAL_ITERS, bool DMA_PARTIAL_BYTES>
  __device__ __forceinline__ void do_copy_elmt(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr,
                                               const int intra_elmt_stride, const int partial_bytes) const
  {
#define MAX_LOADS (BYTES_PER_THREAD/sizeof(BULK_TYPE))
    BULK_TYPE temp[GUARD_ZERO(MAX_LOADS)];
#define PERFORM_MEMORY_OPS(_num_loads)                \
    for (int idx = 0; idx < _num_loads; idx++)        \
    {                                                 \
      perform_load<BULK_TYPE>(src_ptr, &(temp[idx])); \
      src_ptr += intra_elmt_stride;                   \
    }                                                 \
    for (int idx = 0; idx < _num_loads; idx++)        \
    {                                                 \
      perform_store<BULK_TYPE>(&(temp[idx]), dst_ptr);\
      dst_ptr += intra_elmt_stride;                   \
    }
    // First perform as many max iters as we can
    for (int i = 0; i < DMA_MAX_ITERS; i++)
    {
      PERFORM_MEMORY_OPS(MAX_LOADS);
    }
#undef MAX_LOADS
    if (DMA_PARTIAL_ITERS)
    {
      for (int i = 0; i < DMA_PARTIAL_ITERS; i++)
      {
        PERFORM_MEMORY_OPS(DMA_PART_ITERS);
      }
    }
    if (DMA_PARTIAL_BYTES)
    {
      switch (partial_bytes)
      {
        case 0:
          break;
        case 4:
          {
            float tmp;
            perform_load<float>(src_ptr,&tmp);
            perform_store<float>(&tmp,dst_ptr);
            break;
          }
        case 8:
          {
            float2 tmp;
            perform_load<float2>(src_ptr,&tmp);
            perform_store<float2>(&tmp,dst_ptr);
            break; 
          }
        case 12:
          {
            float3 tmp;
            perform_load<float3>(src_ptr,&tmp);
            perform_store<float3>(&tmp,dst_ptr);
            break;
          }
        case 16:
          {
            float4 tmp;
            perform_load<float4>(src_ptr,&tmp);
            perform_store<float4>(&tmp,dst_ptr);
            break;
          }
#ifdef CUDADMA_DEBUG_ON
        default:
          printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
          assert(false);
#endif
      }
    }
#undef PERFORM_MEMORY_OPS
  }

  template<bool DO_SYNC, int DMA_ROW_ITERS>
  __device__ __forceinline__ void do_strided_across(const char *RESTRICT src_ptr,
  						    char       *RESTRICT dst_ptr,
						    const int src_elmt_stride,
						    const int dst_elmt_stride,
                                                    const int partial_bytes) const
  {
    // Note we need to have two switch statements in here because we can't
    // have branches that might cause warp divergence with synchronization
    // calls inside of them, otherwise we get deadlock.
    float temp[GUARD_ZERO(DMA_ROW_ITERS*ALIGNMENT/sizeof(float))];
    // Perform the loads
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
        {
          float *temp_ptr = (float*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_load<float>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 8:
        {
          float2 *temp_ptr = (float2*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_load<float2>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 12:
        {
          float3 *temp_ptr = (float3*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_load<float3>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 16:
        {
          float4 *temp_ptr = (float4*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_load<float4>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
    // Check for the synchronization 
    if (DO_SYNC)
      CUDADMA_BASE::wait_for_dma_start();
    // Perform the stores
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
        {
          const float *temp_ptr = (const float*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_store<float>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 8:
        {
          const float2 *temp_ptr = (const float2*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_store<float2>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 12:
        {
          const float3 *temp_ptr = (const float3*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_store<float3>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 16:
        {
          const float4 *temp_ptr = (const float4*)temp;
          for (int idx = 0; idx < DMA_ROW_ITERS; idx++)
          {
            perform_store<float4>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
  }

  // A similar function to the one above, but upper bounded on the number of rows instead
  // of giving the exact number
  template<typename BULK_TYPE, bool DO_SYNC, bool DMA_ALL_ACTIVE, int DMA_ROW_ITERS_UPPER, int DMA_COL_ITERS>
  __device__ __forceinline__ void do_strided_upper(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr,
  					     const int src_elmt_stride, const int dst_elmt_stride,
					     const int intra_elmt_stride, const int partial_bytes,
					     const int row_iters) const
  {
#ifdef CUDADMA_DEBUG_ON
    assert(row_iters < GUARD_ZERO(DMA_ROW_ITERS_UPPER));
#endif
    BULK_TYPE temp[GUARD_ZERO(DMA_ROW_ITERS_UPPER*DMA_COL_ITERS)];
    // Perform loads from the source
    {
      unsigned idx = 0;
      const char *src_row_ptr = src_ptr;
      for (int row = 0; row < row_iters; row++)
      {
        const char *src_col_ptr = src_row_ptr;
        for (int col = 0; col < DMA_COL_ITERS; col++)
        {
  	  perform_load<BULK_TYPE>(src_col_ptr, &(temp[idx++]));
	  src_col_ptr += intra_elmt_stride;
        }
        src_row_ptr += src_elmt_stride; 
      }
    }
    if (!DMA_ALL_ACTIVE)
    {
      // Can't have a branch which might split a warp have a sync operation inside it
      // because this will cause things to deadlock.
      do_strided_across_upper<DO_SYNC,DMA_ROW_ITERS_UPPER>(src_ptr+dma_partial_offset,
                                                           dst_ptr+dma_partial_offset,
                                                           src_elmt_stride, dst_elmt_stride,
                                                           row_iters, partial_bytes);
    }
    else if (DO_SYNC) // Otherwise check to see if we should do the sync here
    {
      CUDADMA_BASE::wait_for_dma_start();
    }
    // Perform the destination stores
    {
      unsigned idx = 0;
      char *dst_row_ptr = dst_ptr;
      for (int row = 0; row < row_iters; row++)
      {
	char *dst_col_ptr = dst_row_ptr;
	for (int col = 0; col < DMA_COL_ITERS; col++)
	{
	  perform_store<BULK_TYPE>(&(temp[idx++]), dst_col_ptr);
	  dst_col_ptr += intra_elmt_stride;
	}
	dst_row_ptr += dst_elmt_stride;
      }
    }
  }

  template<bool DO_SYNC, int DMA_ROW_ITERS_UPPER>
  __device__ __forceinline__ void do_strided_across_upper(const char *RESTRICT src_ptr,
  						    char       *RESTRICT dst_ptr,
						    const int src_elmt_stride,
						    const int dst_elmt_stride,
						    const int row_iters,
                                                    const int partial_bytes) const
  {
#ifdef CUDADMA_DEBUG_ON
    assert(row_iters < GUARD_ZERO(DMA_ROW_ITERS_UPPER));
#endif
    // Note this is an upper bound on the number of registers needed
    float temp[GUARD_ZERO(DMA_ROW_ITERS_UPPER*ALIGNMENT/sizeof(float))];
    // Perform the loads
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
        {
          float *temp_ptr = (float*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_load<float>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 8:
        {
          float2 *temp_ptr = (float2*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_load<float2>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 12:
        {
          float3 *temp_ptr = (float3*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_load<float3>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 16:
        {
          float4 *temp_ptr = (float4*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_load<float4>(src_ptr, temp_ptr);
            src_ptr += src_elmt_stride;
            temp_ptr++;
          }
          break;
        }
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
    // Check for synchronization
    if (DO_SYNC)
      CUDADMA_BASE::wait_for_dma_start();
    // Perform the stores
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
        {
          const float *temp_ptr = (const float*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_store<float>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 8:
        {
          const float2 *temp_ptr = (const float2*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_store<float2>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 12:
        {
          const float3 *temp_ptr = (const float3*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_store<float3>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
      case 16:
        {
          const float4 *temp_ptr = (const float4*)temp;
          for (int idx = 0; idx < row_iters; idx++)
          {
            perform_store<float4>(temp_ptr, dst_ptr);
            dst_ptr += dst_elmt_stride;
            temp_ptr++;
          }
          break;
        }
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
  }

private:
  const unsigned int dma_src_offset;
  const unsigned int dma_dst_offset;
  const unsigned int dma_src_step_stride;
  const unsigned int dma_dst_step_stride;
  const unsigned int dma_src_elmt_stride;
  const unsigned int dma_dst_elmt_stride;
  const unsigned int dma_intra_elmt_stride;
  const unsigned int dma_partial_bytes;
  const unsigned int dma_partial_offset;
  const unsigned int dma_partial_elmts;
  const bool         dma_active_warp;
};

