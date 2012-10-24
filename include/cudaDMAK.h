/*
 *  Copyright 2012 NVIDIA Corporation
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
#define GUARD_UNDERFLOW(expr) (((expr) < 0) ? 0 : (expr))
#define GUARD_ZERO(expr) (((expr) == 0) ? 1 : (expr))
#define GUARD_OVERFLOW(index,max) ((index < max) ? index : max-1)

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
#define ENABLE_LDG

#ifdef ENABLE_LDG
template<typename T>
__device__ __forceinline__
T __ldg_intr(const T* ptr)
{
  return __ldg(ptr);
}

// Special case for float3 because the compiler team hates me 
template<>
__device__ __forceinline__
float3 __ldg_intr(const float3* ptr)
{
  float3 result; 
  asm volatile("ld.global.nc.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(ptr) : "memory");
  asm volatile("ld.global.nc.f32 %0, [%1+8];" : "=f"(result.z) : "l"(ptr) : "memory");
  return result;
}
#endif

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
public:
  template<typename T>
  static __device__ __forceinline__ void perform_load(const void *src_ptr, void *dst_ptr)
  {
#ifdef ENABLE_LDG
    *((T*)dst_ptr) = __ldg_intr(((T*)src_ptr));
#else
    *((T*)dst_ptr) = *((T*)src_ptr);
#endif
  }
  template<typename T>
  static __device__ __forceinline__ void perform_store(const void *src_ptr, void *dst_ptr)
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


/**
 * CudaDMAStrided
 * 
 * This is a simple description of the strategy for CudaDMAStrided.  There
 * are three primary cases in which problems are placed depending on the
 * size of the elements, the number of DMA threads, and the number of 
 * registers available for outstanding loads.  We define a 'step' to
 * be the issuing a group of loads up to the maximum number permitted
 * by the user-specified number of available registers and then 
 * writing them back into memory.
 *
 * The first case that we handle is the 'split' case.  This occurs 
 * whenever elements require 32 or fewer loads to be transferred.  In
 * this case we can split a warp across multiple elements.  Warps 
 * are still split into powers of 2 to avoid warp divergence (see THREADS_PER_ELMT).
 *
 * The second case that we handle is the 'big' elements case.  Big
 * here is a relative term dependent on the number of DMA threads
 * and the number of permitted outstanding loads.  We define a big
 * element to be one that can't be loaded by all the DMA warps
 * in a single step.  In this case we assign all the DMA warps to
 * a single element at a time and have them perform as many steps
 * as necessary to load the element.  All the warps then move onto
 * the next element.
 *
 * The final case ('full') handles all the remaining problems.  We know in this
 * case that we always have enough warps to 'cover' an element: that is
 * we can always assign enough warps to an element to load it in a single
 * step.  We assign the minimum number of warps necessary to cover an
 * element unless this will result in unused warps due to a low element
 * count.  If we find that there would be unused warps due to a low
 * element count, we assign as many warps as possible to an element
 * such that they will all be busy to maximize memory-level parallelism.
 *
 * After assigning warps for the 'full' case, warps figure out how
 * many total loads they need to perform for each element, and then
 * based on the number of registers they have, they can compute
 * how many elements they can handle in a single step.
 *
 * To optimize we differentiate code paths.  In the case where
 * an entire transfer can be performed in a single step, we have
 * a fast path to enable pre-loading of elements into registers.
 * For cases where we know the exact size of transfers, we
 * have optimized paths as well.  For the unoptimized path
 * we still place upper bounds on the number of required registers
 * to enable the compiler to optimize register allocation.
 */
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
#define HAS_PARTIAL_BYTES_SPLIT ((BYTES_PER_ELMT % (THREADS_PER_ELMT*ALIGNMENT)) != 0)
#define COL_ITERS_SPLIT  ((BYTES_PER_ELMT == (THREADS_PER_ELMT*ALIGNMENT)) ? 1 : 0)
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
                        ((NUM_WARPS/MINIMUM_COVER) <= NUM_ELMTS) ? MINIMUM_COVER : \
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
#define HAS_PARTIAL_BYTES_FULL ((BYTES_PER_ELMT % (WARPS_PER_ELMT*WARP_SIZE*ALIGNMENT)) != 0)
#define COL_ITERS_FULL (BYTES_PER_ELMT/(WARPS_PER_ELMT*WARP_SIZE*ALIGNMENT))
#define STEP_ITERS_FULL (NUM_ELMTS/ELMT_PER_STEP_FULL)

#define HAS_PARTIAL_BYTES (SPLIT_WARP ? HAS_PARTIAL_BYTES_SPLIT : \
                           BIG_ELMTS ? HAS_PARTIAL_BYTES_BIG : HAS_PARTIAL_BYTES_FULL)
#define HAS_PARTIAL_ELMTS (SPLIT_WARP ? HAS_PARTIAL_ELMTS_SPLIT : \
                           BIG_ELMTS ? HAS_PARTIAL_ELMTS_BIG : HAS_PARTIAL_ELMTS_FULL)

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
#define REMAINING_BYTES_FULL (BYTES_PER_ELMT - (COL_ITERS_FULL*WARPS_PER_ELMT*WARP_SIZE*ALIGNMENT))
#define REMAINING_LOADS_FULL (FULL_LDS_PER_ELMT % (WARPS_PER_ELMT * WARP_SIZE))
// Same three cases as for split
#define INIT_PARTIAL_BYTES_FULL ((REMAINING_BYTES_FULL==0) ? 0 : \
                                 (FULL_GROUP_TID > (REMAINING_BYTES_FULL/ALIGNMENT)) ? 0 : \
                                 (FULL_GROUP_TID == (REMAINING_BYTES_FULL/ALIGNMENT)) ? \
                                                  (BYTES_PER_ELMT - (FULL_LDS_PER_ELMT*ALIGNMENT)) : ALIGNMENT)
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
  PRINT_VAR(MINIMUM_COVER);
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
protected:
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
    if (DO_SYNC)
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
      // Perform the synchronization
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
    else // The case with no synchronization so we only need one switch statement
    {
#define PERFORM_LOADS(_type)                                \
      _type temp[DMA_ROW_ITERS];                            \
      for (int idx = 0; idx < DMA_ROW_ITERS; idx++)         \
      {                                                     \
        perform_load<_type>(src_ptr,&(temp[idx]));          \
        src_ptr += src_elmt_stride;                         \
      }                                                     \
      for (int idx = 0; idx < DMA_ROW_ITERS; idx++)         \
      {                                                     \
        perform_store<_type>(&(temp[idx]),dst_ptr);         \
        dst_ptr += dst_elmt_stride;                         \
      }
      switch (partial_bytes)
      {
        case 0:
          break;
        case 4:
          {
            PERFORM_LOADS(float);
            break;
          }
        case 8:
          {
            PERFORM_LOADS(float2);
            break;
          }
        case 12:
          {
            PERFORM_LOADS(float3);
            break;
          }
        case 16:
          {
            PERFORM_LOADS(float4);
            break;
          }
#ifdef CUDADMA_DEBUG_ON
        default:
          printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
          assert(false);
#endif
      }
#undef PERFORM_LOADS
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
    assert(row_iters < GUARD_ZERO(DMA_ROW_ITERS_UPPER*DMA_COL_ITERS));
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
    if (DO_SYNC)
    {
      // See the note above in do_strided across about why we
      // need two separate switch statements
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
      // Synchronize
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
    else // The case without synchronization so we need only one switch statement
    {
#define PERFORM_LOADS(_type)                                \
      _type temp[DMA_ROW_ITERS_UPPER];                      \
      for (int idx = 0; idx < row_iters; idx++)             \
      {                                                     \
        perform_load<_type>(src_ptr,&(temp[idx]));          \
        src_ptr += src_elmt_stride;                         \
      }                                                     \
      for (int idx = 0; idx < row_iters; idx++)             \
      {                                                     \
        perform_store<_type>(&(temp[idx]),dst_ptr);         \
        dst_ptr += dst_elmt_stride;                         \
      }
      switch (partial_bytes)
      {
        case 0:
          break;
        case 4:
          {
            PERFORM_LOADS(float);
            break;
          }
        case 8:
          {
            PERFORM_LOADS(float2);
            break;
          }
        case 12:
          {
            PERFORM_LOADS(float3);
            break;
          }
        case 16:
          {
            PERFORM_LOADS(float4);
            break;
          }
#ifdef CUDADMA_DEBUG_ON
        default:
          printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
          assert(false);
#endif
      }
#undef PERFORM_LOADS
    }
  }

protected:
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

// Helper namespace for some template meta-programming structures that we need
// for the two phase implementations of the code so that the compiler generates good code.
namespace CudaDMAMeta {
    template<typename ET, unsigned LS>
    class DMABuffer {
    public:
      template<unsigned IDX>
      __device__ __forceinline__
      ET& get_ref(void) { return get_ref_impl<GUARD_OVERFLOW(IDX,LS)>(); }
      template<unsigned IDX>
      __device__ __forceinline__
      const ET& get_ref(void) const { return get_ref_impl<GUARD_OVERFLOW(IDX,LS)>(); }
      template<unsigned IDX>
      __device__ __forceinline__
      ET* get_ptr(void) { return get_ptr_impl<GUARD_OVERFLOW(IDX,LS)>(); }
      template<unsigned IDX>
      __device__ __forceinline__
      const ET* get_ptr(void) const { return get_ptr_impl<GUARD_OVERFLOW(IDX,LS)>(); }
    public:
      template<unsigned IDX>
      __device__ __forceinline__
      void perform_load(const void *src_ptr) { perform_load_impl<GUARD_OVERFLOW(IDX,LS)>(src_ptr); }
      template<unsigned IDX>
      __device__ __forceinline__
      void perform_store(void *dst_ptr) const { perform_store_impl<GUARD_OVERFLOW(IDX,LS)>(dst_ptr); }
    private:
      template<unsigned IDX>
      __device__ __forceinline__
      ET& get_ref_impl(void) { return buffer[IDX]; }
      template<unsigned IDX>
      __device__ __forceinline__
      const ET& get_ref_impl(void) const { return buffer[IDX]; }
      template<unsigned IDX>
      __device__ __forceinline__
      ET* get_ptr_impl(void) { return &buffer[IDX]; }
      template<unsigned IDX>
      __device__ __forceinline__
      const ET* get_ptr_impl(void) const { return &buffer[IDX]; }
    private:
      template<unsigned IDX>
      __device__ __forceinline__
      void perform_load_impl(const void *src_ptr)
      {
#ifdef ENABLE_LDG
	buffer[IDX] = __ldg_intr(((const ET*)src_ptr));
#else
	buffer[IDX] = *((const ET*)src_ptr);
#endif
      }
      template<unsigned IDX>
      __device__ __forceinline__
      void perform_store_impl(void *dst_ptr) const
      {
      	*((ET*)dst_ptr) = buffer[IDX];
      }
    private:
      ET buffer[LS];
    };

    // These following templates are ways of statically forcing the compiler to unroll the
    // loops for loading and storing into/from the DMABuffer objects.
    template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, unsigned IDX>
    struct BufferLoader {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned stride)
      {
	buffer.template perform_load<OFFSET+MAX-IDX>(src);
	BufferLoader<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE>::load_all(buffer, src+stride, stride);
      }
    };

    template<typename BUFFER, unsigned OFFSET, unsigned STRIDE>
    struct BufferLoader<BUFFER,OFFSET,STRIDE,0,0>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned stride)
      {
        // do nothing
      }
    };

    template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX>
    struct BufferLoader<BUFFER,OFFSET,STRIDE,MAX,1>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned stride)
      {
	buffer.template perform_load<OFFSET+MAX-1>(src);
      }
    };

    template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, unsigned IDX>
    struct BufferStorer {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned stride)
      {
	buffer.template perform_store<OFFSET+MAX-IDX>(dst);
	BufferStorer<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE>::store_all(buffer, dst+stride, stride);
      }
    };

    template<typename BUFFER, unsigned OFFSET, unsigned STRIDE>
    struct BufferStorer<BUFFER,OFFSET,STRIDE,0,0>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned stride)
      {
        // do nothing
      }
    };

    template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX>
    struct BufferStorer<BUFFER,OFFSET,STRIDE,MAX,1>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned stride)
      {
	buffer.template perform_store<OFFSET+MAX-1>(dst);
      }
    };

    // Higher-order static looping constructs for supporting nested loops
    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE, 
    		unsigned OUT_MAX, unsigned OUT_IDX>
    struct NestedBufferLoader {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned outer_stride, unsigned inner_stride)
      {
	BufferLoader<BUFFER,(OUT_MAX-OUT_IDX)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::load_all(buffer, src, inner_stride);
	NestedBufferLoader<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,OUT_IDX-OUT_STRIDE>::load_all(buffer,src+outer_stride,outer_stride,inner_stride);
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE>
    struct NestedBufferLoader<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,0,0>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned outer_stride, unsigned inner_stride)
      {
	// do nothing
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE, unsigned OUT_MAX>
    struct NestedBufferLoader<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,1>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned outer_stride, unsigned inner_stride)
      {
	BufferLoader<BUFFER,(OUT_MAX-1)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::load_all(buffer, src, inner_stride);
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE,
    		unsigned OUT_MAX, unsigned OUT_IDX>
    struct NestedBufferStorer {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned outer_stride, unsigned inner_stride)
      {
	BufferStorer<BUFFER,(OUT_MAX-OUT_IDX)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::store_all(buffer, dst, inner_stride);
	NestedBufferStorer<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,OUT_IDX-OUT_STRIDE>::store_all(buffer,dst+outer_stride,outer_stride,inner_stride);
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE>
    struct NestedBufferStorer<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,0,0>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned outer_stride, unsigned inner_stride)
      {
	// do nothing
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE, unsigned OUT_MAX>
    struct NestedBufferStorer<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,1>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned outer_stride, unsigned inner_stride)
      {
	BufferStorer<BUFFER,(OUT_MAX-1)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::store_all(buffer, dst, inner_stride);
      }
    };

    // Conditional versions of the structs above iterating statically
    // and evaluating a condition
    template<typename BUFFER, int STRIDE, int MAX, int IDX>
    struct ConditionalLoader {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, int stride, int actual_max)
      {
	if ((MAX-IDX) < actual_max)
	{
	  buffer.template perform_load<MAX-IDX>(src);
	  ConditionalLoader<BUFFER,STRIDE,MAX,IDX-STRIDE>::load_all(buffer, src+stride, stride, actual_max);
	}
      }
    };

    template<typename BUFFER, int STRIDE>
    struct ConditionalLoader<BUFFER,STRIDE,0,0>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, int stride, int actual_max)
      {
	// do nothing
      }
    };

    template<typename BUFFER, int STRIDE, int MAX>
    struct ConditionalLoader<BUFFER,STRIDE,MAX,1>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, int stride, int actual_max)
      {
	if ((MAX-1) < actual_max)
	  buffer.template perform_load<MAX-1>(src);
      }
    };

    template<typename BUFFER, int STRIDE, int MAX, int IDX>
    struct ConditionalStorer {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, int stride, int actual_max)
      {
	if ((MAX-IDX) < actual_max)
	{
	  buffer.template perform_store<MAX-IDX>(dst);
	  ConditionalStorer<BUFFER,STRIDE,MAX,IDX-STRIDE>::store_all(buffer, dst+stride, stride, actual_max);
	}
      }
    };

    template<typename BUFFER, int STRIDE>
    struct ConditionalStorer<BUFFER,STRIDE,0,0>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, int stride, int actual_max)
      {
	// do nothing
      }
    };

    template<typename BUFFER, int STRIDE, int MAX>
    struct ConditionalStorer<BUFFER,STRIDE,MAX,1>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, int stride, int actual_max)
      {
	if ((MAX-1) < actual_max)
	  buffer.template perform_store<MAX-1>(dst);
      }
    };

    // Higher-order static looping constructs for supporting conditional nested loops
    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE, 
    		unsigned OUT_MAX, unsigned OUT_IDX>
    struct NestedConditionalLoader {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned outer_stride, unsigned inner_stride, unsigned actual_max)
      {
        if ((OUT_MAX-OUT_IDX) < actual_max)
	{
	  BufferLoader<BUFFER,(OUT_MAX-OUT_IDX)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::load_all(buffer, src, inner_stride);
	  NestedConditionalLoader<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,OUT_IDX-OUT_STRIDE>::load_all(buffer,src+outer_stride,outer_stride,inner_stride,actual_max);
        }
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE>
    struct NestedConditionalLoader<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,0,0>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned outer_stride, unsigned inner_stride, unsigned actual_max)
      {
	// do nothing
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE, unsigned OUT_MAX>
    struct NestedConditionalLoader<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,1>
    {
      static __device__ __forceinline__
      void load_all(BUFFER &buffer, const char *src, unsigned outer_stride, unsigned inner_stride, unsigned actual_max)
      {
        if ((OUT_MAX-1) < actual_max)
	  BufferLoader<BUFFER,(OUT_MAX-1)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::load_all(buffer, src, inner_stride);
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE,
    		unsigned OUT_MAX, unsigned OUT_IDX>
    struct NestedConditionalStorer {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned outer_stride, unsigned inner_stride, unsigned actual_max)
      {
        if ((OUT_MAX-OUT_IDX) < actual_max)
	{
	  BufferStorer<BUFFER,(OUT_MAX-OUT_IDX)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::store_all(buffer, dst, inner_stride);
	  NestedConditionalStorer<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,OUT_IDX-OUT_STRIDE>::store_all(buffer,dst+outer_stride,outer_stride,inner_stride,actual_max);
	}
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE>
    struct NestedConditionalStorer<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,0,0>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned outer_stride, unsigned inner_stride, unsigned actual_max)
      {
	// do nothing
      }
    };

    template<typename BUFFER, unsigned IN_STRIDE, unsigned IN_MAX, unsigned OUT_SCALE, unsigned OUT_STRIDE, unsigned OUT_MAX>
    struct NestedConditionalStorer<BUFFER,IN_STRIDE,IN_MAX,OUT_SCALE,OUT_STRIDE,OUT_MAX,1>
    {
      static __device__ __forceinline__
      void store_all(const BUFFER &buffer, char *dst, unsigned outer_stride, unsigned inner_stride, unsigned actual_max)
      {
        if ((OUT_MAX-1) < actual_max)
	  BufferStorer<BUFFER,(OUT_MAX-1)*OUT_SCALE,IN_STRIDE,IN_MAX,IN_MAX>::store_all(buffer, dst, inner_stride);
      }
    };
};

/**
 * This is a different version of CudaDMAStrided in the new model that instead of having
 * a single execute_dma call will break the code into separate begin_xfer and commit_xfer
 * calls that will break the transfer into two parts: one for issuing the loads and
 * one for writing into shared memory.
 */

#define CUDADMA_STRIDED_BASE cudaDMAStrided<DO_SYNC_TOP,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
#define MIN_UPDATE(type) (((sizeof(type) % ALIGNMENT) == 0) ? sizeof(type) : ALIGNMENT*((sizeof(type)+ALIGNMENT-1)/ALIGNMENT))

template<bool DO_SYNC_TOP, typename ALIGNMENT_TYPE, int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class cudaDMAStridedTwoPhase : public CUDADMA_STRIDED_BASE
{
public:
  // Constructor for when dst_stride == BYTES_PER_ELMT
  __device__ cudaDMAStridedTwoPhase (const int dmaID,
			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int el_stride)
    : CUDADMA_STRIDED_BASE(dmaID,num_compute_threads,dma_threadIdx_start,el_stride)
  {
  }

  __device__ cudaDMAStridedTwoPhase (const int dmaID,
  			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int src_stride,
			     const int dst_stride)
    : CUDADMA_STRIDED_BASE(dmaID,num_compute_threads,dma_threadIdx_start,src_stride,dst_stride)
  {
  }
public:
  __device__ __forceinline__ void begin_xfer_async(const void *RESTRICT src_ptr)  
  {
    switch (ALIGNMENT)
    {
      case 4:
      	{
          execute_begin_xfer<float,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(src_ptr);
	  break;
	}
    case 8:
      	{
	  execute_begin_xfer<float2,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(src_ptr);
	  break;
	}
    case 16:
      	{
	  execute_begin_xfer<float4,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(src_ptr);
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

  __device__ __forceinline__ void wait_xfer_commit(void *RESTRICT dst_ptr) 
  {
    switch (ALIGNMENT)
    {
      case 4:
      	{
          execute_xfer_commit<float,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(dst_ptr);
	  break;
	}
    case 8:
      	{
	  execute_xfer_commit<float2,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(dst_ptr);
	  break;
	}
    case 16:
      	{
	  execute_xfer_commit<float4,SPLIT_WARP,BIG_ELMTS,
                                  STEP_ITERS_SPLIT,ROW_ITERS_SPLIT,COL_ITERS_SPLIT,
                                  STEP_ITERS_BIG,MAX_ITERS_BIG,PART_ITERS_BIG,
	  			  STEP_ITERS_FULL,ROW_ITERS_FULL,COL_ITERS_FULL,
				  HAS_PARTIAL_BYTES,HAS_PARTIAL_ELMTS,ALL_WARPS_ACTIVE>(dst_ptr);
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
protected: // begin xfer functions
  template<typename BULK_TYPE, bool DMA_IS_SPLIT, bool DMA_IS_BIG,
           int DMA_STEP_ITERS_SPLIT, int DMA_ROW_ITERS_SPLIT, int DMA_COL_ITERS_SPLIT,
           int DMA_STEP_ITERS_BIG,   int DMA_MAX_ITERS_BIG,   int DMA_PART_ITERS_BIG,
           int DMA_STEP_ITERS_FULL,  int DMA_ROW_ITERS_FULL,  int DMA_COL_ITERS_FULL,
	   bool DMA_PARTIAL_BYTES, bool DMA_PARTIAL_ROWS, bool DMA_ALL_WARPS_ACTIVE>
  __device__ __forceinline__ void execute_begin_xfer(const void *RESTRICT src_ptr)
  {
    // Note we can only issue at most one step here so we have to save the src_off_ptr
    this->dma_src_off_ptr = ((const char*)src_ptr) + this->dma_src_offset;
    if (DMA_IS_SPLIT)
    {
      if (DMA_STEP_ITERS_SPLIT == 0)
      {
	load_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
				DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(this->dma_src_off_ptr);
      }
      else
      {
	load_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,false/*partial rows*/,
				DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(this->dma_src_off_ptr);
      }
    }
    else if (DMA_IS_BIG)
    {
      // No optimized case here we have at least one full column iteration to perform
      // so we'll just wait until we can do everything
    }
    else // Not split and not big
    {
      if (DMA_ALL_WARPS_ACTIVE)
      {
	if (DMA_STEP_ITERS_FULL == 0)
	{
	  load_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
	  			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr);
	}
	else
	{
	  load_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,false/*partial rows*/,
	   			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr);
	}
      }
      else if (this->dma_active_warp)
      {
	if (DMA_STEP_ITERS_FULL == 0)
	{
	  load_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
	  			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr);
	}
	else
	{
	  load_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,false/*partial rows*/,
	  			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr);
	}
      }
    }
  }

  template<typename BULK_TYPE, bool DMA_PARTIAL_BYTES, bool DMA_PARTIAL_ROWS,
  		int DMA_ROW_ITERS, int DMA_COL_ITERS>
  __device__ __forceinline__ void load_all_partial_cases(const char *RESTRICT src_ptr)
  {
    if (!DMA_PARTIAL_BYTES)
    {
      if (!DMA_PARTIAL_ROWS)
      {
	load_strided<BULK_TYPE,true/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, this->dma_src_elmt_stride,
			 this->dma_intra_elmt_stride, 0/*no partial bytes*/);
      }
      else
      {
	load_strided_upper<BULK_TYPE,true/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, this->dma_src_elmt_stride,
			 this->dma_intra_elmt_stride, 0/*no partial bytes*/, this->dma_partial_elmts);
      }
    }
    else
    {
      if (!DMA_PARTIAL_ROWS)
      {
        load_strided<BULK_TYPE,false/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, this->dma_src_elmt_stride, 
			 this->dma_intra_elmt_stride, this->dma_partial_bytes);
      }
      else
      {
	load_strided_upper<BULK_TYPE,false/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(src_ptr, this->dma_src_elmt_stride, 
			 this->dma_intra_elmt_stride, this->dma_partial_bytes, this->dma_partial_elmts);
      }
    }
  }

  template<typename BULK_TYPE, bool DMA_ALL_ACTIVE, int DMA_ROW_ITERS, int DMA_COL_ITERS>
  __device__ __forceinline__ void load_strided(const char *RESTRICT src_ptr,
  				  	       const int src_elmt_stride, 
					       const int intra_elmt_stride, const int partial_bytes)
  {
#if 0
    const char *src_row_ptr = src_ptr;
    for (int row = 0; row < DMA_ROW_ITERS; row++)
    {
      CudaDMAMeta::BufferLoader<BulkBuffer,1,GUARD_UNDERFLOW(DMA_COL_ITERS-1),GUARD_UNDERFLOW(DMA_COL_ITERS-1)>::load_all(bulk_buffer, src_row_ptr, intra_elmt_stride);
      src_row_ptr += src_elmt_stride;
    }
#else
    CudaDMAMeta::NestedBufferLoader<BulkBuffer,1,DMA_COL_ITERS,DMA_COL_ITERS,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::load_all(bulk_buffer,src_ptr,src_elmt_stride,intra_elmt_stride);
#endif
    if (!DMA_ALL_ACTIVE)
    {
      load_across<DMA_ROW_ITERS>(src_ptr+this->dma_partial_offset,
                         src_elmt_stride, partial_bytes);
    }
  }

  template<typename BULK_TYPE, bool DMA_ALL_ACTIVE, int DMA_ROW_ITERS_UPPER, int DMA_COL_ITERS>
  __device__ __forceinline__ void load_strided_upper(const char *RESTRICT src_ptr,
  						const int src_elmt_stride, const int intra_elmt_stride,
						const int partial_bytes, const int row_iters)
  {
#if 0
    const char *src_row_ptr = src_ptr;
    for (int row = 0; row < row_iters; row++)
    {
      CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_COL_ITERS-1),GUARD_UNDERFLOW(DMA_COL_ITERS-1)>::load_all(bulk_buffer,src_row_ptr,intra_elmt_stride);
      src_row_ptr += src_elmt_stride;
    }
#else
    CudaDMAMeta::NestedConditionalLoader<BulkBuffer,1,DMA_COL_ITERS,DMA_COL_ITERS,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::load_all(bulk_buffer,src_ptr,src_elmt_stride,intra_elmt_stride,row_iters);
#endif
    if (!DMA_ALL_ACTIVE)
    {
      load_across_upper<DMA_ROW_ITERS_UPPER>(src_ptr+this->dma_partial_offset,
			     src_elmt_stride, partial_bytes, row_iters);
    }
  }

  template<int DMA_ROW_ITERS>
  __device__ __forceinline__ void load_across(const char *RESTRICT src_ptr,
  					      const int src_elmt_stride, const int partial_bytes)
					      //void *temp)
  {
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
	{
	  CudaDMAMeta::BufferLoader<AcrossBuffer1,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::load_all(across_one, src_ptr, src_elmt_stride);
	  break;
	}
      case 8:
	{
	  CudaDMAMeta::BufferLoader<AcrossBuffer2,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::load_all(across_two, src_ptr, src_elmt_stride);
	  break;
	}
      case 12:
      	{
	  CudaDMAMeta::BufferLoader<AcrossBuffer3,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::load_all(across_three, src_ptr, src_elmt_stride);
	  break;
	}
      case 16:
        {
	  CudaDMAMeta::BufferLoader<AcrossBuffer4,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::load_all(across_four, src_ptr, src_elmt_stride);
	  break;
	}
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
  }

  template<int DMA_ROW_ITERS_UPPER>
  __device__ __forceinline__ void load_across_upper(const char *RESTRICT src_ptr,
  					      const int src_elmt_stride, const int partial_bytes,
					      const int row_iters)
  {
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
	{
	  CudaDMAMeta::ConditionalLoader<AcrossBuffer1,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::load_all(across_one,src_ptr,src_elmt_stride,row_iters);
	  break;
	}
      case 8:
	{
	  CudaDMAMeta::ConditionalLoader<AcrossBuffer2,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::load_all(across_two,src_ptr,src_elmt_stride,row_iters);
	  break;
	}
      case 12:
      	{
	  CudaDMAMeta::ConditionalLoader<AcrossBuffer3,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::load_all(across_three,src_ptr,src_elmt_stride,row_iters);
	  break;
	}
      case 16:
        {
	  CudaDMAMeta::ConditionalLoader<AcrossBuffer4,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::load_all(across_four,src_ptr,src_elmt_stride,row_iters);
	  break;
	}
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
  }

protected: // commit xfer functions
  template<typename BULK_TYPE, bool DMA_IS_SPLIT, bool DMA_IS_BIG,
           int DMA_STEP_ITERS_SPLIT, int DMA_ROW_ITERS_SPLIT, int DMA_COL_ITERS_SPLIT,
           int DMA_STEP_ITERS_BIG,   int DMA_MAX_ITERS_BIG,   int DMA_PART_ITERS_BIG,
           int DMA_STEP_ITERS_FULL,  int DMA_ROW_ITERS_FULL,  int DMA_COL_ITERS_FULL,
	   bool DMA_PARTIAL_BYTES, bool DMA_PARTIAL_ROWS, bool DMA_ALL_WARPS_ACTIVE>
  __device__ __forceinline__ void execute_xfer_commit(void *RESTRICT dst_ptr) 
  {
    char * dst_off_ptr = ((char*)dst_ptr) + this->dma_dst_offset;
    if (DO_SYNC_TOP)
        CUDADMA_BASE::wait_for_dma_start();
    if (DMA_IS_SPLIT)
    {
      if (DMA_STEP_ITERS_SPLIT == 0)
      {
	store_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
				DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(dst_off_ptr);
      }
      else
      {
        store_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,false/*partial rows*/,
				DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(dst_off_ptr);
	this->dma_src_off_ptr += this->dma_src_step_stride;
	dst_off_ptr += this->dma_dst_step_stride;
	// Only need to do N-1 iterations since we already did the first one
	for (int i = 0; i < (DMA_STEP_ITERS_SPLIT-1); i++)
	{
          CUDADMA_STRIDED_BASE::template all_partial_cases<BULK_TYPE,false/*do sync*/,
                                DMA_PARTIAL_BYTES,false/*partial rows*/,
	  			DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(this->dma_src_off_ptr, dst_off_ptr);
	  this->dma_src_off_ptr += this->dma_src_step_stride;
	  dst_off_ptr += this->dma_dst_step_stride;
        }
	if (DMA_PARTIAL_ROWS)
	{
          CUDADMA_STRIDED_BASE::template all_partial_cases<BULK_TYPE,false/*do sync*/,
                                DMA_PARTIAL_BYTES,true/*partial rows*/,
                                DMA_ROW_ITERS_SPLIT,DMA_COL_ITERS_SPLIT>(this->dma_src_off_ptr, dst_off_ptr);
	}
      }
    }
    else if (DMA_IS_BIG)
    {
      for (int i = 0; i < DMA_STEP_ITERS_BIG; i++)
      {
        CUDADMA_STRIDED_BASE::template do_copy_elmt<BULK_TYPE,DMA_MAX_ITERS_BIG,
                        DMA_PART_ITERS_BIG,DMA_PARTIAL_ROWS,DMA_PARTIAL_BYTES>
            (this->dma_src_off_ptr, dst_off_ptr, this->dma_intra_elmt_stride, this->dma_partial_bytes);
        this->dma_src_off_ptr += this->dma_src_elmt_stride;
        dst_off_ptr += this->dma_dst_elmt_stride;
      }
    }
    else
    {
      if (DMA_ALL_WARPS_ACTIVE)
      {
	if (DMA_STEP_ITERS_FULL == 0)
	{
	  store_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
	  			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(dst_off_ptr);
	}
	else
	{
	  store_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,false/*partial rows*/,
	  			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(dst_off_ptr);
	  this->dma_src_off_ptr += this->dma_src_step_stride;
	  dst_off_ptr += this->dma_dst_step_stride;
	  // Only need to handle N-1 cases now
	  for (int i = 0; i < (DMA_STEP_ITERS_FULL-1); i++)
	  {
            CUDADMA_STRIDED_BASE::template all_partial_cases<BULK_TYPE,false/*do sync*/,
                                DMA_PARTIAL_BYTES,false/*partial rows*/,
	    			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr,dst_off_ptr);
	    this->dma_src_off_ptr += this->dma_src_step_stride;
	    dst_off_ptr += this->dma_dst_step_stride;
	  }
	  if (DMA_PARTIAL_ROWS)
	  {
            CUDADMA_STRIDED_BASE::template all_partial_cases<BULK_TYPE,false/*do sync*/,
                                DMA_PARTIAL_BYTES,true/*partial rows*/,
	    			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr, dst_off_ptr);
	  }
	}
      }
      else if (this->dma_active_warp)
      {
	if (DMA_STEP_ITERS_FULL == 0)
	{
	  store_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,DMA_PARTIAL_ROWS,
				DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(dst_off_ptr);
	}
	else
	{
	  store_all_partial_cases<BULK_TYPE,DMA_PARTIAL_BYTES,false/*partial rows*/,
	  			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(dst_off_ptr);
	  this->dma_src_off_ptr += this->dma_src_step_stride;
	  dst_off_ptr += this->dma_dst_step_stride;
	  // Only need to handle N-1 cases now
	  for (int i = 0; i < (DMA_STEP_ITERS_FULL-1); i++)
	  {
            CUDADMA_STRIDED_BASE::template all_partial_cases<BULK_TYPE,false/*do sync*/,
                                DMA_PARTIAL_BYTES,false/*partial rows*/,
	    			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr, dst_off_ptr);
	    this->dma_src_off_ptr += this->dma_src_step_stride;
	    dst_off_ptr += this->dma_dst_step_stride;
	  }
	  if (DMA_PARTIAL_ROWS)
	  {
            CUDADMA_STRIDED_BASE::template all_partial_cases<BULK_TYPE,false/*do sync*/,
                                DMA_PARTIAL_BYTES,true/*partial rows*/,
	    			DMA_ROW_ITERS_FULL,DMA_COL_ITERS_FULL>(this->dma_src_off_ptr, dst_off_ptr);
	  }
	}
      }
    }
    if (DO_SYNC_TOP)
      CUDADMA_BASE::finish_async_dma();
  }

  template<typename BULK_TYPE, bool DMA_PARTIAL_BYTES, bool DMA_PARTIAL_ROWS,
  		int DMA_ROW_ITERS, int DMA_COL_ITERS>
  __device__ __forceinline__ void store_all_partial_cases(char *RESTRICT dst_ptr)
  {
    if (!DMA_PARTIAL_BYTES)
    {
      if (!DMA_PARTIAL_ROWS)
      {
	store_strided<BULK_TYPE,true/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(dst_ptr, this->dma_dst_elmt_stride,
			 this->dma_intra_elmt_stride, 0/*no partial bytes*/);
      }
      else
      {
#if 1
	store_strided_upper<BULK_TYPE,true/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(dst_ptr, this->dma_dst_elmt_stride,
			 this->dma_intra_elmt_stride, 0/*no partial bytes*/, this->dma_partial_elmts);
#else
	CUDADMA_STRIDED_BASE::template do_strided_upper<BULK_TYPE,false,true,DMA_ROW_ITERS,DMA_COL_ITERS>
		(this->dma_src_off_ptr, dst_ptr, this->dma_src_elmt_stride, this->dma_dst_elmt_stride,
		 this->dma_intra_elmt_stride, 0/*no partial bytes*/, this->dma_partial_elmts);
#endif
      }
    }
    else
    {
      if (!DMA_PARTIAL_ROWS)
      {
	store_strided<BULK_TYPE,false/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(dst_ptr, this->dma_dst_elmt_stride,
			 this->dma_intra_elmt_stride, this->dma_partial_bytes);
      }
      else
      {
#if 1
	store_strided_upper<BULK_TYPE,false/*all active*/,DMA_ROW_ITERS,DMA_COL_ITERS>
			(dst_ptr, this->dma_dst_elmt_stride,
			 this->dma_intra_elmt_stride, this->dma_partial_bytes, this->dma_partial_elmts);
#else
	CUDADMA_STRIDED_BASE::template do_strided_upper<BULK_TYPE,false,false,DMA_ROW_ITERS,DMA_COL_ITERS>
		(this->dma_src_off_ptr, dst_ptr, this->dma_src_elmt_stride, this->dma_dst_elmt_stride,
		 this->dma_intra_elmt_stride, this->dma_partial_bytes, this->dma_partial_elmts);
#endif
      }
    }
  }

  template<typename BULK_TYPE, bool DMA_ALL_ACTIVE, int DMA_ROW_ITERS, int DMA_COL_ITERS>
  __device__ __forceinline__ void store_strided(char *RESTRICT dst_ptr, const int dst_elmt_stride,
  						const int intra_elmt_stride, const int partial_bytes)
  {
#if 0
    char *dst_row_ptr = dst_ptr;
    for (int row = 0; row < DMA_ROW_ITERS; row++)
    {
      CudaDMAMeta::BufferStorer<BulkBuffer,1,GUARD_UNDERFLOW(DMA_COL_ITERS-1),GUARD_UNDERFLOW(DMA_COL_ITERS-1)>::store_all(bulk_buffer,dst_row_ptr, intra_elmt_stride);
      dst_row_ptr += dst_elmt_stride;
    } 
#else
    CudaDMAMeta::NestedBufferStorer<BulkBuffer,1,DMA_COL_ITERS,DMA_COL_ITERS,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::store_all(bulk_buffer,dst_ptr,dst_elmt_stride,intra_elmt_stride);
#endif
    if (!DMA_ALL_ACTIVE)
    { 
      store_across<DMA_ROW_ITERS>(dst_ptr+this->dma_partial_offset,
			      dst_elmt_stride,partial_bytes);
    }
  }

  template<typename BULK_TYPE, bool DMA_ALL_ACTIVE, int DMA_ROW_ITERS_UPPER, int DMA_COL_ITERS>
  __device__ __forceinline__ void store_strided_upper(char *RESTRICT dst_ptr, const int dst_elmt_stride,
  						      const int intra_elmt_stride, const int partial_bytes,
						      const int row_iters)
  {
#if 0
    char *dst_row_ptr = dst_ptr;
    for (int row = 0; row < row_iters; row++)
    {
      CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_COL_ITERS-1),GUARD_UNDERFLOW(DMA_COL_ITERS-1)>::store_all(bulk_buffer,dst_row_ptr,intra_elmt_stride);
      dst_row_ptr += dst_elmt_stride;
    } 
#else
    CudaDMAMeta::NestedConditionalStorer<BulkBuffer,1,DMA_COL_ITERS,DMA_COL_ITERS,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::store_all(bulk_buffer,dst_ptr,dst_elmt_stride,intra_elmt_stride,row_iters);
#endif
    if (!DMA_ALL_ACTIVE)
    { 
      store_across_upper<DMA_ROW_ITERS_UPPER>(dst_ptr+this->dma_partial_offset,
			      dst_elmt_stride,partial_bytes,row_iters);
    } 
  }

  template<int DMA_ROW_ITERS>
  __device__ __forceinline__ void store_across(char *RESTRICT dst_ptr, const int dst_elmt_stride,
  						       const int partial_bytes)// const void *temp)
  {
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
        {
	  CudaDMAMeta::BufferStorer<AcrossBuffer1,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::store_all(across_one, dst_ptr, dst_elmt_stride);
	  break;
	}
      case 8:
	{
	  CudaDMAMeta::BufferStorer<AcrossBuffer2,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::store_all(across_two,dst_ptr, dst_elmt_stride);
	  break;
	}
      case 12:
	{
	  CudaDMAMeta::BufferStorer<AcrossBuffer3,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::store_all(across_three, dst_ptr, dst_elmt_stride);
	  break;
	}
      case 16:
	{
	  CudaDMAMeta::BufferStorer<AcrossBuffer4,0,1,DMA_ROW_ITERS,DMA_ROW_ITERS>::store_all(across_four ,dst_ptr, dst_elmt_stride);
	  break;
	}
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
  }

  template<int DMA_ROW_ITERS_UPPER>
  __device__ __forceinline__ void store_across_upper(char *RESTRICT dst_ptr, const int dst_elmt_stride,
							     const int partial_bytes, const int row_iters)
  {
    switch (partial_bytes)
    {
      case 0:
        break;
      case 4:
        {
	  CudaDMAMeta::ConditionalStorer<AcrossBuffer1,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::store_all(across_one,dst_ptr,dst_elmt_stride,row_iters);
	  break;
	}
      case 8:
	{
	  CudaDMAMeta::ConditionalStorer<AcrossBuffer2,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::store_all(across_two,dst_ptr,dst_elmt_stride,row_iters);
	  break;
	}
      case 12:
	{
	  CudaDMAMeta::ConditionalStorer<AcrossBuffer3,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::store_all(across_three,dst_ptr,dst_elmt_stride,row_iters);
	  break;
	}
      case 16:
	{
	  CudaDMAMeta::ConditionalStorer<AcrossBuffer4,1,DMA_ROW_ITERS_UPPER,DMA_ROW_ITERS_UPPER>::store_all(across_four,dst_ptr,dst_elmt_stride,row_iters);
	  break;
	}
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid partial bytes size (%d) for strided across.\n" partial_bytes);
        assert(false);
#endif
    }
  }

protected:
  const char * dma_src_off_ptr;
  typedef CudaDMAMeta::DMABuffer<ALIGNMENT_TYPE,GUARD_ZERO(BYTES_PER_THREAD/ALIGNMENT)> BulkBuffer;
  BulkBuffer bulk_buffer;
  typedef CudaDMAMeta::DMABuffer<float,GUARD_ZERO(SPLIT_WARP ? ROW_ITERS_SPLIT : 
  				BIG_ELMTS ? 1 : ROW_ITERS_FULL)*ALIGNMENT/sizeof(float)> AcrossBuffer1;
  typedef CudaDMAMeta::DMABuffer<float2,GUARD_ZERO(SPLIT_WARP ? ROW_ITERS_SPLIT : 
  				BIG_ELMTS ? 1 : ROW_ITERS_FULL)*ALIGNMENT/sizeof(float2)> AcrossBuffer2;
  typedef CudaDMAMeta::DMABuffer<float3,GUARD_ZERO(SPLIT_WARP ? ROW_ITERS_SPLIT : 
  				BIG_ELMTS ? 1 : ROW_ITERS_FULL)*ALIGNMENT/sizeof(float3)> AcrossBuffer3;
  typedef CudaDMAMeta::DMABuffer<float4,GUARD_ZERO(SPLIT_WARP ? ROW_ITERS_SPLIT : 
  				BIG_ELMTS ? 1 : ROW_ITERS_FULL)*ALIGNMENT/sizeof(float4)> AcrossBuffer4;
#if 0
  union AcrossBuffer {
    AcrossBuffer1 one;
    AcrossBuffer2 two;
    AcrossBuffer3 three;
    AcrossBuffer4 four;
  } across_buffer;
#else
  AcrossBuffer1 across_one;
  AcrossBuffer2 across_two;
  AcrossBuffer3 across_three;
  AcrossBuffer4 across_four;
#endif
};


