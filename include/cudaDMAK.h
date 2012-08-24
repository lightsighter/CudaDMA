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
  __device__ __forceinline__ void perform_load(const T *src_ptr, T *dst_ptr) const
  {
#ifdef ENABLE_LDG
    *dst_ptr = __ldg(src_ptr);
#else
    *dst_ptr = *src_ptr;
#endif
  }
  template<typename T>
  __device__ __forceinline__ void perform_store(const T *src_ptr, T *dst_ptr) const
  {
    *dst_ptr = *src_ptr;
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
#define LDS_PER_ELMT_PER_THREAD ((LDS_PER_ELMT+WARP_SIZE-1)/WARP_SIZE)
#define FULL_LDS_PER_ELMT (LDS_PER_ELMT/WARP_SIZE)
#define PARTIAL_LDS_PER_ELMT (((LDS_PER_ELMT%WARP_SIZE)+WARP_SIZE-1)/WARP_SIZE) // either 0 or 1
// note FULL_LDS_PER_ELMT + PARTIAL_LDS_PER_ELMT == LDS_PER_ELMT_PER_THREAD
// The condition for when we split a warp across many elements
#define SPLIT_ELMT (LDS_PER_ELMT_PER_THREAD <= MAX_LDS_PER_THREAD)
// The condition for when we split a warp, figure out how many threads there are per elmt
#define SPLIT_WARP (LDS_PER_ELMT <= WARP_SIZE)
#define THREADS_PER_ELMT (LDS_PER_ELMT > (WARP_SIZE/2) ? WARP_SIZE : \
			 LDS_PER_ELMT > (WARP_SIZE/4) ? (WARP_SIZE/2) : \
			 LDS_PER_ELMT > (WARP_SIZE/8) ? (WARP_SIZE/4) : \
			 LDS_PER_ELMT > (WARP_SIZE/16) ? (WARP_SIZE/8) : \
			 LDS_PER_ELMT > (WARP_SIZE/32) ? (WARP_SIZE/16) : WARP_SIZE/32)
#define DMA_COL_ITER_INC_SPLIT (SPLIT_WARP ? THREADS_PER_ELMT*ALIGNMENT : WARP_SIZE*ALIGNMENT)
#define ELMT_PER_STEP_SPLIT (SPLIT_WARP ? (DMA_THREADS/THREADS_PER_ELMT)*MAX_LDS_PER_THREAD : (DMA_THREADS/WARP_SIZE)*MAX_LDS_PER_THREAD) 
#define ELMT_ID_SPLIT (SPLIT_WARP ? CUDADMA_DMA_TID/THREADS_PER_ELMT : CUDADMA_DMA_TID/WARP_SIZE)
#define REMAINING_ELMTS ((NUM_ELMTS==ELMT_PER_STEP_SPLIT) ? ELMT_PER_STEP_SPLIT : NUM_ELMTS%ELMT_PER_STEP_SPLIT) // Handle the optimized case special
#define PARTIAL_ELMTS (SPLIT_WARP ? REMAINING_ELMTS/(DMA_THREADS/THREADS_PER_ELMT) + (int(ELMT_ID_SPLIT) < (REMAINING_ELMTS%(DMA_THREADS/THREADS_PER_ELMT)) ? 1 : 0) : \
				REMAINING_ELMTS/(DMA_THREADS/WARP_SIZE) + (int(ELMT_ID_SPLIT) < (REMAINING_ELMTS%(DMA_THREADS/WARP_SIZE)) ? 1 : 0))

// Now for the case where SPLIT_ELMT is false (i.e. multiple warps per element)
#define MAX_WARPS_PER_ELMT ((BYTES_PER_ELMT+(WARP_SIZE*BYTES_PER_THREAD-1))/(WARP_SIZE*BYTES_PER_THREAD))
// If we can use all the warps on one element, then do it,
// Otherwise check to see if we are wasting DMA warps due to low element count
// If so allocate more warps to a single element than is necessary to improve MLP
#define WARPS_PER_ELMT ((MAX_WARPS_PER_ELMT >= (DMA_THREADS/WARP_SIZE)) ? (DMA_THREADS/WARP_SIZE) : \
	(DMA_THREADS/WARP_SIZE)>(MAX_WARPS_PER_ELMT*NUM_ELMTS) ? (DMA_THREADS/WARP_SIZE)/NUM_ELMTS : MAX_WARPS_PER_ELMT)
#define ELMT_PER_STEP ((DMA_THREADS/WARP_SIZE)/WARPS_PER_ELMT)
#define ELMT_ID ((CUDADMA_DMA_TID/WARP_SIZE)/WARPS_PER_ELMT)
#define CUDADMA_WARP_TID (threadIdx.x - (dma_threadIdx_start + (ELMT_ID*WARPS_PER_ELMT*WARP_SIZE)))

#define STEP_ITERS_FULL ((NUM_ELMTS==ELMT_PER_STEP) ? 0 : NUM_ELMTS/ELMT_PER_STEP)
#define STEP_ITERS_SPLIT ((NUM_ELMTS==ELMT_PER_STEP_SPLIT) ? 0 : NUM_ELMTS/ELMT_PER_STEP_SPLIT) // Handle the optimized case special

// Bytes-per-thread will manage the number of registers available for each thread
// Bytes-per-thread must be divisible by alignment
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_ELMTS>
class cudaDMAStrided : public CUDADMA_BASE
{
public:
  // Constructor for when dst_stride == BYTES_PER_ELMT
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int el_stride)
    : CUDADMA_BASE(dmaID,DMA_THREADS,num_compute_threads,dma_threadIdx_start),
      src_elmt_stride(ELMT_PER_STEP*el_stride),
      dst_elmt_stride(ELMT_PER_STEP*el_stride)
      ld_stride(THREADS_PER_ELMT*ALIGNMENT),
      num_passes(NUM_ELMTS/ELMT_PER_STEP),
      elmt_per_pass(ELMT_PER_STEP),
      lds_per_elmt(LDS_PER_ELMT/THREADS_PER_ELMT)
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
    : CUDADMA_BASE(dmaID,DMA_THREADS,num_compute_threads,dma_threadIdx_start)
      src_elmt_stride(ELMT_PER_STEP*src_stride),
      dst_elmt_stride(ELMT_PER_STEP*dst_stride)

  {

  }
public:
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) const
  {
    switch (ALIGNMENT)
    {
      case 4:
      	{
          execute_internal<float>(src_ptr,dst_ptr);
	  break;
	}
    case 8:
      	{
          execute_internal<float2>(src_ptr,dst_ptr);
	  break;
	}
    case 16:
      	{
          execute_internal<float4>(src_ptr,dst_ptr);
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

  template<typename BULK_TYPE, int ELMT_LDS, int DMA_STEP_ITERS_FULL, int DMA_STEP_ITERS_SPLIT, int DMA_ROW_ITERS_FULL, int DMA_ROW_ITERS_SPLIT, int DMA_COL_ITERS_FULL, int DMA_COL_ITERS_SPLIT>
  __device__ __forceinline__ void execute_internal(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) const
  {
    if (ELMT_LDS == 1)
    {
      // Loop over elements only
      const char * src_row_ptr = ((const char*)src_ptr) + dma_src_offset;
      char * dst_row_ptr = ((char*)dst_ptr) + dma_dst_offset;
      if (DMA_STEP_ITERS_SPLIT == 0)
      {
      	// Single step
	if (all_threads_active) // The optimized case
	{
	  do_strided_across<DO_SYNC>(src_row_ptr, dst_row_ptr, dma_split_partial_elmts, dma_src_elmt_stride, dma_dst_elmt_stride, partial_bytes);
	}
	else
	{
	  if (DO_SYNC)
	    CUDADMA_BASE::wait_for_dma_start();
	  if (dma_split_partial_elmts > 0)
	    do_strided_across<false>(src_row_ptr, dst_row_ptr, dma_split_partial_elmts, dma_src_elmt_stride, dma_dst_elmt_stride, partial_bytes);
	}
      }
      else
      {
        if (DO_SYNC)
	  CUDADMA_BASE::wait_for_dma_start();
      	// Iterate over the steps 
	for (int i = 0; i < DMA_STEP_ITERS_SPLIT; i++)
	{
	  do_strided_across<false>(src_row_ptr, dst_row_ptr, MAX_LDS_PER_THREAD, dma_src_elmt_stride, dma_dst_elmt_stride, partial_bytes);	
	  src_row_ptr += dma_src_step_stride;
	  dst_row_ptr += dma_dst_step_stride;
	}
	if (dma_split_partial_elmts > 0)
	{
	  do_xfer_across<false>(src_row_ptr, dst_row_ptr, dma_split_partial_elmts, dma_src_elmt_stride, dma_ds_elmt_stride, partial_bytes)
	}
      }
    }
    else if (ELMT_LDS <= MAX_LDS_PER_THREAD)
    {
      const char * src_row_ptr = ((const char*)src_ptr) + dma_src_offset;
      char       * dst_row_ptr = ((char*)dst_ptr)       + dma_dst_offset;
      // Double loop over both elements and loads per element per step
      if (DMA_STEP_ITERS_SPLIT == 0)
      {
	// Single step
	if (DO_SYNC)
	  CUDADMA_BASE::wait_for_dma_start();
	perform_strided<BULK_TYPE,DO_SYNC>(src_row_ptr, dst_row_ptr, dma_split_partial_elmts, dma_lds_per_elmt, dma_src_elmt_stride, dma_dst_elmt_stride, dma_ld_stride); 		
      }
      else
      {
      	// Iterate over the steps 
	for (int i = 0; i< DMA_STEP_ITERS_SPLIT; i++)
	{
	  perform_strided<BULK_TYPE,DO_SYNC>(src_row_ptr, dst_row_ptr, DMA_ROW_ITERS_SPLIT, dma_lds_per_elmt, dma_src_elmt_stride, dma_dst_elmt_stride, dma_ld_stride);
	}
      }
    }
    else
    {
      // Loop over a multiple loads for a single element per step
      if (DMA_STEP_ITERS_FULL == 0)
      {
	// Single step	
      }
      else
      {
        // Iterate over the steps
        for (int i = 0; i < DMA_STEP_ITERS_FULL; i++)
	{

	}
      }
    }
  }

  template<bool DO_SYNC>
  __device__ __forceinline__ void do_strided_across(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr, const unsigned num_elmts, const int src_stride, const int dst_stride, const unsigned xfer_size) const
  {
    switch (partial_bytes)
    {
      case 0:
        {
          if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
	  break;
	}
      case 4:
        {
          perform_strided_across<float,DO_SYNC>(src_ptr, dst_ptr, num_elmts, src_stride, dst_stride);
	  break;
	}
      case 8:
        {
	  perform_strided_across<float2,DO_SYNC>(src_ptr, dst_ptr, num_elmts, src_stride, dst_stride);
	  break;
	}
      case 12:
        {
	  perform_strided_across<float3,DO_SYNC>(src_ptr, dst_ptr, num_elmts, src_stride, dst_stride);
	  break;
	}
      case 16:
        {
	  perform_strided_across<float4,DO_SYNC>(src_ptr, dst_ptr, num_elmts, src_stride, dst_stride);
	  break;
	}
#ifdef CUDADMA_DEBUG_ON
      default:
        printf("Invalid xfer size (%d) for xfer across.\n",xfer_size);
        break;
#endif
    }
  }

  template<typename BULK_TYPE, bool DO_SYNC>
  __device__ __forceinline__ void perform_strided_across(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr, const unsigned num_elmts, const int src_stride, const int dst_stride) const
  {
    BULK_TYPE temp[BYTES_PER_THREAD/sizeof(BULK_TYPE)];
    for (unsigned idx = 0; idx < num_elmts; idx++)
    {
      perform_load<BULK_TYPE>(src_ptr, &temp[idx]);
      src_ptr += src_stride;
    }
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    // WRite all the registers into their destinations
    for (unsigned idx = 0; idx < num_elmts; idx++)
    {
      perform_store<BULK_TYPE>(dst_ptr, &temp[idx]);
      dst_ptr += dst_stride;
    }
  }

  template<typename BULK_TYPE, bool DO_SYNC>
  __device__ __forceinline__ void perform_strided(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr, const unsigned num_elmts, const unsigned num_lds, const int src_stride, const int dst_stride, const int ld_stride) const
  {
    BULK_TYPE temp[BYTES_PER_THREAD/sizeof(BULK_TYPE)];
    unsigned idx = 0;
    for (unsigned elmt = 0; elmt < num_elmts; elmt++)
    {
      const char *temp_src = src_ptr;
      for (unsigned ld = 0; ld < num_lds; ld++)
      {
	perform_load<BULK_TYPE>(temp_src, &temp[idx]);
	temp_src += ld_stride;
	idx++;
      }
      src_ptr += src_stride;
    }
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    idx = 0;
    for (unsigned elmt = 0; elmt < num_elmts; elmt++)
    {
      char *temp_dst = dst_ptr;
      for (unsigned ld = 0; ld < num_lds; ld++)
      {
        perform_store<BULK_TYPE>(temp_dst, &temp[idx]);
	temp_dst += ld_stride;
	idx++;
      }
      dst_ptr += dst_stride;
    }
  }

  template<typename BULK_TYPE, bool DO_SYNC>
  __device__ __forceinline__ void perform_strided_along(const char *RESTRICT src_ptr, char *RESTRICT dst_ptr, const unsigned num_lds, const int ld_stride) const
  {
    BULK_TYPE temp[BYTES_PER_THREAD/sizeof(BULK_TYPE)];
    for (unsigned idx = 0; idx < num_lds; idx++)
    {
      perform_load<BULK_TYPE>(src_ptr, &temp[idx]);
      src_ptr += ld_stride;
    }
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    for (unsigned idx = 0; idx < num_lds; idx++)
    {
      perform_store<BULK_TYPE>(dst_ptr, &temp[idx]);
      dst_ptr += ld_stride;
    }
  }

#if 0
  template<typename BULK_TYPE>
  __device__ __forceinline__ void execute_internal(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) const
  {
    const char * dma_src_ptr = ((const char*)src_ptr) + src_offset; 
    char * dma_dst_ptr = ((char*)dst_ptr) + dst_offset;
    CUDADMA_BASE::wait_for_dma_start();
    for (unsigned p = 0; p < num_passes; p++)
    {
      BULK_TYPE registers[BYTES_PER_THREAD/sizeof(BULK_TYPE)];
      unsigned idx = 0;
      // Perform all the loads into registers
      for (unsigned e = 0; e < elmts_per_pass; e++)
      {
        for (unsigned l = 0; l < lds_per_elmt; l++)
	{
  	  perform_load<BULK_TYPE>(dma_src_ptr, &registers[idx]);       
	  dma_src_ptr += ld_stride;
	  idx++;
	}
        dma_src_ptr += src_elmt_stride;
      }
      // Write all the registers into their destinations
      idx = 0;
      for (unsigned e = 0; e < elmts_per_pass; e++)
      {
        for (unsigned l = 0; l < lds_per_elmt; l++)
	{
	  perform_store<BULK_TYPE>(&registers[idx], dma_dst_ptr);
	  dma_dst_ptr += ld_stride;
	  idx++;
	}
	dma_dst_ptr += dst_elmt_stride;
      }
    }
    // TODO: handle the weird cases
    CUDADMA_BASE::finish_async_dma();
  }
#endif
private:
  const unsigned int src_offset;
  const unsigned int dst_offset;
  const unsigned int src_elmt_stride;
  const unsigned int dst_elmt_stride;
  const unsigned int ld_stride;
  const unsigned int num_passes;
  const unsigned int elmt_per_pass;
  const unsigned int lds_per_elmt;
};

