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
#define MAX_BYTES_OUTSTANDING_PER_THREAD (4*ALIGNMENT) 
#define MAX_LDS_OUTSTANDING_PER_THREAD 4
#define CUDADMA_DMA_TID (threadIdx.x-dma_threadIdx_start)

//#define CUDADMA_DEBUG_ON

// Base class - define common variables and functions:
class cudaDMA {

 protected:

  // Synchronization related variables
  const bool is_dma_thread;
  const int barrierID_empty;
  const int barrierID_full;
  const int barrierSize;

  // DMA-thread Synchronization Functions:
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    ptx_cudaDMA_barrier_blocking(barrierID_empty,barrierSize); 
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    ptx_cudaDMA_barrier_nonblocking(barrierID_full,barrierSize); 
  }

 public:

  const int dma_tid;
  const unsigned long int dma1_src_iter_offs; 
  const unsigned long int dma2_src_iter_offs; 
  const unsigned long int dma3_src_iter_offs; 
  const unsigned long int dma4_src_iter_offs; 
  const unsigned long int dma1_src_offs; 
  const unsigned long int dma2_src_offs; 
  const unsigned long int dma3_src_offs; 
  const unsigned long int dma4_src_offs;
  const unsigned long int dma1_dst_iter_offs; 
  const unsigned long int dma2_dst_iter_offs; 
  const unsigned long int dma3_dst_iter_offs; 
  const unsigned long int dma4_dst_iter_offs; 
  const unsigned long int dma1_dst_offs; 
  const unsigned long int dma2_dst_offs; 
  const unsigned long int dma3_dst_offs; 
  const unsigned long int dma4_dst_offs;

  // Constructor without destination offsets:
  __device__ cudaDMA (const int dmaID,
		      const int num_dma_threads,
		      const int num_compute_threads,
		      const int dma_threadIdx_start, 
                      const unsigned long int dma1_src_iter_offs, 
                      const unsigned long int dma2_src_iter_offs, 
                      const unsigned long int dma3_src_iter_offs, 
                      const unsigned long int dma4_src_iter_offs,
		      const unsigned long int dma1_src_offs, 
                      const unsigned long int dma2_src_offs, 
                      const unsigned long int dma3_src_offs, 
                      const unsigned long int dma4_src_offs
		      ) : 
    is_dma_thread ((threadIdx.x>=dma_threadIdx_start) && (threadIdx.x<(dma_threadIdx_start+num_dma_threads))),
    dma_tid (CUDADMA_DMA_TID),
    barrierID_empty ((dmaID<<1)+1),
    barrierID_full (dmaID<<1),
    barrierSize (num_dma_threads+num_compute_threads),
    dma1_src_iter_offs (dma1_src_iter_offs),
    dma2_src_iter_offs (dma2_src_iter_offs),
    dma3_src_iter_offs (dma3_src_iter_offs),
    dma4_src_iter_offs (dma4_src_iter_offs),
    dma1_src_offs (dma1_src_offs),
    dma2_src_offs (dma2_src_offs),
    dma3_src_offs (dma3_src_offs),
    dma4_src_offs (dma4_src_offs),
    dma1_dst_iter_offs (0),
    dma2_dst_iter_offs (0),
    dma3_dst_iter_offs (0),
    dma4_dst_iter_offs (0),
    dma1_dst_offs (0),
    dma2_dst_offs (0),
    dma3_dst_offs (0),
    dma4_dst_offs (0)
      {
      }

  // Constructor with destination offsets:
  __device__ cudaDMA (const int dmaID,
		      const int num_dma_threads,
		      const int num_compute_threads,
		      const int dma_threadIdx_start, 
                      const unsigned long int dma1_src_iter_offs, 
                      const unsigned long int dma2_src_iter_offs, 
                      const unsigned long int dma3_src_iter_offs, 
                      const unsigned long int dma4_src_iter_offs,
		      const unsigned long int dma1_src_offs, 
                      const unsigned long int dma2_src_offs, 
                      const unsigned long int dma3_src_offs, 
                      const unsigned long int dma4_src_offs,
                      const unsigned long int dma1_dst_iter_offs, 
                      const unsigned long int dma2_dst_iter_offs, 
                      const unsigned long int dma3_dst_iter_offs, 
                      const unsigned long int dma4_dst_iter_offs,
		      const unsigned long int dma1_dst_offs, 
                      const unsigned long int dma2_dst_offs, 
                      const unsigned long int dma3_dst_offs, 
                      const unsigned long int dma4_dst_offs
		      ) : 
    is_dma_thread ((threadIdx.x>=dma_threadIdx_start) && (threadIdx.x<(dma_threadIdx_start+num_dma_threads))),
    dma_tid (CUDADMA_DMA_TID),
    barrierID_empty ((dmaID<<1)+1),
    barrierID_full (dmaID<<1),
    barrierSize (num_dma_threads+num_compute_threads),
    dma1_src_iter_offs (dma1_src_iter_offs),
    dma2_src_iter_offs (dma2_src_iter_offs),
    dma3_src_iter_offs (dma3_src_iter_offs),
    dma4_src_iter_offs (dma4_src_iter_offs),
    dma1_src_offs (dma1_src_offs),
    dma2_src_offs (dma2_src_offs),
    dma3_src_offs (dma3_src_offs),
    dma4_src_offs (dma4_src_offs),
    dma1_dst_iter_offs (dma1_dst_iter_offs),
    dma2_dst_iter_offs (dma2_dst_iter_offs),
    dma3_dst_iter_offs (dma3_dst_iter_offs),
    dma4_dst_iter_offs (dma4_dst_iter_offs),
    dma1_dst_offs (dma1_dst_offs),
    dma2_dst_offs (dma2_dst_offs),
    dma3_dst_offs (dma3_dst_offs),
    dma4_dst_offs (dma4_dst_offs)
      {
      }
    
  // Compute-thread Synchronization Functions:
  __device__ __forceinline__ void start_async_dma() const
  {
    ptx_cudaDMA_barrier_nonblocking(barrierID_empty,barrierSize); 
  }
  __device__ __forceinline__ void wait_for_dma_finish() const
  {
    ptx_cudaDMA_barrier_blocking(barrierID_full,barrierSize); 
  }
  
  // Intraspective Functions
  __device__ __forceinline__ bool owns_this_thread() const { return is_dma_thread; }

  // Transfer primitives used by more than one subclass
  template<bool DO_SYNC, int ALIGNMENT>
    __device__ __forceinline__ void do_xfer( void * src_ptr, void * dst_ptr, unsigned int xfer_size) const
  {
     switch (ALIGNMENT)
       {
          case 4:
	    {
	      do_xfer_alignment_04<DO_SYNC>(src_ptr,dst_ptr,xfer_size);
	      break;
	    }
          case 8:
            {
	      do_xfer_alignment_08<DO_SYNC>(src_ptr,dst_ptr,xfer_size);
	      break;
            }
          case 16:
            {
	      do_xfer_alignment_16<DO_SYNC>(src_ptr,dst_ptr,xfer_size);
	      break;
	    }
          default:
            printf("Illegal alignment size (%d).  Must be one of (4,8,16).\n",ALIGNMENT);
            break;
       }
  }

  // Manage transfers only aligned to 4 bytes
  template<bool DO_SYNC>
   __device__ __forceinline__ void do_xfer_alignment_04( void * src_ptr, void * dst_ptr, unsigned int xfer_size) const
  {
    switch (xfer_size)
      {
        case 16:
          {
	    perform_four_xfers<float,float,DO_SYNC,true> (src_ptr,dst_ptr);
	    break;
          }
	case 12:
          {
            perform_three_xfers<float,float,DO_SYNC> (src_ptr,dst_ptr);
            break;
          }
	case 8:
          {
            perform_two_xfers<float,float,DO_SYNC> (src_ptr,dst_ptr);
	    break;
          }
	case 4:
          {
	    perform_one_xfer<float,DO_SYNC> (src_ptr,dst_ptr);
	    break;
          }
        case 0:
	 {
	   if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start(); 
	   break;
	 }
         default:
           printf("Invalid xfer size (%u) for dma_tid = %d\n",xfer_size, CUDADMA_BASE::dma_tid);
	   break;
      }
  }

  // Manage transfers aligned to 8 byte boundary
  template<bool DO_SYNC>
   __device__ __forceinline__ void do_xfer_alignment_08( void * src_ptr, void * dst_ptr, unsigned int xfer_size) const
  {
    switch (xfer_size)
      {
        case 32:
         {
           perform_four_xfers<float2,float2,DO_SYNC,true> (src_ptr,dst_ptr);  
           break;
         }
         case 28:
         {
           perform_four_xfers<float2,float,DO_SYNC,true> (src_ptr,dst_ptr);   
           break;
         }
         case 24:
         {
           perform_three_xfers<float2,float2,DO_SYNC> (src_ptr,dst_ptr);  
           break;
         }
         case 20:
         {
           perform_three_xfers<float2,float,DO_SYNC> (src_ptr,dst_ptr);  
           break;
         }
         case 16:
         {
           perform_two_xfers<float2,float2,DO_SYNC> (src_ptr,dst_ptr); 
           break;
         }
         case 12:
         {
           perform_two_xfers<float2,float,DO_SYNC> (src_ptr,dst_ptr); 
           break;
         }
         case 8:
         {
           perform_one_xfer<float2,DO_SYNC> (src_ptr,dst_ptr); 
           break;
         }
         case 4:
         {
           perform_one_xfer<float,DO_SYNC> (src_ptr,dst_ptr); 
           break;
         }
         case 0:
	 {
	   if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start(); 
	   break;
	 }
         default:
           printf("Invalid xfer size (%u) for dma_tid = %d\n",xfer_size, CUDADMA_BASE::dma_tid);
	   break;
      }
  }

  // Manage transfers aligned to 16 byte boundary
  template<bool DO_SYNC>
   __device__ __forceinline__ void do_xfer_alignment_16( void * src_ptr, void * dst_ptr, unsigned int xfer_size) const
  {
    switch (xfer_size) 
      {
      case 64: 
	{
	  perform_four_xfers<float4,float4,DO_SYNC,true> (src_ptr,dst_ptr);
	  break;
	}
      case 60: 
	{
	  perform_four_xfers<float4,float3,DO_SYNC,true> (src_ptr,dst_ptr);
	  break;
	}
      case 56: 
	{
	  perform_four_xfers<float4,float2,DO_SYNC,true> (src_ptr,dst_ptr);
	  break;
	}
      case 52:
	{
	  perform_four_xfers<float4,float,DO_SYNC,true> (src_ptr,dst_ptr);
	  break;
	}
      case 48:
	{
	  perform_three_xfers<float4,float4,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 44:
	{
	  perform_three_xfers<float4,float3,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 40:
	{
	  perform_three_xfers<float4,float2,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 36:
	{
	  perform_three_xfers<float4,float,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 32:
	{
	  perform_two_xfers<float4,float4,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 28:
	{
	  perform_two_xfers<float4,float3,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 24:
	{
	  perform_two_xfers<float4,float2,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 20:
	{
	  perform_two_xfers<float4,float,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 16:
	{
	  perform_one_xfer<float4,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 12:
	{
	  perform_one_xfer<float3,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 8:
	{
	  perform_one_xfer<float2,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 4:
	{
	  perform_one_xfer<float,DO_SYNC> (src_ptr,dst_ptr);
	  break;
	}
      case 0:
	{
	  if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start(); 
	  break;
	}
      default:
	printf("Invalid xfer size (%u) for dma_tid = %d\n",xfer_size, CUDADMA_BASE::dma_tid);
	break;
      }
  }

  /*
   * These functions are used to emit vector loads of the appropriate size at
   * the predefined offsets.
   */
  template<typename TYPE1, bool DO_SYNC>
    __device__ __forceinline__ void perform_one_xfer(void *src_ptr, void *dst_ptr) const
  {
    TYPE1 tmp1 = *(TYPE1 *)((char *)src_ptr + dma1_src_offs);
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    *(TYPE1 *)((char *)dst_ptr+dma1_dst_offs) = tmp1;
  }
  template<typename TYPE1, typename TYPE2, bool DO_SYNC>
    __device__ __forceinline__ void perform_two_xfers(void *src_ptr, void *dst_ptr) const
  {
    TYPE1 tmp1 = *(TYPE1 *)((char *)src_ptr + dma1_src_offs);
    TYPE2 tmp2 = *(TYPE2 *)((char *)src_ptr + dma2_src_offs);
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    *(TYPE1 *)((char *)dst_ptr+dma1_dst_offs) = tmp1;
    *(TYPE2 *)((char *)dst_ptr+dma2_dst_offs) = tmp2;
  }
  template<typename TYPE1, typename TYPE2, bool DO_SYNC>
    __device__ __forceinline__ void perform_three_xfers(void *src_ptr, void *dst_ptr) const
  {
    TYPE1 tmp1 = *(TYPE1 *)((char *)src_ptr + dma1_src_offs);
    TYPE1 tmp2 = *(TYPE1 *)((char *)src_ptr + dma2_src_offs);
    TYPE2 tmp3 = *(TYPE2 *)((char *)src_ptr + dma3_src_offs);
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    *(TYPE1 *)((char *)dst_ptr+dma1_dst_offs) = tmp1;
    *(TYPE1 *)((char *)dst_ptr+dma2_dst_offs) = tmp2;
    *(TYPE2 *)((char *)dst_ptr+dma3_dst_offs) = tmp3;
  }
  template <typename TYPE1, typename TYPE2, bool DO_SYNC, bool LAST_XFER>
    __device__ __forceinline__ void perform_four_xfers(void *src_ptr, void *dst_ptr) const
  {
    TYPE1 tmp1 = *(TYPE1 *)((char *)src_ptr + (LAST_XFER ? dma1_src_offs : dma1_src_iter_offs));
    TYPE1 tmp2 = *(TYPE1 *)((char *)src_ptr + (LAST_XFER ? dma2_src_offs : dma2_src_iter_offs));
    TYPE1 tmp3 = *(TYPE1 *)((char *)src_ptr + (LAST_XFER ? dma3_src_offs : dma3_src_iter_offs));
    TYPE2 tmp4 = *(TYPE2 *)((char *)src_ptr + (LAST_XFER ? dma4_src_offs : dma4_src_iter_offs));
    if (DO_SYNC) CUDADMA_BASE::wait_for_dma_start();
    *(TYPE1 *)((char *)dst_ptr+(LAST_XFER ? dma1_dst_offs : dma1_dst_iter_offs)) = tmp1;
    *(TYPE1 *)((char *)dst_ptr+(LAST_XFER ? dma2_dst_offs : dma2_dst_iter_offs)) = tmp2;
    *(TYPE1 *)((char *)dst_ptr+(LAST_XFER ? dma3_dst_offs : dma3_dst_iter_offs)) = tmp3;
    *(TYPE2 *)((char *)dst_ptr+(LAST_XFER ? dma4_dst_offs : dma4_dst_iter_offs)) = tmp4;
  }
};

#define CUDADMASEQUENTIAL_DMA_ITERS (BYTES_PER_THREAD-4)/MAX_BYTES_OUTSTANDING_PER_THREAD
// All of these values below will be used as byte address offsets:
#define CUDADMASEQUENTIAL_DMA_ITER_INC MAX_BYTES_OUTSTANDING_PER_THREAD*num_dma_threads
#define CUDADMASEQUENTIAL_DMA1_ITER_OFFS 1*ALIGNMENT*CUDADMA_DMA_TID
#define CUDADMASEQUENTIAL_DMA2_ITER_OFFS 1*ALIGNMENT*num_dma_threads+ALIGNMENT*CUDADMA_DMA_TID
#define CUDADMASEQUENTIAL_DMA3_ITER_OFFS 2*ALIGNMENT*num_dma_threads+ALIGNMENT*CUDADMA_DMA_TID
#define CUDADMASEQUENTIAL_DMA4_ITER_OFFS 3*ALIGNMENT*num_dma_threads+ALIGNMENT*CUDADMA_DMA_TID
#if 0
#define CUDADMASEQUENTIAL_DMA1_OFFS \
  (((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)<(1*ALIGNMENT))&&((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)!=0)) ? \
  (BYTES_PER_THREAD%ALIGNMENT)*CUDADMA_DMA_TID : \
  ALIGNMENT*CUDADMA_DMA_TID
#define CUDADMASEQUENTIAL_DMA2_OFFS \
  (((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)<(2*ALIGNMENT))&&((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)!=0)) ?		\
  (ALIGNMENT*num_dma_threads+(BYTES_PER_THREAD%ALIGNMENT)*CUDADMA_DMA_TID) :	\
    (ALIGNMENT*num_dma_threads+ALIGNMENT*CUDADMA_DMA_TID)
#define CUDADMASEQUENTIAL_DMA3_OFFS \
  (((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)<(3*ALIGNMENT))&&((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)!=0)) ? \
  (2*ALIGNMENT*num_dma_threads+(BYTES_PER_THREAD%ALIGNMENT)*CUDADMA_DMA_TID) : \
    (2*ALIGNMENT*num_dma_threads+ALIGNMENT*CUDADMA_DMA_TID) 
#define CUDADMASEQUENTIAL_DMA4_OFFS \
  ((BYTES_PER_THREAD%MAX_BYTES_OUTSTANDING_PER_THREAD)!=0) ? \
  (3*ALIGNMENT*num_dma_threads+(BYTES_PER_THREAD%ALIGNMENT)*CUDADMA_DMA_TID) : \
    (3*ALIGNMENT*num_dma_threads+ALIGNMENT*CUDADMA_DMA_TID)
#endif
#define CUDADMASEQUENTIAL_DMA1_OFFS (ALIGNMENT*CUDADMA_DMA_TID)
#define CUDADMASEQUENTIAL_DMA2_OFFS (1*ALIGNMENT*num_dma_threads + ALIGNMENT*CUDADMA_DMA_TID)
#define CUDADMASEQUENTIAL_DMA3_OFFS (2*ALIGNMENT*num_dma_threads + ALIGNMENT*CUDADMA_DMA_TID)
#define CUDADMASEQUENTIAL_DMA4_OFFS (3*ALIGNMENT*num_dma_threads + ALIGNMENT*CUDADMA_DMA_TID)
	      
template <int BYTES_PER_THREAD, int ALIGNMENT>
class cudaDMASequential : public CUDADMA_BASE {

 public:

  // DMA Addressing variables
  const unsigned int dma_iters;
  const unsigned int dma_iter_inc;  // Precomputed offset for next copy iteration
  bool all_threads_active; // If true, then we know that all threads are guaranteed to be active (needed for sync/divergence functionality guarantee)
  bool is_active_thread;   // If true, then all of BYTES_PER_THREAD will be transferred for this thread
  bool is_partial_thread;  // If true, then only some of BYTES_PER_THREAD will be transferred for this thread
  int partial_thread_bytes;

  // Constructor for when (sz = BYTES_PER_THREAD * num_dma_threads)
  __device__ cudaDMASequential (const int dmaID,
				const int num_dma_threads,
				const int num_compute_threads,
				const int dma_threadIdx_start)
    : CUDADMA_BASE (dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
                    CUDADMASEQUENTIAL_DMA1_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA2_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA3_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA4_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA1_OFFS,
                    CUDADMASEQUENTIAL_DMA2_OFFS,
                    CUDADMASEQUENTIAL_DMA3_OFFS,
                    CUDADMASEQUENTIAL_DMA4_OFFS,
                    CUDADMASEQUENTIAL_DMA1_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA2_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA3_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA4_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA1_OFFS,
                    CUDADMASEQUENTIAL_DMA2_OFFS,
                    CUDADMASEQUENTIAL_DMA3_OFFS,
                    CUDADMASEQUENTIAL_DMA4_OFFS
                   ),
    dma_iters (CUDADMASEQUENTIAL_DMA_ITERS),
    dma_iter_inc (CUDADMASEQUENTIAL_DMA_ITER_INC),
    all_threads_active (true)
    {
      is_active_thread = true;
      is_partial_thread = false;
      partial_thread_bytes = 0;
    }

  // Constructor for when (sz <= BYTES_PER_THREAD * num_dma_threads)
  __device__ cudaDMASequential (const int dmaID,
				const int num_dma_threads,
				const int num_compute_threads,
				const int dma_threadIdx_start, 
				const int sz)
    : CUDADMA_BASE (dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
                    CUDADMASEQUENTIAL_DMA1_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA2_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA3_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA4_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA1_OFFS,
                    CUDADMASEQUENTIAL_DMA2_OFFS,
                    CUDADMASEQUENTIAL_DMA3_OFFS,
                    CUDADMASEQUENTIAL_DMA4_OFFS,
                    CUDADMASEQUENTIAL_DMA1_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA2_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA3_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA4_ITER_OFFS,
                    CUDADMASEQUENTIAL_DMA1_OFFS,
                    CUDADMASEQUENTIAL_DMA2_OFFS,
                    CUDADMASEQUENTIAL_DMA3_OFFS,
                    CUDADMASEQUENTIAL_DMA4_OFFS
                   ),
    dma_iters ( (sz-4)/(MAX_BYTES_OUTSTANDING_PER_THREAD*num_dma_threads) ),
    dma_iter_inc (CUDADMASEQUENTIAL_DMA_ITER_INC)
    //all_threads_active (sz==(BYTES_PER_THREAD*num_dma_threads))
    {

      // Do a bunch of arithmetic based on total size of the xfer:
      int num_vec4_loads = sz / (ALIGNMENT*num_dma_threads);
      int leftover_bytes = sz % (ALIGNMENT*num_dma_threads);

#ifdef CUDADMA_DEBUG_ON
      if ((CUDADMA_DMA_TID==1)&&(CUDADMA_BASE::barrierID_full==2)&&(CUDADMA_BASE::is_dma_thread)) {
	printf("leftover_bytes = %d\n",leftover_bytes);
	printf("num_vec4_loads = %d\n",num_vec4_loads);
      }
#endif

      // After computing leftover_bytes, figure out the cutoff point in dma_tid:
      // Note, all threads are either active, partial, or inactive
      if (leftover_bytes==0) {
	all_threads_active = true;
	// Transfer is perfectly divisible by 16 bytes...only have to worry about not using all of BYTES_PER_THREAD
	partial_thread_bytes = num_vec4_loads*ALIGNMENT; 
	is_partial_thread = (partial_thread_bytes!=BYTES_PER_THREAD);
	is_active_thread = (partial_thread_bytes==BYTES_PER_THREAD);
      } else {
	all_threads_active = false;
	// Threads below partial thread dma_tid will do 16-byte (or BYTES_PER_THREAD leftover) xfers, above should be inactive
	//int max_thread_bytes = min(ALIGNMENT,BYTES_PER_THREAD-(num_vec4_loads*ALIGNMENT));
	int max_thread_bytes = ALIGNMENT;
#ifdef CUDADMA_DEBUG_ON
	if ((CUDADMA_DMA_TID==1)&&(CUDADMA_BASE::barrierID_full==2)&&(CUDADMA_BASE::is_dma_thread)) {
	  printf("max_thread_bytes = %d\n",max_thread_bytes);
	}
#endif
	if (leftover_bytes>=(max_thread_bytes*(CUDADMA_DMA_TID+1))) {
	  // Below:  Do 16-byte xfers
	  partial_thread_bytes = (num_vec4_loads)*ALIGNMENT + max_thread_bytes;
	  is_partial_thread = (partial_thread_bytes!=BYTES_PER_THREAD);
	  is_active_thread = (partial_thread_bytes==BYTES_PER_THREAD);
	} else if (leftover_bytes<(max_thread_bytes*CUDADMA_DMA_TID)) {
	  // Above:  Do 0-byte xfers on last vec_load, do max_bytes_per_thread xfer on all previous loads
	  is_active_thread = false;
	  partial_thread_bytes = (num_vec4_loads-(dma_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT;
	  is_partial_thread = (num_vec4_loads!=0);
	} else {
	  // Do less than max_thread_bytes on last vec_load in this thread, do max_bytes_per_thread xfer on all previous loads
	  partial_thread_bytes = (num_vec4_loads-(dma_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + (leftover_bytes%max_thread_bytes);
	  //partial_thread_bytes = (num_vec4_loads*ALIGNMENT) + (leftover_bytes%max_thread_bytes);
	  is_partial_thread = true;
	  is_active_thread = false;
	}
      }
      
#ifdef CUDADMA_DEBUG_ON
      if ((CUDADMA_BASE::barrierID_full==2)&&(CUDADMA_BASE::is_dma_thread)) {
	if (is_partial_thread==true) {
	  printf("PARTIAL THREAD:  tid = %d  dma_tid = %d,\tpartial_thread_bytes=%d\n",threadIdx.x, CUDADMA_DMA_TID,partial_thread_bytes);
	} else if (is_active_thread==false) {
	  printf("INACTIVE THREAD: tid = %d  dma_tid = %d\n",threadIdx.x, CUDADMA_DMA_TID);
	} else {
	  printf("ACTIVE THREAD:  tid = %d  dma_tid = %d,\tactive_thread_bytes=%d\n",threadIdx.x, CUDADMA_DMA_TID,BYTES_PER_THREAD);
	}
      }
#endif
    }
  
  // DMA-thread Data Transfer Functions:
    __device__ __forceinline__ void execute_dma ( void * src_ptr, void * dst_ptr ) const
  {

#ifdef CUDADMA_DEBUG_ON
      /*
      if ((CUDADMA_BASE::dma_tid==0)&&(CUDADMA_BASE::barrierID_full==2)&&(CUDADMA_BASE::is_dma_thread)) {
	printf("dma_tid = %d\nnum_dma_threads = %d\nbytes_per_thread = %d\ndma_iters = %d\ndma_iter_inc = %d\ndma1_src_offs = %lu\ndma2_offs = %lu\ndma3_offs = %lu\ndma4_offs = %lu\ndma1_iter_offs = %lu\ndma2_iter_offs = %lu\ndma3_iter_offs = %lu\ndma4_iter_offs = %lu\n", CUDADMA_BASE::dma_tid, dma_iter_inc/MAX_BYTES_OUTSTANDING_PER_THREAD, BYTES_PER_THREAD, dma_iters, dma1_src_offs, dma1_src_offs, dma2_src_offs, dma3_src_offs, dma4_src_offs, dma1_src_iter_offs, dma2_src_iter_offs, dma3_src_iter_offs, dma4_src_iter_offs);
      }
	*/
#endif

    int this_thread_bytes = is_active_thread ? BYTES_PER_THREAD : is_partial_thread ? partial_thread_bytes : 0;
    if ((dma_iters>0) || (!all_threads_active)) {
      CUDADMA_BASE::wait_for_dma_start(); 
    }
    // Slightly less optimized case
    char * src_temp = (char *)src_ptr;
    char * dst_temp = (char *)dst_ptr;
    switch (ALIGNMENT)
      {
	case 4:
	  {
	    for(int i = 0; i < dma_iters; i++) {
	      CUDADMA_BASE::template perform_four_xfers<float,float,false,false> (src_temp,dst_temp);
	      src_temp += dma_iter_inc;
	      dst_temp += dma_iter_inc;
	    }
	    break;
	  }
        case 8:
          {
            for(int i = 0; i < dma_iters ; i++) {
	      CUDADMA_BASE::template perform_four_xfers<float2,float2,false,false> (src_temp,dst_temp);
	      src_temp += dma_iter_inc;
	      dst_temp += dma_iter_inc;
	    }
	    break;
          }
        case 16:
          {
            for(int i = 0; i < dma_iters ; i++) {
              CUDADMA_BASE::template perform_four_xfers<float4,float4,false,false> (src_temp,dst_temp);
              src_temp += dma_iter_inc;
              dst_temp += dma_iter_inc;
            }
            break;
          }
        default:
          printf("ALIGNMENT must be one of (4,8,16)\n");
          break;
      }
    // Handle the leftovers
    if (all_threads_active) {
      CUDADMA_BASE::template do_xfer<CUDADMASEQUENTIAL_DMA_ITERS==0, ALIGNMENT> ( src_temp, dst_temp, 
						(this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ? 
						(this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) : 
						MAX_BYTES_OUTSTANDING_PER_THREAD );
    } else {
#ifdef CUDADMA_DEBUG_ON
      if ((CUDADMA_BASE::dma_tid==1)&&(CUDADMA_BASE::barrierID_full==4)&&(CUDADMA_BASE::is_dma_thread)) {
	printf("src1 addr = %x\n",((char *)src_temp + dma1_src_offs));
	printf("src2 addr = %x\n",((char *)src_temp + dma2_src_offs));
	printf("src3 addr = %x\n",((char *)src_temp + dma3_src_offs));
	printf("src4 addr = %x\n",((char *)src_temp + dma4_src_offs));
	printf("dst1 addr = %x\n",((char *)dst_temp + dma1_dst_offs));
	printf("dst2 addr = %x\n",((char *)dst_temp + dma2_dst_offs));
	printf("dst3 addr = %x\n",((char *)dst_temp + dma3_dst_offs));
	printf("dst4 addr = %x\n",((char *)dst_temp + dma4_dst_offs));
	printf("bytes = %d\n",
	       (this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ? 
	       (this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) : 
	       (this_thread_bytes ? MAX_BYTES_OUTSTANDING_PER_THREAD : 0));
      }
#endif
      CUDADMA_BASE::template do_xfer<false, ALIGNMENT> ( src_temp, dst_temp, 
		       (this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ? 
		       (this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) : 
		       (this_thread_bytes ? MAX_BYTES_OUTSTANDING_PER_THREAD : 0));
    }
    CUDADMA_BASE::finish_async_dma();
  }

  // Public DMA-thread Synchronization Functions:
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    CUDADMA_BASE::wait_for_dma_start();
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    CUDADMA_BASE::finish_async_dma();
  }

};
#undef CUDADMASEQUENTIAL_DMA_ITERS
#undef CUDADMASEQUENTIAL_DMA_ITER_INC
#undef CUDADMASEQUENTIAL_DMA1_ITER_OFFS
#undef CUDADMASEQUENTIAL_DMA2_ITER_OFFS
#undef CUDADMASEQUENTIAL_DMA3_ITER_OFFS
#undef CUDADMASEQUENTIAL_DMA4_ITER_OFFS
#undef CUDADMASEQUENTIAL_DMA1_OFFS
#undef CUDADMASEQUENTIAL_DMA2_OFFS
#undef CUDADMASEQUENTIAL_DMA3_OFFS
#undef CUDADMASEQUENTIAL_DMA4_OFFS

#if 0
#define CUDADMASTRIDED_DMA_COL_ITERS ( (el_sz-4)/(MAX_BYTES_OUTSTANDING_PER_THREAD*num_dma_threads) )
#define CUDADMASTRIDED_DMA_COL_ITER_INC (MAX_BYTES_OUTSTANDING_PER_THREAD*num_dma_threads)
#define CUDADMASTRIDED_DMA1_ITER_OFFS (16*CUDADMA_DMA_TID)
#define CUDADMASTRIDED_DMA2_ITER_OFFS (16*num_dma_threads+16*CUDADMA_DMA_TID)
#define CUDADMASTRIDED_DMA3_ITER_OFFS (32*num_dma_threads+16*CUDADMA_DMA_TID)
#define CUDADMASTRIDED_DMA4_ITER_OFFS (48*num_dma_threads+16*CUDADMA_DMA_TID)
#define CUDADMASTRIDED_DMA1_OFFS(x) \
  ((((x)%64)<16)&&(((x)%64)!=0)) ? \
  (((x)%16)*CUDADMA_DMA_TID) : \
  (16*CUDADMA_DMA_TID)
#define CUDADMASTRIDED_DMA2_OFFS(x) \
  ((((x)%64)<32)&&(((x)%64)!=0)) ?		\
  (16*num_dma_threads+((x)%16)*CUDADMA_DMA_TID) :	\
    (16*num_dma_threads+16*CUDADMA_DMA_TID)
#define CUDADMASTRIDED_DMA3_OFFS(x) \
  ((((x)%64)<48)&&(((x)%64)!=0)) ? \
  (32*num_dma_threads+((x)%16)*CUDADMA_DMA_TID) : \
    (32*num_dma_threads+16*CUDADMA_DMA_TID) 
#define CUDADMASTRIDED_DMA4_OFFS(x) \
  (((x)%64)!=0) ? \
  (48*num_dma_threads+((x)%16)*CUDADMA_DMA_TID) : \
    (48*num_dma_threads+16*CUDADMA_DMA_TID)

template <int BYTES_PER_THREAD, int ALIGNMENT>
class cudaDMAStrided : public CUDADMA_BASE {

 public:

  // DMA Addressing variables
  const int el_sz;
  const int bytes_per_thread_per_el; //Bytes loaded per thread within one 'element'/'row'
  const int dma_col_iters; // Number of iters to do for within one 'element'/'row'
  const int dma_row_iters; // Number of iters required to gather all strided 'elements'/'rows'
  const int dma_col_iter_inc; // Precomputed offset for next column iteration
  const int dma_src_row_iter_inc; // Precomputed source offset for next row iteration
  const int dma_dst_row_iter_inc; // Precomputed dest offset for next row iteration
  const bool all_threads_active; // If true, then we know that all threads are guaranteed to be active (needed for sync/divergence functionality guarantee)
  bool is_active_thread;   // If true, then all of BYTES_PER_THREAD will be transferred for this thread
  bool is_partial_thread;  // If true, then only some of BYTES_PER_THREAD will be transferred for this thread
  int partial_thread_bytes;

  // Constructor for when el_cnt == num_dma_threads*el_per_thread 
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_dma_threads,
			     const int num_compute_threads,
			     const int dma_threadIdx_start, 
			     const int el_sz, 
			     //In this case, we are assuming that
			     //el_cnt == num_dma_threads*el_per_thread
			     //       == num_dma_threads*(BYTES_PER_THREAD/el_sz)
			     const int el_stride)
    : CUDADMA_BASE (dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
                    CUDADMASTRIDED_DMA1_ITER_OFFS,
                    CUDADMASTRIDED_DMA2_ITER_OFFS,
                    CUDADMASTRIDED_DMA3_ITER_OFFS,
                    CUDADMASTRIDED_DMA4_ITER_OFFS,
                    CUDADMASTRIDED_DMA1_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA2_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA3_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA4_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA1_ITER_OFFS,
                    CUDADMASTRIDED_DMA2_ITER_OFFS,
                    CUDADMASTRIDED_DMA3_ITER_OFFS,
                    CUDADMASTRIDED_DMA4_ITER_OFFS,
                    CUDADMASTRIDED_DMA1_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA2_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA3_OFFS(el_sz/num_dma_threads),
                    CUDADMASTRIDED_DMA4_OFFS(el_sz/num_dma_threads)
                   ),
// Assuming  4*sizeof(float)*num_dma_threads <= el_size
// && el_sz <= 4*sizeof(float4)*num_dma_threads 
// && el_sz/num_dma_threads % 4 == 0
    bytes_per_thread_per_el (el_sz/num_dma_threads),
    el_sz (el_sz),
    dma_col_iters (CUDADMASTRIDED_DMA_COL_ITERS),
    dma_col_iter_inc (CUDADMASTRIDED_DMA_COL_ITER_INC),
    dma_row_iters (BYTES_PER_THREAD/(el_sz/num_dma_threads)),
    dma_src_row_iter_inc (el_stride),
    dma_dst_row_iter_inc (el_sz),
    all_threads_active (true)
    {
      is_active_thread = true;
      is_partial_thread = false;
      partial_thread_bytes = 0;
    }

  // Constructor for when el_cnt != num_dma_threads*el_per_thread 
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_dma_threads,
			     const int num_compute_threads,
			     const int dma_threadIdx_start, 
			     const int el_sz, 
                             const int el_cnt,
			     const int el_stride)
    : CUDADMA_BASE (dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
                    CUDADMASTRIDED_DMA1_ITER_OFFS,
                    CUDADMASTRIDED_DMA2_ITER_OFFS,
                    CUDADMASTRIDED_DMA3_ITER_OFFS,
                    CUDADMASTRIDED_DMA4_ITER_OFFS,
                    CUDADMASTRIDED_DMA1_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA2_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA3_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA4_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA1_ITER_OFFS,
                    CUDADMASTRIDED_DMA2_ITER_OFFS,
                    CUDADMASTRIDED_DMA3_ITER_OFFS,
                    CUDADMASTRIDED_DMA4_ITER_OFFS,
                    CUDADMASTRIDED_DMA1_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA2_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA3_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA4_OFFS(BYTES_PER_THREAD/el_cnt)
                   ),
    el_sz (el_sz),
    bytes_per_thread_per_el (BYTES_PER_THREAD/el_cnt),
    dma_col_iters (CUDADMASTRIDED_DMA_COL_ITERS),
    dma_col_iter_inc (CUDADMASTRIDED_DMA_COL_ITER_INC),
    dma_row_iters (el_cnt),
    dma_src_row_iter_inc (el_stride),
    dma_dst_row_iter_inc (el_sz),
    all_threads_active (el_sz==(BYTES_PER_THREAD*num_dma_threads/el_cnt))
    {
      int num_vec4_loads_per_el = el_sz / (16*num_dma_threads);
      int leftover_bytes_per_el = el_sz % (16*num_dma_threads);
      if (leftover_bytes_per_el==0) {
	// Transfer is perfectly divisible by 16 bytes...only have to worry about not using all of BYTES_PER_THREAD
	partial_thread_bytes = num_vec4_loads_per_el*16; 
	is_partial_thread = (partial_thread_bytes!=bytes_per_thread_per_el);
	is_active_thread = (partial_thread_bytes==bytes_per_thread_per_el);
      } else {
	// Threads below partial thread dma_tid will do 16-byte (or BYTES_PER_THREAD leftover) xfers, above should be inactive
	int max_thread_bytes_per_el = min(16,bytes_per_thread_per_el-(num_vec4_loads_per_el*16));
	if (leftover_bytes_per_el>=(max_thread_bytes_per_el*(CUDADMA_DMA_TID+1))) {
	  // Below:  Do 16-byte xfers
	  partial_thread_bytes = num_vec4_loads_per_el*16 + max_thread_bytes_per_el;
	  is_partial_thread = (partial_thread_bytes!=bytes_per_thread_per_el);
	  is_active_thread = (partial_thread_bytes==bytes_per_thread_per_el);
	} else if (leftover_bytes_per_el<(max_thread_bytes_per_el*(CUDADMA_DMA_TID+1))) {
	  // Above:  Do 0-byte xfers
	  is_active_thread = false;
	  partial_thread_bytes = num_vec4_loads_per_el*16;
	  is_partial_thread = (num_vec4_loads_per_el != 0);
	} else {
	  // Less than 16 bytes on this thread
	  partial_thread_bytes = (num_vec4_loads_per_el*16) + (leftover_bytes_per_el%max_thread_bytes_per_el);
	  is_partial_thread = true;
	  is_active_thread = false;
        } 
      }
    }

  // Constructor for when el_cnt = num_dma_threads*el_per_thread 
  // and when a destination stride is required in addition to a source stride
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_dma_threads,
			     const int num_compute_threads,
			     const int dma_threadIdx_start, 
			     const int el_sz, 
                             const int el_cnt,
			     const int el_src_stride,
                             const int el_dst_stride)
    : CUDADMA_BASE (dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
                    CUDADMASTRIDED_DMA1_ITER_OFFS,
                    CUDADMASTRIDED_DMA2_ITER_OFFS,
                    CUDADMASTRIDED_DMA3_ITER_OFFS,
                    CUDADMASTRIDED_DMA4_ITER_OFFS,
                    CUDADMASTRIDED_DMA1_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA2_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA3_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA4_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA1_ITER_OFFS,
                    CUDADMASTRIDED_DMA2_ITER_OFFS,
                    CUDADMASTRIDED_DMA3_ITER_OFFS,
                    CUDADMASTRIDED_DMA4_ITER_OFFS,
                    CUDADMASTRIDED_DMA1_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA2_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA3_OFFS(BYTES_PER_THREAD/el_cnt),
                    CUDADMASTRIDED_DMA4_OFFS(BYTES_PER_THREAD/el_cnt)
                   ),
    el_sz (el_sz),
    bytes_per_thread_per_el (BYTES_PER_THREAD/el_cnt),
    dma_col_iters (CUDADMASTRIDED_DMA_COL_ITERS),
    dma_col_iter_inc (CUDADMASTRIDED_DMA_COL_ITER_INC),
    dma_row_iters (el_cnt),
    dma_src_row_iter_inc (el_src_stride),
    dma_dst_row_iter_inc (el_dst_stride),
    all_threads_active (el_sz==(BYTES_PER_THREAD*num_dma_threads/el_cnt))
    {
      int num_vec4_loads_per_el = el_sz / (16*num_dma_threads);
      int leftover_bytes_per_el = el_sz % (16*num_dma_threads);
      if (leftover_bytes_per_el==0) {
	// Transfer is perfectly divisible by 16 bytes...only have to worry about not using all of BYTES_PER_THREAD
	partial_thread_bytes = num_vec4_loads_per_el*16; 
	is_partial_thread = (partial_thread_bytes!=bytes_per_thread_per_el);
	is_active_thread = (partial_thread_bytes==bytes_per_thread_per_el);
      } else {
	// Threads below partial thread dma_tid will do 16-byte (or BYTES_PER_THREAD leftover) xfers, above should be inactive
	int max_thread_bytes_per_el = min(16,bytes_per_thread_per_el-(num_vec4_loads_per_el*16));
	if (leftover_bytes_per_el>=(max_thread_bytes_per_el*(CUDADMA_DMA_TID+1))) {
	  // Below:  Do 16-byte xfers
	  partial_thread_bytes = num_vec4_loads_per_el*16 + max_thread_bytes_per_el;
	  is_partial_thread = (partial_thread_bytes!=bytes_per_thread_per_el);
	  is_active_thread = (partial_thread_bytes==bytes_per_thread_per_el);
	} else if (leftover_bytes_per_el<(max_thread_bytes_per_el*(CUDADMA_DMA_TID+1))) {
	  // Above:  Do 0-byte xfers
	  is_active_thread = false;
	  partial_thread_bytes = num_vec4_loads_per_el*16;
	  is_partial_thread = (num_vec4_loads_per_el != 0);
	} else {
	  // Less than 16 bytes on this thread
	  partial_thread_bytes = (num_vec4_loads_per_el*16) + (leftover_bytes_per_el%max_thread_bytes_per_el);
	  is_partial_thread = true;
	  is_active_thread = false;
        } 
      }
    }

  // DMA-thread Data Transfer Functions:
  __device__ void execute_dma ( void * src_ptr, void * dst_ptr) const
  {
#ifdef CUDADMA_DEBUG_ON
    /*
      if ((dma_tid==0)&&(CUDADMA_BASE::barrierID_full==4)&&(CUDADMA_BASE::is_dma_thread)) {
	printf("dma_tid = %d\nnum_dma_threads = %d\nbytes_per_thread_per_el = %d\ndma_col_iters = %d\ndma_col_iter_inc = %d\ndma1_offs = %lu\ndma2_offs = %lu\ndma3_offs = %lu\ndma4_offs = %lu\ndma1_iter_offs = %lu\ndma2_iter_offs = %lu\ndma3_iter_offs = %lu\ndma4_iter_offs = %lu\n", CUDADMA_BASE::dma_tid, dma_col_iter_inc/MAX_BYTES_OUTSTANDING_PER_THREAD, bytes_per_thread_per_el, dma_col_iters, dma_col_iter_inc, dma1_offs, dma2_offs, dma3_offs, dma4_offs, dma1_iter_offs, dma2_iter_offs, dma3_iter_offs, dma4_iter_offs);
      }
    */
#endif
    int this_thread_bytes = is_active_thread ? bytes_per_thread_per_el : is_partial_thread ? partial_thread_bytes : 0;
    //if ((dma_col_iters>0) || (dma_row_iters>1) || (!all_threads_active)) {
      CUDADMA_BASE::wait_for_dma_start(); 
    //}
    // Slightly less optimized case
    char * src_row_ptr = (char *)src_ptr;
    char * dst_row_ptr = (char *)dst_ptr;
    char * src_temp    = (char *)src_ptr;
    char * dst_temp    = (char *)dst_ptr;

    for(int i = 0; i < dma_row_iters; i++) {
      for(int j = 0; j < dma_col_iters; j++) {
        CUDADMA_BASE::template perform_four_xfers<float4,float4,false,false> (src_temp,dst_temp);
        src_temp += dma_col_iter_inc;
        dst_temp += dma_col_iter_inc;
      }
      // Handle the col leftovers
      if (all_threads_active) {
        CUDADMA_BASE::template do_xfer<false, ALIGNMENT> ( src_temp, dst_temp, 
	  					(this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ? 
						(this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) : 
						MAX_BYTES_OUTSTANDING_PER_THREAD );
      } else {
        CUDADMA_BASE::template do_xfer<false, ALIGNMENT> ( src_temp, dst_temp, 
		       (this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ? 
		       (this_thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) : 
		       (this_thread_bytes ? MAX_BYTES_OUTSTANDING_PER_THREAD : 0));
      }
      src_row_ptr += dma_src_row_iter_inc;
      src_temp = src_row_ptr;
      dst_row_ptr += dma_dst_row_iter_inc;
      dst_temp = dst_row_ptr;
    }

    //No row leftovers

    CUDADMA_BASE::finish_async_dma();
  }

  // Public DMA-thread Synchronization Functions:
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    CUDADMA_BASE::wait_for_dma_start();
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    CUDADMA_BASE::finish_async_dma();
  }

};
    
#define MAX_SMALL_ELEMENTS_SIZE 64
template<int ALIGNMENT>
class cudaDMAStridedSmallElements : public CUDADMA_BASE {

  const int bytes_per_thread_per_el; //Bytes loaded per thread within one 'element'/'row'
  const int dma_row_iters; // Number of iters required to gather all strided 'elements'/'rows'
  const int dma_src_row_iter_inc; // Precomputed source offset for next row iteration
  const int dma_dst_row_iter_inc; // Precomputed dest offset for next row iteration
  const bool all_threads_active; // Precomputed enabler of optimized synchronization
  bool is_active_row_leftover_thread; //Determines whether thread is used to handle leftover rows
  

 public:

  // Constructor for when 4 <= el_sz <= 64
  // and when a destination stride is required in addition to a source stride
  __device__ cudaDMAStridedSmallElements (const int dmaID,
			     const int num_dma_threads,
			     const int num_compute_threads,
			     const int dma_threadIdx_start, 
			     const int el_sz, //; = BYTES_PER_EL_PER_THREAD * num_dma_threads 
                             const int el_cnt,
			     const int el_src_stride,
                             const int el_dst_stride)
    : CUDADMA_BASE (
                    dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
                    CUDADMA_DMA_TID*el_src_stride, 
                    CUDADMA_DMA_TID*el_src_stride+16, 
                    CUDADMA_DMA_TID*el_src_stride+32, 
                    CUDADMA_DMA_TID*el_src_stride+48,
                    CUDADMA_DMA_TID*el_src_stride, 
                    CUDADMA_DMA_TID*el_src_stride+16, 
                    CUDADMA_DMA_TID*el_src_stride+32, 
                    CUDADMA_DMA_TID*el_src_stride+48,
                    CUDADMA_DMA_TID*el_dst_stride, 
                    CUDADMA_DMA_TID*el_dst_stride+16, 
                    CUDADMA_DMA_TID*el_dst_stride+32, 
                    CUDADMA_DMA_TID*el_dst_stride+48,
                    CUDADMA_DMA_TID*el_dst_stride, 
                    CUDADMA_DMA_TID*el_dst_stride+16, 
                    CUDADMA_DMA_TID*el_dst_stride+32, 
                    CUDADMA_DMA_TID*el_dst_stride+48
                   ),
    bytes_per_thread_per_el (el_sz), 
    dma_row_iters ((el_cnt-1)/num_dma_threads), 
    all_threads_active (el_cnt%num_dma_threads==0),
    dma_src_row_iter_inc (el_src_stride*num_dma_threads),
    dma_dst_row_iter_inc (el_dst_stride*num_dma_threads)
    {
      is_active_row_leftover_thread = CUDADMA_BASE::dma_tid < el_cnt % num_dma_threads;
    }

  // DMA-thread Data Transfer Functions:
  __device__ void execute_dma ( void * src_ptr, void * dst_ptr) const
  {
#ifdef CUDADMA_DEBUG_ON
      if ((dma_tid==0)&&(CUDADMA_BASE::barrierID_full==4)&&(CUDADMA_BASE::is_dma_thread)) {
	printf("dma_tid = %d\nbytes_per_thread_per_el = %d\ndma_row_iters = %d\ndma_src_row_iter_inc = %d\ndma1_src_offs = %lu\ndma2_src_offs = %lu\ndma3_src_offs = %lu\ndma4_src_offs = %lu\ndma1_dst_offs = %lu\ndma2_dst_offs = %lu\ndma3_dst_offs = %lu\ndma4_dst_offs = %lu\n", CUDADMA_BASE::dma_tid, bytes_per_thread_per_el, dma_row_iters, dma_src_row_iter_inc,dma1_src_offs, dma2_src_offs, dma3_src_offs, dma4_src_offs, dma1_dst_offs, dma2_dst_offs, dma3_dst_offs, dma4_dst_offs);
      }
#endif
    if ((dma_row_iters>0) || (!all_threads_active)) {
      CUDADMA_BASE::wait_for_dma_start(); 
    }

    char * src_temp = (char *)src_ptr;
    char * dst_temp = (char *)dst_ptr;
    for(int i = 0; i < dma_row_iters; i++) {
      CUDADMA_BASE::template do_xfer<false, ALIGNMENT> ( src_temp, dst_temp, bytes_per_thread_per_el );
      src_temp += dma_src_row_iter_inc;
      dst_temp += dma_dst_row_iter_inc;
    }
    if (all_threads_active && !dma_row_iters) CUDADMA_BASE::template do_xfer<true, ALIGNMENT> ( src_temp, dst_temp, bytes_per_thread_per_el );
    else if (all_threads_active && dma_row_iters) CUDADMA_BASE::template do_xfer<false, ALIGNMENT> ( src_temp, dst_temp, bytes_per_thread_per_el );
    else if (is_active_row_leftover_thread) CUDADMA_BASE::template do_xfer<false, ALIGNMENT> ( src_temp, dst_temp, bytes_per_thread_per_el );

    CUDADMA_BASE::finish_async_dma();
  }

  // Public DMA-thread Synchronization Functions:
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    CUDADMA_BASE::wait_for_dma_start();
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    CUDADMA_BASE::finish_async_dma();
  }

};

#else // The old cudaDMAStrided

// The idea behind this version of cudaDMAStrided is to check to see how many warps are
// required per element and then figure out how many elements can be loaded at a time.
#define MAX_WARPS_PER_ELMT ((el_sz+(warpSize*MAX_BYTES_OUTSTANDING_PER_THREAD-1))/(warpSize*MAX_BYTES_OUTSTANDING_PER_THREAD))
#define ELMT_PER_STEP ((num_dma_threads/warpSize+MAX_WARPS_PER_ELMT-1)/MAX_WARPS_PER_ELMT)
#define ELMT_ID ((CUDADMA_DMA_TID/warpSize)/MAX_WARPS_PER_ELMT)
// For a given elmt, figure out how many warps there are loading it
// If there is just one element, then all the warps will load it
// Otherwise all elements get MAX_WARPS_PER_ELMT except the last one which gets the left over warps
#define WARPS_PER_EL (ELMT_PER_STEP == 1 ? (num_dma_threads/warpSize) : (ELMT_ID < (ELMT_PER_STEP-1) ? MAX_WARPS_PER_ELMT : (num_dma_threads/warpSize) - (ELMT_PER_STEP-1)*MAX_WARPS_PER_ELMT))
#define CUDADMA_WARP_TID (threadIdx.x - (dma_threadIdx_start + (ELMT_ID*MAX_WARPS_PER_ELMT*warpSize)))
#define CUDADMASTRIDED_DMA1_ITER_OFFS (ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMASTRIDED_DMA2_ITER_OFFS (1*ALIGNMENT*WARPS_PER_EL*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMASTRIDED_DMA3_ITER_OFFS (2*ALIGNMENT*WARPS_PER_EL*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMASTRIDED_DMA4_ITER_OFFS (3*ALIGNMENT*WARPS_PER_EL*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMASTRIDED_DMA1_OFFS (ALIGNMENT*CUDADMA_WARP_TID)	
#define CUDADMASTRIDED_DMA2_OFFS (1*ALIGNMENT*WARPS_PER_EL*warpSize + ALIGNMENT*CUDADMA_WARP_TID)	
#define CUDADMASTRIDED_DMA3_OFFS (2*ALIGNMENT*WARPS_PER_EL*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMASTRIDED_DMA4_OFFS (3*ALIGNMENT*WARPS_PER_EL*warpSize + ALIGNMENT*CUDADMA_WARP_TID)

template<int ALIGNMENT>
class cudaDMAStrided : public CUDADMA_BASE
{
private:
	const int el_sz;
	const int dma_col_iters;
	const int dma_col_iter_inc;
	const int dma_row_iters;
	const int dma_src_row_iter_inc;
	const int dma_dst_row_iter_inc;
	const int dma_src_offset;
	const int dma_dst_offset;
	int thread_bytes;
public:
  // Constructor for when dst_stride is the same as el_sz
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_dma_threads,
			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int el_sz,
			     const int el_cnt,
			     const int el_stride)	
	: CUDADMA_BASE (dmaID,
			num_dma_threads,
			num_compute_threads,
			dma_threadIdx_start,
			CUDADMASTRIDED_DMA1_ITER_OFFS,
			CUDADMASTRIDED_DMA2_ITER_OFFS,
			CUDADMASTRIDED_DMA3_ITER_OFFS,
			CUDADMASTRIDED_DMA4_ITER_OFFS,
			CUDADMASTRIDED_DMA1_OFFS,
			CUDADMASTRIDED_DMA2_OFFS,
			CUDADMASTRIDED_DMA3_OFFS,
			CUDADMASTRIDED_DMA4_OFFS,			
			CUDADMASTRIDED_DMA1_ITER_OFFS,
			CUDADMASTRIDED_DMA2_ITER_OFFS,
			CUDADMASTRIDED_DMA3_ITER_OFFS,
			CUDADMASTRIDED_DMA4_ITER_OFFS,
			CUDADMASTRIDED_DMA1_OFFS,
			CUDADMASTRIDED_DMA2_OFFS,
			CUDADMASTRIDED_DMA3_OFFS,
			CUDADMASTRIDED_DMA4_OFFS),
		el_sz (el_sz),	
		dma_col_iters ((el_sz-4)/(MAX_BYTES_OUTSTANDING_PER_THREAD*WARPS_PER_EL*warpSize)),
		dma_col_iter_inc (MAX_BYTES_OUTSTANDING_PER_THREAD*WARPS_PER_EL*warpSize),
		dma_row_iters (el_cnt/ELMT_PER_STEP + (ELMT_ID<(el_cnt%ELMT_PER_STEP) ? 1 : 0)),
		dma_src_row_iter_inc (el_stride*ELMT_PER_STEP),
		dma_dst_row_iter_inc (el_sz*ELMT_PER_STEP),
		dma_src_offset (ELMT_ID*el_stride),
		dma_dst_offset (ELMT_ID*el_sz)
	{
		int num_vec_loads  = el_sz / (ALIGNMENT*WARPS_PER_EL*warpSize);
		int leftover_bytes = el_sz % (ALIGNMENT*WARPS_PER_EL*warpSize);

		if (leftover_bytes==0)
		{
			thread_bytes = num_vec_loads*ALIGNMENT;
		}
		else
		{
			int max_thread_bytes = ALIGNMENT;
			if (leftover_bytes>=(max_thread_bytes*(CUDADMA_WARP_TID+1)))
			{
				// Below: do alignment-byte xfers
				thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + max_thread_bytes;
			}
			else if (leftover_bytes<(max_thread_bytes*CUDADMA_WARP_TID))
			{
				// Above: do 0-byte xfers on last load
				thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT;
			}
			else
			{
				// This is the partial thread
				thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + (leftover_bytes%max_thread_bytes);
			}
		}
	}

  // Constructor for different source and destination strides
  __device__ cudaDMAStrided (const int dmaID,
			     const int num_dma_threads,
			     const int num_compute_threads,
			     const int dma_threadIdx_start,
			     const int el_sz,
			     const int el_cnt,
			     const int src_stride,
			     const int dst_stride)	
	: CUDADMA_BASE (dmaID,
			num_dma_threads,
			num_compute_threads,
			dma_threadIdx_start,
			CUDADMASTRIDED_DMA1_ITER_OFFS,
			CUDADMASTRIDED_DMA2_ITER_OFFS,
			CUDADMASTRIDED_DMA3_ITER_OFFS,
			CUDADMASTRIDED_DMA4_ITER_OFFS,
			CUDADMASTRIDED_DMA1_OFFS,
			CUDADMASTRIDED_DMA2_OFFS,
			CUDADMASTRIDED_DMA3_OFFS,
			CUDADMASTRIDED_DMA4_OFFS,			
			CUDADMASTRIDED_DMA1_ITER_OFFS,
			CUDADMASTRIDED_DMA2_ITER_OFFS,
			CUDADMASTRIDED_DMA3_ITER_OFFS,
			CUDADMASTRIDED_DMA4_ITER_OFFS,
			CUDADMASTRIDED_DMA1_OFFS,
			CUDADMASTRIDED_DMA2_OFFS,
			CUDADMASTRIDED_DMA3_OFFS,
			CUDADMASTRIDED_DMA4_OFFS),
		el_sz (el_sz),	
		dma_col_iters ((el_sz-4)/(MAX_BYTES_OUTSTANDING_PER_THREAD*WARPS_PER_EL*warpSize)),
		dma_col_iter_inc (MAX_BYTES_OUTSTANDING_PER_THREAD*WARPS_PER_EL*warpSize),
		dma_row_iters (el_cnt/ELMT_PER_STEP + (ELMT_ID<(el_cnt%ELMT_PER_STEP) ? 1 : 0)),
		dma_src_row_iter_inc (src_stride*ELMT_PER_STEP),
		dma_dst_row_iter_inc (dst_stride*ELMT_PER_STEP),
		dma_src_offset (ELMT_ID*src_stride),
		dma_dst_offset (ELMT_ID*dst_stride)
	{
		int num_vec_loads  = el_sz / (ALIGNMENT*WARPS_PER_EL*warpSize);
		int leftover_bytes = el_sz % (ALIGNMENT*WARPS_PER_EL*warpSize);

		if (leftover_bytes==0)
		{
			thread_bytes = num_vec_loads*ALIGNMENT;
		}
		else
		{
			int max_thread_bytes = ALIGNMENT;
			if (leftover_bytes>=(max_thread_bytes*(CUDADMA_WARP_TID+1)))
			{
				// Below: do alignment-byte xfers
				thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + max_thread_bytes;
			}
			else if (leftover_bytes<(max_thread_bytes*CUDADMA_WARP_TID))
			{
				// Above: do 0-byte xfers on last load
				thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT;
			}
			else
			{
				// This is the partial thread
				thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + (leftover_bytes%max_thread_bytes);
			}
		}
#ifdef CUDADMA_DEBUG_ON 
		if ((CUDADMA_BASE::barrierID_full==2)&&(CUDADMA_BASE::is_dma_thread))
		{
			printf("DMA id %d: element_id %d  src_offset %d  dst_offset %d  row_iters %d  warps_per_elem %d\n",CUDADMA_DMA_TID,ELMT_ID,dma_src_offset,dma_dst_offset,dma_row_iters,WARPS_PER_EL);
		} 
#endif
	}
public:
  __device__ __forceinline__ void execute_dma( void * src_ptr, void * dst_ptr) const
  {
	// Wait for the transfer to begin
	CUDADMA_BASE::wait_for_dma_start();

	char * src_row_ptr = ((char*)src_ptr)+dma_src_offset;
	char * dst_row_ptr = ((char*)dst_ptr)+dma_dst_offset;
	char * src_temp    = src_row_ptr;
	char * dst_temp    = dst_row_ptr;
	for (int i = 0; i < dma_row_iters; i++)
	{
	  switch (ALIGNMENT)
	    {
	      case 4:
	        {
		  for (int j = 0; j < dma_col_iters; j++)
		  {
		    CUDADMA_BASE::template perform_four_xfers<float,float,false,false> (src_temp, dst_temp);
		    src_temp += dma_col_iter_inc;
		    dst_temp += dma_col_iter_inc;
		  }
		  break;
	        }
	      case 8:
	        {
		  for (int j = 0; j < dma_col_iters; j++)
		  {
		    CUDADMA_BASE::template perform_four_xfers<float2,float2,false,false> (src_temp, dst_temp);
		    src_temp += dma_col_iter_inc;
		    dst_temp += dma_col_iter_inc;
		  }
		  break;
	        }
	      case 16:
	        {
		  for (int j = 0; j < dma_col_iters; j++)
		  {
		    CUDADMA_BASE::template perform_four_xfers<float4,float4,false,false> (src_temp, dst_temp);
		    src_temp += dma_col_iter_inc;
		    dst_temp += dma_col_iter_inc;
		  }
		  break;
	        }
	      default:
	        printf("ALIGNMENT must be one of (4,8,16)\n");
		break;
	    }
	  // Now handle the column leftovers
	  CUDADMA_BASE::template do_xfer<false,ALIGNMENT> (src_temp, dst_temp,
				(thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ?
				(thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) :
				(thread_bytes ? MAX_BYTES_OUTSTANDING_PER_THREAD : 0));
	  // Now set up for the next element
	  src_row_ptr += dma_src_row_iter_inc;
	  src_temp = src_row_ptr;
	  dst_row_ptr += dma_dst_row_iter_inc;
	  dst_temp = dst_row_ptr;
	}
	// Now we're finished, indicate we're done
	CUDADMA_BASE::finish_async_dma();
  }
public:
  // Public DMA-thread Synchronization Functions
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    CUDADMA_BASE::wait_for_dma_start();
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    CUDADMA_BASE::finish_async_dma();
  }
};
#undef MAX_WARPS_PER_ELMT
#undef ELMT_PER_STEP
#undef ELMT_ID
#undef WARPS_PER_EL
#undef CUDADMA_WARP_TID
#undef CUDADMASTRIDED_DMA1_ITER_OFFS
#undef CUDADMASTRIDED_DMA2_ITER_OFFS
#undef CUDADMASTRIDED_DMA3_ITER_OFFS
#undef CUDADMASTRIDED_DMA4_ITER_OFFS
#undef CUDADMASTRIDED_DMA1_OFFS
#undef CUDADMASTRIDED_DMA2_OFFS
#undef CUDADMASTRIDED_DMA3_OFFS
#undef CUDADMASTRIDED_DMA4_OFFS
#endif // Old cudaDMAStrided

#define ROW_BYTES (CORNERS ? (dimx+2*RADIUS)*sizeof(ELMT_TYPE) : dimx*sizeof(ELMT_TYPE))
#define MAX_WARPS_PER_ROW ((ROW_BYTES+(warpSize*MAX_BYTES_OUTSTANDING_PER_THREAD-1))/(warpSize*MAX_BYTES_OUTSTANDING_PER_THREAD))
#define ROWS_PER_STEP ((num_dma_threads/warpSize+MAX_WARPS_PER_ROW-1)/MAX_WARPS_PER_ROW)
#define ROW_ID ((CUDADMA_DMA_TID/warpSize)/MAX_WARPS_PER_ROW)
// For a given row, figure out how many warps there are loading it
// There will always be at least 2 rows
#define WARPS_PER_ROW (ROWS_PER_STEP == 1 ? (num_dma_threads/warpSize) : (ROW_ID < (ROWS_PER_STEP-1) ? MAX_WARPS_PER_ROW : (num_dma_threads/warpSize) - (ROWS_PER_STEP-1)*MAX_WARPS_PER_ROW))
#define CUDADMA_WARP_TID (threadIdx.x - (dma_threadIdx_start + (ROW_ID*MAX_WARPS_PER_ROW*warpSize)))
#define CUDADMAHALO_DMA1_ITER_OFFS (ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMAHALO_DMA2_ITER_OFFS (1*ALIGNMENT*WARPS_PER_ROW*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMAHALO_DMA3_ITER_OFFS (2*ALIGNMENT*WARPS_PER_ROW*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMAHALO_DMA4_ITER_OFFS (3*ALIGNMENT*WARPS_PER_ROW*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMAHALO_DMA1_OFFS (ALIGNMENT*CUDADMA_WARP_TID)	
#define CUDADMAHALO_DMA2_OFFS (1*ALIGNMENT*WARPS_PER_ROW*warpSize + ALIGNMENT*CUDADMA_WARP_TID)	
#define CUDADMAHALO_DMA3_OFFS (2*ALIGNMENT*WARPS_PER_ROW*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
#define CUDADMAHALO_DMA4_OFFS (3*ALIGNMENT*WARPS_PER_ROW*warpSize + ALIGNMENT*CUDADMA_WARP_TID)
// We also need to figure out how many loads need to be performed for the sides
// Figure out the biggest size loads we can perform for a given radius and alignment
#define SIDE_BYTES (RADIUS*sizeof(ELMT_TYPE))
// This value should be statically determined 
#define SIDE_XFER_SIZE ((SIDE_BYTES%ALIGNMENT)==0 ? ALIGNMENT : (SIDE_BYTES%(ALIGNMENT/2))==0 ? ALIGNMENT/2 : ALIGNMENT/4)
#define TOTAL_SIDE_LOADS (SIDE_BYTES/SIDE_XFER_SIZE)
// We want threads per side to be a multiple of 2 so it should divide evenly into warpSize
#define THREADS_PER_SIDE ((TOTAL_SIDE_LOADS==1) ? 1 : (TOTAL_SIDE_LOADS==2) ? 2 : (TOTAL_SIDE_LOADS <=4) ? 4 : 8)
// Total number of sides that can be handled by a warp, each thread can handle MAX_LDS_OUTSTANDING sides 
#define SIDES_PER_WARP (warpSize/THREADS_PER_SIDE)*MAX_LDS_OUTSTANDING_PER_THREAD
// The remaining warps after the warps have been allocated to the rows
#define REMAINING_WARPS ((ROWS_PER_STEP*MAX_WARPS_PER_ROW < (num_dma_threads/warpSize)) ? \
				(num_dma_threads/warpSize) - (ROWS_PER_STEP*MAX_WARPS_PER_ROW) : 0 )

// Asserting sizeof(ELMT_TYPE) <= ALIGNMENT
template<typename ELMT_TYPE, int RADIUS, bool CORNERS, int ALIGNMENT>
class cudaDMAHalo : public CUDADMA_BASE
{
private:
	// Values for moving rows
	const int row_sz;
	const int dma_col_iters;
	const int dma_col_iter_inc;
	const int dma_row_id;
	const int dma_row_iters;
	const int dma_src_row_iter_inc;
	const int dma_dst_row_iter_inc;
	const int dma_src_top_offset;
	const int dma_src_bot_offset;
	const int dma_dst_top_offset;
	const int dma_dst_bot_offset;
	int thread_bytes;
	// Values for moving the sides
	int side_id;
	int side_load;
	int side_src_iter_inc;
	int side_dst_iter_inc;
	int side_src_offset;
	int side_dst_offset;
	int side_iters;
	bool side_active;

	// Optimizations for different loading techniques
	const bool split; // If we split warps into row warps and side warps
	const bool optimized; // If everything can be loaded in one step, pre-load into registers
	const bool row_thread; // true if responsible for loading rows, otherwise a side thread in split cases
public:
  __device__ cudaDMAHalo (const int dmaID,
			  const int num_dma_threads,
			  const int num_compute_threads,
			  const int dma_threadIdx_start,
			  const int dimx,
			  const int dimy,
			  const int pitch)
	: CUDADMA_BASE (dmaID,
			num_dma_threads,
			num_compute_threads,
			dma_threadIdx_start,
			CUDADMAHALO_DMA1_ITER_OFFS,
			CUDADMAHALO_DMA2_ITER_OFFS,
			CUDADMAHALO_DMA3_ITER_OFFS,
			CUDADMAHALO_DMA4_ITER_OFFS,
			CUDADMAHALO_DMA1_OFFS,
			CUDADMAHALO_DMA2_OFFS,
			CUDADMAHALO_DMA3_OFFS,
			CUDADMAHALO_DMA4_OFFS,	
			CUDADMAHALO_DMA1_ITER_OFFS,
			CUDADMAHALO_DMA2_ITER_OFFS,
			CUDADMAHALO_DMA3_ITER_OFFS,
			CUDADMAHALO_DMA4_ITER_OFFS,
			CUDADMAHALO_DMA1_OFFS,
			CUDADMAHALO_DMA2_OFFS,
			CUDADMAHALO_DMA3_OFFS,
			CUDADMAHALO_DMA4_OFFS),	
		row_sz (ROW_BYTES),
		dma_col_iters ((ROW_BYTES-4)/(MAX_BYTES_OUTSTANDING_PER_THREAD*WARPS_PER_ROW*warpSize)),
		dma_col_iter_inc (MAX_BYTES_OUTSTANDING_PER_THREAD*WARPS_PER_ROW*warpSize),
		dma_row_id (ROW_ID),	
		dma_row_iters (ROWS_PER_STEP),
		dma_src_row_iter_inc (pitch*sizeof(ELMT_TYPE)),
		dma_dst_row_iter_inc ((2*RADIUS+dimx)*sizeof(ELMT_TYPE)),
		dma_src_top_offset (CORNERS ? (-(RADIUS*pitch + RADIUS)*sizeof(ELMT_TYPE)) : 
						(-(RADIUS*pitch)*sizeof(ELMT_TYPE))),
		dma_src_bot_offset (CORNERS ? ((dimy*pitch - RADIUS)*sizeof(ELMT_TYPE)) : 
						(dimy*pitch)*sizeof(ELMT_TYPE)),
		dma_dst_top_offset (CORNERS ? (-(RADIUS*(dimx+2*RADIUS)+RADIUS)*sizeof(ELMT_TYPE)) : 
						(-(RADIUS*(dimx+2*RADIUS)*sizeof(ELMT_TYPE)))),
		dma_dst_bot_offset (CORNERS ? ((dimy*(dimx+2*RADIUS)-RADIUS)*sizeof(ELMT_TYPE)) : 
						(dimy*(dimx+2*RADIUS)*sizeof(ELMT_TYPE))),
		split (REMAINING_WARPS>0),	
		optimized ((REMAINING_WARPS>0) && ((REMAINING_WARPS*SIDES_PER_WARP)>=2*dimy)),
		row_thread (ROW_ID<(2*RADIUS))
	{
		if (REMAINING_WARPS>0)
		{
			// In this case we will have seperate warps for handling the rows and the sides
			// This encompasses both the split and optimized cases
			if (!(ROW_ID<(2*RADIUS)))
			{
				int local_id = threadIdx.x - (dma_threadIdx_start + num_dma_threads 
									- (REMAINING_WARPS*warpSize));	
				side_id = local_id/THREADS_PER_SIDE;
				side_load = local_id%THREADS_PER_SIDE;
				side_active = (side_load < TOTAL_SIDE_LOADS) && (side_id < (2*dimy));
				side_src_iter_inc = ((REMAINING_WARPS*warpSize/THREADS_PER_SIDE)/2)*pitch*sizeof(ELMT_TYPE);
				side_dst_iter_inc = ((REMAINING_WARPS*warpSize/THREADS_PER_SIDE)/2)*(2*RADIUS+dimx)*sizeof(ELMT_TYPE);
				side_src_offset = ((side_id%2 ? dimx+side_load : -(THREADS_PER_SIDE-side_load)) + (side_id/2)*pitch)*sizeof(ELMT_TYPE);
				side_dst_offset = ((side_id%2 ? dimx+side_load : -(THREADS_PER_SIDE-side_load)) + (side_id/2)*(2*RADIUS+dimx))*sizeof(ELMT_TYPE);
			}
			else // These are the row threads, finish computing their bytes
			{
				int num_vec_loads  = ROW_BYTES / (ALIGNMENT*WARPS_PER_ROW*warpSize);
				int leftover_bytes = ROW_BYTES % (ALIGNMENT*WARPS_PER_ROW*warpSize);

				if (leftover_bytes==0)
					thread_bytes = num_vec_loads*ALIGNMENT;
				else
				{
					int max_thread_bytes = ALIGNMENT;
					if (leftover_bytes>=(max_thread_bytes*(CUDADMA_WARP_TID+1)))
					{
						// Below: do alignment-byte xfers
						thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + max_thread_bytes;
					}
					else if (leftover_bytes<(max_thread_bytes*CUDADMA_WARP_TID))
					{
						// Above: do 0-byte xfers on last load
						thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT;
					}
					else
					{
						// This is the partial thread
						thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + (leftover_bytes%max_thread_bytes);
					}
				}
			}
		}
		else
		{
			// Unoptimized case where all threads will do both rows and sides
			side_id   = CUDADMA_DMA_TID / THREADS_PER_SIDE;
			side_load = CUDADMA_DMA_TID % THREADS_PER_SIDE;
			side_active = (side_load < TOTAL_SIDE_LOADS) && (side_id < (2*dimy));
			// Compute side_stride from the number of rows that can be handled
			side_src_iter_inc = ((num_dma_threads/THREADS_PER_SIDE)/2)*pitch*sizeof(ELMT_TYPE);
			side_dst_iter_inc = ((num_dma_threads/THREADS_PER_SIDE)/2)*(2*RADIUS+dimx)*sizeof(ELMT_TYPE);
			side_iters = (2*dimy - side_id +(num_dma_threads/THREADS_PER_SIDE)-1)/(num_dma_threads/THREADS_PER_SIDE);
			side_src_offset = ((side_id%2 ? dimx+side_load : -(THREADS_PER_SIDE-side_load)) + (side_id/2)*pitch)*sizeof(ELMT_TYPE);
			side_dst_offset = ((side_id%2 ? dimx+side_load : -(THREADS_PER_SIDE-side_load)) + (side_id/2)*(2*RADIUS+dimx))*sizeof(ELMT_TYPE);

			// Finally set up the row information
			int num_vec_loads  = ROW_BYTES / (ALIGNMENT*WARPS_PER_ROW*warpSize);
			int leftover_bytes = ROW_BYTES % (ALIGNMENT*WARPS_PER_ROW*warpSize);

			if (leftover_bytes==0)
				thread_bytes = num_vec_loads*ALIGNMENT;
			else
			{
				int max_thread_bytes = ALIGNMENT;
				if (leftover_bytes>=(max_thread_bytes*(CUDADMA_WARP_TID+1)))
				{
					// Below: do alignment-byte xfers
					thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + max_thread_bytes;
				}
				else if (leftover_bytes<(max_thread_bytes*CUDADMA_WARP_TID))
				{
					// Above: do 0-byte xfers on last load
					thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT;
				}
				else
				{
					// This is the partial thread
					thread_bytes = (num_vec_loads-(dma_col_iters*MAX_LDS_OUTSTANDING_PER_THREAD))*ALIGNMENT + (leftover_bytes%max_thread_bytes);
				}
			}	
		}
	}
public:
  __device__ __forceinline__ void execute_dma(void * src_origin, void * dst_origin) const
  {
	// First check if this is the optimized case
	if (optimized)
	{
		if (dma_row_id < (2*RADIUS)) // These are the row loader threads
		{
			
		}
		else // These are the side loader threads
		{

		}
	}
	else
	{
		// If this isn't optimized check to see if the warps are split or not
		if (split)
		{
			if (dma_row_id < (2*RADIUS)) // These are the row loader threads
			{

			}
			else // These are the side loader threads
			{

			}
		}
		else  // In this case all threads have to load both rows and sides (the general case)
		{
			// Wait for the transfer to begin
			CUDADMA_BASE::wait_for_dma_start();
			// Do the top first
			char * src_row_ptr = ((char*)src_origin) + dma_src_top_offset;
			char * dst_row_ptr = ((char*)dst_origin) + dma_dst_top_offset;
			int row_id = dma_row_id;
			load_rows<RADIUS>(row_id, src_row_ptr, dst_row_ptr);
			// Now do the bottom set of rows
			src_row_ptr = ((char*)src_origin) + dma_src_bot_offset;
			dst_row_ptr = ((char*)dst_origin) + dma_dst_bot_offset;
			load_rows<2*RADIUS>(row_id, src_row_ptr, dst_row_ptr);
			// We've finished the rows, now do the sides
			char * src_side_ptr = ((char*)src_origin) + side_src_offset;
			char * dst_side_ptr = ((char*)dst_origin) + side_dst_offset;
			if (side_active)
				load_sides(src_side_ptr, dst_side_ptr);	
			// We're finally finished, indicate we're done
			CUDADMA_BASE::finish_async_dma();
		}
	}
  }
private: // Helper methods
  template<int MAX_ROW>
  __device__ __forceinline__ void load_rows(int &row_id, char *src_row_ptr, char *dst_row_ptr) const
  {
	char * src_temp = src_row_ptr;
	char * dst_temp = dst_row_ptr;
	for ( ; row_id<MAX_ROW; row_id+=dma_row_iters)
	{
	  switch(ALIGNMENT)
	    {
	      case 4:
		{
		  for (int j=0; j < dma_col_iters; j++)
		  {
		    CUDADMA_BASE::template perform_four_xfers<float,float,false,false> (src_temp,dst_temp);
		    src_temp += dma_col_iter_inc;
		    dst_temp += dma_col_iter_inc;
		  }
		  break;
		}
	      case 8:
		{
		  for (int j=0; j < dma_col_iters; j++)
		  {
		    CUDADMA_BASE::template perform_four_xfers<float2,float2,false,false> (src_temp,dst_temp);	
		    src_temp += dma_col_iter_inc;
		    dst_temp += dma_col_iter_inc;
		  }
		  break;
		}
	      case 16:
		{
		  for (int j=0; j < dma_col_iters; j++)
		  {
		    CUDADMA_BASE::template perform_four_xfers<float4,float4,false,false> (src_temp,dst_temp);
		    src_temp += dma_col_iter_inc;
		    dst_temp += dma_col_iter_inc;
		  }
		  break;
		}
	      default:
		printf("ALIGNMENT must be one of (4,8,16)\n");
		break;
	    }
	  // Now handle the column leftovers
	  CUDADMA_BASE::template do_xfer<false,ALIGNMENT> (src_temp, dst_temp,
			(thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) ?
			(thread_bytes%MAX_BYTES_OUTSTANDING_PER_THREAD) :
			(thread_bytes ? MAX_BYTES_OUTSTANDING_PER_THREAD : 0));
	  // Now set up for the next row
	  src_row_ptr += (dma_row_iters*dma_src_row_iter_inc);
	  src_temp = src_row_ptr;
	  dst_row_ptr += (dma_row_iters*dma_dst_row_iter_inc);
	  dst_temp = dst_row_ptr;
	}
  }
  __device__ __forceinline__ void load_sides(char * src_side_ptr, char * dst_side_ptr) const
  {
	for (int i = 0; i < side_iters; i++)
	{
		switch (SIDE_XFER_SIZE)
		  {
		    case 4:
		      {
			float temp = (*((float*)(src_side_ptr)));
			(*((float*)(dst_side_ptr))) = temp;
			break;
		      }
		    case 8:
		      {
			float2 temp = (*((float2*)(src_side_ptr)));
			(*((float2*)(dst_side_ptr))) = temp;	
			break;
		      }
		    case 16:
		      {
			float4 temp = (*((float4*)(src_side_ptr)));
			(*((float4*)(dst_side_ptr))) = temp;
			break;
		      }
		    default:
		      printf("Warning CUDA_DMA internal error, invalid side xfer size: %d\n",SIDE_XFER_SIZE);
		      break;
		  }	
		// Get ready for the next transfer
		src_side_ptr += side_src_iter_inc;
		dst_side_ptr += side_dst_iter_inc;
	}
  }
public:
  // Public DMA-thread Synchronization Functions:
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    CUDADMA_BASE::wait_for_dma_start();
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    CUDADMA_BASE::finish_async_dma();
  }
};
#undef ROW_BYTES
#undef MAX_WARPS_PER_ROW
#undef ROWS_PER_STEP
#undef ROW_ID
#undef WARPS_PER_ROW
#undef CUDADMA_WARP_TID
#undef CUDADMAHALO_DMA1_ITER_OFFS
#undef CUDADMAHALO_DMA2_ITER_OFFS
#undef CUDADMAHALO_DMA3_ITER_OFFS
#undef CUDADMAHALO_DMA4_ITER_OFFS
#undef CUDADMAHALO_DMA1_OFFS
#undef CUDADMAHALO_DMA2_OFFS
#undef CUDADMAHALO_DMA3_OFFS
#undef CUDADMAHALO_DMA4_OFFS
#undef SIDE_BYTES
#undef SIDE_XFER_SIZE
#undef TOTAL_SIDE_LOADS
#undef THREADS_PER_SIDE
#undef SIDES_PER_WARP
#undef REMAINING_WARPS
    
template<int ALIGNMENT>
class cudaDMACustom : public CUDADMA_BASE {

  __device__ cudaDMACustom (const int dmaID,
				const int num_dma_threads,
				const int num_compute_threads,
				const int dma_threadIdx_start)
    : CUDADMA_BASE (dmaID,
		    num_dma_threads,
		    num_compute_threads,
		    dma_threadIdx_start, 
		    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    {
    }

  public:
  // Public DMA-thread Synchronization Functions:
  __device__ __forceinline__ void wait_for_dma_start() const
  {
    CUDADMA_BASE::wait_for_dma_start();
  }
  __device__ __forceinline__ void finish_async_dma() const
  {
    CUDADMA_BASE::finish_async_dma();
  }
};

#undef CUDADMA_BASE
#undef MAX_BYTES_OUTSTANDING_PER_THREAD
#undef MAX_LDS_OUTSTANDING_PER_THREAD
#undef CUDADMA_DMA_TID

// EOF

