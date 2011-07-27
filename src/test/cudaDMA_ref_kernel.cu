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

/* Reference kernel code for efficient DMAs
 */

#ifndef _SW_DMA_KERNEL_H_
#define _SW_DMA_KERNEL_H_

#include <stdio.h>

#define NUM_ELEMENTS 512
#define NUM_ITERS 8192
#define NUM_FLOATS 4*NUM_ELEMENTS

//#define DEBUG_PRINT 1

__device__ __forceinline__ void ptx_barrier_blocking (int name, unsigned int num_barriers)
{
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(num_barriers) : "memory" );
}

__device__ __forceinline__ void ptx_barrier_nonblocking (int name, unsigned int num_barriers)
{
  //  __threadfence_block();
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(num_barriers) : "memory" );
}

__global__ void
dma_4ld( float4* g_idata, float4* g_odata, clock_t * timer_vals) 
{
  // shared memory
  __shared__  float sdata_x_i0[NUM_ELEMENTS];
  __shared__  float sdata_y_i0[NUM_ELEMENTS];
  __shared__  float sdata_z_i0[NUM_ELEMENTS];
  __shared__  float sdata_w_i0[NUM_ELEMENTS];
  __shared__  float sdata_x_i1[NUM_ELEMENTS];
  __shared__  float sdata_y_i1[NUM_ELEMENTS];
  __shared__  float sdata_z_i1[NUM_ELEMENTS];
  __shared__  float sdata_w_i1[NUM_ELEMENTS];

  // Constants
  unsigned int num_dma_hb_threads = NUM_ELEMENTS / 4;

  // access thread id
  unsigned int tid = threadIdx.x ;
  unsigned int num_barriers = NUM_ELEMENTS + num_dma_hb_threads;


  unsigned int dma2_offs = num_dma_hb_threads;
  unsigned int dma3_offs = 2*num_dma_hb_threads;
  unsigned int dma4_offs = 3*num_dma_hb_threads;
  
  unsigned int offset = 2 * NUM_ELEMENTS;

  // Preamble
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (tid<NUM_ELEMENTS) {
    sdata_x_i0[tid] = 0.0f;
    sdata_y_i0[tid] = 0.0f;
    sdata_z_i0[tid] = 0.0f;
    sdata_w_i0[tid] = 0.0f;
    sdata_x_i1[tid] = 0.0f;
    sdata_y_i1[tid] = 0.0f;
    sdata_z_i1[tid] = 0.0f;
    sdata_w_i1[tid] = 0.0f;
  }
  __syncthreads();

  if (tid<NUM_ELEMENTS) {

    float4 tmp;

    // This is the compute code

    // Pre-amble:
    // Non-blocking sync signalling sdata_i1 being empty:
    ptx_barrier_nonblocking(2,num_barriers);


    for (unsigned int i = 0; i < NUM_ITERS; i+=2) {

      // Phase 1:
      // Consumer blocking sync waiting on sdata_i0 being full:
      ptx_barrier_blocking(0, num_barriers);
      float tmp0_x = sdata_x_i0[tid];
      float tmp0_y = sdata_y_i0[tid];
      float tmp0_z = sdata_z_i0[tid];
      float tmp0_w = sdata_w_i0[tid];
      acc.x += tmp0_x;
      acc.y += tmp0_y;
      acc.z += tmp0_z;
      acc.w += tmp0_w;

      // Consumer non-blocking sync signaling sdata_i0 being empty:
      ptx_barrier_nonblocking(1,num_barriers);


      // Phase 2:
      // Consumer blocking sync waiting on sdata_i1 being full:
      ptx_barrier_blocking(3, num_barriers);

      float tmp1_x = sdata_x_i1[tid];
      float tmp1_y = sdata_y_i1[tid];
      float tmp1_z = sdata_z_i1[tid];
      float tmp1_w = sdata_w_i1[tid];
      

      // Consumer non-blocking sync on sdata_i1 being empty:
      ptx_barrier_nonblocking(2,num_barriers);

      acc.x += tmp1_x;
      acc.y += tmp1_y;
      acc.z += tmp1_z;
      acc.w += tmp1_w;

    }

    // Postamble:
    ptx_barrier_blocking(0, num_barriers);
    tmp.x = sdata_x_i0[tid];
    tmp.y = sdata_y_i0[tid];
    tmp.z = sdata_z_i0[tid];
    tmp.w = sdata_w_i0[tid];
    acc.x += tmp.x;
    acc.y += tmp.y;
    acc.z += tmp.z;
    acc.w += tmp.w;
    g_odata[tid] = acc;

  } else {

    // This is the DMA code:

    // This is one set of DMA threads for one half-buffer:
    if (tid<num_barriers) {

      unsigned int dma_tid = tid - NUM_ELEMENTS;

      float * ptr_x_i1_dma1 = &sdata_x_i1[dma_tid];
      float * ptr_y_i1_dma1 = &sdata_y_i1[dma_tid];
      float * ptr_z_i1_dma1 = &sdata_z_i1[dma_tid];
      float * ptr_w_i1_dma1 = &sdata_w_i1[dma_tid];

      unsigned int idx = dma_tid;
      float4 * tmp1_ptr = &g_idata[idx];
      float4 * tmp2_ptr = &g_idata[idx + dma2_offs];
      float4 * tmp3_ptr = &g_idata[idx + dma3_offs];
      float4 * tmp4_ptr = &g_idata[idx + dma4_offs];


      for (unsigned int j = 0; j < NUM_ITERS; j+=2) {

	// PHASE 1 - global load to registers
  	volatile float4 tmp1 = *tmp1_ptr;
  	volatile float4 tmp2 = *tmp2_ptr;
  	volatile float4 tmp3 = *tmp3_ptr;
  	volatile float4 tmp4 = *tmp4_ptr;

	// PHASE 2 - sync with compute warps, blast st.shared's from RF, sync with compute threads
	// Producer blocking sync waiting on sdata_i1 being empty:
	ptx_barrier_blocking(2,num_barriers);


	*ptr_x_i1_dma1 = tmp1.x;
	*ptr_y_i1_dma1 = tmp1.y;
	*ptr_z_i1_dma1 = tmp1.z;
	*ptr_w_i1_dma1 = tmp1.w;
	*(ptr_x_i1_dma1+dma2_offs) = tmp2.x;
	*(ptr_y_i1_dma1+dma2_offs) = tmp2.y;
	*(ptr_z_i1_dma1+dma2_offs) = tmp2.z;
	*(ptr_w_i1_dma1+dma2_offs) = tmp2.w;
	*(ptr_x_i1_dma1+dma3_offs) = tmp3.x;
	*(ptr_y_i1_dma1+dma3_offs) = tmp3.y;
	*(ptr_z_i1_dma1+dma3_offs) = tmp3.z;
	*(ptr_w_i1_dma1+dma3_offs) = tmp3.w;
	*(ptr_x_i1_dma1+dma4_offs) = tmp4.x;
	*(ptr_y_i1_dma1+dma4_offs) = tmp4.y;
	*(ptr_z_i1_dma1+dma4_offs) = tmp4.z;
	*(ptr_w_i1_dma1+dma4_offs) = tmp4.w;

	tmp1_ptr += offset;
	tmp2_ptr += offset;
	tmp3_ptr += offset;
	tmp4_ptr += offset;

	// Consumer non-blocking sync signalling sdata_i1 being full:
	ptx_barrier_nonblocking(3,num_barriers);


      }

      // Post-amble:
      ptx_barrier_blocking(2,num_barriers);

    } else {

      unsigned int dma_tid = tid - NUM_ELEMENTS - num_dma_hb_threads;
      
      float * ptr_x_i0_dma1 = &sdata_x_i0[dma_tid];
      float * ptr_y_i0_dma1 = &sdata_y_i0[dma_tid];
      float * ptr_z_i0_dma1 = &sdata_z_i0[dma_tid];
      float * ptr_w_i0_dma1 = &sdata_w_i0[dma_tid];


      // Pre-amble:
      // Producer non-blocking sync signalling sdata_i0 being full:
      ptx_barrier_nonblocking(0,num_barriers);

      unsigned int idx = NUM_ELEMENTS + dma_tid;

      float4 * tmp1_ptr = &g_idata[idx];
      float4 * tmp2_ptr = &g_idata[idx + dma2_offs];
      float4 * tmp3_ptr = &g_idata[idx + dma3_offs];
      float4 * tmp4_ptr = &g_idata[idx + dma4_offs];

      for (unsigned int j = 1; j < NUM_ITERS; j+=2) {

	// PHASE 1 - global load to registers
  	volatile float4 tmp1 = *tmp1_ptr;
  	volatile float4 tmp2 = *tmp2_ptr;
  	volatile float4 tmp3 = *tmp3_ptr;
  	volatile float4 tmp4 = *tmp4_ptr;



	// PHASE 2 - sync with compute warps, blast st.shared's from RF, sync with compute threads
	// Producer blocking sync waiting on sdata_i0 being empty:
	ptx_barrier_blocking(1,num_barriers);

	*ptr_x_i0_dma1 = tmp1.x;
	*ptr_y_i0_dma1 = tmp1.y;
	*ptr_z_i0_dma1 = tmp1.z;
	*ptr_w_i0_dma1 = tmp1.w;
	*(ptr_x_i0_dma1+dma2_offs) = tmp2.x;
	*(ptr_y_i0_dma1+dma2_offs) = tmp2.y;
	*(ptr_z_i0_dma1+dma2_offs) = tmp2.z;
	*(ptr_w_i0_dma1+dma2_offs) = tmp2.w;
	*(ptr_x_i0_dma1+dma3_offs) = tmp3.x;
	*(ptr_y_i0_dma1+dma3_offs) = tmp3.y;
	*(ptr_z_i0_dma1+dma3_offs) = tmp3.z;
	*(ptr_w_i0_dma1+dma3_offs) = tmp3.w;
	*(ptr_x_i0_dma1+dma4_offs) = tmp4.x;
	*(ptr_y_i0_dma1+dma4_offs) = tmp4.y;
	*(ptr_z_i0_dma1+dma4_offs) = tmp4.z;
	*(ptr_w_i0_dma1+dma4_offs) = tmp4.w;

	tmp1_ptr += offset;
	tmp2_ptr += offset;
	tmp3_ptr += offset;
	tmp4_ptr += offset;

	// Producer non-blocking sync signalling sdata_i0 being full:
	ptx_barrier_nonblocking(0,num_barriers);

      }

    }

  }

}


#endif // #ifndef _SW_DMA_KERNEL_H_

