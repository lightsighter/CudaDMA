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

/* 
 * SAXPY test code...used for both functionality and performance testing
 * 
 * Kernel descriptions:
 *   saxpy_baseline: 
 *       Simplest possible SAXPY code (should achieve peak performance with max threads)
 *   saxpy_float4s:  
 *       Uses LD.E.128 instead of LD.E.32 to achieve better MLP at low thread count
 *   saxpy_float4s_unroll2:
 *       Same as saxpy_float4s but unrolled twice to to try to maximize MLP/thread
 *       by having four in-flight LD.E.128's per thread.
 *   saxpy_shmem: (still unimplemented)
 *       Stage input data through shmem using __syncthreads, then process results
 *   saxpy_shmem_doublebuffer:
 *       Same as saxpy_shmem except with manual double buffering
 *   saxpy_cudaDMA:
 *       Uses cudaDMA library for staging data in shmem with separate DMA threads
 *   saxpy_cudaDMA_doublebuffer:
 *       Same as saxpy_cudaDMA except with manual double buffering
 */

#pragma once



#include <stdio.h>
//#define CUDADMA_DEBUG_ON 1
#include "../../include/cudaDMA.h"
#include "params.h"
#include <stdlib.h>

/*
 * This baseline version of saxpy is to demonstrate the CUDA C Best Practices version of writing saxpy.  
 * One element per thread with perfectly coalesced accesses.
 */
__global__ void saxpy_baseline ( float* y, float* x, float a, clock_t * timer_vals) 
{
  for (int i=0; i < NUM_ITERS; i++) {
    unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * x[idx] + y[idx];
  }
}

/*
 * This next version of saxpy is to achieved max mem BW with the least # of threads with 
 * the use of float4's.
 */

__global__ void saxpy_float4s ( float* y, float* x, float a, clock_t * timer_vals) 
{
  for (int i=0; i < NUM_ITERS/4; i++) {
    unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;

    float4 * x_as_float4 = (float4 *)x;
    float4 * y_as_float4 = (float4 *)y;

    float4 tmp1_x, tmp1_y;
    tmp1_x = x_as_float4[idx];
    tmp1_y = y_as_float4[idx];

    float4 result_y;
    result_y.x = a * tmp1_x.x + tmp1_y.x;
    result_y.y = a * tmp1_x.y + tmp1_y.y;
    result_y.z = a * tmp1_x.z + tmp1_y.z;
    result_y.w = a * tmp1_x.w + tmp1_y.w;
    y_as_float4[idx] = result_y;
  }
}

/*
 * This version of saxpy stages input data through shmem using __syncthreads, then process results.
 *    Requires 2 CTAs/SM for double buffering and optimal mem BW.
 */
__global__ void saxpy_shmem ( float* y, float* x, float a, clock_t * timer_vals) 
{
  volatile __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
  int tid = threadIdx.x ;
  for (int i=0; i < NUM_ITERS; ++i) {
    unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
    __syncthreads();
    sdata_x0[tid] = x[idx];
    sdata_y0[tid] = y[idx];
    __syncthreads();
    y[idx] = a * sdata_x0[tid] + sdata_y0[tid];
  }
}

/*
 * This version of saxpy stages input data through shmem using __syncthreads, then process results.
 *    This is the same as the above implementation but includes manual double buffering.
 */
__global__ void saxpy_shmem_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals) 
{
  volatile __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y1 [COMPUTE_THREADS_PER_CTA];
  int tid = threadIdx.x ;
  unsigned int idx0, idx1;
  idx0 = blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
  idx1 = COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + tid; 
  for (int i=0; i < NUM_ITERS; i+=2) {
    __syncthreads();
    sdata_x0[tid] = x[idx0];
    sdata_y0[tid] = y[idx0];
    if (i!=0) {
      y[idx1] = a * sdata_x1[tid] + sdata_y1[tid];
      idx1 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
    }
    __syncthreads();
    sdata_x1[tid] = x[idx1];
    sdata_y1[tid] = y[idx1];
    y[idx0] = a * sdata_x0[tid] + sdata_y0[tid];
    idx0 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
  }
  __syncthreads();
  y[idx1] = a * sdata_x1[tid] + sdata_y1[tid];
}


/*
 * This version of saxpy stages input data through shmem using __syncthreads, then process results.
 *    Requires 2 CTAs/SM for double buffering and optimal mem BW.
 */
__global__ void saxpy_float4s_shmem ( float* y, float* x, float a, clock_t * timer_vals) 
{
  volatile __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x2 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x3 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y2 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y3 [COMPUTE_THREADS_PER_CTA];
  int tid = threadIdx.x ;

  for (int i=0; i < NUM_ITERS/4; i++) {
    unsigned int idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;

    __syncthreads();
    float4 * x_as_float4 = (float4 *)x;
    float4 * y_as_float4 = (float4 *)y;
    float4 tmp1_x, tmp1_y;
    tmp1_x = x_as_float4[idx];
    tmp1_y = y_as_float4[idx];
    sdata_x0[tid] = tmp1_x.x;
    sdata_x1[tid] = tmp1_x.y;
    sdata_x2[tid] = tmp1_x.z;
    sdata_x3[tid] = tmp1_x.w;
    sdata_y0[tid] = tmp1_y.x;
    sdata_y1[tid] = tmp1_y.y;
    sdata_y2[tid] = tmp1_y.z;
    sdata_y3[tid] = tmp1_y.w;
    __syncthreads();

    float4 result_y;
    result_y.x = a * sdata_x0[tid] + sdata_y0[tid];
    result_y.y = a * sdata_x1[tid] + sdata_y1[tid];
    result_y.z = a * sdata_x2[tid] + sdata_y2[tid];
    result_y.w = a * sdata_x3[tid] + sdata_y3[tid];
    y_as_float4[idx] = result_y;
  }

}


/*
 * This version of saxpy stages input data through shmem using __syncthreads, then process results.
 *    Requires 2 CTAs/SM for double buffering and optimal mem BW.
 */
__global__ void saxpy_float4s_shmem_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals) 
{
  volatile __shared__ float sdata_x0_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x1_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x2_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x3_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y0_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y1_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y2_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y3_0 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x0_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x1_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x2_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_x3_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y0_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y1_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y2_1 [COMPUTE_THREADS_PER_CTA];
  volatile __shared__ float sdata_y3_1 [COMPUTE_THREADS_PER_CTA];
  int tid = threadIdx.x ;

  unsigned int idx0, idx1;
  idx0 = blockIdx.x * COMPUTE_THREADS_PER_CTA + tid;
  idx1 = COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + tid; 

  float4 * x_as_float4 = (float4 *)x;
  float4 * y_as_float4 = (float4 *)y;
  float4 result_y;

  for (int i=0; i < NUM_ITERS/4; i+=2) {
    float4 tmp1_x, tmp1_y;

    __syncthreads();
    tmp1_x = x_as_float4[idx0];
    tmp1_y = y_as_float4[idx0];
    if (i!=0) {
      result_y.x = a * sdata_x0_1[tid] + sdata_y0_1[tid];
      result_y.y = a * sdata_x1_1[tid] + sdata_y1_1[tid];
      result_y.z = a * sdata_x2_1[tid] + sdata_y2_1[tid];
      result_y.w = a * sdata_x3_1[tid] + sdata_y3_1[tid];
      y_as_float4[idx1] = result_y;
      idx1 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
    }
    sdata_x0_0[tid] = tmp1_x.x;
    sdata_x1_0[tid] = tmp1_x.y;
    sdata_x2_0[tid] = tmp1_x.z;
    sdata_x3_0[tid] = tmp1_x.w;
    sdata_y0_0[tid] = tmp1_y.x;
    sdata_y1_0[tid] = tmp1_y.y;
    sdata_y2_0[tid] = tmp1_y.z;
    sdata_y3_0[tid] = tmp1_y.w;
    __syncthreads();
    tmp1_x = x_as_float4[idx1];
    tmp1_y = y_as_float4[idx1];
    result_y.x = a * sdata_x0_0[tid] + sdata_y0_0[tid];
    result_y.y = a * sdata_x1_0[tid] + sdata_y1_0[tid];
    result_y.z = a * sdata_x2_0[tid] + sdata_y2_0[tid];
    result_y.w = a * sdata_x3_0[tid] + sdata_y3_0[tid];
    y_as_float4[idx0] = result_y;
    idx0 += 2 * COMPUTE_THREADS_PER_CTA * CTA_COUNT ;
    sdata_x0_1[tid] = tmp1_x.x;
    sdata_x1_1[tid] = tmp1_x.y;
    sdata_x2_1[tid] = tmp1_x.z;
    sdata_x3_1[tid] = tmp1_x.w;
    sdata_y0_1[tid] = tmp1_y.x;
    sdata_y1_1[tid] = tmp1_y.y;
    sdata_y2_1[tid] = tmp1_y.z;
    sdata_y3_1[tid] = tmp1_y.w;
  }
  __syncthreads();
  result_y.x = a * sdata_x0_1[tid] + sdata_y0_1[tid];
  result_y.y = a * sdata_x1_1[tid] + sdata_y1_1[tid];
  result_y.z = a * sdata_x2_1[tid] + sdata_y2_1[tid];
  result_y.w = a * sdata_x3_1[tid] + sdata_y3_1[tid];
  y_as_float4[idx1] = result_y;

}


/*
 * This version of saxpy uses cudaDMA for DMAs (but requires 2 CTAs/SM) for double buffering.
 */
__global__ void saxpy_cudaDMA ( float* y, float* x, float a, clock_t * timer_vals) 
{
  __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];

  cudaDMASequential<BYTES_PER_DMA_THREAD,16>
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA, DMA_SZ);
  cudaDMASequential<BYTES_PER_DMA_THREAD,16>
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD, DMA_SZ);

  int tid = threadIdx.x ;

  if ( tid < COMPUTE_THREADS_PER_CTA ) {
    unsigned int idx;
    int i;
    float tmp_x;
    float tmp_y;
    
    // Preamble:
    dma_ld_x_0.start_async_dma();
    dma_ld_y_0.start_async_dma();
    for (i = 0; i < NUM_ITERS-1; ++i) {
      dma_ld_x_0.wait_for_dma_finish();
      tmp_x = sdata_x0[tid];
      dma_ld_x_0.start_async_dma();
      dma_ld_y_0.wait_for_dma_finish();
      tmp_y = sdata_y0[tid];
      dma_ld_y_0.start_async_dma();
      idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
    // Postamble:
    dma_ld_x_0.wait_for_dma_finish();
    tmp_x = sdata_x0[tid];
    dma_ld_y_0.wait_for_dma_finish();
    tmp_y = sdata_y0[tid];
    idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;

  } else if (dma_ld_x_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; ++j) {
      // idx is a pointer to the base of the chunk of memory to copy
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_x_0.execute_dma( &x[idx], sdata_x0 );
    }
  } else if (dma_ld_y_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; ++j) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_y_0.execute_dma( &y[idx], sdata_y0 );
    }
  }
}



/*
 * This version of saxpy uses cudaDMA for DMAs with manual double buffering.
 */
__global__ void saxpy_cudaDMA_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals) 
{
  __shared__ float sdata_x0 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_x1 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_y0 [COMPUTE_THREADS_PER_CTA];
  __shared__ float sdata_y1 [COMPUTE_THREADS_PER_CTA];

  cudaDMASequential<BYTES_PER_DMA_THREAD,16>
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA);
  cudaDMASequential<BYTES_PER_DMA_THREAD,16>
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD);
  cudaDMASequential<BYTES_PER_DMA_THREAD,16>
    dma_ld_x_1 (3, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + 2*DMA_THREADS_PER_LD);
  cudaDMASequential<BYTES_PER_DMA_THREAD,16>
    dma_ld_y_1 (4, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + 3*DMA_THREADS_PER_LD);

  int tid = threadIdx.x ;

  if ( tid < COMPUTE_THREADS_PER_CTA ) {
    unsigned int idx;
    int i;
    float tmp_x;
    float tmp_y;
    
    // Preamble:
    dma_ld_x_0.start_async_dma();
    dma_ld_y_0.start_async_dma();
    dma_ld_x_1.start_async_dma();
    dma_ld_y_1.start_async_dma();
    for (i = 0; i < NUM_ITERS-2; i += 2) {
      
      // Phase 1:
      dma_ld_x_0.wait_for_dma_finish();
      tmp_x = sdata_x0[tid];
      dma_ld_x_0.start_async_dma();
      dma_ld_y_0.wait_for_dma_finish();
      tmp_y = sdata_y0[tid];
      dma_ld_y_0.start_async_dma();
      idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;

      // Phase 2:
      dma_ld_x_1.wait_for_dma_finish();
      tmp_x = sdata_x1[tid];
      dma_ld_x_1.start_async_dma();
      dma_ld_y_1.wait_for_dma_finish();
      tmp_y = sdata_y1[tid];
      dma_ld_y_1.start_async_dma();
      idx = (i+1) * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
      
    // Postamble
    dma_ld_x_0.wait_for_dma_finish();
    tmp_x = sdata_x0[tid];
    dma_ld_y_0.wait_for_dma_finish();
    tmp_y = sdata_y0[tid];
    idx = i * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;
    dma_ld_x_1.wait_for_dma_finish();
    tmp_x = sdata_x1[tid];
    dma_ld_y_1.wait_for_dma_finish();
    tmp_y = sdata_y1[tid];
    idx = (i+1) * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;

  } else if (dma_ld_x_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      // idx is a pointer to the base of the chunk of memory to copy
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_x_0.execute_dma( &x[idx], sdata_x0 );
    }
  } else if (dma_ld_y_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_y_0.execute_dma( &y[idx], sdata_y0 );
    }
  } else if (dma_ld_x_1.owns_this_thread()) {
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_x_1.execute_dma( &x[idx], sdata_x1 );
    }
  } else if (dma_ld_y_1.owns_this_thread()) {
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      unsigned int idx = j * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA;
      dma_ld_y_1.execute_dma( &y[idx], sdata_y1 );
    }
  }
  
}
