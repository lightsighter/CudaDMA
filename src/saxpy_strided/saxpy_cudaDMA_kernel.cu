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
    //unsigned int idx = i%ALLOC_ITERS * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * x[idx] + y[idx];
  }
}

/*
 * This version of saxpy uses cudaDMA for DMAs (but requires 2 CTAs/SM) for double buffering.
 */
__global__ void saxpy_cudaDMA ( float* y, float* x, float a, clock_t * timer_vals) 
{
  __shared__ float sdata_x0 [DMA_SZ_IN_FS];
  __shared__ float sdata_y0 [DMA_SZ_IN_FS];

#ifdef USE_SMALL_EL_OPT
  cudaDMAStridedSmallElements
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA, EL_SZ, DMA_SZ/EL_SZ, EL_SZ, EL_SZ);
  cudaDMAStridedSmallElements
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ, EL_SZ);
#else
  cudaDMAStrided<BYTES_PER_DMA_THREAD>
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA, EL_SZ, DMA_SZ/EL_SZ, EL_SZ);
  cudaDMAStrided<BYTES_PER_DMA_THREAD>
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ);
#endif
  int tid = threadIdx.x ;

  if ( tid < COMPUTE_THREADS_PER_CTA) {
    unsigned int idx;
    int i;
    int k;
    float tmp_x;
    float tmp_y;
    
    // Preamble:
    dma_ld_x_0.start_async_dma();
    dma_ld_y_0.start_async_dma();
    for (i = 0; i < NUM_ITERS-1; ++i) {
      dma_ld_x_0.wait_for_dma_finish();
#ifdef DO_COMPUTE
      tmp_x = sdata_x0[tid];
#endif
      dma_ld_x_0.start_async_dma();
      dma_ld_y_0.wait_for_dma_finish();
#ifdef DO_COMPUTE
      tmp_y = sdata_y0[tid];
#endif
      dma_ld_y_0.start_async_dma();
#ifdef DO_COMPUTE
      idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
      //idx = i%ALLOC_ITERS * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
      for(k = 1; k < ITERS_PER_COMPUTE_THREAD && k*COMPUTE_THREADS_PER_CTA+tid < DMA_SZ_IN_FS; ++k) {
        tmp_x = sdata_x0[k*COMPUTE_THREADS_PER_CTA+tid];
        tmp_y = sdata_y0[k*COMPUTE_THREADS_PER_CTA+tid];
        idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
        //idx = i%ALLOC_ITERS * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
        y[idx] = a * tmp_x + tmp_y;
      }
#endif
    }
    // Postamble:
    dma_ld_x_0.wait_for_dma_finish();
    dma_ld_y_0.wait_for_dma_finish();
#ifdef DO_COMPUTE
    tmp_x = sdata_x0[tid];
    tmp_y = sdata_y0[tid];
    idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
    //idx = i%ALLOC_ITERS * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;
    for(k = 1; k < ITERS_PER_COMPUTE_THREAD && k*COMPUTE_THREADS_PER_CTA+tid < DMA_SZ_IN_FS; ++k) {
      tmp_x = sdata_x0[k*COMPUTE_THREADS_PER_CTA+tid];
      tmp_y = sdata_y0[k*COMPUTE_THREADS_PER_CTA+tid];
      idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
      //idx = i%ALLOC_ITERS * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
#endif
  } else if (dma_ld_x_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; ++j) {
      // idx is a pointer to the base of the chunk of memory to copy
      unsigned int idx = j * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      //unsigned int idx = j%ALLOC_ITERS * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      dma_ld_x_0.execute_dma( &x[idx], sdata_x0 );
    }
  } else if (dma_ld_y_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; ++j) {
      unsigned int idx = j * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      //unsigned int idx = j%ALLOC_ITERS * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      dma_ld_y_0.execute_dma( &y[idx], sdata_y0 );
    }
  }
}



/*
 * This version of saxpy uses cudaDMA for DMAs with manual double buffering.
 */
__global__ void saxpy_cudaDMA_doublebuffer ( float* y, float* x, float a, clock_t * timer_vals) 
{
  __shared__ float sdata_x0 [DMA_SZ_IN_FS/2];
  __shared__ float sdata_x1 [DMA_SZ_IN_FS/2];
  __shared__ float sdata_y0 [DMA_SZ_IN_FS/2];
  __shared__ float sdata_y1 [DMA_SZ_IN_FS/2];

#ifdef USE_SMALL_EL_OPT
  cudaDMAStridedSmallElements
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA, EL_SZ, DMA_SZ/EL_SZ, EL_SZ, EL_SZ);
  cudaDMAStridedSmallElements
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ, EL_SZ);
  cudaDMAStridedSmallElements
    dma_ld_x_1 (3, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + 2*DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ, EL_SZ);
  cudaDMAStridedSmallElements
    dma_ld_y_1 (4, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + 3*DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ, EL_SZ);
#else
  cudaDMAStrided<BYTES_PER_DMA_THREAD>
    dma_ld_x_0 (1, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA, EL_SZ, DMA_SZ/EL_SZ, EL_SZ);
  cudaDMAStrided<BYTES_PER_DMA_THREAD>
    dma_ld_y_0 (2, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ);
  cudaDMAStrided<BYTES_PER_DMA_THREAD>
    dma_ld_x_1 (3, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + 2*DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ);
  cudaDMAStrided<BYTES_PER_DMA_THREAD>
    dma_ld_y_1 (4, DMA_THREADS_PER_LD, COMPUTE_THREADS_PER_CTA,
		COMPUTE_THREADS_PER_CTA + 3*DMA_THREADS_PER_LD, EL_SZ, DMA_SZ/EL_SZ, EL_SZ);
#endif
  int tid = threadIdx.x ;

  if ( tid < COMPUTE_THREADS_PER_CTA ) {
    unsigned int idx;
    int i;
    int k;
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
#ifdef DO_COMPUTE
      tmp_x = sdata_x0[tid];
#endif
      dma_ld_x_0.start_async_dma();
      dma_ld_y_0.wait_for_dma_finish();
#ifdef DO_COMPUTE
      tmp_y = sdata_y0[tid];
#endif
      dma_ld_y_0.start_async_dma();
#ifdef DO_COMPUTE
      //idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
      idx = blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
      for(k = 1; k < ITERS_PER_COMPUTE_THREAD && k*COMPUTE_THREADS_PER_CTA+tid < DMA_SZ_IN_FS; ++k) {
        tmp_x = sdata_x0[k*COMPUTE_THREADS_PER_CTA+tid];
        tmp_y = sdata_y0[k*COMPUTE_THREADS_PER_CTA+tid];
        //idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
        idx = blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
        y[idx] = a * tmp_x + tmp_y;
      }
#endif

      // Phase 2:
      dma_ld_x_1.wait_for_dma_finish();
#ifdef DO_COMPUTE
      tmp_x = sdata_x1[tid];
#endif
      dma_ld_x_1.start_async_dma();
      dma_ld_y_1.wait_for_dma_finish();
#ifdef DO_COMPUTE
      tmp_y = sdata_y1[tid];
#endif
      dma_ld_y_1.start_async_dma();
#ifdef DO_COMPUTE
      //idx = (i+1) * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
      idx = blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
      for(k = 1; k < ITERS_PER_COMPUTE_THREAD && k*COMPUTE_THREADS_PER_CTA+tid < DMA_SZ_IN_FS; ++k) {
        tmp_x = sdata_x1[k*COMPUTE_THREADS_PER_CTA+tid];
        tmp_y = sdata_y1[k*COMPUTE_THREADS_PER_CTA+tid];
        //idx = (i+1) * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
        idx = blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
        y[idx] = a * tmp_x + tmp_y;
      }
#endif
    }
      
    // Postamble
    dma_ld_x_0.wait_for_dma_finish();
#ifdef DO_COMPUTE
    tmp_x = sdata_x0[tid];
#endif
    dma_ld_y_0.wait_for_dma_finish();
#ifdef DO_COMPUTE
    tmp_y = sdata_y0[tid];
    //idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
    idx = blockIdx.x * DMA_SZ_IN_FS + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;
    for(k = 1; k < ITERS_PER_COMPUTE_THREAD && k*COMPUTE_THREADS_PER_CTA+tid < DMA_SZ_IN_FS; ++k) {
      tmp_x = sdata_x0[k*COMPUTE_THREADS_PER_CTA+tid];
      tmp_y = sdata_y0[k*COMPUTE_THREADS_PER_CTA+tid];
      //idx = i * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
      idx = blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
#endif

    dma_ld_x_1.wait_for_dma_finish();
#ifdef DO_COMPUTE
    tmp_x = sdata_x1[tid];
#endif
    dma_ld_y_1.wait_for_dma_finish();
#ifdef DO_COMPUTE
    tmp_y = sdata_y1[tid];
    //idx = (i+1) * COMPUTE_THREADS_PER_CTA * CTA_COUNT + blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    idx = blockIdx.x * COMPUTE_THREADS_PER_CTA + threadIdx.x;
    y[idx] = a * tmp_x + tmp_y;
    for(k = 1; k < ITERS_PER_COMPUTE_THREAD && k*COMPUTE_THREADS_PER_CTA+tid < DMA_SZ_IN_FS; ++k) {
      tmp_x = sdata_x1[k*COMPUTE_THREADS_PER_CTA+tid];
      tmp_y = sdata_y1[k*COMPUTE_THREADS_PER_CTA+tid];
      //idx = (i+1) * CTA_COUNT * DMA_SZ_IN_FS + blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
      idx = blockIdx.x * DMA_SZ_IN_FS + k*COMPUTE_THREADS_PER_CTA + threadIdx.x;
      y[idx] = a * tmp_x + tmp_y;
    }
#endif

  } else if (dma_ld_x_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      // idx is a pointer to the base of the chunk of memory to copy
      //unsigned int idx = j * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      unsigned int idx = blockIdx.x * DMA_SZ_IN_FS;
      dma_ld_x_0.execute_dma( &x[idx], sdata_x0 );
    }
  } else if (dma_ld_y_0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      //unsigned int idx = j * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      unsigned int idx = blockIdx.x * DMA_SZ_IN_FS;
      dma_ld_y_0.execute_dma( &y[idx], sdata_y0 );
    }
  } else if (dma_ld_x_1.owns_this_thread()) {
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      //unsigned int idx = j * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      unsigned int idx = blockIdx.x * DMA_SZ_IN_FS;
      dma_ld_x_1.execute_dma( &x[idx], sdata_x1 );
    }
  } else if (dma_ld_y_1.owns_this_thread()) {
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      //unsigned int idx = j * DMA_SZ_IN_FS * CTA_COUNT + blockIdx.x * DMA_SZ_IN_FS;
      unsigned int idx = blockIdx.x * DMA_SZ_IN_FS;
      dma_ld_y_1.execute_dma( &y[idx], sdata_y1 );
    }
  }
  
}
