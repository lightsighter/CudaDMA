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
   Test kernel code for efficient DMAs
 */

#pragma once

#include <stdio.h>
#include "../../include/cudaDMA.h"

#define NUM_ITERS 1000
#define F4S_IN_SHMEM 256
#define NUM_ELEMENTS F4S_IN_SHMEM
#define NUM_DMA_THREADS 32
//#define DEBUG_PRINT 1

__global__ void
dma_4ld_strided( float4* g_idata, float4* g_odata, clock_t * timer_vals, const unsigned int el_cnt, const unsigned int el_size, const unsigned int el_stride ) 
{
//  cudaDMAStrided<128> dma0 (1, NUM_DMA_THREADS, NUM_ELEMENTS,
//	  NUM_ELEMENTS, el_size, el_cnt, el_stride, el_size);
//  cudaDMAStrided<128> dma1 (2, NUM_DMA_THREADS, NUM_ELEMENTS,
//	  NUM_ELEMENTS+NUM_DMA_THREADS, el_size, el_cnt, el_stride, el_size);
  cudaDMAStridedSmallElements dma0 (1, NUM_DMA_THREADS, NUM_ELEMENTS,
        NUM_ELEMENTS, el_size, el_cnt, el_stride, el_size);
  cudaDMAStridedSmallElements dma1 (2, NUM_DMA_THREADS, NUM_ELEMENTS,
        NUM_ELEMENTS+NUM_DMA_THREADS, el_size, el_cnt,
        el_stride, el_size);


  // shared memory
  __shared__  float4 sdata_i0[F4S_IN_SHMEM];
  __shared__  float4 sdata_i1[F4S_IN_SHMEM];
  // access thread id
  const unsigned int tid = threadIdx.x ;
//  if(tid == 0) printf("%u %u %u\n", el_size, el_cnt, el_stride);

  // Preamble
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (tid<F4S_IN_SHMEM) {
    sdata_i0[tid] = acc;
    sdata_i1[tid] = acc;
  }
  __syncthreads();


  if (tid<F4S_IN_SHMEM) {

    // Pre-amble:
    dma1.start_async_dma();
    dma0.start_async_dma();
    dma1.wait_for_dma_finish();
    float4 tmp1 = sdata_i1[tid];
    dma1.start_async_dma();
    acc.x += tmp1.x;
    acc.y += tmp1.y;
    acc.z += tmp1.z;
    acc.w += tmp1.w;

    for (unsigned int i = 0; i < NUM_ITERS-2; i+=2) {

      // Phase 1:
      dma0.wait_for_dma_finish();
      float4 tmp0 = sdata_i0[tid];
      dma0.start_async_dma();
      acc.x += tmp0.x;
      acc.y += tmp0.y;
      acc.z += tmp0.z;
      acc.w += tmp0.w;

      // Phase 2:
      dma1.wait_for_dma_finish();
      float4 tmp1 = sdata_i1[tid];
      dma1.start_async_dma();
      acc.x += tmp1.x;
      acc.y += tmp1.y;
      acc.z += tmp1.z;
      acc.w += tmp1.w;

    }

    // Postamble:
    dma0.wait_for_dma_finish();
    float4 tmp = sdata_i0[tid];
    acc.x += tmp.x;
    acc.y += tmp.y;
    acc.z += tmp.z;
    acc.w += tmp.w;
    g_odata[tid] = acc;

/*
    if(tid == 0) {
      printf("\n");
      for(unsigned int j = 0; j < NUM_ELEMENTS; j++)
        printf("%1.0f", sdata_i0[j].x);
      printf("\n");
      for(unsigned int j = 0; j < NUM_ELEMENTS; j++)
        printf("%1.0f", sdata_i1[j].x);
      printf("\n\n");
    }
*/

  } else if (dma0.owns_this_thread()) {
    for (unsigned int j = 0; j < NUM_ITERS; j+=2) {
      float4 * base_ptr = &g_idata[0];//[j*el_stride*el_cnt];
      dma0.execute_dma( base_ptr, &sdata_i0[0]);
    }
  } else if (dma1.owns_this_thread()) {
    for (unsigned int j = 1; j < NUM_ITERS; j+=2) {
      float4 * base_ptr = &g_idata[0];//[j*el_stride*el_cnt];
      dma1.execute_dma( base_ptr, &sdata_i1[0]);
    }
  }
}

