/*
 *  Copyright 2013 NVIDIA Corporation
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

#ifndef __WARP_SPECIALIZED_DOUBLE_BUFFER__
#define __WARP_SPECIALIZED_DOUBLE_BUFFER__

#include "../../../include/cudaDMAv2.h"

// The stencil function to be performed
#include "stencil_math.h"

template<int ALIGNMENT, int BYTES_PER_THREAD, int TILE_X, int TILE_Y, int RADIUS, int DMA_THREADS_PER_LD>
__global__
void stencil_2D_warp_specialized_double_buffer(const float2 *src_buffer, float2 *dst_buffer, const int offset,
                                               const int row_stride, const int slice_stride, const int z_steps)
{
  // Two buffers
  __shared__ float2 buffer0[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];
  __shared__ float2 buffer1[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];

  // Two DMA instances
  CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,
                (TILE_X+2*RADIUS)*sizeof(float2), DMA_THREADS_PER_LD, (TILE_Y+2*RADIUS)>
    dma_ld0(0, (TILE_X*TILE_Y), (TILE_X*TILE_Y), 
            row_stride*sizeof(float2), (TILE_X+2*RADIUS)*sizeof(float2));
  CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,
                (TILE_X+2*RADIUS)*sizeof(float2), DMA_THREADS_PER_LD, (TILE_Y+2*RADIUS)>
    dma_ld1(1, (TILE_X*TILE_Y), (TILE_X*TILE_Y) + DMA_THREADS_PER_LD/*extra offset*/, 
            row_stride*sizeof(float2), (TILE_X+2*RADIUS)*sizeof(float2));

  const unsigned block_offset = offset + blockIdx.x*TILE_X + blockIdx.y*TILE_Y*row_stride;

  if (dma_ld0.owns_this_thread())
  {
    // 1st set of DMA warps
    const float2 *src_ptr = src_buffer + block_offset - (RADIUS*row_stride + RADIUS);
    // Do every other iteration
    for (int iz = 0; iz < z_steps; iz+=2)
    {
#if __CUDA_ARCH__ >= 350
      dma_ld0.execute_dma<true>(src_ptr, buffer0);
#else
      dma_ld0.execute_dma(src_ptr, buffer0);
#endif
      src_ptr += (2*slice_stride);
    }
  }
  else if (dma_ld1.owns_this_thread())
  {
    // 2nd set of DMA warps
    const float2 *src_ptr = src_buffer + block_offset + slice_stride - (RADIUS*row_stride + RADIUS);
    // Do every other iteration
    for (int iz = 0; iz < z_steps; iz+=2)
    {
#if __CUDA_ARCH__ >= 350
      dma_ld1.execute_dma<true>(src_ptr, buffer1);
#else
      dma_ld1.execute_dma(src_ptr, buffer1);
#endif
      src_ptr += (2*slice_stride);
    }
  }
  else
  {
    // Compute warps
    dma_ld0.start_async_dma();
    dma_ld1.start_async_dma();
    const unsigned tx = threadIdx.x % TILE_X;
    const unsigned ty = threadIdx.x / TILE_X;

    float2 *dst_ptr = dst_buffer + block_offset + ty*row_stride + tx;
    // Unroll the loop one time to handle the two sets of synchronization
    for (int iz = 0; iz < (z_steps-2); iz+=2)
    {
      // First iteration
      dma_ld0.wait_for_dma_finish();
      perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer0, tx, ty, dst_ptr);
      dma_ld0.start_async_dma();
      dst_ptr += slice_stride;
      // Second iteration
      dma_ld1.wait_for_dma_finish();
      perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer1, tx, ty, dst_ptr);
      dma_ld1.start_async_dma();
      dst_ptr += slice_stride;
    }
    // Handle the last two iterations
    dma_ld0.wait_for_dma_finish();
    perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer0, tx, ty, dst_ptr);
    dst_ptr += slice_stride;
    dma_ld1.wait_for_dma_finish();
    perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer1, tx, ty, dst_ptr);
  }
}

#endif // __WARP_SPECIALIZED_DOUBLE_BUFFER__

