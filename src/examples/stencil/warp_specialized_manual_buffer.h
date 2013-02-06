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

#ifndef __WARP_SPECIALIZED_MANUAL_BUFFER__
#define __WARP_SPECIALIZED_MANUAL_BUFFER__

#include "../../../include/cudaDMAv2.h"

// The stencil function to be performed
#include "stencil_math.h"

template<int ALIGNMENT, int BYTES_PER_THREAD, int TILE_X, int TILE_Y, int RADIUS, int DMA_THREADS_PER_LD>
__global__
void stencil_2D_warp_specialized_manual_buffer(const float2 *src_buffer, float2 *dst_buffer, const int offset,
                                               const int row_stride, const int slice_stride, const int z_steps)
{
  // Two buffers
  __shared__ float2 buffer0[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];
  __shared__ float2 buffer1[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];

  // Two DMA instances that share the same set of DMA threads
  CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,
                (TILE_X+2*RADIUS)*sizeof(float2), DMA_THREADS_PER_LD, (TILE_Y+2*RADIUS)>
    dma_ld0(0, (TILE_X*TILE_Y), (TILE_X*TILE_Y), 
            row_stride*sizeof(float2), (TILE_X+2*RADIUS)*sizeof(float2));
  CudaDMAStrided<true,ALIGNMENT,BYTES_PER_THREAD,
                (TILE_X+2*RADIUS)*sizeof(float2), DMA_THREADS_PER_LD, (TILE_Y+2*RADIUS)>
    dma_ld1(1, (TILE_X*TILE_Y), (TILE_X*TILE_Y), 
            row_stride*sizeof(float2), (TILE_X+2*RADIUS)*sizeof(float2));

  const unsigned block_offset = offset + blockIdx.x*TILE_X + blockIdx.y*TILE_Y*row_stride;

  if (dma_ld0.owns_this_thread())
  {
    // DMA threads
    const float2 *src_ptr = src_buffer + block_offset - (RADIUS*row_stride + RADIUS);
    // Two versions here, one for Kepler that uses the two-phase API to 
    // gut multiple sets of LDG loads in flight at a time, and the simpler
    // approach for Fermi.  Note that either version will run correctly on
    // the other architecture.  This is about performance.
#if __CUDA_ARCH__ >= 350
    // Kepler Version
    dma_ld0.start_xfer_async<true>(src_ptr); // launch first set of LDGs
    src_ptr += slice_stride;
    dma_ld1.start_xfer_async<true>(src_ptr); // launch second set of LDGs
    src_ptr += slice_stride;
    for (int iz = 0; iz < (z_steps-2); iz+=2)
    {
      dma_ld0.wait_xfer_finish(buffer0); // texture barrier for first set of LDGs 
      dma_ld0.start_xfer_async<true>(src_ptr); // launch more LDGs
      src_ptr += slice_stride;
      dma_ld1.wait_xfer_finish(buffer1); // texture barrier for second set of LDGs
      dma_ld1.start_xfer_async<true>(src_ptr); // launch more LDGs
      src_ptr += slice_stride;
    }
    // Wait for last loop iteration
    dma_ld0.wait_xfer_finish(buffer0);
    dma_ld1.wait_xfer_finish(buffer1);
#else
    // Fermi Version
    for (int iz = 0; iz < z_steps; iz+=2)
    {
      dma_ld0.execute_dma(src_ptr, buffer0);
      src_ptr += slice_stride;
      dma_ld1.execute_dma(src_ptr, buffer1);
      src_ptr += slice_stride;
    }
#endif
  }
  else
  {
    // Compute threads
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

#endif // __WARP_SPECIALIZED_MANUAL_BUFFER__

