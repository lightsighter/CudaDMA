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

#ifndef __WARP_SPECIALIZED_SINGLE_BUFFER__
#define __WARP_SPECIALIZED_SINGLE_BUFFER__

#include "../../../include/cudaDMAv2.h"

// The stencil function to be performed
#include "stencil_math.h"

template<int ALIGNMENT, int BYTES_PER_THREAD, int TILE_X, int TILE_Y, int RADIUS, int DMA_THREADS_PER_LD>
__global__
void stencil_2D_warp_specialized_single_buffer(const float2 *src_buffer, float2 *dst_buffer, const int offset,
                                               const int row_stride, const int slice_stride, const int z_steps)
{
  // Declare our shared memory buffer, base tile + halo region 
  __shared__ float2 buffer[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];

  // CudaDMAStrided instance for managing this buffer
  // In this case a strided element is a row in our buffer
  CudaDMAStrided<true/*warp specialized*/, ALIGNMENT, BYTES_PER_THREAD,
                  (TILE_X+2*RADIUS)*sizeof(float2)/*elmt size*/, 
                  DMA_THREADS_PER_LD, (TILE_Y+2*RADIUS)/*num elements*/>
    dma_ld(0/*dmaID*/, (TILE_X*TILE_Y)/*num compute threads*/, (TILE_X*TILE_Y)/*dma_threadIdx_start*/, 
            row_stride*sizeof(float2)/*src stride*/, (TILE_X+2*RADIUS)*sizeof(float2)/*dst stride*/);

  // Offset for this threadblock
  const unsigned block_offset = offset + blockIdx.x*TILE_X + blockIdx.y*TILE_Y*row_stride;

  if (dma_ld.owns_this_thread())
  {
    // DMA warps
    // Don't forget to backup to the upper left corner of the slice including the halo region
    const float2 *src_ptr = src_buffer + block_offset - (RADIUS*row_stride + RADIUS);
    for (int iz = 0; iz < z_steps; iz++)
    {
#if __CUDA_ARCH__ >= 350
      // Say that we are loading from global memory so we get LDG loads on K20
      dma_ld.execute_dma<true>(src_ptr, buffer);
#else
      dma_ld.execute_dma(src_ptr, buffer);
#endif
      src_ptr += slice_stride;
    }
  }
  else
  {
    // Compute warps
    // Start the DMA threads going
    dma_ld.start_async_dma();
    const unsigned tx = threadIdx.x % TILE_X;
    const unsigned ty = threadIdx.x / TILE_X;

    float2 *dst_ptr = dst_buffer + block_offset + ty*row_stride + tx;
    // Note we unroll the loop one time to facilitate
    // the right number of synchronizations with the DMA threads
    for (int iz = 0; iz < z_steps-1; iz++)
    {
      // Wait for the DMA threads to finish loading
      dma_ld.wait_for_dma_finish();
      // Process the buffer;
      perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer, tx, ty, dst_ptr);
      // Start the next transfer
      dma_ld.start_async_dma();
      // Update our out index
      dst_ptr += slice_stride;
    }
    // Handle the last loop iteration
    dma_ld.wait_for_dma_finish();
    perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer, tx, ty, dst_ptr);
  }
}

#endif // __WARP_SPECIALIZED_SINGLE_BUFFER__

