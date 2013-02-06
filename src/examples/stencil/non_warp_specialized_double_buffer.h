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

#ifndef __NON_WARP_SPECIALIZED_DOUBLE_BUFFER__
#define __NON_WARP_SPECIALIZED_DOUBLE_BUFFER__

#include "../../../include/cudaDMAv2.h"

// The stencil function to be performed
#include "stencil_math.h"

template<int ALIGNMENT, int BYTES_PER_THREAD, int TILE_X, int TILE_Y, int RADIUS>
__global__
void stencil_2D_non_warp_specialized_double_buffer(const float2 *src_buffer, float2 *dst_buffer, const int offset,
                                                   const int row_stride, const int slice_stride, const int z_steps)
{
  // Need two barriers to avoid extra synchronization
  // If you're willing to add extra __synchtreads calls you can get away with one barrier
  // See below for where to put in extra __synchtreads calls
  __shared__ float2 buffer0[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];
  __shared__ float2 buffer1[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];

  // Two CudaDMA objects to get twice as many loads in flight at a time
  CudaDMAStrided<false/*non warp specialized*/, ALIGNMENT, BYTES_PER_THREAD,
                 (TILE_X+2*RADIUS)*sizeof(float2)/*elmt size*/,
                 (TILE_X*TILE_Y)/*num available threads*/, (TILE_Y+2*RADIUS)/*num elements*/>
    dma_ld0(row_stride*sizeof(float2)/*src stride*/, (TILE_X+2*RADIUS)*sizeof(float2)/*dst stride*/);
  CudaDMAStrided<false/*non warp specialized*/, ALIGNMENT, BYTES_PER_THREAD,
                 (TILE_X+2*RADIUS)*sizeof(float2)/*elmt size*/,
                 (TILE_X*TILE_Y)/*num available threads*/, (TILE_Y+2*RADIUS)/*num elements*/>
    dma_ld1(row_stride*sizeof(float2)/*src stride*/, (TILE_X+2*RADIUS)*sizeof(float2)/*dst stride*/);

  const unsigned block_offset = offset + blockIdx.x*TILE_X + blockIdx.y*TILE_Y*row_stride;
  const float2 *src_ptr = src_buffer + block_offset - (RADIUS*row_stride + RADIUS);
  const unsigned tx = threadIdx.x % TILE_X;
  const unsigned ty = threadIdx.x / TILE_X;
  float2 *dst_ptr = dst_buffer + block_offset + ty*row_stride + tx;

  // Launch the first set of LDGs
  dma_ld0.start_xfer_async<true>(src_ptr);
  src_ptr += slice_stride;
  // Launch the second set of LDGs
  dma_ld1.start_xfer_async<true>(src_ptr);
  src_ptr += slice_stride;
  for (int iz = 0; iz < (z_steps-2); iz+=2)
  {
    // Wait for the first tranfer to finish
    dma_ld0.wait_xfer_finish<true>(buffer0); // texture barrier
    dma_ld0.start_xfer_async<true>(src_ptr); // launch more LDGs
    src_ptr += slice_stride;
    // Wait for buffer to be full
    __syncthreads();
    // Consume the buffer
    perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer0, tx, ty, dst_ptr);
    // Need extra synchronization here if you only want to use one shared memory buffer
    dst_ptr += slice_stride;

    // Now do the second iteration
    dma_ld1.wait_xfer_finish<true>(buffer1); // texture barrier
    dma_ld1.start_xfer_async<true>(src_ptr);
    src_ptr += slice_stride;
    // Wait for the barrier to be full
    __syncthreads();
    perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer1, tx, ty, dst_ptr);
    // Need extra synchronization here if you only want to use one shared memory buffer
    dst_ptr += slice_stride;
  }
  // Loop cleanup
  dma_ld0.wait_xfer_finish<true>(buffer0);
  __syncthreads();
  perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer0, tx, ty, dst_ptr);
  // Need extra synchronization here if you only want to use one shared memory buffer
  dst_ptr += slice_stride;
  dma_ld1.wait_xfer_finish<true>(buffer1);
  __syncthreads();
  perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer1, tx, ty, dst_ptr);
}

#endif // __NON_WARP_SPECIALIZED_DOUBLE_BUFFER__

