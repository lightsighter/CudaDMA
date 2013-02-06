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

#ifndef __NON_WARP_SPECIALIZED_SINGLE_BUFFER__
#define __NON_WARP_SPECIALIZED_SINGLE_BUFFER__

#include "../../../include/cudaDMAv2.h"

// The stencil function to be performed
#include "stencil_math.h"

template<int ALIGNMENT, int BYTES_PER_THREAD, int TILE_X, int TILE_Y, int RADIUS>
__global__
void stencil_2D_non_warp_specialized_single_buffer(const float2 *src_buffer, float2 *dst_buffer, const int offset,
                                                   const int row_stride, const int slice_stride, const int z_steps)
{
  // Declare our shared memory buffer, base tile + halo region 
  __shared__ float2 buffer[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)];

  // Allocate a non-warp specialized instance
  CudaDMAStrided<false/*non warp specialized*/, ALIGNMENT, BYTES_PER_THREAD,
                 (TILE_X+2*RADIUS)*sizeof(float2)/*elmt size*/,
                 (TILE_X*TILE_Y)/*num available threads*/, (TILE_Y+2*RADIUS)/*num elements*/>
    dma_ld(row_stride*sizeof(float2)/*src stride*/, (TILE_X+2*RADIUS)*sizeof(float2)/*dst stride*/);

  const unsigned block_offset = offset + blockIdx.x*TILE_X + blockIdx.y*TILE_Y*row_stride;
  const float2 *src_ptr = src_buffer + block_offset - (RADIUS*row_stride + RADIUS);
  const unsigned tx = threadIdx.x % TILE_X;
  const unsigned ty = threadIdx.x / TILE_X;
  float2 *dst_ptr = dst_buffer + block_offset + ty*row_stride + tx;

  for (int iz = 0; iz < z_steps; iz++)
  {
    // Perform the transfer
#if __CUDA_ARCH__ >= 350
    dma_ld.execute_dma<true>(src_ptr, buffer); // Make sure we get LDGs
#else
    dma_ld.execute_dma(src_ptr, buffer);
#endif
    src_ptr += slice_stride;
    // Wait for the transfer to finish
    __syncthreads();
    // Do the stencil operation
    perform_stencil<TILE_X,TILE_Y,RADIUS>(buffer, tx, ty, dst_ptr);
    dst_ptr += slice_stride;
  }
}

#endif // __NON_WARP_SPECIALIZED_SINGLE_BUFFER__

