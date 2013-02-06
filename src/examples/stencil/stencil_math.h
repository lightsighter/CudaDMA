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

#ifndef __STENCIL_MATH__
#define __STENCIL_MATH__

// This is the stencil computation that is performed
// by all of the different variants of CudaDMA kernels.
//
// There are two variants, one for dense math and one
// that shows of CudaDMA better and is more memory bound
//#define DENSE_MATH

template<int TILE_X, int TILE_Y, int RADIUS>
__device__ __forceinline__
void perform_stencil(const float2 buffer[(TILE_Y+2*RADIUS)*(TILE_X+2*RADIUS)],
                     const unsigned tx, const unsigned ty,
                     float2 *dst_ptr)
{
  // Average all the points in X and Y dimensions within
  // the given RADIUS of our thread's point
#ifdef DENSE_MATH
  float2 accum;
  accum.x = 0.0f;
  accum.y = 0.0f;
  for (int dy = -RADIUS; dy <= RADIUS; dy++)
  {
    for (int dx = -RADIUS; dx <= RADIUS; dx++)
    {
      const float2 &value = buffer[(ty+RADIUS+dy)*(TILE_X+2*RADIUS) + (tx+RADIUS+dx)];
      accum.x += value.x;
      accum.y += value.y;
    }
  }
  // Average out the values
  accum.x /= ((2*RADIUS+1)*(2*RADIUS+1));
  accum.y /= ((2*RADIUS+1)*(2*RADIUS+1));
#else
  float2 accum = buffer[(ty+RADIUS)*(TILE_X+2*RADIUS) + (tx+RADIUS)];
  // 1 to RADIUS for +X, -X, +Y, -Y
  #pragma unroll
  for (int dx = 1; dx < RADIUS; dx++)
  {
    const float2 &value = buffer[(ty+RADIUS)*(TILE_X+2*RADIUS) + (tx+RADIUS-dx)];
    accum.x += value.x;
    accum.y += value.y;
  }
  #pragma unroll
  for (int dx = 1; dx < RADIUS; dx++)
  {
    const float2 &value = buffer[(ty+RADIUS)*(TILE_X+2*RADIUS) + (tx+RADIUS+dx)];
    accum.x += value.x;
    accum.y += value.y;
  }
  #pragma unroll
  for (int dy = 1; dy < RADIUS; dy++)
  {
    const float2 &value = buffer[(ty+RADIUS-dy)*(TILE_X+2*RADIUS) + (tx+RADIUS)];
    accum.x += value.x;
    accum.y += value.y;
  }
  #pragma unroll
  for (int dy = 1; dy < RADIUS; dy++)
  {
    const float2 &value = buffer[(ty+RADIUS+dy)*(TILE_X+2*RADIUS) + (tx+RADIUS)];
    accum.x += value.x;
    accum.y += value.y;
  }
  accum.x /= (4*RADIUS+1);
  accum.y /= (4*RADIUS+1);
#endif
  // Write the output value
  *dst_ptr = accum;
}

#endif // __STENCIL_MATH__

