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

#pragma once

// For doing static assertions.  If you get a static assertion that
// means that there is something wrong with the way you instantiated
// a CudaDMA instance.
template<bool COND> struct CudaDMAStaticAssert;
template<> struct CudaDMAStaticAssert<true> { };
#define STATIC_ASSERT(condition) do { CudaDMAStaticAssert<(condition)>(); } while (0)

#define GUARD_ZERO(expr) (((expr) == 0) ? 1 : (expr))

// Enumeration types for load and store qualifiers.
// For more information on how these work, see the PTX manual.
enum CudaDMALoadQualifier {
  LOAD_CACHE_ALL, // cache at all levels
  LOAD_CACHE_GLOBAL, // cache only in L2
  LOAD_CACHE_STREAMING, // cache all levels, but mark evict first
  LOAD_CACHE_LAST_USE, // invalidates line after use
  LOAD_CACHE_VOLATILE, // don't cache at any level
};

enum CudaDMAStoreQualifier {
  STORE_WRITE_BACK,
  STORE_CACHE_GLOBAL,
  STORE_CACHE_STREAMING,
  STORE_CACHE_WRITE_THROUGH,
};

__device__ __forceinline__ 
void ptx_cudaDMA_barrier_blocking (const int name, const int num_barriers)
{
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(num_barriers) : "memory" );
}

__device__ __forceinline__ 
void ptx_cudaDMA_barrier_nonblocking (const int name, const int num_barriers)
{
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(num_barriers) : "memory" );
}

/*****************************************************/
/*           Load functions                          */
/*****************************************************/
template<typename T, bool GLOBAL_LOAD, int LOAD_QUAL>
__device__ __forceinline__
T ptx_cudaDMA_load(const T *src_ptr)
{
  T result;
  STATIC_ASSERT(false); // If you get here implement the right ptx version for your type
  return result;
}

template<bool GLOBAL_LOAD, int LOAD_QUAL>
__device__ __forceinline__
float ptx_cudaDMA_load<float,GLOBAL_LOAD,LOAD_QUAL>(const float *src_ptr)
{
  float result;
  // Handle LDG loads for GK110
#if __CUDA_ARCH__ == 350 
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        // LDG load
        asm volatile("ld.global.nc.ca.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_GLOBAL:
        // LDG load
        asm volatile("ld.global.nc.cg.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        // LDG load
        asm volatile("ld.global.nc.cs.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.global.lu,f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#else
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_GLOBAL:
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.global.lu,f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#endif
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        asm volatile("ld.ca.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_GLOBAL:
        asm volatile("ld.cg.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        asm volatile("ld.cs.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.lu,f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.cv.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  return result;
}

template<bool GLOBAL_LOAD, int LOAD_QUAL>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,GLOBAL_LOAD,LOAD_QUAL>(const float2 *src_ptr)
{
  float2 result;
  // Handle LDG loads for GK110
#if __CUDA_ARCH__ == 350
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        // LDG load
        asm volatile("ld.global.nc.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
        break;
      case LOAD_CACHE_GLOBAL:
        // LDG load
        asm volatile("ld.global.nc.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        // LDG load
        asm volatile("ld.global.nc.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }

  }
  else
#else
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        asm volatile("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
        break;
      case LOAD_CACHE_GLOBAL:
        asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        asm volatile("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#endif
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        asm volatile("ld.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
        break;
      case LOAD_CACHE_GLOBAL:
        asm volatile("ld.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        asm volatile("ld.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  return result;
}

template<bool GLOBAL_LOAD, int LOAD_QUAL>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,GLOBAL_LOAD,LOAD_QUAL>(const float3 *src_ptr)
{
  float3 result;
  // Handle LDG loads for GK110
#if __CUDA_ARCH__ == 350
  if (GLOBAL_LOAD)
  {
    // LDG loads
    switch (LOAD_QUAL)
    {
      case load_cache_all:
        // LDG loads
        asm volatile("ld.global.nc.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
        asm volatile("ld.global.nc.ca.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_global:
        // LDG loads
        asm volatile("ld.global.nc.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.nc.cg.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_streaming:
        // LDG loads
        asm volatile("ld.global.nc.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.nc.cs.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_last_use:
        asm volatile("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.lu.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_volatile:
        asm volatile("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.cv.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#else
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case load_cache_all:
        asm volatile("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
        asm volatile("ld.global.ca.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_global:
        asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.cg.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_streaming:
        asm volatile("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.cs.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_last_use:
        asm volatile("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.lu.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_volatile:
        asm volatile("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.global.cv.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#endif
  {
    switch (LOAD_QUAL)
    {
      case load_cache_all:
        asm volatile("ld.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
        asm volatile("ld.ca.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_global:
        asm volatile("ld.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.cg.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_streaming:
        asm volatile("ld.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.cs.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_last_use:
        asm volatile("ld.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.lu.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
      case load_cache_volatile:
        asm volatile("ld.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
        asm volatile("ld.cv.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  return result;
}

template<bool GLOBAL_LOAD, int LOAD_QUAL>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,GLOBAL_LOAD,LOAD_QUAL>(const float4 *src_ptr)
{
  float4 result;
  // Handle LDG loads for GK110
#if __CUDA_ARCH__ == 350
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        // LDG load
        asm volatile("ld.global.nc.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_GLOBAL:
        // LDG load
        asm volatile("ld.global.nc.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        // LDG load
        asm volatile("ld.global.nc.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#else
  if (GLOBAL_LOAD)
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_GLOBAL:
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  else
#endif
  {
    switch (LOAD_QUAL)
    {
      case LOAD_CACHE_ALL:
        asm volatile("ld.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_GLOBAL:
        asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_STREAMING:
        asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_LAST_USE:
        asm volatile("ld.lu.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
      case LOAD_CACHE_VOLATILE:
        asm volatile("ld.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
        break;
#ifdef DEBUG_CUDADMA
      default:
        assert(false);
#endif
    }
  }
  return result;
}

/*****************************************************/
/*           Store functions                         */
/*****************************************************/
template<typename T, int STORE_QUAL>
__device__ __forceinline__
void ptx_cudaDMA_store(const T &src_val, T *dst_ptr)
{
  STATIC_ASSERT(false); // If you get here implement the right ptx version for your type
}

template<int STORE_QUAL>
void ptx_cudaDMA_store<float,STORE_QUAL>(const float &src_val, float *dst_ptr)
{
  switch (STORE_QUAL)
  {
    case STORE_WRITE_BACK:
      asm volatile("st.wb.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
      break;
    case STORE_CACHE_GLOBAL:
      asm volatile("st.cg.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
      break;
    case STORE_CACHE_STREAMING:
      asm volatile("st.cs.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
      break;
    case STORE_CACHE_WRITE_THROUGH:
      asm volatile("st.wt.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
      break;
#ifdef DEBUG_CUDADMA
    default:
      assert(false);
#endif
  }
}

template<int STORE_QUAL>
__device__ __forceinline__
void ptx_cudaDMA_store<float2,STORE_QUAL>(const float2 &src_val, float2 *dst_ptr)
{
  switch (STORE_QUAL)
  {
    case STORE_WRITE_BACK:
      asm volatile("st.wb.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      break;
    case STORE_CACHE_GLOBAL:
      asm volatile("st.cg.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      break;
    case STORE_CACHE_STREAMING:
      asm volatile("st.cs.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      break;
    case STORE_CACHE_WRITE_THROUGH:
      asm volatile("st.wt.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      break;
#ifdef DEBUG_CUDADMA
    default:
      assert(false);
#endif
  }
}

template<int STORE_QUAL>
__device__ __forceinline__
void ptx_cudaDMA_store<float3,STORE_QUAL>(const float3 &src_val, float3 *dst_ptr)
{
  switch (STORE_QUAL)
  {
    case STORE_WRITE_BACK:
      asm volatile("st.wb.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      asm volatile("st.wb.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
      break;
    case STORE_CACHE_GLOBAL:
      asm volatile("st.cg.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      asm volatile("st.cg.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
      break;
    case STORE_CACHE_STREAMING:
      asm volatile("st.cs.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      asm volatile("st.cs.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
      break;
    case STORE_CACHE_WRITE_THROUGH:
      asm volatile("st.wt.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
      asm volatile("st.wt.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
      break;
#ifdef DEBUG_CUDADMA
    default:
      assert(false);
#endif
  }
}

template<int STORE_QUAL>
__device__ __forceinline__
void ptx_cudaDMA_store<float4,STORE_QUAL>(const float4 &src_val, float4 *dst_ptr)
{
  switch (STORE_QUAL)
  {
    case STORE_WRITE_BACK:
      asm volatile("st.wb.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
      break;
    case STORE_CACHE_GLOBAL:
      asm volatile("st.cg.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
      break;
    case STORE_CACHE_STREAMING:
      asm volatile("st.cs.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
      break;
    case STORE_CACHE_WRITE_THROUGH:
      asm volatile("st.wt.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
      break;
#ifdef DEBUG_CUDADMA
    default:
      assert(false);
#endif
  }
}

#define WARP_SIZE 32
#define WARP_MASK 0x1f
#define CUDADMA_DMA_TID (threadIdx.x-dma_threadIdx_start)

// Enable the restrict keyword to allow additional compiler optimizations
// Note that this can increase register pressure (see appendix B.2.4 of
// CUDA Programming Guide)
#define ENABLE_RESTRICT

#ifdef ENABLE_RESTRICT
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// Have a special namespace for our meta-programming objects
// so that we can guarantee that they don't interfere with any
// user level code.  
//
// Why metaprogramming?  I just don't trust the compiler.
namespace CudaDMAMeta {
  // A buffer for guaranteeing static access when loading and storing
  // ET = element type
  // NUM_ELMTS = number of elements in the static buffer
  template<typename ET, unsigned NUM_ELMTS>
  class DMABuffer {
  public:
    template<unsigned IDX>
    __device__ __forceinline__
    ET& get_ref(void) { STATIC_ASSERT(IDX < NUM_ELMTS); return buffer[IDX]; }
    template<unsigned IDX>
    __device__ __forceinline__
    const ET& get_ref(void) const { STATIC_ASSERT(IDX < NUM_ELMTS); return buffer[IDX]; }
    template<unsigned IDX>
    __device__ __forceinline__
    ET* get_ptr(void) { STATIC_ASSERT(IDX < NUM_ELMTS); return &buffer[IDX]; }
    template<unsigned IDX>
    __device__ __forceinline__
    const ET* get_ptr(void) const { STATIC_ASSERT(IDX < NUM_ELMTS); return &buffer[IDX]; }
  public:
    template<unsigned IDX, bool GLOBAL_LOAD = false, int LOAD_QUAL = LOAD_CACHE_ALL>
    __device__ __forceinline__
    void perform_load(const void *src_ptr)
    {
      STATIC_ASSERT(IDX < NUM_ELMTS);
      buffer[IDX] = ptx_cudaDMA_load<ET, GLOBAL_LOAD, LOAD_QUAL>((const ET*)src_ptr);
    }
    template<unsigned IDX, int STORE_QUAL = STORE_WRITE_BACK>
    __device__ __forceinline__
    void perform_store(void *dst_ptr) const
    {
      STATIC_ASSERT(IDX < NUM_ELMTS);
      ptx_cudaDMA_store<ET, STORE_QUAL>(buffer[IDX], (ET*)dst_ptr);
    }
  private:
    ET buffer[NUM_ELMTS];
  };

  /********************************************/
  // BufferLoader
  // A class for loading to DMABuffers
  /********************************************/
  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, unsigned IDX>
  struct BufferLoader {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, unsigned stride)
    {
      buffer.template perform_load<OFFSET+MAX-IDX>(src);
      BufferLoader<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE>::load_all(buffer, src+stride, stride);
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE>
  struct BufferLoader<BUFFER,OFFSET,STRIDE,0,0>
  {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, unsigned stride)
    {
      // do nothing
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX>
  struct BufferLoader<BUFFER,OFFSET,STRIDE,MAX,1>
  {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, unsigned stride)
    {
      buffer.template perform_load<OFFSET+MAX-1>(src);
    }
  };

  /*********************************************/
  // BufferStorer
  // A class for storing from DMABuffers
  /*********************************************/
  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, unsigned IDX>
  struct BufferStorer {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, unsigned stride)
    {
      buffer.template perform_store<OFFSET+MAX-IDX>(dst);
      BufferStorer<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE>::store_all(buffer, dst+stride, stride);
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE>
  struct BufferStorer<BUFFER,OFFSET,STRIDE,0,0>
  {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, unsigned stride)
    {
      // do nothing
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX>
  struct BufferStorer<BUFFER,OFFSET,STRIDE,MAX,1>
  {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, unsigned stride)
    {
      buffer.template perform_store<OFFSET+MAX-1>(dst);
    }
  };
};

/**
 * This is the base class for CudaDMA and contains most of the baseline
 * functionality that is used for synchronizing all of the CudaDMA instances.
 */
class CudaDMA {
public:
  __device__ CudaDMA(const int dmaID,
                     const int num_dma_threads,
                     const int num_compute_threads,
                     const int dma_threadIdx_start);
public:
  __device__ __forceinline__ void start_async_dma(void) const
  {
    ptx_cudaDMA_barrier_nonblocking(m_barrierID_empty,m_barrier_size);
  }
  __device__ __forceinline__ void wait_for_dma_finish(void) const
  {
    ptx_cudaDMA_barrier_blocking(m_barrierID_full,m_barrier_size);
  }
  __device__ __forceinline__
  bool owns_this_thread(void) const { return m_is_dma_thread; }
protected:
  __device__ __forceinline__ void wait_for_dma_start(void) const
  {
    ptx_cudaDMA_barrier_blocking(m_barrierID_empty,m_barrier_size); 
  }
  __device__ __forceinline__ void finish_async_dma(void) const
  {
    ptx_cudaDMA_barrier_nonblocking(m_barrierID_full,m_barrier_size);
  }
protected:
  const bool m_is_dma_thread;
  const int m_barrierID_empty;
  const int m_barrierID_full;
  const int m_barrier_size;
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// CudaDMASequential
//////////////////////////////////////////////////////////////////////////////////////////////////

#define FULL_STEP_STRIDE (DMA_THREADS*(BYTES_PER_THREAD/ALIGNMENT)*ALIGNMENT)
#define FULL_LD_STRIDE (DMA_THREADS*ALIGNMENT)
// The number of steps we need to take doing as many full loads as possible
#define FULL_STEPS (BYTES_PER_ELMT/FULL_STEP_STRIDE)
// Number of partial bytes left after performing as many full strides as possible
#define PARTIAL_BYTES (BYTES_PER_ELMT - (FULL_STEPS*FULL_STEP_STRIDE))
// Number of full loads needed in the single partial step
#define PARTIAL_STEPS (PARTIAL_BYTES/FULL_LD_STRIDE)
// Number of remaining bytes after performing all the partial loads
#define REMAINING_BYTES (PARTIAL_BYTES - (PARTIAL_STEPS*FULL_LD_STRIDE))

/**
 * CudaDMASequential will transfer data in a contiguous block from one location to another.
 * DO_SYNC - is warp-specialized or not
 * ALIGNMENT - guaranteed alignment of all pointers passed to the instance
 * BYTES_PER_THREAD - maximum number of bytes that can be used for buffering inside the instance
 * BYTES_PER_ELMT - the size of the element to be transfered
 * DMA_THREADS - the number of DMA Threads that are going to be statically allocated
 */
template<bool DO_SYNC, int ALIGNMENT, int BYTES_PER_THREAD=4*ALIGNMENT, int BYTES_PER_ELMT=0, int DMA_THREADS=0>
class CudaDMASequential : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start);
public:
  template<bool GLOBAL_LOAD = true, 
           int LOAD_QUAL = LOAD_CACHE_ALL, 
           int STORE_QUAL = STORE_WRITE_BACK>
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr);
  
  template<bool GLOBAL_LOAD = true,
           int LOAD_QUAL = LOAD_CACHE_ALL,
  __device__ __forceinline__ void start_xfer_async(const void *RESTRICT src_ptr);

  template<int STORE_QUAL = STORE_WRITE_BACK>
  __device__ __forceinline__ void wait_xfer_finish(void *RESTRICT dst_ptr);
};

#if 0
// one template paramemters, warp-specialized
template<int ALIGNMENT, int BYTES_PER_THREAD>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int DMA_THREADS,
                               const int num_compute_threads,
                               const int dma_threadIdx_start,
                               const int BYTES_PER_ELMT);
};

// one template parameters, non-warp-specialized
template<int ALIGNMENT, int BYTES_PER_THREAD>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int BYTES_PER_ELMT,
                               const int DMA_THREADS = 0);
};

// two template parameters, warp-specialized
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int DMA_THREADS,
                               const int num_compute_threads,
                               const int dma_threadIdx_start);
};

// two template parameters, non-warp-specialized
template<int ALIGNMENT, int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int DMA_THREADS = 0);
};
#endif

// three template parameters, warp-specialized 
#define LOCAL_TYPENAME  float
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<true,4,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
  }
public:
  template<bool GLOBAL_LOAD = false, int LOAD_QUAL = LOAD_CACHE_ALL, int STORE_QUAL = STORE_WRITE_BACK>
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr)
  {
    start_xfer_async<GLOBAL_LOAD,LOAD_QUAL>(src_ptr);
    wait_xfer_finish<STORE_WRITE_BACK>(dst_ptr);
  }

  template<bool GLOBAL_LOAD = false, int LOAD_QUAL = LOAD_CACHE_ALL>
  __device__ __forceinline__ void start_xfer_async(const void *RESTRICT src_ptr)
  {
    this->dma_src_ptr = ((const char*)src_ptr) + this->dma_offset;
    if (FULL_STEPS == 0)
    {
      // Partial case
      issue_loads<true,DMA_PARTIAL_FULL,GLOBAL_LOAD,LOAD_QUAL>(src_ptr, dma_partial_bytes);
    }
    else
    {
      // Everybody issue their full complement of loads
      issue_loads<false,DMA_ITERS_FULL,GLOBAL_LOAD,LOAD_QUAL>(src_ptr, 0/*no partial bytes*/);
      this->dma_src_ptr += DMA_STEP_STRIDE;
    }
  }

  template<int STORE_QUAL = STORE_WRITE_BACK>
  __device__ __forceinline__ void wait_xfer_finish(void *RESTRICT dst_ptr)
  {
    if (FULL_STEPS == 0)
    {

    }
    else
    {
      
    }
  }
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool GLOBAL_LOAD, int LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr, const int partial_bytes)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS-1),GUARD_UNDERFLOW(DMA_FULL_LOADS-1)>::load_all(bulk_buffer, src_ptr, DMA_LD_STRIDE);   
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
};
#undef LOCAL_TYPENAME

template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<true,8,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start);
private:
  float2 bulk_buffer[BYTES_PER_THREAD/sizeof(float2)];
};

template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<true,16,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start);
private:
  float4 bulk_buffer[BYTES_PER_THREAD/sizeof(float4)];
};

// three template parameters, non-warp-specialized
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<false,4,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequentil(void);
};

template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<false,8,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequentil(void);
};

template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<false,16,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequentil(void);
};

#undef WARP_SIZE
#undef WARP_MASK
#undef CUDADMA_DMA_TID
#undef RESTRICT
#ifdef ENABLE_RESTRICT
#undef ENABLE_RESTRICT
#endif
#undef STATIC_ASSERT
#undef GUARD_ZERO

// EOF

