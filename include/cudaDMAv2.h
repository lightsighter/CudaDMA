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

// For doing static assertions.  If you get a static assertion that
// means that there is something wrong with the way you instantiated
// a CudaDMA instance.
template<bool COND> struct CudaDMAStaticAssert;
template<> struct CudaDMAStaticAssert<true> { };
#define STATIC_ASSERT(condition) do { CudaDMAStaticAssert<(condition)>(); } while (0)

#define GUARD_ZERO(expr) (((expr) == 0) ? 1 : (expr))
#define GUARD_UNDERFLOW(expr) (((expr) < 0) ? 0 : (expr))

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
  STATIC_ASSERT(GLOBAL_LOAD && !GLOBAL_LOAD);
  return result;
}

// No partial function specialization, so just do them all explicitly
/////////////////////////////
// FLOAT
/////////////////////////////
template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,true,LOAD_CACHE_ALL>(const float *src_ptr)
{
  float result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.ca.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,false,LOAD_CACHE_ALL>(const float *src_ptr)
{
  float result;
  asm volatile("ld.ca.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,true,LOAD_CACHE_GLOBAL>(const float *src_ptr)
{
  float result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.cg.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,false,LOAD_CACHE_GLOBAL>(const float *src_ptr)
{
  float result;
  asm volatile("ld.cg.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,true,LOAD_CACHE_STREAMING>(const float *src_ptr)
{
  float result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.cs.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,false,LOAD_CACHE_STREAMING>(const float *src_ptr)
{
  float result;
  asm volatile("ld.cs.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,true,LOAD_CACHE_LAST_USE>(const float *src_ptr)
{
  float result;
  asm volatile("ld.global.lu.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,false,LOAD_CACHE_LAST_USE>(const float *src_ptr)
{
  float result;
  asm volatile("ld.lu.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,true,LOAD_CACHE_VOLATILE>(const float *src_ptr)
{
  float result;
  asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float ptx_cudaDMA_load<float,false,LOAD_CACHE_VOLATILE>(const float *src_ptr)
{
  float result;
  asm volatile("ld.cv.f32 %0, [%1];" : "=f"(result) : "l"(src_ptr) : "memory");
  return result;
}

/////////////////////////////
// FLOAT2
/////////////////////////////

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,true,LOAD_CACHE_ALL>(const float2 *src_ptr)
{
  float2 result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
#endif
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,false,LOAD_CACHE_ALL>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,true,LOAD_CACHE_GLOBAL>(const float2 *src_ptr)
{
  float2 result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,false,LOAD_CACHE_GLOBAL>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,true,LOAD_CACHE_STREAMING>(const float2 *src_ptr)
{
  float2 result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,false,LOAD_CACHE_STREAMING>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,true,LOAD_CACHE_LAST_USE>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,false,LOAD_CACHE_LAST_USE>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,true,LOAD_CACHE_VOLATILE>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float2 ptx_cudaDMA_load<float2,false,LOAD_CACHE_VOLATILE>(const float2 *src_ptr)
{
  float2 result;
  asm volatile("ld.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  return result;
}

/////////////////////////////
// FLOAT3
/////////////////////////////

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,true,LOAD_CACHE_ALL>(const float3 *src_ptr)
{
  float3 result;
#if __CUDA_ARCH__ == 350
  // LDG loads
  asm volatile("ld.global.nc.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
  asm volatile("ld.global.nc.ca.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
  asm volatile("ld.global.ca.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,false,LOAD_CACHE_ALL>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.ca.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");   
  asm volatile("ld.ca.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,true,LOAD_CACHE_GLOBAL>(const float3 *src_ptr)
{
  float3 result;
#if __CUDA_ARCH__ == 350
  // LDG loads
  asm volatile("ld.global.nc.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.global.nc.cg.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.global.cg.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,false,LOAD_CACHE_GLOBAL>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.cg.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.cg.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,true,LOAD_CACHE_STREAMING>(const float3 *src_ptr)
{
  float3 result;
#if __CUDA_ARCH__ == 350
  // LDG loads
  asm volatile("ld.global.nc.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.global.nc.cs.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.global.cs.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,false,LOAD_CACHE_STREAMING>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.cs.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.cs.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,true,LOAD_CACHE_LAST_USE>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.global.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.global.lu.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,false,LOAD_CACHE_LAST_USE>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.lu.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.lu.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,true,LOAD_CACHE_VOLATILE>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.global.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.global.cv.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float3 ptx_cudaDMA_load<float3,false,LOAD_CACHE_VOLATILE>(const float3 *src_ptr)
{
  float3 result;
  asm volatile("ld.cv.v2.f32 {%0,%1}, [%2];" : "=f"(result.x), "=f"(result.y) : "l"(src_ptr) : "memory");
  asm volatile("ld.cv.f32 %0, [%1+8];" : "=f"(result.z) : "l"(src_ptr) : "memory");
  return result;
}

/////////////////////////////
// FLOAT4
/////////////////////////////

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,true,LOAD_CACHE_ALL>(const float4 *src_ptr)
{
  float4 result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,false,LOAD_CACHE_ALL>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.ca.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,true,LOAD_CACHE_GLOBAL>(const float4 *src_ptr)
{
  float4 result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,false,LOAD_CACHE_GLOBAL>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,true,LOAD_CACHE_STREAMING>(const float4 *src_ptr)
{
  float4 result;
#if __CUDA_ARCH__ == 350
  // LDG load
  asm volatile("ld.global.nc.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
#else
  asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
#endif
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,false,LOAD_CACHE_STREAMING>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.cs.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,true,LOAD_CACHE_LAST_USE>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,false,LOAD_CACHE_LAST_USE>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.lu.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,true,LOAD_CACHE_VOLATILE>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

template<>
__device__ __forceinline__
float4 ptx_cudaDMA_load<float4,false,LOAD_CACHE_VOLATILE>(const float4 *src_ptr)
{
  float4 result;
  asm volatile("ld.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w) : "l"(src_ptr) : "memory");
  return result;
}

/*****************************************************/
/*           Store functions                         */
/*****************************************************/
template<typename T, int STORE_QUAL>
__device__ __forceinline__
void ptx_cudaDMA_store(const T &src_val, T *dst_ptr)
{
  // This template should never be instantiated
  STATIC_ASSERT(STORE_QUAL < 0);
}

/////////////////////////////
// FLOAT
/////////////////////////////

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float,STORE_WRITE_BACK>(const float &src_val, float *dst_ptr)
{
  asm volatile("st.wb.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float,STORE_CACHE_GLOBAL>(const float &src_val, float *dst_ptr)
{
  asm volatile("st.cg.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float,STORE_CACHE_STREAMING>(const float &src_val, float *dst_ptr)
{
  asm volatile("st.cs.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float,STORE_CACHE_WRITE_THROUGH>(const float &src_val, float *dst_ptr)
{
  asm volatile("st.wt.f32 [%0], %1;" :  : "l"(dst_ptr), "f"(src_val) : "memory");
}

/////////////////////////////
// FLOAT2
/////////////////////////////

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float2,STORE_WRITE_BACK>(const float2 &src_val, float2 *dst_ptr)
{
  asm volatile("st.wb.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float2,STORE_CACHE_GLOBAL>(const float2 &src_val, float2 *dst_ptr)
{
  asm volatile("st.cg.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float2,STORE_CACHE_STREAMING>(const float2 &src_val, float2 *dst_ptr)
{
  asm volatile("st.cs.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float2,STORE_CACHE_WRITE_THROUGH>(const float2 &src_val, float2 *dst_ptr)
{
  asm volatile("st.wt.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
}

/////////////////////////////
// FLOAT3
/////////////////////////////

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float3,STORE_WRITE_BACK>(const float3 &src_val, float3 *dst_ptr)
{
  asm volatile("st.wb.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
  asm volatile("st.wb.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float3,STORE_CACHE_GLOBAL>(const float3 &src_val, float3 *dst_ptr)
{
  asm volatile("st.cg.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
  asm volatile("st.cg.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float3,STORE_CACHE_STREAMING>(const float3 &src_val, float3 *dst_ptr)
{
  asm volatile("st.cs.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
  asm volatile("st.cs.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float3,STORE_CACHE_WRITE_THROUGH>(const float3 &src_val, float3 *dst_ptr)
{
  asm volatile("st.wt.v2.f32 [%0], {%1,%2};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y) : "memory");
  asm volatile("st.wt.f32 [%0+8], %1;" :  : "l"(dst_ptr), "f"(src_val.z) : "memory");
}

/////////////////////////////
// FLOAT4
/////////////////////////////

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float4,STORE_WRITE_BACK>(const float4 &src_val, float4 *dst_ptr)
{
  asm volatile("st.wb.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float4,STORE_CACHE_GLOBAL>(const float4 &src_val, float4 *dst_ptr)
{
  asm volatile("st.cg.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float4,STORE_CACHE_STREAMING>(const float4 &src_val, float4 *dst_ptr)
{
  asm volatile("st.cs.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
}

template<>
__device__ __forceinline__
void ptx_cudaDMA_store<float4,STORE_CACHE_WRITE_THROUGH>(const float4 &src_val, float4 *dst_ptr)
{
  asm volatile("st.wt.v4.f32 [%0], {%1,%2,%3,%4};" :  : "l"(dst_ptr), "f"(src_val.x), "f"(src_val.y), "f"(src_val.z), "f"(src_val.w) : "memory");
}

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
    template<unsigned IDX, bool GLOBAL_LOAD, int LOAD_QUAL>
    __device__ __forceinline__
    void perform_load(const void *src_ptr)
    {
      STATIC_ASSERT(IDX < NUM_ELMTS);
      buffer[IDX] = ptx_cudaDMA_load<ET, GLOBAL_LOAD, LOAD_QUAL>((const ET*)src_ptr);
    }
    template<unsigned IDX, int STORE_QUAL>
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
  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, unsigned IDX, bool GLOBAL_LOAD, int LOAD_QUAL>
  struct BufferLoader {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, unsigned stride)
    {
      buffer.template perform_load<OFFSET+MAX-IDX,GLOBAL_LOAD,LOAD_QUAL>(src);
      BufferLoader<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE,GLOBAL_LOAD,LOAD_QUAL>::load_all(buffer, src+stride, stride);
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, bool GLOBAL_LOAD, int LOAD_QUAL>
  struct BufferLoader<BUFFER,OFFSET,STRIDE,0,0,GLOBAL_LOAD,LOAD_QUAL>
  {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, unsigned stride)
    {
      // do nothing
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, bool GLOBAL_LOAD, int LOAD_QUAL>
  struct BufferLoader<BUFFER,OFFSET,STRIDE,MAX,1,GLOBAL_LOAD,LOAD_QUAL>
  {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, unsigned stride)
    {
      buffer.template perform_load<OFFSET+MAX-1,GLOBAL_LOAD,LOAD_QUAL>(src);
    }
  };

  /*********************************************/
  // BufferStorer
  // A class for storing from DMABuffers
  /*********************************************/
  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, unsigned IDX, int STORE_QUAL>
  struct BufferStorer {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, unsigned stride)
    {
      buffer.template perform_store<OFFSET+MAX-IDX,STORE_QUAL>(dst);
      BufferStorer<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE,STORE_QUAL>::store_all(buffer, dst+stride, stride);
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, int STORE_QUAL>
  struct BufferStorer<BUFFER,OFFSET,STRIDE,0,0,STORE_QUAL>
  {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, unsigned stride)
    {
      // do nothing
    }
  };

  template<typename BUFFER, unsigned OFFSET, unsigned STRIDE, unsigned MAX, int STORE_QUAL>
  struct BufferStorer<BUFFER,OFFSET,STRIDE,MAX,1,STORE_QUAL>
  {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, unsigned stride)
    {
      buffer.template perform_store<OFFSET+MAX-1,STORE_QUAL>(dst);
    }
  };

  /********************************************/
  // ConditionalBufferLoader
  // A class for loading to DMABuffers with a
  // static upper bound on number of iterations,
  // but a dynamically determined actual iteration count
  /********************************************/
  template<typename BUFFER, int OFFSET, int STRIDE, int MAX, int IDX, bool GLOBAL_LOAD, int LOAD_QUAL>
  struct ConditionalBufferLoader {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, int stride, int actual_max)
    {
      if ((MAX-IDX) < actual_max)
      {
        buffer.template perform_load<OFFSET+MAX-IDX,GLOBAL_LOAD,LOAD_QUAL>(src);
        ConditionalBufferLoader<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE,GLOBAL_LOAD,LOAD_QUAL>::load_all(buffer, src+stride, stride, actual_max);
      }
    }
  };

  template<typename BUFFER, int OFFSET, int STRIDE, bool GLOBAL_LOAD, int LOAD_QUAL>
  struct ConditionalBufferLoader<BUFFER,OFFSET,STRIDE,0,0,GLOBAL_LOAD,LOAD_QUAL>
  {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, int stride, int actual_max)
    {
      // do nothing
    }
  };

  template<typename BUFFER, int OFFSET, int STRIDE, int MAX,bool GLOBAL_LOAD, int LOAD_QUAL>
  struct ConditionalBufferLoader<BUFFER,OFFSET,STRIDE,MAX,1,GLOBAL_LOAD,LOAD_QUAL>
  {
    static __device__ __forceinline__
    void load_all(BUFFER &buffer, const char *src, int stride, int actual_max)
    {
      if ((MAX-1) < actual_max)
        buffer.template perform_load<OFFSET+MAX-1,GLOBAL_LOAD,LOAD_QUAL>(src);
    }
  };

  /********************************************/
  // ConditionalBufferStorer
  // A class for storing from DMABuffers with a
  // static upper bound on number of iterations,
  // but a dynamically determined actual iteration count
  /********************************************/
  template<typename BUFFER, int OFFSET, int STRIDE, int MAX, int IDX, int STORE_QUAL>
  struct ConditionalBufferStorer {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, int stride, int actual_max)
    {
      if ((MAX-IDX) < actual_max)
      {
        buffer.template perform_store<OFFSET+MAX-IDX,STORE_QUAL>(dst);
        ConditionalBufferStorer<BUFFER,OFFSET,STRIDE,MAX,IDX-STRIDE,STORE_QUAL>::store_all(buffer, dst+stride, stride, actual_max);
      }
    }
  };

  template<typename BUFFER, int OFFSET, int STRIDE, int STORE_QUAL>
  struct ConditionalBufferStorer<BUFFER,OFFSET,STRIDE,0,0,STORE_QUAL>
  {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, int stride, int actual_max)
    {
      // do nothing
    }
  };

  template<typename BUFFER, int OFFSET, int STRIDE, int MAX, int STORE_QUAL>
  struct ConditionalBufferStorer<BUFFER,OFFSET,STRIDE,MAX,1,STORE_QUAL>
  {
    static __device__ __forceinline__
    void store_all(const BUFFER &buffer, char *dst, int stride, int actual_max)
    {
      if ((MAX-1) < actual_max)
        buffer.template perform_store<OFFSET+MAX-1,STORE_QUAL>(dst);
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
                     const int dma_threadIdx_start)
    : m_is_dma_thread((int(threadIdx.x)>=dma_threadIdx_start) && (int(threadIdx.x)<(dma_threadIdx_start+num_dma_threads))),
      m_barrierID_empty((dmaID<<1)+1),
      m_barrierID_full(dmaID<<1),
      m_barrier_size(num_dma_threads+num_compute_threads)
  {
  }
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
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, DMA_THREADS, num_compute_threads, dma_threadIdx_start)
  {
    // This template should never be instantiated
    STATIC_ASSERT(DO_SYNC && !DO_SYNC);
  }
};

// Number of bytes handled by all DMA threads performing on full load
#define FULL_LD_STRIDE (DMA_THREADS*ALIGNMENT)
// Maximum number of loads that can be performed by a thread based on register constraints
#define BULK_LDS (BYTES_PER_THREAD/ALIGNMENT)
// Number of bytes handled by all DMA threads performing as many loads as possible
// based on the restriction of the number of bytes alloted to them
#define BULK_STEP_STRIDE (FULL_LD_STRIDE*BULK_LDS)
// Number of bulk steps needed
#define BULK_STEPS (BYTES_PER_ELMT/BULK_STEP_STRIDE)
// Number of partial bytes left after performing as many full strides as possible
#define PARTIAL_BYTES (BYTES_PER_ELMT - (BULK_STEPS*BULK_STEP_STRIDE))
// Number of full loads needed in the single partial step
#define PARTIAL_LDS (PARTIAL_BYTES/FULL_LD_STRIDE)
// Number of remaining bytes after performing all the partial loads
#define REMAINING_BYTES (PARTIAL_BYTES - (PARTIAL_LDS*FULL_LD_STRIDE))
// Compute the thread offset
#define THREAD_OFFSET  (CUDADMA_DMA_TID*ALIGNMENT)
// Compute the number of partial bytes that this thread is responsible for
#define THREAD_LEFTOVER (int(REMAINING_BYTES) - int(CUDADMA_DMA_TID*ALIGNMENT))
#define THREAD_PARTIAL_BYTES ((THREAD_LEFTOVER > ALIGNMENT) ? ALIGNMENT : \
                              (THREAD_LEFTOVER < 0) ? 0 : THREAD_LEFTOVER)

#define WARP_SPECIALIZED_UNQUALIFIED_METHODS                                                        \
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) \
  {                                                                                                 \
    start_xfer_async(src_ptr);                                                                      \
    wait_xfer_finish(dst_ptr);                                                                      \
  }                                                                                                 \
  __device__ __forceinline__ void start_xfer_async(const void *RESTRICT src_ptr)                    \
  {                                                                                                 \
    SEQUENTIAL_START_XFER_IMPL(false,LOAD_CACHE_ALL,STORE_WRITE_BACK)                               \
  }                                                                                                 \
  __device__ __forceinline__ void wait_xfer_finish(void *RESTRICT dst_ptr)                          \
  {                                                                                                 \
    CudaDMA::template wait_for_dma_start();                                                         \
    SEQUENTIAL_WAIT_XFER_IMPL(false,LOAD_CACHE_ALL,STORE_WRITE_BACK)                                \
    CudaDMA::template finish_async_dma();                                                           \
  }

#define WARP_SPECIALIZED_QUALIFIED_METHODS                                                          \
  template<bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL, int DMA_STORE_QUAL>                             \
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) \
  {                                                                                                 \
    start_xfer_async<DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL>(src_ptr);                        \
    wait_xfer_finish<DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL>(dst_ptr);                        \
  }                                                                                                 \
  template<bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL, int DMA_STORE_QUAL>                             \
  __device__ __forceinline__ void start_xfer_async(const void *RESTRICT src_ptr)                    \
  {                                                                                                 \
    SEQUENTIAL_START_XFER_IMPL(DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL)                        \
  }                                                                                                 \
  template<bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL, int DMA_STORE_QUAL>                             \
  __device__ __forceinline__ void wait_xfer_finish(void *RESTRICT dst_ptr)                          \
  {                                                                                                 \
    CudaDMA::template wait_for_dma_start();                                                         \
    SEQUENTIAL_WAIT_XFER_IMPL(DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL)                         \
    CudaDMA::template finish_async_dma();                                                           \
  }

#define NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS                                                    \
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) \
  {                                                                                                 \
    start_xfer_async(src_ptr);                                                                      \
    wait_xfer_finish(dst_ptr);                                                                      \
  }                                                                                                 \
  __device__ __forceinline__ void start_xfer_async(const void *RESTRICT src_ptr)                    \
  {                                                                                                 \
    SEQUENTIAL_START_XFER_IMPL(false,LOAD_CACHE_ALL,STORE_WRITE_BACK)                               \
  }                                                                                                 \
  __device__ __forceinline__ void wait_xfer_finish(void *RESTRICT dst_ptr)                          \
  {                                                                                                 \
    SEQUENTIAL_WAIT_XFER_IMPL(false,LOAD_CACHE_ALL,STORE_WRITE_BACK)                                \
  }

#define NON_WARP_SPECIALIZED_QUALIFIED_METHODS                                                      \
  template<bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL, int DMA_STORE_QUAL>                             \
  __device__ __forceinline__ void execute_dma(const void *RESTRICT src_ptr, void *RESTRICT dst_ptr) \
  {                                                                                                 \
    start_xfer_async<DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL>(src_ptr);                        \
    wait_xfer_finish<DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL>(dst_ptr);                        \
  }                                                                                                 \
  template<bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL, int DMA_STORE_QUAL>                             \
  __device__ __forceinline__ void start_xfer_async(const void *RESTRICT src_ptr)                    \
  {                                                                                                 \
    SEQUENTIAL_START_XFER_IMPL(DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL)                        \
  }                                                                                                 \
  template<bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL, int DMA_STORE_QUAL>                             \
  __device__ __forceinline__ void wait_xfer_finish(void *RESTRICT dst_ptr)                          \
  {                                                                                                 \
    SEQUENTIAL_WAIT_XFER_IMPL(DMA_GLOBAL_LOAD,DMA_LOAD_QUAL,DMA_STORE_QUAL)                         \
  }

#ifdef DEBUG_CUDADMA
#define HANDLE_LOAD_4_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                       \
  const char *partial_ptr = ((const char*)src_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                    \
  switch (this->dma_partial_bytes)                                                                  \
  {                                                                                                 \
    case 0:                                                                                         \
      break;                                                                                        \
    case 4:                                                                                         \
      partial_buffer = ptx_cudaDMA_load<float,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float*)partial_ptr);  \
      break;                                                                                        \
    default:                                                                                        \
      assert(false);                                                                                \
  }
#else
#define HANDLE_LOAD_4_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                       \
  const char *partial_ptr = ((const char*)src_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                    \
  switch (this->dma_partial_bytes)                                                                  \
  {                                                                                                 \
    case 0:                                                                                         \
      break;                                                                                        \
    case 4:                                                                                         \
      partial_buffer = ptx_cudaDMA_load<float,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float*)partial_ptr);  \
      break;                                                                                        \
  }
#endif

#ifdef DEBUG_CUDADMA
#define HANDLE_STORE_4_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                \
  char *partial_ptr = ((char*)dst_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                          \
  switch (this->dma_partial_bytes)                                                            \
  {                                                                                           \
    case 0:                                                                                   \
      break;                                                                                  \
    case 4:                                                                                   \
      ptx_cudaDMA_store<float,DMA_STORE_QUAL>(partial_buffer, (float*)partial_ptr);           \
      break;                                                                                  \
    default:                                                                                  \
      assert(false);                                                                          \
  }
#else
#define HANDLE_STORE_4_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                \
  char *partial_ptr = ((char*)dst_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                          \
  switch (this->dma_partial_bytes)                                                            \
  {                                                                                           \
    case 0:                                                                                   \
      break;                                                                                  \
    case 4:                                                                                   \
      ptx_cudaDMA_store<float,DMA_STORE_QUAL>(partial_buffer, (float*)partial_ptr);           \
      break;                                                                                  \
  }
#endif

#ifdef DEBUG_CUDADMA
#define HANDLE_LOAD_8_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                           \
  const char *partial_ptr = ((const char*)src_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                        \
  switch (this->dma_partial_bytes)                                                                      \
  {                                                                                                     \
    case 0:                                                                                             \
      break;                                                                                            \
    case 4:                                                                                             \
      partial_buffer.x = ptx_cudaDMA_load<float,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float*)partial_ptr);    \
      break;                                                                                            \
    case 8:                                                                                             \
      partial_buffer = ptx_cudaDMA_load<float2,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float2*)partial_ptr);    \
      break;                                                                                            \
    default:                                                                                            \
      assert(false);                                                                                    \
  }
#else
#define HANDLE_LOAD_8_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                           \
  const char *partial_ptr = ((const char*)src_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                        \
  switch (this->dma_partial_bytes)                                                                      \
  {                                                                                                     \
    case 0:                                                                                             \
      break;                                                                                            \
    case 4:                                                                                             \
      partial_buffer.x = ptx_cudaDMA_load<float,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float*)partial_ptr);    \
      break;                                                                                            \
    case 8:                                                                                             \
      partial_buffer = ptx_cudaDMA_load<float2,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float2*)partial_ptr);    \
      break;                                                                                            \
  }
#endif

#ifdef DEBUG_CUDADMA
#define HANDLE_STORE_8_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                          \
  char *partial_ptr = ((char*)dst_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                    \
  switch (this->dma_partial_bytes)                                                      \
  {                                                                                     \
    case 0:                                                                             \
      break;                                                                            \
    case 4:                                                                             \
      ptx_cudaDMA_store<float,DMA_STORE_QUAL>(partial_buffer.x, (float*)partial_ptr);   \
      break;                                                                            \
    case 8:                                                                             \
      ptx_cudaDMA_store<float2,DMA_STORE_QUAL>(partial_buffer, (float2*)partial_ptr);   \
      break;                                                                            \
    default:                                                                            \
      assert(false);                                                                    \
  }
#else
#define HANDLE_STORE_8_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                          \
  char *partial_ptr = ((char*)dst_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                    \
  switch (this->dma_partial_bytes)                                                      \
  {                                                                                     \
    case 0:                                                                             \
      break;                                                                            \
    case 4:                                                                             \
      ptx_cudaDMA_store<float,DMA_STORE_QUAL>(partial_buffer.x, (float*)partial_ptr);   \
      break;                                                                            \
    case 8:                                                                             \
      ptx_cudaDMA_store<float2,DMA_STORE_QUAL>(partial_buffer, (float2*)partial_ptr);   \
      break;                                                                            \
  }
#endif

#ifdef DEBUG_CUDADMA
#define HANDLE_LOAD_16_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                          \
  const char *partial_ptr = ((const char*)src_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                        \
  switch (this->dma_partial_bytes)                                                                      \
  {                                                                                                     \
    case 0:                                                                                             \
      break;                                                                                            \
    case 4:                                                                                             \
      partial_buffer.x = ptx_cudaDMA_load<float,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float*)partial_ptr);    \
      break;                                                                                            \
    case 8:                                                                                             \
      {                                                                                                 \
        float2 temp2 = ptx_cudaDMA_load<float2,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float2*)partial_ptr);    \
        partial_buffer.x = temp2.x;                                                                     \
        partial_buffer.y = temp2.y;                                                                     \
        break;                                                                                          \
      }                                                                                                 \
    case 12:                                                                                            \
      {                                                                                                 \
        float3 temp3 = ptx_cudaDMA_load<float3,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float3*)partial_ptr);    \
        partial_buffer.x = temp3.x;                                                                     \
        partial_buffer.y = temp3.y;                                                                     \
        partial_buffer.z = temp3.z;                                                                     \
        break;                                                                                          \
      }                                                                                                 \
    case 16:                                                                                            \
      partial_buffer = ptx_cudaDMA_load<float4,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float4*)partial_ptr);    \
      break;                                                                                            \
    default:                                                                                            \
      assert(false);                                                                                    \
  }
#else
#define HANDLE_LOAD_16_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                                          \
  const char *partial_ptr = ((const char*)src_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                        \
  switch (this->dma_partial_bytes)                                                                      \
  {                                                                                                     \
    case 0:                                                                                             \
      break;                                                                                            \
    case 4:                                                                                             \
      partial_buffer.x = ptx_cudaDMA_load<float,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float*)partial_ptr);    \
      break;                                                                                            \
    case 8:                                                                                             \
      {                                                                                                 \
        float2 temp2 = ptx_cudaDMA_load<float2,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float2*)partial_ptr);    \
        partial_buffer.x = temp2.x;                                                                     \
        partial_buffer.y = temp2.y;                                                                     \
        break;                                                                                          \
      }                                                                                                 \
    case 12:                                                                                            \
      {                                                                                                 \
        float3 temp3 = ptx_cudaDMA_load<float3,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float3*)partial_ptr);    \
        partial_buffer.x = temp3.x;                                                                     \
        partial_buffer.y = temp3.y;                                                                     \
        partial_buffer.z = temp3.z;                                                                     \
        break;                                                                                          \
      }                                                                                                 \
    case 16:                                                                                            \
      partial_buffer = ptx_cudaDMA_load<float4,DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>((float4*)partial_ptr);    \
      break;                                                                                            \
  }
#endif

#ifdef DEBUG_CUDADMA
#define HANDLE_STORE_16_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                         \
  char *partial_ptr = ((char*)dst_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                    \
  switch (this->dma_partial_bytes)                                                      \
  {                                                                                     \
    case 0:                                                                             \
      break;                                                                            \
    case 4:                                                                             \
      ptx_cudaDMA_store<float,DMA_STORE_QUAL>(partial_buffer.x, (float*)partial_ptr);   \
      break;                                                                            \
    case 8:                                                                             \
      {                                                                                 \
        float2 temp2;                                                                   \
        temp2.x = partial_buffer.x;                                                     \
        temp2.y = partial_buffer.y;                                                     \
        ptx_cudaDMA_store<float2,DMA_STORE_QUAL>(temp2, (float2*)partial_ptr);          \
        break;                                                                          \
      }                                                                                 \
    case 12:                                                                            \
      {                                                                                 \
        float3 temp3;                                                                   \
        temp3.x = partial_buffer.x;                                                     \
        temp3.y = partial_buffer.y;                                                     \
        temp3.z = partial_buffer.z;                                                     \
        ptx_cudaDMA_store<float3,DMA_STORE_QUAL>(temp3, (float3*)partial_ptr);          \
        break;                                                                          \
      }                                                                                 \
    case 16:                                                                            \
      ptx_cudaDMA_store<float4,DMA_STORE_QUAL>(partial_buffer, (float4*)partial_ptr);   \
      break;                                                                            \
    default:                                                                            \
      assert(false);                                                                    \
  }
#else
#define HANDLE_STORE_16_PARTIAL_BYTES(NUM_PREV_LOADS,LD_STRIDE)                         \
  char *partial_ptr = ((char*)dst_ptr) + (NUM_PREV_LOADS*LD_STRIDE);                    \
  switch (this->dma_partial_bytes)                                                      \
  {                                                                                     \
    case 0:                                                                             \
      break;                                                                            \
    case 4:                                                                             \
      ptx_cudaDMA_store<float,DMA_STORE_QUAL>(partial_buffer.x, (float*)partial_ptr);   \
      break;                                                                            \
    case 8:                                                                             \
      {                                                                                 \
        float2 temp2;                                                                   \
        temp2.x = partial_buffer.x;                                                     \
        temp2.y = partial_buffer.y;                                                     \
        ptx_cudaDMA_store<float2,DMA_STORE_QUAL>(temp2, (float2*)partial_ptr);          \
        break;                                                                          \
      }                                                                                 \
    case 12:                                                                            \
      {                                                                                 \
        float3 temp3;                                                                   \
        temp3.x = partial_buffer.x;                                                     \
        temp3.y = partial_buffer.y;                                                     \
        temp3.z = partial_buffer.z;                                                     \
        ptx_cudaDMA_store<float3,DMA_STORE_QUAL>(temp3, (float3*)partial_ptr);          \
        break;                                                                          \
      }                                                                                 \
    case 16:                                                                            \
      ptx_cudaDMA_store<float4,DMA_STORE_QUAL>(partial_buffer, (float4*)partial_ptr);   \
      break;                                                                            \
  }
#endif


// one template paramemters, warp-specialized
#define SEQUENTIAL_START_XFER_IMPL(GLOBAL_LOAD,LOAD_QUAL,STORE_QUAL)                                    \
    this->dma_src_ptr = ((const char*)src_ptr) + this->dma_offset;                                      \
    if (BULK_STEPS == 0)                                                                                \
    {                                                                                                   \
      issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,this->dma_src_ptr);   \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
      issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>(dma_src_ptr);                                         \
      this->dma_src_ptr += BULK_STEP_STRIDE;                                                            \
    }

#define SEQUENTIAL_WAIT_XFER_IMPL(GLOBAL_LOAD,LOAD_QUAL,STORE_QUAL)                                     \
    char *dst_off_ptr = ((char*)dst_ptr) + this->dma_offset;                                            \
    if (BULK_STEPS == 0)                                                                                \
    {                                                                                                   \
      issue_stores<BULK_LDS,STORE_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,dst_off_ptr);                   \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
      issue_stores<BULK_LDS,STORE_QUAL>(dst_off_ptr);                                                   \
      dst_off_ptr += BULK_STEP_STRIDE;                                                                  \
      for (int i = 0; i < (BULK_STEPS-1); i++)                                                          \
      {                                                                                                 \
        issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>(this->dma_src_ptr);                                 \
        this->dma_src_ptr += BULK_STEP_STRIDE;                                                          \
        issue_stores<BULK_LDS,STORE_QUAL>(dst_off_ptr);                                                 \
        dst_off_ptr += BULK_STEP_STRIDE;                                                                \
      }                                                                                                 \
      issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,this->dma_src_ptr);   \
      issue_stores<BULK_LDS,STORE_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,dst_off_ptr);                   \
    }

#define LOCAL_TYPENAME float
#define ALIGNMENT 4
template<int BYTES_PER_THREAD>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_dma_threads,
                               const int num_compute_threads,
                               const int dma_threadIdx_start,
                               const int elmt_size_in_bytes)
    : CudaDMA(dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start),
      BYTES_PER_ELMT(elmt_size_in_bytes),
      DMA_THREADS(num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }

private:
  const int BYTES_PER_ELMT;
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float2
#define ALIGNMENT 8
template<int BYTES_PER_THREAD>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_dma_threads,
                               const int num_compute_threads,
                               const int dma_threadIdx_start,
                               const int elmt_size_in_bytes)
    : CudaDMA(dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start),
      BYTES_PER_ELMT(elmt_size_in_bytes),
      DMA_THREADS(num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int BYTES_PER_ELMT;
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float2 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float4
#define ALIGNMENT 16 
template<int BYTES_PER_THREAD>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_dma_threads,
                               const int num_compute_threads,
                               const int dma_threadIdx_start,
                               const int elmt_size_in_bytes)
    : CudaDMA(dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start),
      BYTES_PER_ELMT(elmt_size_in_bytes),
      DMA_THREADS(num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int BYTES_PER_ELMT;
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float4 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

// one template parameters, non-warp-specialized
#define LOCAL_TYPENAME float
#define ALIGNMENT 4
template<int BYTES_PER_THREAD>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int elmt_size_in_bytes,
                               const int num_dma_threads = 0,
                               const int dma_threadIdx_start = 0)
    : CudaDMA(0, num_dma_threads, num_dma_threads, dma_threadIdx_start),
      BYTES_PER_ELMT(elmt_size_in_bytes),
      DMA_THREADS((num_dma_threads <= 0) ? blockDim.x : num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int BYTES_PER_ELMT;
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float2
#define ALIGNMENT 8
template<int BYTES_PER_THREAD>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int elmt_size_in_bytes,
                               const int num_dma_threads= 0,
                               const int dma_threadIdx_start = 0)
    : CudaDMA(0, num_dma_threads, num_dma_threads, dma_threadIdx_start),
      BYTES_PER_ELMT(elmt_size_in_bytes),
      DMA_THREADS((num_dma_threads <= 0) ? blockDim.x : num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int BYTES_PER_ELMT;
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float2 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float4
#define ALIGNMENT 16 
template<int BYTES_PER_THREAD>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,0,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int elmt_size_in_bytes,
                               const int num_dma_threads= 0,
                               const int dma_threadIdx_start = 0)
    : CudaDMA(0, num_dma_threads, num_dma_threads, dma_threadIdx_start),
      BYTES_PER_ELMT(elmt_size_in_bytes),
      DMA_THREADS((num_dma_threads <= 0) ? blockDim.x : num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
public:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int BYTES_PER_ELMT;
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float4 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#undef SEQUENTIAL_START_XFER_IMPL
#undef SEQUENTIAL_WAIT_XFER_IMPL

// two template parameters, warp-specialized
#define SEQUENTIAL_START_XFER_IMPL(GLOBAL_LOAD,LOAD_QUAL,STORE_QUAL)                                    \
    this->dma_src_ptr = ((const char*)src_ptr) + this->dma_offset;                                      \
    if (BULK_STEPS == 0)                                                                                \
    {                                                                                                   \
      issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,this->dma_src_ptr);   \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
      issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>(dma_src_ptr);                                         \
      this->dma_src_ptr += BULK_STEP_STRIDE;                                                            \
    }

#define SEQUENTIAL_WAIT_XFER_IMPL(GLOBAL_LOAD,LOAD_QUAL,STORE_QUAL)                                     \
    char *dst_off_ptr = ((char*)dst_ptr) + this->dma_offset;                                            \
    if (BULK_STEPS == 0)                                                                                \
    {                                                                                                   \
      issue_stores<BULK_LDS,STORE_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,dst_off_ptr);                   \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
      issue_stores<BULK_LDS,STORE_QUAL>(dst_off_ptr);                                                   \
      dst_off_ptr += BULK_STEP_STRIDE;                                                                  \
      for (int i = 0; i < (BULK_STEPS-1); i++)                                                          \
      {                                                                                                 \
        issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>(this->dma_src_ptr);                                 \
        this->dma_src_ptr += BULK_STEP_STRIDE;                                                          \
        issue_stores<BULK_LDS,STORE_QUAL>(dst_off_ptr);                                                 \
        dst_off_ptr += BULK_STEP_STRIDE;                                                                \
      }                                                                                                 \
      issue_loads<BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,this->dma_src_ptr);   \
      issue_stores<BULK_LDS,STORE_QUAL>((REMAINING_BYTES>0),PARTIAL_LDS,dst_off_ptr);                   \
    }

#define LOCAL_TYPENAME float
#define ALIGNMENT 4
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_dma_threads,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start),
      DMA_THREADS(num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float2
#define ALIGNMENT 8
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_dma_threads,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start),
      DMA_THREADS(num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float2 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float4
#define ALIGNMENT 16
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_dma_threads,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start),
      DMA_THREADS(num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float4 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

// two template parameters, non-warp-specialized
#define LOCAL_TYPENAME float
#define ALIGNMENT 4
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int num_dma_threads = 0,
                               const int dma_threadIdx_start = 0)
    : CudaDMA(0, num_dma_threads, num_dma_threads, dma_threadIdx_start),
      DMA_THREADS((num_dma_threads <= 0) ? blockDim.x : num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_4_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float2
#define ALIGNMENT 8
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int num_dma_threads = 0,
                               const int dma_threadIdx_start = 0)
    : CudaDMA(0, num_dma_threads, num_dma_threads, dma_threadIdx_start),
      DMA_THREADS((num_dma_threads <= 0) ? blockDim.x : num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_8_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float2 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float4
#define ALIGNMENT 16 
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,0> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int num_dma_threads = 0,
                               const int dma_threadIdx_start = 0)
    : CudaDMA(0, num_dma_threads, num_dma_threads, dma_threadIdx_start),
      DMA_THREADS((num_dma_threads <= 0) ? blockDim.x : num_dma_threads),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
public:
  template<int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(bool has_partial, int full_loads, const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::ConditionalBufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_LOAD_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
  template<int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
  }
  template<int DMA_MAX_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(bool has_partial, int full_loads, void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::ConditionalBufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_MAX_LOADS),GUARD_UNDERFLOW(DMA_MAX_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE, full_loads);
    if (has_partial)
    {
      HANDLE_STORE_16_PARTIAL_BYTES(full_loads,FULL_LD_STRIDE);
    }
  }
private:
  const int DMA_THREADS;
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float4 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#undef SEQUENTIAL_START_XFER_IMPL
#undef SEQUENTIAL_WAIT_XFER_IMPL

// three template parameters, warp-specialized 
#define SEQUENTIAL_START_XFER_IMPL(GLOBAL_LOAD,LOAD_QUAL,STORE_QUAL)                          \
    this->dma_src_ptr = ((const char*)src_ptr) + this->dma_offset;                            \
    if (BULK_STEPS == 0)                                                                      \
    {                                                                                         \
      issue_loads<(REMAINING_BYTES>0),PARTIAL_LDS,GLOBAL_LOAD,LOAD_QUAL>(this->dma_src_ptr);  \
    }                                                                                         \
    else                                                                                      \
    {                                                                                         \
      issue_loads<false,BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>(dma_src_ptr);                         \
      this->dma_src_ptr += BULK_STEP_STRIDE;                                                  \
    }

#define SEQUENTIAL_WAIT_XFER_IMPL(GLOBAL_LOAD,LOAD_QUAL,STORE_QUAL)                           \
    char *dst_off_ptr = ((char*)dst_ptr) + this->dma_offset;                                  \
    if (BULK_STEPS == 0)                                                                      \
    {                                                                                         \
      issue_stores<(REMAINING_BYTES>0),PARTIAL_LDS,STORE_QUAL>(dst_off_ptr);                  \
    }                                                                                         \
    else                                                                                      \
    {                                                                                         \
      issue_stores<false,BULK_LDS,STORE_QUAL>(dst_off_ptr);                                   \
      dst_off_ptr += BULK_STEP_STRIDE;                                                        \
      for (int i = 0; i < (BULK_STEPS-1); i++)                                                \
      {                                                                                       \
        issue_loads<false,BULK_LDS,GLOBAL_LOAD,LOAD_QUAL>(this->dma_src_ptr);                 \
        this->dma_src_ptr += BULK_STEP_STRIDE;                                                \
        issue_stores<false,BULK_LDS,STORE_QUAL>(dst_off_ptr);                                 \
        dst_off_ptr += BULK_STEP_STRIDE;                                                      \
      }                                                                                       \
      issue_loads<(REMAINING_BYTES>0),PARTIAL_LDS,GLOBAL_LOAD,LOAD_QUAL>(this->dma_src_ptr);  \
      issue_stores<(REMAINING_BYTES>0),PARTIAL_LDS,STORE_QUAL>(dst_off_ptr);                  \
    }

#define LOCAL_TYPENAME  float
#define ALIGNMENT 4
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, DMA_THREADS, num_compute_threads, dma_threadIdx_start),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);   
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_LOAD_4_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_STORE_4_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float partial_buffer;
};
#undef ALIGNMENT
#undef LOCAL_TYPENAME

#define LOCAL_TYPENAME float2
#define ALIGNMENT 8
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, DMA_THREADS, num_compute_threads, dma_threadIdx_start),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);   
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_LOAD_8_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_STORE_8_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float2 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float4
#define ALIGNMENT 16
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dmaID,
                               const int num_compute_threads,
                               const int dma_threadIdx_start)
    : CudaDMA(dmaID, DMA_THREADS, num_compute_threads, dma_threadIdx_start),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  WARP_SPECIALIZED_UNQUALIFIED_METHODS
  WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_LOAD_16_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_STORE_16_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float4 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

// three template parameters, non-warp-specialized
#define LOCAL_TYPENAME float
#define ALIGNMENT 4
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dma_threadIdx_start = 0)
    : CudaDMA(0, DMA_THREADS, DMA_THREADS, dma_threadIdx_start),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);   
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_LOAD_4_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_STORE_4_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float2
#define ALIGNMENT 8
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dma_threadIdx_start = 0)
    : CudaDMA(0, DMA_THREADS, DMA_THREADS, dma_threadIdx_start),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);   
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_LOAD_8_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_STORE_8_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float2 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#define LOCAL_TYPENAME float4
#define ALIGNMENT 16
template<int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
class CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS> : public CudaDMA {
public:
  __device__ CudaDMASequential(const int dma_threadIdx_start = 0)
    : CudaDMA(0, DMA_THREADS, DMA_THREADS, dma_threadIdx_start),
      dma_offset(THREAD_OFFSET),
      dma_partial_bytes(THREAD_PARTIAL_BYTES)
  {
    STATIC_ASSERT((BYTES_PER_THREAD/ALIGNMENT) > 0);
    STATIC_ASSERT((BYTES_PER_THREAD%ALIGNMENT) == 0);
  }
public:
  NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
  NON_WARP_SPECIALIZED_QUALIFIED_METHODS
private:
  // Helper methods
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, bool DMA_GLOBAL_LOAD, int DMA_LOAD_QUAL>
  __device__ __forceinline__ void issue_loads(const void *RESTRICT src_ptr)
  {
    CudaDMAMeta::BufferLoader<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_GLOBAL_LOAD,DMA_LOAD_QUAL>::load_all(bulk_buffer, (const char*)src_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_LOAD_16_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
  template<bool DMA_PARTIAL_BYTES, int DMA_FULL_LOADS, int DMA_STORE_QUAL>
  __device__ __forceinline__ void issue_stores(void *RESTRICT dst_ptr)
  {
    CudaDMAMeta::BufferStorer<BulkBuffer,0,1,GUARD_UNDERFLOW(DMA_FULL_LOADS),GUARD_UNDERFLOW(DMA_FULL_LOADS),DMA_STORE_QUAL>::store_all(bulk_buffer, (char*)dst_ptr, FULL_LD_STRIDE);
    if (DMA_PARTIAL_BYTES)
    {
      HANDLE_STORE_16_PARTIAL_BYTES(DMA_FULL_LOADS,FULL_LD_STRIDE);
    }
  }
private:
  const int dma_offset;
  const int dma_partial_bytes;
  const char *dma_src_ptr;
  typedef CudaDMAMeta::DMABuffer<LOCAL_TYPENAME,BYTES_PER_THREAD/ALIGNMENT> BulkBuffer;
  BulkBuffer bulk_buffer;
  float4 partial_buffer;
};
#undef LOCAL_TYPENAME
#undef ALIGNMENT

#undef SEQUENTIAL_START_XFER_IMPL
#undef SEQUENTIAL_WAIT_XFER_IMPL

#undef FULL_LD_STRIDE
#undef BULK_LDS
#undef BULK_STEP_STRIDE
#undef BULK_STEPS
#undef PARTIAL_BYTES
#undef PARTIAL_LDS
#undef REMAINING_BYTES
#undef THREAD_OFFSET
#undef THREAD_LEFTOVER
#undef THREAD_PARTIAL_BYTES
#undef WARP_SPECIALIZED_UNQUALIFIED_METHODS
#undef WARP_SPECIALIZED_QUALIFIED_METHODS
#undef NON_WARP_SPECIALIZED_UNQUALIFIED_METHODS
#undef NON_WARP_SPECIALIZED_QUALIFIED_METHODS
#undef HANDLE_LOAD_4_PARTIAL_BYTES
#undef HANDLE_STORE_4_PARTIAL_BYTES
#undef HANDLE_LOAD_8_PARTIAL_BYTES
#undef HANDLE_STORE_8_PARTIAL_BYTES
#undef HANDLE_LOAD_16_PARTIAL_BYTES
#undef HANDLE_STORE_16_PARTIAL_BYTES

////////////////////////  End of CudaDMASequential //////////////////////////////////////////////////

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

