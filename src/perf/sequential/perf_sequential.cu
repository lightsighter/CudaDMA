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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime.h"

//#define CUDADMA_DEBUG_ON
#include "cudaDMA.h"

#include "params_directed.h"

#define WARP_SIZE 32

#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(false);				\
		}						\
	}

// Define parameter buffering if it hasn't already been defined
#ifndef PARAM_BUFFERING
#define PARAM_BUFFERING ""
#endif
#ifndef PARAM_DMA_WARPS
#define PARAM_DMA_WARPS 0
#endif

template<typename T, int ALIGNMENT, int BYTES_PER_ELMT, int TOTAL_THREADS, int NUM_LOOPS>
__global__
void non_specialized(const T *idata, T *odata, bool always_false)
{
  __shared__ T buffer[BYTES_PER_ELMT/sizeof(T)];

  cudaDMASequential<false,ALIGNMENT,BYTES_PER_ELMT,TOTAL_THREADS>
    dma;

  const int offset = blockIdx.x * BYTES_PER_ELMT * NUM_LOOPS / sizeof(T);
  const T *base_ptr = &(idata[offset]);
  T *out_ptr = &(odata[offset]);

  for (int i = 0; i < NUM_LOOPS; i++)
  {
    dma.execute_dma(base_ptr, buffer);
    __syncthreads();
    if (always_false)
    {
      out_ptr[threadIdx.x] += buffer[threadIdx.x];
    }
    // Update the base pointer to read from the next location
    base_ptr += (BYTES_PER_ELMT/sizeof(T));
    __syncthreads();
  }
}

template<typename T, int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_LOOPS>
__global__
void single_buffer(const T *idata, T *odata, int num_compute_threads, bool always_false)
{
  __shared__ T buffer[BYTES_PER_ELMT/sizeof(T)];

  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma (0, num_compute_threads, num_compute_threads);

  const int offset = blockIdx.x*BYTES_PER_ELMT*NUM_LOOPS/sizeof(T);

  if (dma.owns_this_thread())
  {
    const T *base_ptr = &(idata[offset]);

    for (int i = 0; i < NUM_LOOPS; i++)
    {
      dma.execute_dma(base_ptr, buffer);
      base_ptr += (BYTES_PER_ELMT/sizeof(T));
    }
  }
  else
  {
    T *out_ptr = &(odata[offset]);

    for (int i = 0; i < NUM_LOOPS; i++)
    {
      dma.start_async_dma();
      dma.wait_for_dma_finish();
      if (always_false)
      {
        out_ptr[threadIdx.x] += buffer[threadIdx.x];    
      }
    }
  }
}

template<typename T, int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_LOOPS>
__global__
void double_buffer(const T *idata, T *odata, int num_compute_threads, bool always_false)
{
  __shared__ T buffer0[BYTES_PER_ELMT/sizeof(T)];
  __shared__ T buffer1[BYTES_PER_ELMT/sizeof(T)];

  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (0, num_compute_threads, num_compute_threads);
  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma1 (1, num_compute_threads, num_compute_threads + DMA_THREADS);

  const int offset = blockIdx.x*BYTES_PER_ELMT*NUM_LOOPS/sizeof(T);

  if (dma0.owns_this_thread())
  {
    const T *base_ptr = &(idata[offset]);
    
    for (int i = 0; i < NUM_LOOPS; i+=2)
    {
      dma0.execute_dma(base_ptr,buffer0);
      base_ptr += (2*BYTES_PER_ELMT/sizeof(T));
    }
    dma0.wait_for_dma_start();
  }
  else if (dma1.owns_this_thread())
  {
    const T *base_ptr = &(idata[offset+BYTES_PER_ELMT/sizeof(T)]);

    for (int i = 0; i < NUM_LOOPS; i+=2)
    {
      dma1.execute_dma(base_ptr,buffer1);
      base_ptr += (2*BYTES_PER_ELMT/sizeof(T));
    }
    dma1.wait_for_dma_start();
  }
  else
  {
    T *out_ptr = &(odata[offset]);
    dma0.start_async_dma();
    dma1.start_async_dma();
    for (int i = 0; i < NUM_LOOPS; i+=2)
    {
      dma0.wait_for_dma_finish();
      if (always_false)
      {
        out_ptr[threadIdx.x] += buffer0[threadIdx.x];
      }
      dma0.start_async_dma();
      dma1.wait_for_dma_finish();
      if (always_false)
      {
        out_ptr[threadIdx.x] += buffer1[threadIdx.x];
      }
      dma1.start_async_dma();
    }
  }
}

template<typename T, int ALIGNMENT, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_LOOPS>
__global__
void manual_buffer(const T *idata, T *odata, int num_compute_threads, bool always_false)
{
  __shared__ T buffer0[BYTES_PER_ELMT/sizeof(T)];
  __shared__ T buffer1[BYTES_PER_ELMT/sizeof(T)];

  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (0, num_compute_threads, num_compute_threads);
  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma1 (1, num_compute_threads, num_compute_threads);

  const int offset = blockIdx.x*BYTES_PER_ELMT*NUM_LOOPS/sizeof(T);

  if (dma0.owns_this_thread())
  {
    const T *base_ptr = &(idata[offset]);
    for (int i = 0; i < NUM_LOOPS; i+=2)
    {
      dma0.execute_dma(base_ptr,buffer0);
      base_ptr += (BYTES_PER_ELMT/sizeof(T));
      dma1.execute_dma(base_ptr,buffer1);
      base_ptr += (BYTES_PER_ELMT/sizeof(T));
    }
    dma0.wait_for_dma_start();
    dma1.wait_for_dma_start();
  }
  else
  {
    T *out_ptr = &(odata[offset]);
    dma0.start_async_dma();
    dma1.start_async_dma();
    for (int i = 0; i < NUM_LOOPS; i+=2)
    {
      dma0.wait_for_dma_finish();
      if (always_false)
      {
        out_ptr[threadIdx.x] += buffer0[threadIdx.x];
      }
      dma0.start_async_dma();
      dma1.wait_for_dma_finish();
      if (always_false)
      {
        out_ptr[threadIdx.x] += buffer1[threadIdx.x];
      }
      dma1.start_async_dma();
    }
  }
}

__host__
void performance_test(int device, bool always_false)
{
  assert(!always_false);

  const int compute_warps = 1;
  int total_warps = PARAM_DMA_WARPS;
  // If we're doing specialized, need an extra warp for compute threads
  if (PARAM_SPECIALIZED)
  {
    total_warps += compute_warps;
    // If we're double buffered, we need an extra set of warps
    if (strcmp(PARAM_BUFFERING,"double") == 0)
    {
      total_warps += PARAM_DMA_WARPS;
    }
  }
  int total_ctas;
  long total_mem;
  {
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp,device));
    if(deviceProp.major == 9999 && deviceProp.minor == 9999)
    {
      fprintf(stderr,"ERROR: There is no device supporting CUDA with ID %d.\n",device);
      exit(1);
    }
    fprintf(stdout,"\tRunning on device %d called %s with %d SMs\n", 
                    device, deviceProp.name, deviceProp.multiProcessorCount);
    total_ctas = deviceProp.multiProcessorCount * PARAM_CTA_PER_SM;
    fprintf(stdout,"\tTotal CTAS - %d\n", total_ctas);
    // Make sure we don't overflow the one dimension maximum of the grid
    if (total_ctas >= deviceProp.maxGridSize[0])
    {
      fprintf(stderr,"ERROR: %d CTAs requested for one dimensional grid with maximum %d\n",
                      total_ctas, deviceProp.maxGridSize[0]);
      exit(1);
    }
    // Figure out how much memory we need
    total_mem = total_ctas * PARAM_ELMT_SIZE * PARAM_LOOP_ITERS;
    fprintf(stdout,"\tTotal memory - %ld bytes\n", total_mem);
    if (total_mem >= deviceProp.totalGlobalMem)
    {
      fprintf(stderr,"ERROR: %ld bytes of global memory requested but only %ld available\n",
                      total_mem, deviceProp.totalGlobalMem);
      exit(1);
    }
    // Check to make sure we have the right amount of shared memory
    if (!PARAM_SPECIALIZED || (strcmp(PARAM_BUFFERING,"single") == 0))
    {
      if (PARAM_ELMT_SIZE > deviceProp.sharedMemPerBlock)
      {
        fprintf(stderr,"ERROR: %d bytes of shared memory requested but only %ld available\n",
                        PARAM_ELMT_SIZE, deviceProp.sharedMemPerBlock);
        exit(1);
      }
    }
    else
    {
      // These need twice as much shared memory
      if ((2*PARAM_ELMT_SIZE) > deviceProp.sharedMemPerBlock)
      {
        fprintf(stderr,"ERROR: %d bytes of shared memory requested but only %ld available\n",
                        2*PARAM_ELMT_SIZE, deviceProp.sharedMemPerBlock);
        exit(1); 
      }
    }
    // Check to make sure we don't ask for too many threads
    if ((total_warps * WARP_SIZE) > deviceProp.maxThreadsPerBlock)
    {
      fprintf(stderr,"ERROR: %d threads per CTA requested but only %d available\n",
                      total_warps*WARP_SIZE, deviceProp.maxThreadsPerBlock);
      exit(1);
    }
  }

  // Allocate some memory
  PARAM_ELMT_TYPE *d_src, *d_dst;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_src,total_mem));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_dst,sizeof(PARAM_ELMT_TYPE))); // almost nothing

  // Timing information
  cudaStream_t timingStream;
  CUDA_SAFE_CALL(cudaStreamCreate(&timingStream));
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));

  CUDA_SAFE_CALL(cudaEventRecord(start,timingStream));
  if (PARAM_SPECIALIZED)
  {
    if (strcmp(PARAM_BUFFERING,"single") == 0)
    {
      single_buffer<PARAM_ELMT_TYPE, PARAM_ALIGNMENT, PARAM_ELMT_SIZE,
                    PARAM_DMA_WARPS*WARP_SIZE, PARAM_LOOP_ITERS>
                   <<<total_ctas,total_warps*WARP_SIZE,0,timingStream>>>
                   (d_src, d_dst, compute_warps*WARP_SIZE, always_false);
    }
    else if (strcmp(PARAM_BUFFERING,"double") == 0)
    {
      double_buffer<PARAM_ELMT_TYPE, PARAM_ALIGNMENT, PARAM_ELMT_SIZE,
                    PARAM_DMA_WARPS*WARP_SIZE, PARAM_LOOP_ITERS>
                   <<<total_ctas,total_warps*WARP_SIZE,0,timingStream>>>
                   (d_src, d_dst, compute_warps*WARP_SIZE, always_false);
    }
    else if (strcmp(PARAM_BUFFERING,"manual") == 0)
    {
      manual_buffer<PARAM_ELMT_TYPE, PARAM_ALIGNMENT, PARAM_ELMT_SIZE,
                    PARAM_DMA_WARPS*WARP_SIZE, PARAM_LOOP_ITERS>
                   <<<total_ctas,total_warps*WARP_SIZE,0,timingStream>>>
                   (d_src, d_dst, compute_warps*WARP_SIZE, always_false);
    }
    else
    {
      // Should never get here
      assert(false);
    }
  }
  else
  {
    non_specialized<PARAM_ELMT_TYPE, PARAM_ALIGNMENT, PARAM_ELMT_SIZE, 
                    PARAM_DMA_WARPS*WARP_SIZE, PARAM_LOOP_ITERS>
                   <<<total_ctas,total_warps*WARP_SIZE,0,timingStream>>>
                   (d_src, d_dst, always_false);
  }
  CUDA_SAFE_CALL(cudaEventRecord(stop,timingStream));
  CUDA_SAFE_CALL(cudaStreamSynchronize(timingStream));

  // Do the performance calculation
  {
    float exec_time; // in milliseconds
    CUDA_SAFE_CALL(cudaEventElapsedTime(&exec_time,start,stop));
    double bandwidth_gbs = (double(total_mem) / double(exec_time)) * 1e-6;

    fprintf(stdout,"\nPerformance - %.3lf (GB/s)\n\n", bandwidth_gbs);
  }

  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
  CUDA_SAFE_CALL(cudaStreamDestroy(timingStream));

  CUDA_SAFE_CALL(cudaFree(d_src));
  CUDA_SAFE_CALL(cudaFree(d_dst));
}

__host__
int main(int argc, char **argv)
{
  int device = 0;
  if (argc > 1)
  {
    device = atoi(argv[1]);
  }
  fprintf(stdout,"CudaDMA Sequential Performance Test\n");
  fprintf(stdout,"\tALIGNMENT - %d bytes\n",PARAM_ALIGNMENT);
  fprintf(stdout,"\tELEMENT SIZE - %d bytes\n",PARAM_ELMT_SIZE);
  fprintf(stdout,"\tWARP SPECIALIZED - %s\n",(PARAM_SPECIALIZED?"true":"false"));
  if (PARAM_SPECIALIZED)
  {
    fprintf(stdout,"\tBUFFERING - %s\n", PARAM_BUFFERING);  
    fprintf(stdout,"\tDMA WARPS - %d\n", PARAM_DMA_WARPS);
  }
  fprintf(stdout,"\tCTAs/SM - %d\n",PARAM_CTA_PER_SM);
  fprintf(stdout,"\tLOOP ITERATIONS - %d\n",PARAM_LOOP_ITERS);

  // Hopefully this is always false
  performance_test(device,argc > 10000);

  return 0;
}

// EOF

