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

/* Software DMA project
*
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <set>

#include "cuda.h"
#include "cuda_runtime.h"

//#define CUDADMA_DEBUG_ON
#include "cudaDMA.h"

#include "params_directed.h"

#define WARP_SIZE 32

#define MAX_TRIES 8192

// includes, project

// includes, kernels

#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(false);				\
		}						\
	}

// I hate global variables, but whatever
long total_experiments = 0;

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_gather_four( float *idata, float *odata, int *offsets, int buffer_size, int num_compute_threads)
{
  extern __shared__ float buffer[];	

  cudaDMAIndirect<true,true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
    dma0 (1, num_compute_threads,
          num_compute_threads);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
  }
  else
  {
    // Zero out the buffer
    int iters = buffer_size/num_compute_threads;	
    int index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      buffer[index] = 0.0f;
      index += num_compute_threads;
    }
    if (index < buffer_size)
      buffer[index] = 0.0f;
    dma0.start_async_dma();
    dma0.wait_for_dma_finish();
    // Now read the buffer out of shared and write the results back
    index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      float res = buffer[index];
      odata[index] = res;
      index += num_compute_threads;
    }
    if (index < buffer_size)
    {
      float res = buffer[index];
      odata[index] = res;
    }
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_gather_three(float *idata, float *odata, int *offsets, int buffer_size, int num_compute_threads, int num_elmts)
{
  extern __shared__ float buffer[];	

  cudaDMAIndirect<true,true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (1, num_compute_threads,
          num_compute_threads,
          num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
  }
  else
  {
    // Zero out the buffer
    int iters = buffer_size/num_compute_threads;	
    int index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      buffer[index] = 0.0f;
      index += num_compute_threads;
    }
    if (index < buffer_size)
      buffer[index] = 0.0f;
    dma0.start_async_dma();
    dma0.wait_for_dma_finish();
    // Now read the buffer out of shared and write the results back
    index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      float res = buffer[index];
      odata[index] = res;
      index += num_compute_threads;
    }
    if (index < buffer_size)
    {
      float res = buffer[index];
      odata[index] = res;
    }
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
dma_gather_two(float *idata, float *odata, int *offsets, int buffer_size, int num_compute_threads, int num_dma_threads, int num_elmts)
{
  extern __shared__ float buffer[];	

  cudaDMAIndirect<true,true,ALIGNMENT,BYTES_PER_ELMT>
    dma0 (1, num_dma_threads,
          num_compute_threads,
          num_compute_threads,
          num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
  }
  else
  {
    // Zero out the buffer
    int iters = buffer_size/num_compute_threads;	
    int index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      buffer[index] = 0.0f;
      index += num_compute_threads;
    }
    if (index < buffer_size)
      buffer[index] = 0.0f;
    dma0.start_async_dma();
    dma0.wait_for_dma_finish();
    // Now read the buffer out of shared and write the results back
    index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      float res = buffer[index];
      odata[index] = res;
      index += num_compute_threads;
    }
    if (index < buffer_size)
    {
      float res = buffer[index];
      odata[index] = res;
    }
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
dma_gather_one(float *idata, float *odata, int *offsets, int buffer_size, int num_compute_threads, int num_dma_threads, int bytes_per_elmt, int num_elmts)
{
  extern __shared__ float buffer[];	

  cudaDMAIndirect<true,true,ALIGNMENT>
    dma0 (1, num_dma_threads,
          num_compute_threads,
          num_compute_threads,
          bytes_per_elmt,
          num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, base_ptr, &(buffer[ALIGN_OFFSET]));
#endif
  }
  else
  {
    // Zero out the buffer
    int iters = buffer_size/num_compute_threads;	
    int index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      buffer[index] = 0.0f;
      index += num_compute_threads;
    }
    if (index < buffer_size)
      buffer[index] = 0.0f;
    dma0.start_async_dma();
    dma0.wait_for_dma_finish();
    // Now read the buffer out of shared and write the results back
    index = threadIdx.x;
    for (int i=0; i<iters; i++)
    {
      float res = buffer[index];
      odata[index] = res;
      index += num_compute_threads;
    }
    if (index < buffer_size)
    {
      float res = buffer[index];
      odata[index] = res;
    }
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_scatter_four( float *odata, int *offsets, int buffer_size, int num_compute_threads)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
    dma0 (1, num_compute_threads,
            num_compute_threads);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(odata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
#endif
  }
  else
  {
    // Initialize the shared memory buffer
    if (threadIdx.x == 0)
    {
      for (int i=0; i<buffer_size; i++)
      {
        buffer[i] = float(i);
      }
    }
    dma0.start_async_dma();
    dma0.wait_for_dma_start();
    // No need to write anything back, it's already been done
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_scatter_three( float *odata, int *offsets, int buffer_size, int num_compute_threads, int NUM_ELMTS)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (1, num_compute_threads,
            num_compute_threads,
            NUM_ELMTS);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(odata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
#endif
  }
  else
  {
    // Initialize the shared memory buffer
    if (threadIdx.x == 0)
    {
      for (int i=0; i<buffer_size; i++)
      {
        buffer[i] = float(i);
      }
    }
    dma0.start_async_dma();
    dma0.wait_for_dma_start();
    // No need to write anything back, it's already been done
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
dma_scatter_two( float *odata, int *offsets, int buffer_size, int num_compute_threads, int num_dma_threads, int NUM_ELMTS)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,true,ALIGNMENT,BYTES_PER_ELMT>
    dma0 (1, num_dma_threads,
            num_compute_threads,
            num_compute_threads,
            NUM_ELMTS);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(odata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
#endif
  }
  else
  {
    // Initialize the shared memory buffer
    if (threadIdx.x == 0)
    {
      for (int i=0; i<buffer_size; i++)
      {
        buffer[i] = float(i);
      }
    }
    dma0.start_async_dma();
    dma0.wait_for_dma_start();
    // No need to write anything back, it's already been done
  }
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
dma_scatter_one( float *odata, int *offsets, int buffer_size, int num_compute_threads, int num_dma_threads, int bytes_per_elmt, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,true,ALIGNMENT>
    dma0 (1, num_dma_threads,
            num_compute_threads,
            num_compute_threads,
            bytes_per_elmt,
            num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(odata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
#endif
  }
  else
  {
    // Initialize the shared memory buffer
    if (threadIdx.x == 0)
    {
      for (int i=0; i<buffer_size; i++)
      {
        buffer[i] = float(i);
      }
    }
    dma0.start_async_dma();
    dma0.wait_for_dma_start();
    // No need to write anything back, it's already been done
  }
}

__device__
void zero_buffer(float *buffer, const int buffer_size)
{
  int iters = buffer_size/blockDim.x;
  int index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    buffer[index] = 0.0f;
    index += blockDim.x;
  }
  if (index < buffer_size)
    buffer[index] = 0.0f;
}

__device__
void copy_buffer(float *buffer, float *dst, const int buffer_size)
{
  int iters = buffer_size/blockDim.x;
  int index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    dst[index] = buffer[index];
    index += blockDim.x;
  }
  if (index < buffer_size)
    dst[index] = buffer[index];
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
simple_gather_four(float *idata, float *odata, int *offsets, int buffer_size)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<true,false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
    dma0;

  zero_buffer(buffer, buffer_size);
  __syncthreads();
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets,base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  copy_buffer(buffer, odata, buffer_size);
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
simple_gather_three(float *idata, float *odata, int *offsets, int buffer_size, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<true,false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0(num_elmts);

  zero_buffer(buffer,buffer_size);
  __syncthreads();
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets,base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  copy_buffer(buffer, odata, buffer_size);
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
simple_gather_two(float *idata, float *odata, int *offsets, int buffer_size, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<true,false,ALIGNMENT,BYTES_PER_ELMT>
    dma0(num_elmts);

  zero_buffer(buffer,buffer_size);
  __syncthreads();
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets,base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  copy_buffer(buffer, odata, buffer_size);
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
simple_gather_one(float *idata, float *odata, int *offsets, int buffer_size, int bytes_per_elmt, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<true,false,ALIGNMENT>
    dma0(bytes_per_elmt,num_elmts);

  zero_buffer(buffer,buffer_size);
  __syncthreads();
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets,base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  copy_buffer(buffer, odata, buffer_size);
}

__device__
void initialize_buffer(float *buffer, const int buffer_size)
{
  int iters = buffer_size/blockDim.x;
  int index = threadIdx.x;
  for (int i=0; i<iters; i++)
  {
    buffer[index] = index;
    index += blockDim.x;
  }
  if (index < buffer_size)
    buffer[index] = index;
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
simple_scatter_four(float *odata, int *offsets, int buffer_size)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
    dma0;

  initialize_buffer(buffer, buffer_size);
  __syncthreads();
  float *base_ptr = &(odata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
simple_scatter_three(float *odata, int *offsets, int buffer_size, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0(num_elmts);

  initialize_buffer(buffer, buffer_size);
  __syncthreads();
  float *base_ptr = &(odata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
simple_scatter_two(float *odata, int *offsets, int buffer_size, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,false,ALIGNMENT,BYTES_PER_ELMT>
    dma0(num_elmts);

  initialize_buffer(buffer, buffer_size);
  __syncthreads();
  float *base_ptr = &(odata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets, &(buffer[ALIGN_OFFSET]), base_ptr);
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
simple_scatter_one(float *odata, int *offsets, int buffer_size, int bytes_per_elmt, int num_elmts)
{
  extern __shared__ float buffer[];

  cudaDMAIndirect<false,false,ALIGNMENT>
    dma0(bytes_per_elmt,num_elmts);

  initialize_buffer(buffer, buffer_size);
  __syncthreads();
  float *base_ptr = &(odata[ALIGN_OFFSET]);
  dma0.execute_dma(offsets,&(buffer[ALIGN_OFFSET]), base_ptr);
}

template<bool SPECIALIZED, int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__host__ bool run_gather_experiment(int *h_offsets, int max_index/*floats*/, int num_templates)
{
  int shared_buffer_size = (NUM_ELMTS*BYTES_PER_ELMT/sizeof(float) + ALIGN_OFFSET);

  // Check to see if we'll overflow the shared buffer
  if ((shared_buffer_size*sizeof(float)) > 49152)
    return true;

  // Allocate the inputs
  int input_size = (max_index + ALIGN_OFFSET);
  float *h_idata = (float*)malloc(input_size*sizeof(float));
	for (int i=0; i<input_size; i++)
		h_idata[i] = float(i);

  // Allocate device memory and copy down
  float *d_idata;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_idata, input_size*sizeof(float)));
  CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, input_size*sizeof(float), cudaMemcpyHostToDevice));
  // Allocate the output size
  int output_size = (NUM_ELMTS*BYTES_PER_ELMT/sizeof(float) + ALIGN_OFFSET);
  float *h_odata = (float*)malloc(output_size*sizeof(float));
  for (int i=0; i<output_size; i++)
          h_odata[i] = 0.0f;
  // Allocate device memory and copy down
  float *d_odata;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_odata, output_size*sizeof(float)));
  CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, output_size*sizeof(float), cudaMemcpyHostToDevice));
  // Finally copy down the offsets
  int *d_offsets;
  CUDA_SAFE_CALL( cudaMalloc( (void**)&d_offsets, NUM_ELMTS*sizeof(int)));
  CUDA_SAFE_CALL( cudaMemcpy( d_offsets, h_offsets, NUM_ELMTS*sizeof(int), cudaMemcpyHostToDevice));

  int num_compute_warps = 1;
  int total_threads = 0;
  if (SPECIALIZED)
    total_threads = (num_compute_warps)*WARP_SIZE + DMA_THREADS;
  else
    total_threads = DMA_THREADS;
  assert(total_threads > 0);

  switch (num_templates)
  {
    case 4:
      {
        if (SPECIALIZED)
        {
          dma_gather_four<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, NUM_ELMTS, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE); 
        }
        else
        {
          simple_gather_four<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, NUM_ELMTS, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size);
        }
        break;
      }
    case 3:
      {
        if (SPECIALIZED)
        {
          dma_gather_three<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE, NUM_ELMTS);
        }
        else
        {
          simple_gather_three<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, NUM_ELMTS);
        }
        break;
      }
    case 2:
      {
        if (SPECIALIZED)
        {
          dma_gather_two<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS, NUM_ELMTS);
        }
        else
        {
          simple_gather_two<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, NUM_ELMTS); 
        }
        break;
      }
    case 1:
      {
        if (SPECIALIZED)
        {
          dma_gather_one<ALIGNMENT,ALIGN_OFFSET>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS, BYTES_PER_ELMT, NUM_ELMTS);
        }
        else
        {
          simple_gather_one<ALIGNMENT,ALIGN_OFFSET>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_idata, d_odata, d_offsets, shared_buffer_size, BYTES_PER_ELMT, NUM_ELMTS);
        }
        break;
      }
    default:
      assert(false);
  }

  CUDA_SAFE_CALL( cudaThreadSynchronize());

  CUDA_SAFE_CALL( cudaMemcpy (h_odata, d_odata, output_size*sizeof(float), cudaMemcpyDeviceToHost));

  // Check the result
  bool pass = true;
  for (int i=0; i<NUM_ELMTS && pass; i++)
  {
    int in_index = ALIGN_OFFSET+(h_offsets[i]*BYTES_PER_ELMT/sizeof(float));
    int out_index = ALIGN_OFFSET+i*BYTES_PER_ELMT/sizeof(float);
#if 0
    printf("elment %d result: ",i);
    for (int j=0; j<(BYTES_PER_ELMT/sizeof(float)); j++)
      printf("%f ", h_odata[out_index+j]);
    printf("\n");
    printf("expected element %d: ",i);
    for (int j=0; j<(BYTES_PER_ELMT/sizeof(float)); j++)
      printf("%f ", h_idata[in_index+j]);
    printf("\n");
#endif
    for (int j=0; j<(BYTES_PER_ELMT/sizeof(float)); j++)
    {
      //printf("%f ",h_odata[out_index+j]);
      if (h_idata[in_index+j] != h_odata[out_index+j])
      {
        //fprintf(stderr,"Experiment: %d element bytes, %d elements, %d DMA warps, %d alignment, %d offset, ",BYTES_PER_ELMT,NUM_ELMTS,DMA_THREADS/WARP_SIZE,ALIGNMENT,ALIGN_OFFSET);
        fprintf(stderr,"\nOffsets: ");
        for (int k=0; k<NUM_ELMTS; k++)
          fprintf(stderr,"%d ", h_offsets[k]);
        fprintf(stderr,"\nIndex %d of element %d was expecting %f but received %f\n", j, i, h_idata[in_index+j], h_odata[out_index+j]);
        pass = false;
        break;
      }
    }
    //printf("\n");
  }
  if (!pass)
  {
    fprintf(stdout,"Result - %s\n",(pass?"SUCCESS":"FAILURE"));
    fflush(stdout);
  }
  CUDA_SAFE_CALL( cudaFree(d_idata));
  CUDA_SAFE_CALL( cudaFree(d_odata));
  CUDA_SAFE_CALL( cudaFree(d_offsets));
  free(h_idata);
  free(h_odata);

  total_experiments++;

  return pass;
}

template<bool SPECIALIZED, int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int NUM_ELMTS, int DMA_THREADS>
__host__ bool run_scatter_experiment(int *h_offsets, int max_index/*floats*/, int num_templates)
{
  int shared_buffer_size = (NUM_ELMTS*BYTES_PER_ELMT/sizeof(float) + ALIGN_OFFSET);

  // Check to see if we'll overflow the shared buffer
  if ((shared_buffer_size*sizeof(float)) > 49152)
    return true;

  // Allocate the destination
  int output_size = (max_index + ALIGN_OFFSET);
  float *h_odata = (float*)malloc(output_size*sizeof(float));
  for (int i = 0; i<output_size; i++)
  {
    h_odata[i] = 0.0f;
  }

  // Allocate device memory and copy down
  float *d_odata;
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_odata, output_size*sizeof(float)));
  CUDA_SAFE_CALL( cudaMemcpy(d_odata, h_odata, output_size*sizeof(float), cudaMemcpyHostToDevice));

  // Copy down the offsets
  int *d_offsets;
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_offsets, NUM_ELMTS*sizeof(int)));
  CUDA_SAFE_CALL( cudaMemcpy(d_offsets, h_offsets, NUM_ELMTS*sizeof(int), cudaMemcpyHostToDevice));

  int num_compute_warps = 1;
  int total_threads = 0;
  if (SPECIALIZED)
    total_threads = (num_compute_warps)*WARP_SIZE + DMA_THREADS;
  else
    total_threads = DMA_THREADS;
  assert(total_threads > 0);

  switch (num_templates)
  {
    case 4:
      {
        if (SPECIALIZED)
        {
          dma_scatter_four<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, NUM_ELMTS, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE);
        }
        else
        {
          simple_scatter_four<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, NUM_ELMTS, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size);
        }
        break;
      }
    case 3:
      {
        if (SPECIALIZED)
        {
          dma_scatter_three<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE,NUM_ELMTS);
        }
        else
        {
          simple_scatter_three<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, DMA_THREADS>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, NUM_ELMTS);
        }
        break;
      }
    case 2:
      {
        if (SPECIALIZED)
        {
          dma_scatter_two<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE,DMA_THREADS,NUM_ELMTS);
        }
        else
        {
          simple_scatter_two<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, NUM_ELMTS);
        }
        break;
      }
    case 1:
      {
        if (SPECIALIZED)
        {
          dma_scatter_one<ALIGNMENT, ALIGN_OFFSET>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE,DMA_THREADS,BYTES_PER_ELMT,NUM_ELMTS);
        }
        else
        {
          simple_scatter_one<ALIGNMENT,ALIGN_OFFSET>
            <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
            (d_odata, d_offsets, shared_buffer_size, BYTES_PER_ELMT, NUM_ELMTS);
        }
        break;
      }
    default:
      assert(false);
      break;
  }

  CUDA_SAFE_CALL(cudaThreadSynchronize());

  // Copy the data back
  CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, output_size*sizeof(float), cudaMemcpyDeviceToHost));

  // Check the result
  bool pass = true;
  for (int i = 0; i<NUM_ELMTS && pass; i++)
  {
    float out_value = ALIGN_OFFSET+i*BYTES_PER_ELMT/sizeof(float);
    int out_index = ALIGN_OFFSET+(h_offsets[i]*BYTES_PER_ELMT/sizeof(float));
    for (int j = 0; j<(BYTES_PER_ELMT/sizeof(float)); j++)
    {
      if (h_odata[out_index] != (out_value+float(j)))
      {
        fprintf(stderr,"\nIndex %d of element %d was expecting %f but received %f\n", j, i, out_value+float(j), h_odata[out_index]);  
        pass = false;
        break;
      }
    }
  }
  if (!pass)
  {
    fprintf(stdout,"Result - %s\n",(pass?"SUCCESS":"FAILURE"));
    fflush(stdout);
  }
  CUDA_SAFE_CALL(cudaFree(d_odata));
  CUDA_SAFE_CALL(cudaFree(d_offsets));
  free(h_odata);

  total_experiments++;

  return pass;
}

__host__
int main()
{
  // Generate a set of offsets that meet the necessary alignment
  int *offsets = (int*)malloc(PARAM_NUM_ELMTS*sizeof(int));
  int max_offset = -1;
  // Initialize the random number generator
  srand(PARAM_RAND_SEED);
  printf("Offsets: ");
#if 0
  for (int i = 0; i < PARAM_NUM_ELMTS; i++)
  {
    int off = (rand() % (1024)); 
    off *= (PARAM_ALIGNMENT);
    assert((off % PARAM_ALIGNMENT) == 0);
    offsets[i] = off;
    printf("%d ",off);
    // Keep max_offset in terms of number of floats
    if ((off/int(sizeof(float))) > max_offset)
      max_offset = off;
  }
#else
  // We have to check for overlaps
  {
    int tries = 0; 
    std::set<int> off_set;
    int cnt = 0;
    while (cnt < PARAM_NUM_ELMTS)
    {
      if (tries > MAX_TRIES)
      {
        // Give up if we can't find 
        // good offsets
        fprintf(stderr,"Warning: couldn't find good offsets, retry\n");
        fflush(stderr);
        return true;
      }
      tries++;
      int off = (rand() % (16384));
      off *= PARAM_ALIGNMENT;
      // Check if the offset is good
      bool good = true;
      for (std::set<int>::iterator it = off_set.begin();
            it != off_set.end(); it++)
      {
        if (!((off+PARAM_ELMT_SIZE) <= (*it)) && 
            !(((*it)+PARAM_ELMT_SIZE) <= off))
        {
          good = false;
          break;
        }
      }
      if (good)
      {
        offsets[cnt++] = off;
        printf("%d ", off);
        if ((off*PARAM_ELMT_SIZE/int(sizeof(float))) > max_offset)
          max_offset = off*PARAM_ELMT_SIZE/int(sizeof(float));
        off_set.insert(off);
      }
    }
  }
#endif
  printf("\n");
  assert(max_offset != -1);
  if (PARAM_SPECIALIZED)
    fprintf(stdout,"%s Warp-Specialized Experiment: ALIGNMENT-%2d OFFSET-%d ELMT_SIZE-%5d NUM_ELMTS-%2d DMA_WARPS-%2d NUM_TEMPLATES-%d ",(PARAM_GATHER?"GATHER":"SCATTER"),PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS/WARP_SIZE,PARAM_NUM_TEMPLATES); 
  else
    fprintf(stdout,"%s Non-Warp-Specialized Experiment: ALIGNMENT-%2d OFFSET-%d ELMT_SIZE-%5d NUM_ELMTS-%2d TOTAL_WARPS-%2d NUM_TEMPLATES-%d ",(PARAM_GATHER?"GATHER":"SCATTER"),PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS/WARP_SIZE,PARAM_NUM_TEMPLATES);
  fflush(stdout);
  bool result;
  if (PARAM_GATHER)
    result = run_gather_experiment<PARAM_SPECIALIZED,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS>(offsets, max_offset+PARAM_ELMT_SIZE/sizeof(float),PARAM_NUM_TEMPLATES);
  else
    result = run_gather_experiment<PARAM_SPECIALIZED,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS>(offsets, max_offset+PARAM_ELMT_SIZE/sizeof(float),PARAM_NUM_TEMPLATES);
  fprintf(stdout,"RESULT: %s\n",(result?"SUCCESS":"FAILURE"));
  fflush(stdout);

  free(offsets);

  return result;
}
