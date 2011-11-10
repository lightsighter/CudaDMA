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

#include "cuda.h"
#include "cuda_runtime.h"

//#define CUDADMA_DEBUG_ON
#include "cudaDMA.h"

#include "params_directed.h"

#define WARP_SIZE 32

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

  cudaDMAIndirect<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS,NUM_ELMTS>
    dma0 (1, num_compute_threads,
          num_compute_threads,
          offsets);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
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

  cudaDMAIndirect<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (1, num_compute_threads,
          num_compute_threads,
          offsets,
          num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
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

  cudaDMAIndirect<true,ALIGNMENT,BYTES_PER_ELMT>
    dma0 (1, num_dma_threads,
          num_compute_threads,
          num_compute_threads,
          offsets,
          num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
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

  cudaDMAIndirect<true,ALIGNMENT>
    dma0 (1, num_dma_threads,
          num_compute_threads,
          num_compute_threads,
          offsets,
          bytes_per_elmt,
          num_elmts);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
#ifdef CUDADMA_DEBUG_ON
    dma0.wait_for_dma_start();
    dma0.finish_async_dma();
#else
    dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
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
  int total_threads = (num_compute_warps)*WARP_SIZE + DMA_THREADS;

  switch (num_templates)
  {
    case 4:
      {
        dma_gather_four<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, NUM_ELMTS, DMA_THREADS>
          <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
          (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE); 
        break;
      }
    case 3:
      {
        dma_gather_three<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT, DMA_THREADS>
          <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
          (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE, NUM_ELMTS);
        break;
      }
    case 2:
      {
        dma_gather_two<ALIGNMENT, ALIGN_OFFSET, BYTES_PER_ELMT>
          <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
          (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS, NUM_ELMTS);
        break;
      }
    case 1:
      {
        dma_gather_one<ALIGNMENT,ALIGN_OFFSET>
          <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
          (d_idata, d_odata, d_offsets, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS, BYTES_PER_ELMT, NUM_ELMTS);
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
    int in_index = ALIGN_OFFSET+(h_offsets[i]/sizeof(float));
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

__host__
int main()
{
  // Generate a set of offsets that meet the necessary alignment
  int *offsets = (int*)malloc(PARAM_NUM_ELMTS*sizeof(int));
  int max_offset = -1;
  // Initialize the random number generator
  srand(PARAM_RAND_SEED);
  printf("Offsets: ");
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
  printf("\n");
  assert(max_offset != -1);
  fprintf(stdout,"Experiment: ALIGNMENT-%2d OFFSET-%d ELMT_SIZE-%5d NUM_ELMTS-%2d DMA_WARPS-%2d NUM_TEMPLATES-%d ",PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS/32,PARAM_NUM_TEMPLATES); 
  fflush(stdout);
  bool result = run_gather_experiment<PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_NUM_ELMTS,PARAM_DMA_THREADS>(offsets, max_offset+PARAM_ELMT_SIZE/sizeof(float),PARAM_NUM_TEMPLATES);
  fprintf(stdout,"RESULT: %s\n",(result?"SUCCESS":"FAILURE"));
  fflush(stdout);

  free(offsets);

  return result;
}
