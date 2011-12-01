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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
dma_xfer_three( float *idata, float *odata, int buffer_size, int num_compute_threads)
{
  extern __shared__ float buffer[];

  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (1, num_compute_threads,
             num_compute_threads);

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
dma_xfer_two( float *idata, float *odata, int buffer_size, int num_compute_threads, int num_dma_threads)
{
  extern __shared__ float buffer[];

  cudaDMASequential<true,ALIGNMENT,BYTES_PER_ELMT>
    dma0 (1, num_dma_threads, num_compute_threads,
             num_compute_threads);

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
dma_xfer_one( float *idata, float *odata, int buffer_size, int num_compute_threads, int num_dma_threads, int bytes_per_elmt)
{
  extern __shared__ float buffer[];

  cudaDMASequential<true,ALIGNMENT>
    dma0 (1, num_dma_threads, num_compute_threads,
             num_compute_threads, bytes_per_elmt);

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
simple_xfer_three( float *idata, float *odata, int buffer_size )
{
  extern __shared__ float buffer[];

  cudaDMASequential<false,ALIGNMENT,BYTES_PER_ELMT,DMA_THREADS>
    dma0;

  // Zero out the buffer
  int iters = buffer_size/blockDim.x;
  int index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    buffer[index] = 0.0f;
    index += blockDim.x;
  }
  if (index < buffer_size)
    buffer[index] = 0.0f;
  __syncthreads();
  // Perform the transfer
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  // Now read the buffer out of shared and write the results back
  index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    odata[index] = buffer[index];
    index += blockDim.x;
  }
  if (index < buffer_size)
    odata[index] = buffer[index];
}

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
simple_xfer_two( float *idata, float *odata, int buffer_size)
{
  extern __shared__ float buffer[];

  cudaDMASequential<false,ALIGNMENT,BYTES_PER_ELMT>
    dma0;

  // Zero out the buffer
  int iters = buffer_size/blockDim.x;
  int index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    buffer[index] = 0.0f;
    index += blockDim.x;
  }
  if (index < buffer_size)
    buffer[index] = 0.0f;
  __syncthreads();
  // Perform the transfer
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  // Now read the buffer out of shared and write the results back
  index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    odata[index] = buffer[index];
    index += blockDim.x;
  }
  if (index < buffer_size)
    odata[index] = buffer[index];
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
simple_xfer_one( float *idata, float *odata, int buffer_size, int bytes_per_elmt)
{
  extern __shared__ float buffer[];

  cudaDMASequential<false,ALIGNMENT>
    dma0(bytes_per_elmt);

  // Zero out the buffer
  int iters = buffer_size/blockDim.x;
  int index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    buffer[index] = 0.0f;
    index += blockDim.x;
  }
  if (index < buffer_size)
    buffer[index] = 0.0f;
  __syncthreads();
  // perform the transfer
  float *base_ptr = &(idata[ALIGN_OFFSET]);
  dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
  __syncthreads();
  index = threadIdx.x;
  for (int i = 0; i<iters; i++)
  {
    odata[index] = buffer[index];
    index += blockDim.x;
  }
  if (index < buffer_size)
    odata[index] = buffer[index];
}

template<bool SPECIALIZED, int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_ELMT, int DMA_THREADS, int NUM_TEMPLATE_PARAMS>
__host__ bool run_experiment(void)
{
  int shared_buffer_size = (BYTES_PER_ELMT/sizeof(float)+ALIGN_OFFSET);
  if ((shared_buffer_size*sizeof(float)) > 41952)
    return true;

  // Allocate the input data
  int input_size = (BYTES_PER_ELMT/sizeof(float)+ALIGN_OFFSET);
  float *h_idata = (float*)malloc(input_size*sizeof(float));
  for (int i=0; i<input_size; i++)
    h_idata[i] = float(i);
  // Allocate device memory and copy down
  float *d_idata;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_idata, input_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_idata, h_idata, input_size*sizeof(float), cudaMemcpyHostToDevice));

  // Allocate the output size
  int output_size = (BYTES_PER_ELMT/sizeof(float)+ALIGN_OFFSET);
  float *h_odata = (float*)malloc(output_size*sizeof(float));
  for (int i=0; i<output_size; i++)
    h_odata[i] = 0.0f;
  // Allocate device memory and copy down
  float *d_odata;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, output_size*sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_odata, h_odata, output_size*sizeof(float),cudaMemcpyHostToDevice));

  int num_compute_warps = 1;
  int total_threads = 0;
  if (SPECIALIZED)
    total_threads = (num_compute_warps)*WARP_SIZE + DMA_THREADS;
  else
    total_threads = DMA_THREADS;
  assert(total_threads>0);

  switch (NUM_TEMPLATE_PARAMS)
  {
  case 3:
    {
    if (SPECIALIZED)
    {
      dma_xfer_three<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,DMA_THREADS>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, num_compute_warps*WARP_SIZE);
    }
    else
    {
      simple_xfer_three<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT,DMA_THREADS>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size);
    }
    break;
    }
  case 2:
    {
    if (SPECIALIZED)
    {
      dma_xfer_two<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS);
    }
    else
    {
      simple_xfer_two<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_ELMT>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size);
    }
    break;
    }
  case 1:
    {
    if (SPECIALIZED)
    {
      dma_xfer_one<ALIGNMENT,ALIGN_OFFSET>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS, BYTES_PER_ELMT);
    }
    else
    {
      simple_xfer_one<ALIGNMENT,ALIGN_OFFSET>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, BYTES_PER_ELMT);
    }
    break;
    }
  default:
    assert(false);
  }

  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, output_size*sizeof(float), cudaMemcpyDeviceToHost));

  // Check the result
  bool pass = true;
  int in_index = ALIGN_OFFSET;
  int out_index = ALIGN_OFFSET;
  for (int j = 0; j < (BYTES_PER_ELMT/sizeof(float)); j++)
  {
    if (h_idata[in_index+j] != h_odata[out_index+j])
    {
      fprintf(stderr,"Index %d was expecting %f but received %f\n", j, h_idata[in_index+j], h_odata[out_index+j]);
      pass = false;
      break;
    }
  }
  if (!pass)
  {
    fprintf(stdout,"Result - %s\n",(pass?"SUCCESS":"FAILURE"));
    fflush(stdout);
  }

  CUDA_SAFE_CALL(cudaFree(d_idata));
  CUDA_SAFE_CALL(cudaFree(d_odata));
  free(h_idata);
  free(h_odata);

  return pass;
}

__host__
int main()
{
  if (PARAM_SPECIALIZED)
    fprintf(stdout,"Warp-Specialized Experiment: ALIGNMENT-%2d OFFSET-%d ELMT_SIZE-%5d DMA_WARPS-%2d NUM_TEMPLATES-%d ",PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_DMA_THREADS/WARP_SIZE,PARAM_NUM_TEMPLATES);
  else
    fprintf(stdout,"Non-Warp-Specialized Experiment: ALIGNMENT-%2d OFFSET-%d ELMT_SIZE-%5d TOTAL_WARPS-%2d NUM_TEMPLATES-%d ",PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_DMA_THREADS/WARP_SIZE,PARAM_NUM_TEMPLATES);
  fflush(stdout);
  bool result = run_experiment<PARAM_SPECIALIZED,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_ELMT_SIZE,PARAM_DMA_THREADS,PARAM_NUM_TEMPLATES>();
  fprintf(stdout,"RESULT: %s\n",(result?"SUCCESS":"FAILURE"));
  fflush(stdout);

  return result;
}
