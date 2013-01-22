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

#include "cudaDMAv2.h"

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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
special_xfer_three( float *idata, float *odata, int buffer_size, int num_compute_threads, const bool single, const bool qualified)
{
  extern __shared__ float buffer[];

  CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS>
    dma0 (1, num_compute_threads,
             num_compute_threads);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
    if (single)
    {
      if (qualified)
        dma0.template execute_dma<true,LOAD_CACHE_GLOBAL,STORE_CACHE_GLOBAL>(base_ptr, &(buffer[ALIGN_OFFSET])); 
      else
        dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
    }
    else
    {
      if (qualified)
      {
        dma0.template start_xfer_async<true,LOAD_CACHE_GLOBAL,STORE_CACHE_STREAMING>(base_ptr);
        dma0.template wait_xfer_finish<true,LOAD_CACHE_GLOBAL,STORE_CACHE_STREAMING>(&(buffer[ALIGN_OFFSET]));
      }
      else
      {
        dma0.start_xfer_async(base_ptr);
        dma0.wait_xfer_finish(&(buffer[ALIGN_OFFSET]));
      }
    }
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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
special_xfer_two( float *idata, float *odata, int buffer_size, int num_compute_threads, int num_dma_threads, const bool single, const bool qualified)
{
  extern __shared__ float buffer[];

  CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT>
    dma0 (1, num_dma_threads, num_compute_threads,
             num_compute_threads);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
    if (single)
    {
      if (qualified)
        dma0.template execute_dma<true,LOAD_CACHE_GLOBAL,STORE_CACHE_WRITE_THROUGH>(base_ptr, &(buffer[ALIGN_OFFSET])); 
      else
        dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
    }
    else
    {
      if (qualified)
      {
        dma0.template start_xfer_async<true,LOAD_CACHE_STREAMING,STORE_CACHE_GLOBAL>(base_ptr);
        dma0.template wait_xfer_finish<true,LOAD_CACHE_STREAMING,STORE_CACHE_GLOBAL>(&(buffer[ALIGN_OFFSET]));
      }
      else
      {
        dma0.start_xfer_async(base_ptr);
        dma0.wait_xfer_finish(&(buffer[ALIGN_OFFSET]));
      }
    }
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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD>
__global__ void __launch_bounds__(1024,1)
special_xfer_one( float *idata, float *odata, int buffer_size, int num_compute_threads, int num_dma_threads, int bytes_per_elmt, const bool single, const bool qualified)
{
  extern __shared__ float buffer[];

  CudaDMASequential<true,ALIGNMENT,BYTES_PER_THREAD>
    dma0 (1, num_dma_threads, num_compute_threads,
             num_compute_threads, bytes_per_elmt);

  if (dma0.owns_this_thread())
  {
    float *base_ptr = &(idata[ALIGN_OFFSET]);
    if (single)
    {
      if (qualified)
        dma0.template execute_dma<true,LOAD_CACHE_STREAMING,STORE_CACHE_STREAMING>(base_ptr, &(buffer[ALIGN_OFFSET])); 
      else
        dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
    }
    else
    {
      if (qualified)
      {
        dma0.template start_xfer_async<true,LOAD_CACHE_STREAMING,STORE_CACHE_WRITE_THROUGH>(base_ptr);
        dma0.template wait_xfer_finish<true,LOAD_CACHE_STREAMING,STORE_CACHE_WRITE_THROUGH>(&(buffer[ALIGN_OFFSET]));
      }
      else
      {
        dma0.start_xfer_async(base_ptr);
        dma0.wait_xfer_finish(&(buffer[ALIGN_OFFSET]));
      }
    }
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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
__global__ void __launch_bounds__(1024,1)
nonspec_xfer_three( float *idata, float *odata, int buffer_size, const bool single, const bool qualified)
{
  extern __shared__ float buffer[];

  CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS>
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
  if (single)
  {
    if (qualified)
      dma0.template execute_dma<true,LOAD_CACHE_LAST_USE,STORE_CACHE_GLOBAL>(base_ptr,&(buffer[ALIGN_OFFSET])); 
    else
      dma0.execute_dma(base_ptr,&(buffer[ALIGN_OFFSET]));
  }
  else
  {
    if (qualified)
    {
      dma0.template start_xfer_async<true,LOAD_CACHE_LAST_USE,STORE_CACHE_STREAMING>(base_ptr);
      dma0.template wait_xfer_finish<true,LOAD_CACHE_LAST_USE,STORE_CACHE_STREAMING>(&(buffer[ALIGN_OFFSET]));
    }
    else
    {
      dma0.start_xfer_async(base_ptr);
      dma0.wait_xfer_finish(&(buffer[ALIGN_OFFSET]));
    }
  }
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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD, int BYTES_PER_ELMT>
__global__ void __launch_bounds__(1024,1)
nonspec_xfer_two( float *idata, float *odata, int buffer_size, const bool single, const bool qualified)
{
  extern __shared__ float buffer[];

  CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD,BYTES_PER_ELMT>
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
  if (single)
  {
    if (qualified)
      dma0.template execute_dma<true,LOAD_CACHE_LAST_USE,STORE_CACHE_WRITE_THROUGH>(base_ptr,&(buffer[ALIGN_OFFSET])); 
    else
      dma0.execute_dma(base_ptr,&(buffer[ALIGN_OFFSET]));
  }
  else
  {
    if (qualified)
    {
      dma0.template start_xfer_async<true,LOAD_CACHE_VOLATILE,STORE_CACHE_GLOBAL>(base_ptr);
      dma0.template wait_xfer_finish<true,LOAD_CACHE_VOLATILE,STORE_CACHE_GLOBAL>(&(buffer[ALIGN_OFFSET]));
    }
    else
    {
      dma0.start_xfer_async(base_ptr);
      dma0.wait_xfer_finish(&(buffer[ALIGN_OFFSET]));
    }
  }
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

template<int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD>
__global__ void __launch_bounds__(1024,1)
nonspec_xfer_one( float *idata, float *odata, int buffer_size, int bytes_per_elmt, const bool single, const bool qualified)
{
  extern __shared__ float buffer[];

  CudaDMASequential<false,ALIGNMENT,BYTES_PER_THREAD>
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
  if (single)
  {
    if (qualified)
      dma0.template execute_dma<true,LOAD_CACHE_VOLATILE,STORE_CACHE_STREAMING>(base_ptr,&(buffer[ALIGN_OFFSET])); 
    else
      dma0.execute_dma(base_ptr,&(buffer[ALIGN_OFFSET]));
  }
  else
  {
    if (qualified)
    {
      dma0.template start_xfer_async<true,LOAD_CACHE_VOLATILE,STORE_CACHE_WRITE_THROUGH>(base_ptr);
      dma0.template wait_xfer_finish<true,LOAD_CACHE_VOLATILE,STORE_CACHE_WRITE_THROUGH>(&(buffer[ALIGN_OFFSET]));
    }
    else
    {
      dma0.start_xfer_async(base_ptr);
      dma0.wait_xfer_finish(&(buffer[ALIGN_OFFSET]));
    }
  }
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

template<bool SPECIALIZED, int ALIGNMENT, int ALIGN_OFFSET, int BYTES_PER_THREAD, int BYTES_PER_ELMT, int DMA_THREADS>
__host__ bool run_experiment(bool single, bool qualified, int num_templates)
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

  switch (num_templates)
  {
  case 3:
    {
    if (SPECIALIZED)
    {
      special_xfer_three<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, num_compute_warps*WARP_SIZE,single,qualified);
    }
    else
    {
      nonspec_xfer_three<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_THREAD,BYTES_PER_ELMT,DMA_THREADS>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size,single,qualified);
    }
    break;
    }
  case 2:
    {
    if (SPECIALIZED)
    {
      special_xfer_two<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_THREAD,BYTES_PER_ELMT>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS,single,qualified);
    }
    else
    {
      nonspec_xfer_two<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_THREAD,BYTES_PER_ELMT>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size,single,qualified);
    }
    break;
    }
  case 1:
    {
    if (SPECIALIZED)
    {
      special_xfer_one<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_THREAD>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, num_compute_warps*WARP_SIZE, DMA_THREADS, BYTES_PER_ELMT,single,qualified);
    }
    else
    {
      nonspec_xfer_one<ALIGNMENT,ALIGN_OFFSET,BYTES_PER_THREAD>
        <<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
        (d_idata, d_odata, shared_buffer_size, BYTES_PER_ELMT,single,qualified);
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
  {
    fprintf(stdout," - %s\n",(pass?"PASS":"FAIL"));
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
  bool result = true;
  fprintf(stdout,"Running all experiments for ALIGNMENT-%2d OFFSET-%d BYTES_PER_THREAD-%3d ELMT_SIZE-%5d DMA_WARPS-%2d\n",PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS/WARP_SIZE);
  fprintf(stdout,"  Warp-Specialized Experiments\n");
  fprintf(stdout,"    Single-Phase Experiments\n");
  fprintf(stdout,"      Unqualified experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,false/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,false/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,false/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"      Qualified Experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,true/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,true/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,true/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"    Two-Phase Experiments\n");
  fprintf(stdout,"      Unqualified experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,false/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,false/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,false/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"      Qualified Experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,true/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,true/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<true,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,true/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"  Non-Warp-Specialized Experiments\n");
  fprintf(stdout,"    Single-Phase Experiments\n");
  fprintf(stdout,"      Unqualified experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,false/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,false/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,false/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"      Qualified Experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,true/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,true/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(true/*single*/,true/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"    Two-Phase Experiments\n");
  fprintf(stdout,"      Unqualified experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,false/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,false/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,false/*qualified*/,3);
  if (!result) return result;
  fprintf(stdout,"      Qualified Experiments\n");
  fprintf(stdout,"        Templates-1");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,true/*qualified*/,1);
  if (!result) return result;
  fprintf(stdout,"        Templates-2");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,true/*qualified*/,2);
  if (!result) return result;
  fprintf(stdout,"        Templates-3");
  result = run_experiment<false,PARAM_ALIGNMENT,PARAM_OFFSET,PARAM_BYTES_PER_THREAD,PARAM_ELMT_SIZE,PARAM_DMA_THREADS>(false/*single*/,true/*qualified*/,3);
  if (!result) return result;
  fflush(stdout);

  return result;
}
