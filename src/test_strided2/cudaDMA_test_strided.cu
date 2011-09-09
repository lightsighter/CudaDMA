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

#include "../../include/cudaDMA.h"

#define WARP_SIZE 32

// includes, project

// includes, kernels

#define CUDA_SAFE_CALL(x)					\
	{							\
		cudaError_t err = (x);				\
		if (err != cudaSuccess)				\
		{						\
			printf("Cuda error: %s\n", cudaGetErrorString(err));	\
			exit(1);				\
		}						\
	}

// I hate global variables, but whatever
long total_experiments = 0;

template<int ALIGNMENT, int ALIGN_OFFSET>
__global__ void __launch_bounds__(1024,1)
dma_ld_test ( float *idata, float *odata, int element_size /*in number of floats*/, int num_elements, int src_stride/*bytes*/, int dst_stride/*bytes*/, int buffer_size /*number of floats*/, int num_compute_threads, int num_dma_threads_per_ld)
{
	extern __shared__ float buffer[];	

	cudaDMAStrided<ALIGNMENT>
	  dma0 (1, num_dma_threads_per_ld, num_compute_threads,
		num_compute_threads, element_size*sizeof(float),
		num_elements, src_stride*sizeof(float), dst_stride*sizeof(float));

	if (dma0.owns_this_thread())
	{
		float *base_ptr = &(idata[ALIGN_OFFSET]);
		dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET]));
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
__host__ bool run_experiment(int element_size, int element_count,
			 	int src_stride, int dst_stride,
				int dma_warps)
{
	// check some assertions
	assert(element_size <= src_stride);
	assert(element_size <= dst_stride);
	int shared_buffer_size = (element_count*dst_stride + ALIGN_OFFSET);
	// Check to see if we're using more shared memory than there is, if so return
	if ((shared_buffer_size*sizeof(float)) > 49152)
		return true;

	// Allocate the inpute data
	int input_size = (element_count*src_stride+ALIGN_OFFSET);
	float *h_idata = (float*)malloc(input_size*sizeof(float));
	for (int i=0; i<input_size; i++)
		h_idata[i] = float(i);
	// Allocate device memory and copy down
	float *d_idata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_idata, input_size*sizeof(float)));
	CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, input_size*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate the output size
	int output_size = (element_count*dst_stride+ALIGN_OFFSET);
	float *h_odata = (float*)malloc(output_size*sizeof(float));
	for (int i=0; i<output_size; i++)
		h_odata[i] = 0.0f;
	// Allocate device memory and copy down
	float *d_odata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_odata, output_size*sizeof(float)));
	CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, output_size*sizeof(float), cudaMemcpyHostToDevice));

	int num_compute_warps = 1;
	int total_threads = (num_compute_warps + dma_warps)*WARP_SIZE;

	dma_ld_test<ALIGNMENT,ALIGN_OFFSET>
		<<<1,total_threads,shared_buffer_size*sizeof(float),0>>>
		(d_idata, d_odata, element_size, element_count, src_stride,
		 dst_stride, shared_buffer_size, num_compute_warps*WARP_SIZE,
		 dma_warps*WARP_SIZE);
	CUDA_SAFE_CALL( cudaThreadSynchronize());

	CUDA_SAFE_CALL( cudaMemcpy (h_odata, d_odata, output_size*sizeof(float), cudaMemcpyDeviceToHost));

	// Check the result
	bool pass = true;
	for (int i=0; i<element_count && pass; i++)
	{
		int in_index = ALIGN_OFFSET+i*src_stride;
		int out_index = ALIGN_OFFSET+i*dst_stride;
		for (int j=0; j<element_size; j++)
		{
			//printf("%f ",h_odata[out_index+j]);
			if (h_idata[in_index+j] != h_odata[out_index+j])
			{
				fprintf(stdout,"Experiment: %ld element bytes, %d elements, %ld source stride, %ld destination stride, %d DMA warps, %d alignment, %d offset, ",element_size*sizeof(float),element_count,src_stride*sizeof(float),dst_stride*sizeof(float),dma_warps,ALIGNMENT,ALIGN_OFFSET);
				fprintf(stdout,"Index %d of element %d was expecting %f but received %f, ", j, i, h_idata[in_index+j], h_odata[out_index+j]);
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

	// Free up the remaining memory
	CUDA_SAFE_CALL( cudaFree(d_idata));
	CUDA_SAFE_CALL( cudaFree(d_odata));
	free(h_idata);
	free(h_odata);
	
	total_experiments++;
	
	return pass;
}

template<int ALIGNMENT, int ALIGN_OFFSET>
__host__
bool run_all_experiments(int max_element_size, int max_element_count,
			int max_dma_warps)
{
	bool pass = true;
	for (int element_size=1; element_size <= max_element_size; element_size++)
	{
	  fprintf(stdout,"Testing cases with element_size %ld - alignment %d - offset %d...\n",element_size*sizeof(float), ALIGNMENT, ALIGN_OFFSET);
	  fflush(stdout);
	  for (int element_count=1; element_count <= max_element_count; element_count++)
	  {
		// Get the initial source stride from the element size with the given alignment
		const int min_stride = element_size + (element_size%(ALIGNMENT/sizeof(float)) ? 
					((ALIGNMENT/sizeof(float))-(element_size%(ALIGNMENT/sizeof(float)))) : 0);
		// Let's only check full stride cases if element_size is divisible by 31 so we can
		// make the search space a little sparser and also test lots of potential strides
		// on weird alignment offsets
		if ((element_size%31)==0)
		{
			// Make each of the strides range from min_stride to 2*min_stride
			// This should cover all of the cases for a given element size
			// Anything larger is modulo equivalent to a smaller stride
			for (int src_stride=min_stride; src_stride <= (2*min_stride); src_stride += (ALIGNMENT/sizeof(float)))
			  for (int dst_stride=min_stride; dst_stride <= (2*min_stride); dst_stride += (ALIGNMENT/sizeof(float)))
			  {
				for (int dma_warps=1; dma_warps <= max_dma_warps; dma_warps++)
				{
					pass = pass && run_experiment<ALIGNMENT,ALIGN_OFFSET>(element_size, 
							element_count,src_stride,dst_stride,dma_warps);
				}
			  }
		}
		else
		{
			// Just test the variable number of dma_warps
			for (int dma_warps=1; dma_warps <= max_dma_warps; dma_warps++)
			{
				pass = pass && run_experiment<ALIGNMENT,ALIGN_OFFSET>(element_size,
						element_count,min_stride,min_stride,dma_warps);
			}
		}
	  }
	}
	return pass;
}

__host__
int main()
{
	bool success16_0 = true;
	bool success08_0 = true;
	bool success08_2 = true;
	bool success04_0 = true;
	bool success04_1 = true;
	bool success04_2 = true;
	bool success04_3 = true;
	// An 8192*sizeof(float) element is almost all of shared memory (can only fit one)
	// We probably want to test up to 32 elements per stride if we're going to have up to 16 dma warps
	success16_0 = success16_0 && run_all_experiments<16,0>(8192,32,16);
	success08_0 = success08_0 && run_all_experiments<8,0>(8192,32,16);
	success08_2 = success08_2 && run_all_experiments<8,2>(8192,32,16);
	success04_0 = success04_0 && run_all_experiments<4,0>(8192,32,16);
	success04_1 = success04_1 && run_all_experiments<4,1>(8192,32,16);
	success04_2 = success04_2 && run_all_experiments<4,2>(8192,32,16);
	success04_3 = success04_3 && run_all_experiments<4,3>(8192,32,16);

	fprintf(stdout,"\nResults:\n");
	fprintf(stdout,"\tAlignment16-Offset0: %s\n",(success16_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment08-Offset0: %s\n",(success08_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment08-Offset2: %s\n",(success08_2?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset0: %s\n",(success04_0?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset1: %s\n",(success04_1?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset2: %s\n",(success04_2?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\tAlignment04-Offset3: %s\n",(success04_3?"SUCCESS":"FAILURE"));
	fprintf(stdout,"\n\tTotal Experiments - %ld\n", total_experiments);
	return 0;
}

