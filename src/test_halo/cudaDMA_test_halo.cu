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

#define CUDADMA_DEBUG_ON
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

template<typename TYPE>
__host__ __device__ __forceinline__ void initialize(TYPE &t, float val) { }

template<>
__host__ __device__ __forceinline__ void initialize(float &t, float val) { t = val; }

template<>
__host__ __device__ __forceinline__ void initialize(float2 &t, float val) { t.x = val; t.y = val; }

template<>
__host__ __device__ __forceinline__ void initialize(float4 &t, float val) {t.x = val; t.y = val; t.z = val; t.w = val;}

template<typename TYPE>
__host__ __device__ __forceinline__ float get_val(TYPE &t) { return -1.0f; }

template<>
__host__ __device__ __forceinline__ float get_val(float &t) { return t; }

template<>
__host__ __device__ __forceinline__ float get_val(float2 &t) { return t.x; }

template<>
__host__ __device__ __forceinline__ float get_val(float4 &t) { return t.x; }


// I hate global variables, but whatever
long total_experiments = 0;

template<typename ELMT_TYPE, int RADIUS, bool CORNERS>
__host__ bool check_halo(ELMT_TYPE *idata, ELMT_TYPE *odata, int dimx, int dimy, int pitch)
{
	return false;
}

template<typename ELMT_TYPE>
__host__ void print_2d_array(ELMT_TYPE *data, int rows, int pitch)
{
	for (int i=0; i<rows; i++)
	{
		printf("\t");
		for (int j=0; j<pitch; j++)
		{
			printf("%1.0f ",get_val<ELMT_TYPE>(data[i*pitch+j]));
		}
		printf("\n");
	}	
}

template<int ALIGNMENT, int ALIGN_OFFSET, typename ELMT_TYPE, int RADIUS, bool CORNERS>
__global__ void __launch_bounds__(1024,1)
dma_ld_test ( ELMT_TYPE * idata, ELMT_TYPE * odata, int dimx, int dimy, int pitch, int buffer_size /*number of ELMT_TYPE*/, int num_compute_threads, int num_dma_threads_per_ld)
{
	extern __shared__ ELMT_TYPE buffer[];	

	cudaDMAHalo<ELMT_TYPE,RADIUS,CORNERS,ALIGNMENT>
	  dma0 (1, num_dma_threads_per_ld, num_compute_threads,
		num_compute_threads, dimx, dimy, pitch);

	if (dma0.owns_this_thread())
	{
		// Get the pointers to the origin
		//ELMT_TYPE *base_ptr = &(idata[ALIGN_OFFSET + RADIUS + RADIUS*pitch]);
		//dma0.execute_dma(base_ptr, &(buffer[ALIGN_OFFSET + RADIUS + RADIUS*(dimx+2*RADIUS)]));
		dma0.wait_for_dma_start();
		dma0.finish_async_dma();
	}
	else
	{
		// Zero out the buffer
		int iters = buffer_size/num_compute_threads;	
		int index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			initialize<ELMT_TYPE>(buffer[index],0.0f);
			index += num_compute_threads;
		}
		if (index < buffer_size)
			initialize<ELMT_TYPE>(buffer[index],0.0f);
		dma0.start_async_dma();
		dma0.wait_for_dma_finish();
		// Now read the buffer out of shared and write the results back
		index = threadIdx.x;
		for (int i=0; i<iters; i++)
		{
			ELMT_TYPE res = buffer[index];
			odata[index] = res;
			index += num_compute_threads;
		}
		if (index < buffer_size)
		{
			ELMT_TYPE res = buffer[index];
			odata[index] = res;
		}
	}
}	

template<int ALIGNMENT, int ALIGN_OFFSET, typename ELMT_TYPE, int RADIUS, bool CORNERS>
__host__ bool run_experiment(int dimx, int dimy, int pitch,
				int dma_warps)
{
	int shared_buffer_size = ((dimx+2*RADIUS)*(dimy+2*RADIUS) + ALIGN_OFFSET);
	// Check to see if we're using more shared memory than there is, if so return
	if ((shared_buffer_size*sizeof(ELMT_TYPE)) > 49152)
		return true;

	// Allocate the inpute data
	int input_size = (pitch*(dimy+2*RADIUS)+ALIGN_OFFSET);
	ELMT_TYPE *h_idata = (float*)malloc(input_size*sizeof(ELMT_TYPE));
	for (int i=0; i<input_size; i++)
		initialize<ELMT_TYPE>(h_idata[i],(rand()%10));
	// Allocate device memory and copy down
	float *d_idata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_idata, input_size*sizeof(ELMT_TYPE)));
	CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, input_size*sizeof(ELMT_TYPE), cudaMemcpyHostToDevice));

	// Allocate the output size
	float *h_odata = (float*)malloc(shared_buffer_size*sizeof(float));
	for (int i=0; i<shared_buffer_size; i++)
		initialize<ELMT_TYPE>(h_odata[i],0.0f);
	// Allocate device memory and copy down
	float *d_odata;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_odata, shared_buffer_size*sizeof(float)));
	CUDA_SAFE_CALL( cudaMemcpy( d_odata, h_odata, shared_buffer_size*sizeof(float), cudaMemcpyHostToDevice));

	int num_compute_warps = 1;
	int total_threads = (num_compute_warps + dma_warps)*WARP_SIZE;

	dma_ld_test<ALIGNMENT,ALIGN_OFFSET,ELMT_TYPE,RADIUS,CORNERS>
		<<<1,total_threads,shared_buffer_size*sizeof(ELMT_TYPE),0>>>
		(d_idata, d_odata, dimx, dimy, pitch, shared_buffer_size,
		 num_compute_warps*WARP_SIZE, dma_warps*WARP_SIZE);
	CUDA_SAFE_CALL( cudaThreadSynchronize());

	CUDA_SAFE_CALL( cudaMemcpy (h_odata, d_odata, shared_buffer_size*sizeof(float), cudaMemcpyDeviceToHost));

	// Print out the arrays
	printf("Source array:\n");
	print_2d_array<ELMT_TYPE>(&h_idata[ALIGN_OFFSET],dimy+2*RADIUS,pitch);
	printf("Result array:\n");
	print_2d_array<ELMT_TYPE>(&h_odata[ALIGN_OFFSET],dimy+2*RADIUS,dimx+2*RADIUS);

	// Check the result
	bool pass = check_halo<ELMT_TYPE, RADIUS, CORNERS>(h_idata, h_odata, dimx, dimy, pitch);
	//if (!pass)
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

	run_experiment<16,0,float,4,true>(32,8,64,1);

	// An 8192*sizeof(float) element is almost all of shared memory (can only fit one)
	// We probably want to test up to 32 elements per stride if we're going to have up to 16 dma warps
	//success16_0 = success16_0 && run_all_experiments<16,0>(8192,32,16);
	//success08_0 = success08_0 && run_all_experiments<8,0>(8192,32,16);
	//success08_2 = success08_2 && run_all_experiments<8,2>(8192,32,16);
	//success04_0 = success04_0 && run_all_experiments<4,0>(8192,32,16);
	//success04_1 = success04_1 && run_all_experiments<4,1>(8192,32,16);
	//success04_2 = success04_2 && run_all_experiments<4,2>(8192,32,16);
	//success04_3 = success04_3 && run_all_experiments<4,3>(8192,32,16);

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

